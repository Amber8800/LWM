import itertools
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from lwm.audio import RATE, chunks
from lwm.common import lock, read, save
from lwm.evaluate import pick, rates, speaker_scores, text_counts
from lwm.minutes import batches, consolidate, markdown, summarize, validate
from lwm.speech import windows
from lwm.transcript import UNKNOWN, attribute, stitch


def w(text, a, b):
    return {"text": text, "start_ms": a, "end_ms": b}


def t(name, a, b):
    return {"speaker": name, "start_ms": a, "end_ms": b}


class PipelineTests(unittest.TestCase):
    def test_turns_tie_overlap_and_unknown(self):
        turns = [t("A", 0, 100), t("B", 100, 210), t("C", 170, 210)]
        words = [
            w(" 첫", 0, 40),
            w(" 교대", 90, 110),
            w(" 다음", 115, 155),
            w(" 겹말", 165, 200),
            w(" 공백", 220, 250),
        ]
        result = attribute(words, turns)
        self.assertEqual(
            [u["speaker"] for u in result], ["A", UNKNOWN, "B", UNKNOWN, UNKNOWN]
        )
        self.assertTrue(result[3]["overlap"])
        self.assertEqual(result[3]["words"][0]["candidates"], ["B", "C"])

    def test_timestamp_ownership_dedup_preserves_genuine_repetition(self):
        first = {
            "core": [0, RATE],
            "words": [w(" 네", 880, 1070), w(" 밖", 1020, 1050)],
        }
        second = {
            "core": [RATE, RATE * 2],
            "words": [w(" 네", 935, 1090), w(" 네", 1140, 1280)],
        }
        self.assertEqual(len(stitch([first, second])), 2)
        self.assertEqual(stitch([first, second])[-1]["start_ms"], 1140)

    def test_two_hour_window_coverage(self):
        length = 2 * 3600 * RATE
        pieces = list(windows(length, 10 * RATE))
        self.assertEqual(pieces[0][1], 0)
        self.assertEqual(pieces[-1][2], length)
        self.assertTrue(all(a[2] == b[1] for a, b in itertools.pairwise(pieces)))
        self.assertTrue(
            all(start <= lo < hi <= start + 10 * RATE for start, lo, hi in pieces)
        )

    def test_short_stt_chunks(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "audio.wav"
            with wave.open(str(path), "wb") as out:
                out.setparams((1, 2, RATE, 0, "NONE", ""))
                out.writeframes(bytes(RATE * 2 * 5))
            parts = list(chunks(path, seconds=2, pad=1))
            self.assertEqual(parts[0][0], 0)
            self.assertEqual(parts[-1][1], 5 * RATE)
            self.assertTrue(all(a[1] == b[0] for a, b in itertools.pairwise(parts)))

    def test_lock_releases_after_exception(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "work.lock"
            with self.assertRaises(ValueError), lock(path):
                with self.assertRaises(RuntimeError), lock(path):
                    pass
                raise ValueError()
            with lock(path):
                save(Path(root) / "done", True)
            self.assertTrue(read(Path(root) / "done"))

    def test_technical_alias_and_spurious_occurrence(self):
        counts = text_counts(
            "오늘 GPU를 씁니다.", "오늘 지피유를 씁니다. GPU", [["GPU", "지피유"]]
        )
        self.assertEqual(counts["term_hits"], 1)
        self.assertEqual(counts["false_terms"], 1)
        self.assertEqual(rates(text_counts("가 나!", "가나", []))["cer"], 0)

    def test_version_and_acronym_boundaries(self):
        counts = text_counts(
            "CUDA 12.4와 CPU", "CUDA 12.4.1과 vCPU", [["CUDA 12.4"], ["CPU"]]
        )
        self.assertEqual(counts["terms"], 2)
        self.assertEqual(counts["term_hits"], 0)

    def test_choose_faster_only_within_both_margins(self):
        table = {
            "large-v3": {"cer": 0.08, "term_accuracy": 0.91, "rtf": 0.3},
            "komix-v2": {"cer": 0.084, "term_accuracy": 0.905, "rtf": 0.2},
        }
        self.assertEqual(pick(table), "komix-v2")
        table["komix-v2"]["term_accuracy"] = 0.89
        self.assertEqual(pick(table), "large-v3")

    def test_der_maps_anonymous_ids(self):
        def doc(label):
            turns = [t(label, 0, 1000)]
            return {
                "turns": turns,
                "utterances": attribute([w(" 안녕", 0, 1000)], turns),
            }

        scores = speaker_scores(doc("민수"), doc("SPEAKER_01"))
        self.assertEqual(scores["der"], 0)
        self.assertEqual(scores["word_speaker_accuracy"], 1)

    def test_evidence_owner_due(self):
        sources = {
            "U1": {
                "text": "제가 배포할게요.",
                "speaker": "SPEAKER_01",
                "overlap": False,
                "start_ms": 5000,
            }
        }
        item = {
            "kind": "할 일",
            "text": "배포",
            "owner": "SPEAKER_01",
            "due": "금요일",
            "evidence": [{"id": "U1", "quote": "제가 배포할게요."}],
        }
        items = validate([item], sources)
        self.assertEqual(items[0]["due"], "미정")
        self.assertIn(
            "### 민수",
            markdown(items, [{"id": "U1", **sources["U1"]}], {"SPEAKER_01": "민수"}),
        )
        self.assertEqual(
            validate(
                [{**item, "text": "제가", "evidence": [{"id": "U1", "quote": "제가"}]}],
                sources,
            ),
            [],
        )
        request = {
            **item,
            "text": "보고서 작성해 주세요.",
            "owner": "민수",
            "due": "목요일",
            "evidence": [{"id": "U2", "quote": "민수님은 보고서 작성해 주세요."}],
        }
        pledge = {
            **item,
            "text": "보고서 작성하겠습니다.",
            "owner": "미정",
            "due": "목요일",
            "evidence": [{"id": "U3", "quote": "제가 보고서 작성하겠습니다."}],
        }
        new_sources = {
            "U2": {"speaker": "SPEAKER_02", "overlap": False},
            "U3": {"speaker": "SPEAKER_01", "overlap": False},
        }
        merged = consolidate([request, pledge], new_sources, {"SPEAKER_01": "민수"})
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0]["evidence"]), 2)
        sources["U1"]["overlap"] = True
        self.assertEqual(validate([item], sources)[0]["owner"], "미정")
        with self.assertRaises(ValueError):
            validate([{**item, "evidence": [{"id": "U2", "quote": "임의"}]}], sources)

    def test_generator_selects_real_ids_and_copies_verbatim_evidence(self):
        import json
        from unittest.mock import Mock

        from lwm.minutes import LocalServer

        client = LocalServer(1, "test")
        client.prompt_tokens = Mock(return_value=[1, 2])
        source = {
            "U1": {"text": "금요일에 배포합니다.", "speaker": "A", "overlap": False}
        }
        item = {
            "kind": "결정 사항",
            "text": "금요일 배포",
            "owner": "미정",
            "due": "금요일",
            "evidence": ["U1"],
        }
        client.call = Mock(
            return_value={"content": json.dumps({"items": [item]}), "stop_type": "eos"}
        )
        result = client.generate("test", 128, source)
        self.assertEqual(
            result[0]["evidence"], [{"id": "U1", "quote": source["U1"]["text"]}]
        )
        schema = client.call.call_args.args[1]["json_schema"]
        self.assertEqual(
            schema["properties"]["items"]["items"]["properties"]["evidence"]["items"][
                "enum"
            ],
            ["U1"],
        )

    def test_context_and_hierarchical_summary_with_stub(self):
        from contextlib import nullcontext

        source = {
            "id": "U000001",
            "text": "다음 주에 검토하겠습니다.",
            "speaker": "A",
            "overlap": False,
        }
        item = {
            "kind": "주요 논의",
            "text": "다음 주 검토",
            "owner": "미정",
            "due": "미정",
            "evidence": [{"id": source["id"], "quote": source["text"]}],
        }

        class Stub:
            def tokens(self, prompt):
                return len(prompt)

            def generate(self, prompt, limit, sources):
                return [item]

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "transcript.json"
            save(path, {"utterances": [source]})
            job = {"transcript": str(path), "context": 4096, "cache": root}
            with patch("lwm.minutes.server", return_value=nullcontext(Stub())):
                self.assertEqual(summarize(job), [item])
        with self.assertRaises(ValueError):
            list(batches(Stub(), [source], 5))
