import re
import statistics
import tempfile
import unicodedata
from pathlib import Path

from .audio import RATE, frames
from .common import digest, offline, read, save
from .models import ASR
from .runner import asr_job, audio_cache, gpu_lock, model_job, target_gpu
from .transcript import UNKNOWN, attribute


def normal(text, spaces=False):
    text = unicodedata.normalize("NFC", text)
    chars = [
        " " if c.isspace() else c for c in text if unicodedata.category(c)[0] != "P"
    ]
    if spaces:
        return " ".join("".join(chars).split())
    return "".join(c for c in chars if c != " ")


def text_counts(ref, hyp, terms):
    from rapidfuzz.distance import Levenshtein

    result = {}
    for suffix, spaces in [("", False), ("_spaced", True)]:
        a, b = normal(ref, spaces), normal(hyp, spaces)
        result["edits" + suffix] = Levenshtein.distance(a, b)
        result["chars" + suffix] = len(a)
    # Canonical markers keep equivalent spellings aligned at their occurrence,
    # including repeated mentions. Preserve version punctuation for term scoring.
    aliases = {}
    for i, spellings in enumerate(terms):
        if not isinstance(spellings, list) or not spellings:
            raise ValueError(
                "각 기술 용어를 하나 이상의 허용 표기를 담은 배열로 지정하세요."
            )
        for alias in spellings:
            alias = unicodedata.normalize("NFC", alias).casefold().strip()
            if not alias or alias in aliases:
                raise ValueError("용어 허용 표기는 비어 있거나 중복될 수 없습니다.")
            aliases[alias] = chr(0xF0000 + i)

    def mark(text):
        text = unicodedata.normalize("NFC", text).casefold()
        if not aliases:
            return text
        patterns = []
        for alias in sorted(aliases, key=len, reverse=True):
            left = (
                r"(?<![a-z0-9_])" if alias[0].isascii() and alias[0].isalnum() else ""
            )
            right = (
                r"(?![a-z0-9_])" if alias[-1].isascii() and alias[-1].isalnum() else ""
            )
            if alias[-1].isdigit():
                right += r"(?!\.\d)"
            patterns.append(left + re.escape(alias) + right)
        pattern = "|".join(patterns)
        return re.sub(pattern, lambda m: aliases[m.group()], text)

    a, b = mark(ref), mark(hyp)
    symbols = set(aliases.values())
    hits = sum(
        c in symbols
        for tag, lo, hi, _, _ in Levenshtein.opcodes(a, b)
        if tag == "equal"
        for c in a[lo:hi]
    )
    result.update(
        terms=sum(c in symbols for c in a),
        term_hits=hits,
        false_terms=sum(c in symbols for c in b) - hits,
    )
    return result


def ratio(a, b):
    return a / b if b else None


def rates(counts):
    return {
        "cer": ratio(counts["edits"], counts["chars"]),
        "cer_spaced": ratio(counts["edits_spaced"], counts["chars_spaced"]),
        "term_accuracy": ratio(counts["term_hits"], counts["terms"]),
    }


def pick(table):
    best = min(ASR, key=lambda m: table[m]["cer"])
    ref = table[best]
    candidates = [
        m
        for m in ASR
        if table[m]["cer"] <= ref["cer"] + 0.005
        and (
            ref["term_accuracy"] is None
            or table[m]["term_accuracy"] >= ref["term_accuracy"] - 0.01
        )
    ]
    return min(candidates, key=lambda m: (table[m]["rtf"], table[m]["cer"]))


def benchmark(args):
    offline()
    manifest = Path(args.dataset).resolve()
    data = read(manifest)
    meetings = data["meetings"]
    ids = [m["id"] for m in meetings]
    if len(ids) != len(set(ids)):
        raise ValueError("회의 ID 중복: 한 회의는 한 분할에만 포함하세요.")
    selected = [m for m in meetings if m["split"] == args.split]
    if not selected or args.repeats < 1:
        raise ValueError("평가 회의 및 반복 횟수를 확인하세요.")
    gpu = target_gpu(args.gpu)
    records, baselines = [], []
    # Reject identical recordings even if the manifest renames them across splits.
    processed = {
        digest(manifest.parent / m["audio"])
        for m in meetings
        if m["split"] != args.split
    }
    with gpu_lock(gpu):
        models = {m: model_job(args.models, m) for m in ASR}
        for meeting in selected:
            audio, identity = audio_cache(
                manifest.parent / meeting["audio"], args.output
            )
            if identity in processed:
                raise ValueError("평가 회의 오디오 중복")
            processed.add(identity)
            ref = meeting["reference"]
            terms = meeting.get("terms", data.get("terms", []))
            seconds = frames(audio) / RATE
            if "clova_text" in meeting:
                counts = text_counts(ref, meeting["clova_text"], terms)
                baselines.append({"id": meeting["id"], **counts, **rates(counts)})
            for model in ASR:
                trials = []
                for _ in range(args.repeats):
                    # Each trial loads a fresh worker and cannot reuse ASR chunks.
                    with tempfile.TemporaryDirectory(dir=audio.parent) as tmp:
                        result, stats = asr_job(
                            audio,
                            Path(tmp),
                            models[model],
                            gpu,
                            args.precision,
                            args.prompt,
                        )
                    hyp = "".join(w["text"] for w in result["words"]).strip()
                    counts = text_counts(ref, hyp, terms)
                    trials.append({**stats, **counts, **rates(counts), "text": hyp})
                representative = sorted(trials, key=lambda r: r["elapsed_s"])[
                    len(trials) // 2
                ]
                records.append(
                    {
                        "id": meeting["id"],
                        "model": model,
                        "duration_s": seconds,
                        "counts": {k: representative[k] for k in counts},
                        "median_s": statistics.median(t["elapsed_s"] for t in trials),
                        "trials": trials,
                    }
                )
    table = {}
    for model in ASR:
        group = [r for r in records if r["model"] == model]
        totals = {k: sum(r["counts"][k] for r in group) for k in group[0]["counts"]}
        if not totals["chars"]:
            raise ValueError("평가셋에는 발화가 있는 정답 전사가 필요합니다.")
        table[model] = {
            **totals,
            **rates(totals),
            "rtf": sum(r["median_s"] for r in group)
            / sum(r["duration_s"] for r in group),
            "sampled_peak_mib": max(
                t["sampled_peak_mib"] for r in group for t in r["trials"]
            ),
        }
    report = {
        "split": args.split,
        "gpu": gpu,
        "precision": args.precision,
        "prompt": args.prompt,
        "repeats": args.repeats,
        "meeting_count": len(selected),
        "table": table,
        "selected": pick(table),
        "meetings": records,
        "clova": baselines,
        "model_fingerprints": {m: v[1] for m, v in models.items()},
    }
    path = Path(args.output) / "benchmark.json"
    save(path, report)
    return path


def speaker_scores(reference, hypothesis):
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate
    from rapidfuzz.distance import Levenshtein

    def annotation(turns):
        out = Annotation()
        for i, t in enumerate(turns):
            if t["end_ms"] > t["start_ms"] and t["speaker"] != UNKNOWN:
                out[Segment(t["start_ms"] / 1000, t["end_ms"] / 1000), i] = t["speaker"]
        return out

    ref, hyp = annotation(reference["turns"]), annotation(hypothesis["turns"])
    metric = DiarizationErrorRate(collar=0, skip_overlap=False)
    details = metric(ref, hyp, detailed=True)
    mapping = metric.optimal_mapping(ref, hyp)
    result = {
        "der": float(details[metric.name]),
        "components_s": {k: float(v) for k, v in details.items() if k != metric.name},
    }

    def words(doc):
        return [w for u in doc["utterances"] for w in u["words"] if normal(w["text"])]

    a, b = words(reference), words(hypothesis)
    groups = {name: [0, 0, 0] for name in ("all", "overlap", "short_switch")}
    for tag, lo, hi, other, _ in Levenshtein.opcodes(
        [normal(w["text"]) for w in a], [normal(w["text"]) for w in b]
    ):
        if tag != "equal":
            continue
        for i in range(hi - lo):
            pos = lo + i
            word, guess = a[pos], b[other + i]["speaker"]
            if word["speaker"] == UNKNOWN:
                continue
            labels = ["all"]
            if word.get("overlap"):
                labels.append("overlap")
            if (
                pos
                and a[pos - 1]["speaker"] != word["speaker"]
                and word["end_ms"] - word["start_ms"] <= 500
            ):
                labels.append("short_switch")
            for label in labels:
                counts = groups[label]
                counts[0] += 1
                counts[1] += guess != UNKNOWN and mapping.get(guess) == word["speaker"]
                counts[2] += guess == UNKNOWN

    def summary(counts):
        aligned, correct, unknown = counts
        return {
            "aligned_words": aligned,
            "word_speaker_accuracy": ratio(correct, aligned),
            "unassigned_rate": ratio(unknown, aligned),
        }

    result.update(summary(groups.pop("all")))
    result["subsets"] = {k: summary(v) for k, v in groups.items()}
    # With reference word times, isolate diarization/attribution error from STT drift.
    oracle = attribute(a, hypothesis["turns"])
    oracle_words = [w for u in oracle for w in u["words"]]
    known = [(r, h) for r, h in zip(a, oracle_words) if r["speaker"] != UNKNOWN]
    result["reference_timing_speaker_accuracy"] = ratio(
        sum(
            h["speaker"] != UNKNOWN and mapping.get(h["speaker"]) == r["speaker"]
            for r, h in known
        ),
        len(known),
    )
    return result
