import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lwm.audio import RATE
from lwm.speech import transcribe


class RecoveryTests(unittest.TestCase):
    def test_mid_chunk_failure_resumes_only_unfinished_work(self):
        calls = []

        class Model:
            def transcribe(self, audio, **settings):
                calls.append(len(audio))
                if len(calls) == 2:
                    raise RuntimeError("simulated interruption")
                word = SimpleNamespace(start=0, end=0.6, word=" 배포")
                return [SimpleNamespace(words=[word])], None

        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "test.wav"
            with wave.open(str(source), "wb") as wav:
                wav.setparams((1, 2, RATE, 0, "NONE", ""))
                wav.writeframes(b"\x01\x00" * (2 * RATE))
            job = {
                "audio": str(source),
                "model": "test",
                "cache": root,
                "precision": "float16",
            }
            cores = [(0, RATE, 0, RATE), (RATE, 2 * RATE, RATE, 2 * RATE)]
            with (
                patch("faster_whisper.WhisperModel"),
                patch("faster_whisper.BatchedInferencePipeline", return_value=Model()),
                patch("lwm.speech.chunks", return_value=cores),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    transcribe(job)
                self.assertTrue((Path(root) / "chunk-0000.json").is_file())
                self.assertFalse((Path(root) / "chunk-0001.json").exists())
                result = transcribe(job)
                self.assertEqual(len(calls), 3)
                self.assertEqual([w["start_ms"] for w in result["words"]], [0, 1000])
