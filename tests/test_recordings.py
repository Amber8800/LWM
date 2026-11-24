import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from lwm.common import save
from lwm.library import PURGE_DIR
from lwm.web import Jobs, LocalUI, ThreadingHTTPServer


class RecordingHistoryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.out = self.root / "runs/web/session"
        digest = "a" * 64
        save(self.out / "transcript.json", {
            "source": "audio.mp3", "audio_sha256": digest, "duration_s": 3.4,
            "stt": "komix-v2", "speaker_names": {"S1": "발언자"},
            "utterances": [{"speaker": "S1", "start_ms": 0, "text": "회의 시작"}],
        })
        self.wav = self.out.parent / ".work" / digest / "audio.wav"
        self.wav.parent.mkdir(parents=True)
        self.wav.write_bytes(b"0123456789")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), LocalUI)
        self.server.jobs = Jobs(self.root)
        self.server.jobs.library.edit(self.out, source="주간회의.mp3")
        self.base = f"http://127.0.0.1:{self.server.server_port}/api/recordings"
        self.client = build_opener(ProxyHandler({}))
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def get(self, suffix="", **headers):
        return self.client.open(Request(self.base + suffix, headers=headers))

    def update(self, suffix, **changes):
        request = Request(self.base + suffix, json.dumps(changes).encode(), {"X-LWM-Upload": "1"})
        with self.client.open(request) as response:
            return json.load(response)

    def test_reopen_rename_trash_restore_and_legacy_text(self):
        with self.get() as response:
            item, = json.load(response)
        self.assertEqual(item["title"], "주간회의.mp3")
        suffix = "/" + item["id"]
        self.assertEqual(self.update(suffix, title="주간 계획")["title"], "주간 계획")
        self.update(suffix, trashed=True)
        self.server.jobs = Jobs(self.root)
        with self.get() as response:
            reopened, = json.load(response)
        self.assertTrue(reopened["trashed"])
        self.assertEqual(reopened["title"], "주간 계획")
        self.assertFalse(self.update(suffix, trashed=False)["trashed"])
        with self.get(suffix + "/transcript") as response:
            self.assertIn("발언자  회의 시작", response.read().decode())
        with self.assertRaises(HTTPError) as error:
            self.update(suffix, title=" ")
        self.assertEqual(error.exception.code, 400)

    def test_audio_range_and_minutes(self):
        (self.out / "minutes.md").write_text("# 회의록", encoding="utf8")
        with self.get() as response:
            item, = json.load(response)
        suffix = "/" + item["id"]
        self.assertTrue(item["minutes"] and item["audio"])
        with self.get(suffix + "/audio", Range="bytes=3-5") as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.headers["Content-Range"], "bytes 3-5/10")
            self.assertEqual(response.read(), b"345")
        with self.get(suffix + "/minutes") as response:
            self.assertEqual(response.read().decode(), "# 회의록")

    def test_permanent_delete_requires_trash_and_only_removes_managed_outputs(self):
        for name in ("transcript.txt", "minutes.md", "minutes.json", "metrics.json", "my-notes.txt"):
            (self.out / name).write_text("saved", encoding="utf8")
        with self.get() as response:
            item, = json.load(response)
        suffix = "/" + item["id"]
        with self.assertRaises(HTTPError) as error:
            self.update(suffix, delete=True)
        self.assertEqual(error.exception.code, 409)
        self.update(suffix, trashed=True)
        self.assertTrue(self.update(suffix, delete=True)["deleted"])
        self.assertEqual([f.name for f in self.out.iterdir()], ["my-notes.txt"])
        self.assertTrue(self.wav.is_file())
        self.server.jobs = Jobs(self.root)
        with self.get() as response:
            self.assertEqual(json.load(response), [])
        with self.assertRaises(HTTPError) as error:
            self.get(suffix + "/transcript")
        self.assertEqual(error.exception.code, 404)

    def test_reopen_interrupted_delete(self):
        text = self.out / "transcript.txt"
        text.write_text("text", encoding="utf8")
        staging = self.out / PURGE_DIR
        staging.mkdir()
        (staging / ".pending").touch()
        text.replace(staging / text.name)
        with self.get() as response:
            self.assertEqual(len(json.load(response)), 1)
        self.assertEqual(text.read_text(), "text")
        staging.mkdir()
        (staging / ".pending").touch()
        text.replace(staging / text.name)
        (self.out / "transcript.json").replace(staging / "transcript.json")
        with self.get() as response:
            self.assertEqual(json.load(response), [])
        self.assertFalse(staging.exists())
