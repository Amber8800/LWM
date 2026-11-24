"""Persistent recording titles and trash state alongside local transcript outputs."""

import os
import re
import threading
from pathlib import Path

from .common import key, read, save, stamp

# Move the discovery file last. Its presence in staging commits the deletion.
ARTIFACTS = ("transcript.txt", "minutes.md", "minutes.json", "summary-metrics.json",
             "metrics.json", ".lwm-library.json", "transcript.json")
PURGE_DIR = ".lwm-recording-purge"


class RecordingLibrary:
    def __init__(self, runs):
        self.runs = runs.resolve()
        self.lock = threading.RLock()
        self.folders = {}
        self.headers = {}

    def listing(self):
        with self.lock:
            folders = {}
            records = []
            for parent, dirs, files in os.walk(self.runs):
                folder = Path(parent)
                if PURGE_DIR in dirs:
                    try:
                        self.finish_purge(folder)
                    except OSError:
                        pass
                    if (folder / "transcript.json").is_file() and "transcript.json" not in files:
                        files.append("transcript.json")
                dirs[:] = [d for d in dirs if not d.startswith(".")
                           and (folder / d).resolve() == folder / d]
                if "transcript.json" not in files:
                    continue
                try:
                    record = self.record(folder)
                except (OSError, ValueError, KeyError, TypeError):
                    continue
                folders[record["id"]] = folder
                records.append(record)
            self.folders = folders
            return sorted(records, key=lambda item: item["date"], reverse=True)

    def folder(self, record_id):
        with self.lock:
            if record_id not in self.folders:
                self.listing()
            folder = self.folders.get(record_id)
            if folder is None or folder.resolve() != folder:
                raise FileNotFoundError
            return folder

    @staticmethod
    def local_file(folder, name):
        path = folder / name
        if path.resolve() != path or not path.is_file():
            raise FileNotFoundError
        return path

    def audio(self, folder):
        fingerprint = self.header(folder)["fingerprint"]
        if re.fullmatch("[0-9a-f]{64}", fingerprint or ""):
            try:
                return self.local_file(folder.parent / ".work" / fingerprint, "audio.wav")
            except FileNotFoundError:
                pass

    @staticmethod
    def metadata(folder):
        path = folder / ".lwm-library.json"
        return read(path) if path.is_file() and not path.is_symlink() else {}

    def header(self, folder):
        path = self.local_file(folder, "transcript.json")
        stat = path.stat()
        cached = self.headers.get(folder)
        if not cached or cached[0] != stat.st_mtime_ns:
            doc = read(path)
            if not isinstance(doc, dict) or not isinstance(doc.get("utterances"), list):
                raise ValueError("녹취록 형식 오류")
            cached = (stat.st_mtime_ns, {
                "source": doc.get("source") or "", "model": doc.get("stt", ""),
                "date": stat.st_mtime,
                "duration": doc.get("duration_s"), "diarizer": doc.get("diarizer"),
                "fingerprint": doc.get("audio_sha256"),
            })
            self.headers[folder] = cached
        return cached[1]

    def record(self, folder):
        header = self.header(folder)
        meta = self.metadata(folder)
        source = meta.get("source") or header["source"]
        return {
            "id": key(folder.relative_to(self.runs).as_posix()),
            "title": meta.get("title") or source or "회의 녹음", "source": source,
            "model": header["model"], "duration": header["duration"],
            "diarizer": header["diarizer"], "date": header["date"],
            "trashed": meta.get("trashed", False),
            "minutes": (folder / "minutes.md").is_file(),
            "audio": bool(self.audio(folder)),
        }

    def edit(self, folder, **changes):
        with self.lock:
            self.local_file(folder, "transcript.json")
            meta = self.metadata(folder)
            meta.update(changes)
            save(folder / ".lwm-library.json", meta)

    @staticmethod
    def finish_purge(folder):
        staging = folder / PURGE_DIR
        if staging.resolve() != staging:
            return
        if not (staging / ".pending").is_file():
            if staging.is_dir() and not any(staging.iterdir()):
                staging.rmdir()
            return
        committed = (staging / "transcript.json").is_file()
        for name in ARTIFACTS:
            staged = staging / name
            if not staged.is_file() and not staged.is_symlink():
                continue
            if committed or (folder / name).exists():
                staged.unlink()
            else:
                staged.replace(folder / name)
        (staging / ".pending").unlink()
        staging.rmdir()

    def purge(self, record_id):
        with self.lock:
            folder = self.folder(record_id)
            self.local_file(folder, "transcript.json")
            if not self.metadata(folder).get("trashed"):
                raise ValueError("휴지통에 있는 녹취만 영구 삭제할 수 있습니다.")
            staging = folder / PURGE_DIR
            if staging.exists():
                self.finish_purge(folder)
            staging.mkdir()
            (staging / ".pending").touch()
            try:
                for name in ARTIFACTS:
                    path = folder / name
                    if path.is_file() or path.is_symlink():
                        path.replace(staging / name)
            finally:
                self.finish_purge(folder)
            self.folders.pop(record_id, None)
            self.headers.pop(folder, None)

    def content(self, folder, kind):
        if kind == "minutes":
            return self.local_file(folder, "minutes.md").read_bytes()
        try:
            return self.local_file(folder, "transcript.txt").read_bytes()
        except FileNotFoundError:
            doc = read(self.local_file(folder, "transcript.json"))
            names = doc.get("speaker_names", {})
            lines = (f"[{stamp(u['start_ms'])}] {names.get(u['speaker'], u['speaker'])}  {u['text']}\n"
                     for u in doc["utterances"])
            return "".join(lines).encode()
