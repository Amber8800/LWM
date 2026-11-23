import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, delete=False, mode="w", encoding="utf-8", newline="\n"
    ) as stream:
        temp = Path(stream.name)
        try:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temp.unlink(missing_ok=True)
            raise
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def save(path, value):
    save_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def key(*parts):
    return hashlib.sha256(
        json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()[:24]


@contextmanager
def lock(path):
    """OS releases this lock even if the process crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"다른 작업이 실행 중입니다: {path.name}") from exc
        yield


def offline():
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        PYANNOTE_METRICS_ENABLED="0",
        DO_NOT_TRACK="1",
        SPEECHBRAIN_LOCAL_STRATEGY="copy",
    )


def stamp(ms):
    seconds = ms // 1000
    return f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"
