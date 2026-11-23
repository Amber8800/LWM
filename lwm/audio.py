import math
import os
import shutil
import subprocess
import uuid
import wave
from array import array
from pathlib import Path

RATE = 16000


def ffmpeg():
    exe = os.environ.get("LWM_FFMPEG") or shutil.which("ffmpeg")
    if exe:
        return exe
    from imageio_ffmpeg import get_ffmpeg_exe

    return get_ffmpeg_exe()


def frames(path):
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, RATE):
            raise ValueError("16 kHz mono PCM required")
        return wav.getnframes()


def decode(source, target):
    source, target = Path(source), Path(target)
    if source.suffix.lower() not in {".wav", ".mp3", ".m4a"}:
        raise ValueError("WAV, MP3, M4A 파일을 지정하세요.")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(uuid.uuid4().hex + ".wav")
    try:
        subprocess.run(
            [
                ffmpeg(),
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                "-i",
                str(source.resolve()),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(RATE),
                "-c:a",
                "pcm_s16le",
                str(tmp),
            ],
            check=True,
        )
        if not frames(tmp):
            raise ValueError("빈 오디오 파일입니다.")
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)


def pcm(wav, start, end):
    wav.setpos(start)
    return wav.readframes(end - start)


def chunks(path, seconds=600, pad=3):
    """Prefer quiet 100 ms around each cut. All offsets are PCM sample indices."""
    length = frames(path)
    start = 0
    with wave.open(str(path), "rb") as wav:
        while start < length:
            end = min(length, start + seconds * RATE)
            if end < length:
                step = RATE // 10
                radius = min(5 * RATE, seconds * RATE // 4)
                candidates = []
                for pos in range(
                    end - radius, min(length - step, end + radius) + 1, step
                ):
                    block = array("h", pcm(wav, pos, pos + step))
                    rms = math.sqrt(sum(x * x for x in block) / len(block))
                    candidates.append((rms, abs(end - pos), pos))
                if candidates:
                    end = min(candidates)[2]
            yield start, end, max(0, start - pad * RATE), min(length, end + pad * RATE)
            start = end
