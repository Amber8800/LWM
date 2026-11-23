"""NVIDIA NeMo-Speech.cpp 0.3.0 diarization ABI, isolated in a GPU worker."""

import ctypes as ct
import os
import wave
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

from .audio import RATE
from .models import NEMOTRON
from .progress import Reporter

RUNTIME = "nemo-speech-0.3.0-windows-x86_64-cuda"


class ModelConfig(ct.Structure):
    _fields_ = [
        ("size", ct.c_size_t),
        ("model_path", ct.c_char_p),
        ("gpu", ct.c_int32),
        ("preset", ct.c_char_p),
        ("chunk", ct.c_int32),
        ("right", ct.c_int32),
        ("left", ct.c_int32),
        ("fifo", ct.c_int32),
        ("cache", ct.c_int32),
        ("update", ct.c_int32),
    ]


class Segment(ct.Structure):
    _fields_ = [("start", ct.c_double), ("end", ct.c_double), ("speaker", ct.c_int32)]


def bind(dll):
    handle = ct.c_void_p
    output = ct.POINTER(handle)
    size = ct.c_size_t
    signatures = {
        "create": (ct.c_int, [ct.POINTER(ModelConfig), output]),
        "destroy": (None, [handle]),
        "stream_open": (ct.c_int, [handle, output]),
        "stream_close": (None, [handle]),
        "stream_push_f32": (
            ct.c_int,
            [handle, ct.POINTER(ct.c_float), size, ct.c_int32],
        ),
        "stream_finish": (ct.c_int, [handle]),
        "frame_count": (ct.c_int64, [handle]),
        "seconds_per_frame": (ct.c_double, [handle]),
        "segments": (
            ct.c_int,
            [handle, handle, ct.POINTER(Segment), size, ct.POINTER(size)],
        ),
    }
    for name, (result, args) in signatures.items():
        fn = getattr(dll, "nemo_speech_diar_" + name)
        fn.restype, fn.argtypes = result, args
    dll.nemo_speech_asr_last_error.restype = ct.c_char_p
    dll.nemo_speech_asr_last_error.argtypes = []


def turns_from_segments(segments, duration_ms):
    edges = defaultdict(list)
    for seg in segments:
        start = max(0, round(seg.start * 1000))
        end = min(duration_ms, round(seg.end * 1000))
        if start < end:
            edges[start].append((seg.speaker, 1))
            edges[end].append((seg.speaker, -1))
    active = Counter()
    turns = []
    previous = 0
    for time in sorted(edges):
        for speaker in sorted(active):
            turns.append(
                {
                    "start_ms": previous,
                    "end_ms": time,
                    "speaker": f"SPEAKER_{speaker:02}",
                    "overlap": len(active) > 1,
                }
            )
        for speaker, delta in edges[time]:
            active[speaker] += delta
            if not active[speaker]:
                del active[speaker]
        previous = time
    return turns


def diarize(job):
    import numpy as np

    binary = (
        Path(__file__).resolve().parent.parent
        / ".tools"
        / "nemo-speech"
        / RUNTIME
        / "bin"
    )
    library = binary / "nemo_speech_asr_c.dll"
    if not library.is_file():
        raise RuntimeError("Nemotron CUDA 런타임을 설치하세요: .\\setup-nemotron.ps1")
    progress = Reporter("nemotron", "samples")
    with ExitStack() as stack:
        stack.enter_context(os.add_dll_directory(str(binary)))
        dll = ct.CDLL(str(library))
        bind(dll)

        def check(status):
            if status:
                error = dll.nemo_speech_asr_last_error() or b"Unknown runtime error"
                raise RuntimeError(error.decode("utf-8", errors="replace"))

        config = ModelConfig(
            size=ct.sizeof(ModelConfig),
            gpu=0,
            left=-1,
            preset=b"v3-offline",
            model_path=str(Path(job["model"]) / NEMOTRON).encode(),
        )
        model, stream = ct.c_void_p(), ct.c_void_p()
        check(dll.nemo_speech_diar_create(ct.byref(config), ct.byref(model)))
        stack.callback(dll.nemo_speech_diar_destroy, model)
        check(dll.nemo_speech_diar_stream_open(model, ct.byref(stream)))
        stack.callback(dll.nemo_speech_diar_stream_close, stream)
        wav = stack.enter_context(wave.open(job["audio"], "rb"))
        total = wav.getnframes()
        seconds = dll.nemo_speech_diar_seconds_per_frame(model)
        progress(0, total)
        while block := wav.readframes(5 * RATE):
            audio = np.frombuffer(block, dtype="<i2").astype("float32") / 32768
            check(
                dll.nemo_speech_diar_stream_push_f32(
                    stream,
                    audio.ctypes.data_as(ct.POINTER(ct.c_float)),
                    len(audio),
                    RATE,
                )
            )
            done = round(dll.nemo_speech_diar_frame_count(stream) * seconds * RATE)
            progress(min(done, total), total)
        check(dll.nemo_speech_diar_stream_finish(stream))
        count = ct.c_size_t()
        check(dll.nemo_speech_diar_segments(stream, None, None, 0, ct.byref(count)))
        segments = (Segment * count.value)()
        check(
            dll.nemo_speech_diar_segments(
                stream, None, segments, len(segments), ct.byref(count)
            )
        )
        progress(total, total)
        return turns_from_segments(segments, round(total * 1000 / RATE))
