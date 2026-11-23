import csv
import os
import subprocess
import sys
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .audio import RATE, decode, frames
from .common import digest, key, lock, offline, read, save, save_text, stamp
from .minutes import markdown
from .models import inventory
from .progress import Reporter, compatible_digest
from .transcript import attribute


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


PIPELINE_VERSION = key(
    [
        compatible_digest(digest(Path(__file__).with_name(f)))
        for f in ("audio.py", "speech.py", "transcript.py", "minutes.py", "runner.py")
    ],
    [
        (p, package_version(p))
        for p in (
            "ctranslate2",
            "faster-whisper",
            "pyannote.audio",
            "speechbrain",
            "torch",
        )
    ],
)


def gpu_info():
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [
        dict(zip(("name", "uuid", "memory_mib", "driver"), (s.strip() for s in row)))
        for row in csv.reader(result.splitlines())
    ]


def target_gpu(uuid=None):
    found = [
        g
        for g in gpu_info()
        if (g["uuid"] == uuid if uuid else "4070 Ti SUPER" in g["name"])
    ]
    if len(found) != 1:
        raise ValueError("사용할 GPU의 UUID를 --gpu로 지정하세요.")
    return found[0]


def gpu_lock(gpu):
    return lock(Path(tempfile.gettempdir()) / f"lwm-{gpu['uuid']}.lock")


def run_worker(stage, folder, job, gpu):
    with lock(folder / "stage.lock"):
        return _run_worker(stage, folder, job, gpu)


def _run_worker(stage, folder, job, gpu):
    result, stats = folder / "result.json", folder / "metrics.json"
    if result.exists() and stats.exists():
        Reporter(stage, "cached")()
        return read(result), {**read(stats), "cached": True}
    save(folder / "job.json", {**job, "cache": str(folder)})
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=gpu["uuid"],
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        PYTHONUTF8="1",
        MPLCONFIGDIR=str(folder.parent / ".mpl"),
        HF_HOME=str(folder.parent / ".offline-hub"),
    )
    start = time.perf_counter()
    print(f"[{stage}] {gpu['name']}", file=sys.stderr, flush=True)
    log_path = folder / "worker.log"
    with (
        log_path.open("wb") as log,
        subprocess.Popen(
            [sys.executable, "-m", "lwm.worker", stage, str(folder)],
            env=env,
            stderr=log,
        ) as process,
    ):
        peak = 0
        try:
            while process.poll() is None:
                usage = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        gpu["uuid"],
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                peak = max(peak, int(usage.strip()))
                try:
                    process.wait(timeout=0.25)
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode:
                log.flush()
                tail = log_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()[-3:]
                raise RuntimeError(
                    f"{stage} 실행 실패 ({process.returncode}). {log_path}\n"
                    + "\n".join(tail)
                )
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
    data = read(result)
    metrics = {
        "elapsed_s": time.perf_counter() - start,
        "sampled_peak_mib": peak,
        "gpu": gpu,
        "cached": False,
    }
    if stage == "asr":
        metrics.update({k: data[k] for k in ("load_s", "inference_s")})
    save(stats, metrics)
    return data, metrics


def audio_cache(source, out):
    source = Path(source).resolve()
    identity = digest(source)
    folder = Path(out).resolve() / ".work" / identity
    path = folder / "audio.wav"
    with lock(folder / "decode.lock"):
        if not path.is_file():
            decode(source, path)
    return path, identity


def model_job(models, name):
    path, record = inventory(models, name)
    return str(path), key(record)


def asr_job(audio, folder, model, gpu, precision, prompt):
    directory, fingerprint = model
    job = {
        "audio": str(audio),
        "model": directory,
        "precision": precision,
        "prompt": prompt,
    }
    return run_worker(
        "asr", folder / key(PIPELINE_VERSION, fingerprint, precision, prompt), job, gpu
    )


def names_from(path):
    names = read(path) if path else {}
    if not isinstance(names, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v.strip()
        for k, v in names.items()
    ):
        raise ValueError("이름표는 화자 ID와 이름의 JSON 객체여야 합니다.")
    return names


def summarize_file(transcript, out, models, gpu, context, names):
    doc = read(transcript)
    root = Path(out).resolve()
    if doc["utterances"]:
        folder, model_id = model_job(models, "minutes")
        work = (
            root
            / ".work"
            / key(PIPELINE_VERSION, digest(transcript), model_id, context, names)
        )
        items, metrics = run_worker(
            "minutes",
            work,
            {
                "model": folder,
                "context": context,
                "names": names,
                "transcript": str(Path(transcript).resolve()),
            },
            gpu,
        )
    else:
        items, metrics = [], {"elapsed_s": 0, "sampled_peak_mib": 0, "cached": False}
    root.mkdir(parents=True, exist_ok=True)
    save(root / "minutes.json", items)
    save_text(root / "minutes.md", markdown(items, doc["utterances"], names))
    return metrics


def community_turns(args, audio, work, gpu, times, threshold):
    seg, seg_id = model_job(args.models, "segment")
    emb, emb_id = model_job(args.models, "embed")
    seg_dir = work / key(PIPELINE_VERSION, seg_id)
    _, times["segment"] = run_worker(
        "segment", seg_dir, {"audio": str(audio), "segment": seg}, gpu
    )
    diar_dir = work / key(
        PIPELINE_VERSION,
        seg_id,
        emb_id,
        threshold,
        args.speakers,
        *(["counted-centroids-v1"] if args.speakers else []),
    )
    turns, times["diarize"] = run_worker(
        "diarize",
        diar_dir,
        {
            "audio": str(audio),
            "activity": str(seg_dir / "result.json"),
            "embed": emb,
            "threshold": threshold,
            "speakers": args.speakers,
        },
        gpu,
    )
    return turns, diar_dir


def process_file(args):
    offline()
    gpu = target_gpu(args.gpu)
    started = time.perf_counter()
    tuned = args.speakers is not None or args.threshold is not None
    diarizer = getattr(args, "diarizer", None) or ("community" if tuned else "nemotron")
    if diarizer == "nemotron" and tuned:
        raise ValueError("발언자 수와 군집 거리 임계값은 community 모델 전용입니다.")
    threshold = 0.45 if args.threshold is None else args.threshold
    if args.speakers is not None and args.speakers < 1 or not 0 < threshold < 2:
        raise ValueError("화자 수 또는 군집 임계값 오류")
    names = names_from(args.names)
    with gpu_lock(gpu):
        audio, fingerprint = audio_cache(args.audio, args.output)
        work = audio.parent
        times = {}
        if diarizer == "nemotron":
            model, model_id = model_job(args.models, "nemotron")
            diar_dir = work / key(
                PIPELINE_VERSION,
                model_id,
                digest(Path(__file__).with_name("nemotron.py")),
            )
            turns, times["nemotron"] = run_worker(
                "nemotron", diar_dir, {"audio": str(audio), "model": model}, gpu
            )
        else:
            turns, diar_dir = community_turns(args, audio, work, gpu, times, threshold)
        stt = model_job(args.models, args.stt)
        data, times["asr"] = asr_job(audio, work, stt, gpu, args.precision, args.prompt)
        utterances = attribute(data["words"], turns)
        seconds = frames(audio) / RATE
        out = Path(args.output).resolve() / key(
            PIPELINE_VERSION,
            fingerprint,
            stt[1],
            diar_dir.name,
            args.precision,
            args.prompt,
            names,
            args.context,
        )
        doc = {
            "source": Path(args.audio).name,
            "diarizer": diarizer,
            "audio_sha256": fingerprint,
            "duration_s": seconds,
            "stt": args.stt,
            "speaker_names": names,
            "turns": turns,
            "utterances": utterances,
        }
        transcript = out / "transcript.json"
        save(transcript, doc)
        txt_path = out / "transcript.txt"
        lines = (
            f"[{stamp(u['start_ms'])}] {names.get(u['speaker'], u['speaker'])}  {u['text']}\n"
            for u in utterances
        )
        save_text(txt_path, "".join(lines))
        print(f"텍스트 전사본: {txt_path}", flush=True)
        if not args.transcript_only:
            times["minutes"] = summarize_file(
                transcript, out, args.models, gpu, args.context, names
            )
        elapsed = time.perf_counter() - started
        save(
            out / "metrics.json",
            {
                "duration_s": seconds,
                "wall_s": elapsed,
                "rtf": elapsed / seconds,
                "stages": times,
                "sampled_peak_mib": max(s["sampled_peak_mib"] for s in times.values()),
                "reused_stages": [k for k, v in times.items() if v["cached"]],
            },
        )
        return out
