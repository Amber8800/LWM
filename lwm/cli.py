import argparse
import json
import sys
from pathlib import Path

from .audio import RATE, frames
from .common import key, offline, read, save
from .evaluate import benchmark, rates, speaker_scores, text_counts
from .models import ASR, SOURCES, inventory, prepare
from .runner import (
    asr_job,
    audio_cache,
    gpu_info,
    gpu_lock,
    model_job,
    names_from,
    process_file,
    summarize_file,
    target_gpu,
)


def parser():
    cli = argparse.ArgumentParser(description="한국어 회의 로컬 전사 및 회의록")
    commands = cli.add_subparsers(dest="cmd", required=True)
    p = commands.add_parser(
        "prepare", help="공식 모델을 다운로드하고 해시를 기록합니다"
    )
    p.add_argument("names", choices=list(SOURCES), nargs="+")
    p.add_argument("--revision", default="main")
    p = commands.add_parser("doctor", help="사용 가능한 GPU·모델 상태")
    for name, desc in [
        ("run", "화자 분리, 전사, 회의록 생성"),
        ("asr", "단일 모델로 전사"),
    ]:
        p = commands.add_parser(name, help=desc)
        p.add_argument("audio", type=Path)
        group = p.add_mutually_exclusive_group()
        group.add_argument("--stt", choices=ASR)
        group.add_argument(
            "--selection", type=Path, help="benchmark.json의 선정 모델 사용"
        )
        if name == "run":
            p.add_argument(
                "--diarizer",
                choices=("nemotron", "community"),
                help="화자 분리 모델 (기본 Nemotron 3, 기존 군집 옵션 지정 시 community)",
            )
            p.add_argument("--speakers", type=int)
            p.add_argument(
                "--threshold", type=float, help="community 군집 거리 임계값 (기본 0.45)"
            )
            p.add_argument("--transcript-only", action="store_true")
    p = commands.add_parser("summarize", help="저장한 전사본에서 회의록 재생성")
    p.add_argument("transcript", type=Path)
    p = commands.add_parser("benchmark", help="같은 회의로 두 STT 후보 비교")
    p.add_argument("dataset", type=Path)
    p.add_argument("--split", choices=["dev", "test"], default="test")
    p.add_argument("--repeats", type=int, default=3)
    p = commands.add_parser("score", help="사람이 교정한 정답과 화자·전사 비교")
    p.add_argument("reference", type=Path)
    p.add_argument("transcript", type=Path)
    p = commands.add_parser("score-review", help="사람이 검수한 결정·할 일의 품질 집계")
    p.add_argument("review", type=Path)
    for name, p in commands.choices.items():
        if name not in ("score", "score-review"):
            p.add_argument("--models", type=Path, default=Path("models"))
        if name in ("run", "summarize", "asr", "benchmark"):
            p.add_argument("--gpu", help="GPU UUID (기본값 RTX 4070 Ti SUPER)")
            p.add_argument("--output", type=Path, default=Path("runs"))
        if name in ("run", "summarize"):
            p.add_argument("--context", type=int, choices=(4096, 8192), default=8192)
            p.add_argument("--names", type=Path, help="화자 ID → 이름 JSON")
        if name in ("run", "benchmark", "asr"):
            p.add_argument("--prompt", default="", help="STT 공통 용어 프롬프트")
            p.add_argument(
                "--precision", choices=["float16", "int8_float16"], default="float16"
            )
    return cli


def dispatch(args):
    if args.cmd == "prepare":
        for name in args.names:
            prepare(args.models, name, args.revision)
        return
    offline()
    if args.cmd == "doctor":
        from .audio import ffmpeg

        print("FFmpeg:", ffmpeg())
        print(json.dumps(gpu_info(), ensure_ascii=False, indent=2))
        for name in SOURCES:
            try:
                _, record = inventory(args.models, name)
                print(name, record["revision"])
            except (ValueError, OSError) as exc:
                print(name, exc)
        return
    if args.cmd in ("run", "asr"):
        if args.selection:
            selection = read(args.selection)
            args.stt = selection["selected"]
            if (
                args.stt not in ASR
                or model_job(args.models, args.stt)[1]
                != selection["model_fingerprints"][args.stt]
            ):
                raise ValueError("선정 보고서와 설치된 모델이 다릅니다.")
            args.precision, args.prompt = selection["precision"], selection["prompt"]
        if not args.stt:
            args.stt = "large-v3"
            print("비교 보고서 미지정: 기준 모델 large-v3 사용", file=sys.stderr)
    if args.cmd == "run":
        return process_file(args)
    if args.cmd == "benchmark":
        return benchmark(args)
    if args.cmd == "asr":
        gpu = target_gpu(args.gpu)
        with gpu_lock(gpu):
            audio, identity = audio_cache(args.audio, args.output)
            model = model_job(args.models, args.stt)
            data, stats = asr_job(
                audio, audio.parent, model, gpu, args.precision, args.prompt
            )
            path = (
                args.output
                / key(identity, model[1], args.precision, args.prompt)
                / "asr.json"
            )
            save(
                path,
                {
                    "source": args.audio.name,
                    "model": args.stt,
                    "duration_s": frames(audio) / RATE,
                    **data,
                    "metrics": stats,
                },
            )
            return path
    if args.cmd == "summarize":
        gpu = target_gpu(args.gpu)
        names = (
            names_from(args.names)
            if args.names
            else read(args.transcript).get("speaker_names", {})
        )
        from .common import digest

        output = args.output / key(digest(args.transcript), names, args.context)
        with gpu_lock(gpu):
            stats = summarize_file(
                args.transcript, output, args.models, gpu, args.context, names
            )
            save(output / "summary-metrics.json", stats)
        return output / "minutes.md"
    if args.cmd == "score-review":
        from .review import score_review

        print(json.dumps(score_review(read(args.review)), ensure_ascii=False, indent=2))
    if args.cmd == "score":
        ref, hyp = read(args.reference), read(args.transcript)

        def text(doc):
            return " ".join(u["text"] for u in doc["utterances"])

        counts = text_counts(text(ref), text(hyp), ref.get("terms", []))
        print(
            json.dumps(
                {**counts, **rates(counts), **speaker_scores(ref, hyp)},
                ensure_ascii=False,
                indent=2,
            )
        )


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        output = dispatch(args)
        if output:
            print(output)
        return 0
    except (ValueError, KeyError, OSError, RuntimeError, ImportError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
