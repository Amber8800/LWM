"""Online provisioning is explicit. Runtime workers only receive verified local paths."""

import os
import shutil
import sys
from importlib.metadata import version
from pathlib import Path

from .common import digest, lock, read, save

ASR = ("large-v3", "komix-v2", "buzzasr-korean")
ASR_BASE = {"komix-v2": "openai/whisper-medium", "buzzasr-korean": "openai/whisper-large-v3"}
SOURCES = {
    "large-v3": "openai/whisper-large-v3",
    "komix-v2": "seastar105/whisper-medium-komixv2",
    "buzzasr-korean": "BuzzASR/korean",
    "segment": "pyannote/speaker-diarization-community-1",
    "embed": "speechbrain/spkrec-ecapa-voxceleb",
    "nemotron": "nvidia/Nemotron-3-Diarization",
    "minutes": "google/gemma-3-12b-it-qat-q4_0-gguf",
}
GGUF = "gemma-3-12b-it-q4_0.gguf"
SEGMENT = "segmentation/pytorch_model.bin"
EMBED = "embedding_model.ckpt"
NEMOTRON = "Nemotron-3-Diarization.q8_0.gguf"
WEIGHTS = {"segment": SEGMENT, "embed": EMBED, "minutes": GGUF, "nemotron": NEMOTRON}


def runtime_folder(root, name):
    # Keep the existing Transformers checkpoint alongside its CTranslate2 conversion.
    return Path(root).resolve() / ("buzzasr-korean-ct2" if name == "buzzasr-korean" else name)


def inventory(root, name):
    folder = runtime_folder(root, name)
    manifest = folder / "manifest.json"
    if not manifest.is_file():
        raise ValueError(
            f'모델 준비 필요: python -m lwm prepare {name} --models "{root}"'
        )
    record = read(manifest)
    if record["source"] != SOURCES[name] or not record["sha256"]:
        raise ValueError(f"잘못된 모델 목록: {manifest}")
    required = (
        {"model.bin", "config.json", "tokenizer.json", "preprocessor_config.json"}
        if name in ASR
        else {WEIGHTS[name]}
    )
    if not required.issubset(record["sha256"]):
        raise ValueError(f"필수 파일이 누락되었습니다: {name}")
    for rel, expected in record["sha256"].items():
        asset = (folder / rel).resolve()
        if not asset.is_relative_to(folder) or digest(asset) != expected:
            raise ValueError(f"모델 무결성 오류: {asset}")
    return folder, record


def hub_token():
    names = ("HF_TOKEN", "HUGGING_FACE_ACCESS_TOKEN", "HUGGINGFACE_HUB_TOKEN")
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
    path = Path(".env")
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            field, sep, value = line.strip().removeprefix("export ").partition("=")
            if sep and field.strip() in names:
                return value.strip().strip("\x22\x27") or None
    return None


def prepare(root, name, revision="main"):
    from huggingface_hub import HfApi, snapshot_download

    root = Path(root).resolve()
    with lock(root / f".{name}.lock"):
        dst = runtime_folder(root, name)
        if dst.exists():
            _, record = inventory(root, name)
            if revision not in ("main", record["revision"]):
                raise ValueError("다른 리비전은 별도 --models 경로에 준비하세요.")
            return record
        token = hub_token()
        info = HfApi(endpoint="https://huggingface.co", token=token).model_info(
            SOURCES[name], revision=revision
        )
        commit = info.sha
        files = [f.rfilename for f in info.siblings]
        stage = root / f".{name}-{commit}"
        stage.mkdir(exist_ok=True)
        if name in ASR:
            patterns = [
                "model.safetensors",
                "model-*-of-*.safetensors",
                "*.json",
                "merges.txt",
                "vocab.json",
                "README.md",
                "LICENSE*",
            ]
        else:
            patterns = [WEIGHTS[name], "README.md", "LICENSE*"]
        cache = root / ".downloads"
        source = Path(
            snapshot_download(
                SOURCES[name],
                revision=commit,
                local_dir=cache / name / commit,
                allow_patterns=patterns,
                endpoint="https://huggingface.co",
                token=token,
            )
        )
        if name in ASR:
            from ctranslate2.converters import TransformersConverter
            from transformers import AutoTokenizer

            TransformersConverter(str(source), low_cpu_mem_usage=True).convert(
                str(stage), quantization="float16", force=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                source, local_files_only=True, use_fast=True
            )
            tokenizer.backend_tokenizer.save(str(stage / "tokenizer.json"))
            shutil.copy2(source / "preprocessor_config.json", stage)
        else:
            rel = WEIGHTS[name]
            (stage / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / rel, stage / rel)
        for file in files:
            if (
                "/" not in file
                and (file == "README.md" or file.startswith("LICENSE"))
                and (source / file).is_file()
            ):
                shutil.copy2(source / file, stage)
        record = {
            "source": SOURCES[name],
            "revision": commit,
            "license": (info.card_data or {}).get("license") or "확인 필요",
            "base": ASR_BASE.get(name),
            "converter": version("ctranslate2") if name in ASR else None,
            "sha256": {
                p.relative_to(stage).as_posix(): digest(p)
                for p in stage.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            },
        }
        save(stage / "manifest.json", record)
        stage.rename(dst)
        print(f"{name}: {commit}", file=sys.stderr)
        return record
