"""Compare a short recording on upstream Transformers and the converted runtime."""

import argparse
import os
import tempfile
import wave
from pathlib import Path

from lwm.audio import RATE, decode
from lwm.common import offline, save
from lwm.evaluate import rates, text_counts
from lwm.models import ASR, inventory
from lwm.runner import gpu_lock, target_gpu


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio", type=Path)
    p.add_argument("--stt", choices=ASR, required=True)
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    offline()
    gpu = target_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    import numpy as np
    import torch
    from transformers import pipeline

    from lwm.speech import transcribe

    folder, record = inventory(args.models, args.stt)
    source = args.models / ".downloads" / args.stt / record["revision"]
    with gpu_lock(gpu), tempfile.TemporaryDirectory() as temp:
        wavpath = Path(temp) / "audio.wav"
        decode(args.audio, wavpath)
        with wave.open(str(wavpath), "rb") as wav:
            if wav.getnframes() > 30 * RATE:
                raise ValueError("Use a regression excerpt of 30 seconds or less.")
            audio = (
                np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(
                    "float32"
                )
                / 32768
            )
        pipe = pipeline(
            "automatic-speech-recognition",
            model=str(source.resolve()),
            device=0,
            dtype=torch.float16,
            model_kwargs={"attn_implementation": "eager"},
        )
        # PCM is already decoded by the production FFmpeg path.
        features = pipe.feature_extractor(
            audio,
            sampling_rate=RATE,
            return_tensors="pt",
            return_token_timestamps=True,
            return_attention_mask=True,
        )
        features["input_features"] = features["input_features"].to(dtype=torch.float16)
        result = pipe.forward(
            {"is_last": True, **features},
            return_timestamps="word",
            language="ko",
            task="transcribe",
            num_beams=5,
            do_sample=False,
        )
        native = pipe.postprocess([result], return_timestamps="word")
        del pipe
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        converted = transcribe(
            {
                "audio": str(wavpath),
                "model": str(folder),
                "precision": "float16",
                "cache": temp,
            }
        )
    hypothesis = "".join(w["text"] for w in converted["words"])
    result = {
        "upstream": native,
        "converted": converted,
        "agreement": rates(text_counts(native["text"], hypothesis, [])),
    }
    save(args.output, result)


if __name__ == "__main__":
    main()
