import itertools
import os
import time
import wave
from pathlib import Path

from .audio import RATE, chunks, pcm
from .common import read, save
from .models import EMBED, SEGMENT
from .progress import Reporter
from .transcript import UNKNOWN, stitch


def decode_windows(decoder, audio, offset, prompt, on_window=None):
    """Decode each Whisper window exactly once. Keep only the words owned by its core."""
    import numpy as np

    parts = []
    lo = 0
    while lo < len(audio):
        # Up to 26 s of core plus 2 s of acoustic context on each side.
        hi = min(lo + 25 * RATE, len(audio))
        if hi < len(audio):
            step = RATE // 10
            candidates = range(hi - RATE, min(len(audio) - step, hi + RATE) + 1, step)
            hi = min(
                candidates,
                key=lambda pos: (np.mean(audio[pos : pos + step] ** 2), abs(pos - hi)),
            )
        start = max(0, lo - 2 * RATE)
        end = min(len(audio), hi + 2 * RATE)
        clip = audio[start:end]
        words = []
        if np.any(clip):
            # Preserve real audio and its absolute offset when a recorder inserts long digital gaps.
            voiced = clip != 0
            first = max(0, int(voiced.argmax()) - RATE)
            last = min(len(clip), len(clip) - int(voiced[::-1].argmax()) + RATE)
            clip = clip[first:last]
            start += first
            end = start + len(clip)
            segments, _ = decoder.transcribe(
                clip,
                batch_size=1,
                language="ko",
                task="transcribe",
                beam_size=5,
                temperature=0,
                word_timestamps=True,
                without_timestamps=False,
                vad_filter=False,
                clip_timestamps=[{"start": 0, "end": len(clip) / RATE}],
                initial_prompt=prompt or None,
            )
            for seg in segments:
                for word in seg.words or []:
                    if not word.word.strip():
                        continue
                    a = max(start, min(end - 1, start + round(word.start * RATE)))
                    b = max(a + 1, min(end, start + round(word.end * RATE)))
                    ms = round(1000 * (offset + a) / RATE)
                    words.append(
                        {
                            "text": word.word,
                            "start_ms": ms,
                            "end_ms": max(ms + 1, round(1000 * (offset + b) / RATE)),
                        }
                    )
        parts.append({"core": [offset + lo, offset + hi], "words": words})
        lo = hi
        if on_window:
            on_window(offset + hi)
    return stitch(parts)


def transcribe(job):
    import numpy as np

    # The CUDA Torch wheel supplies the matching cuBLAS/cuDNN DLLs on Windows.
    import torch

    if os.name == "nt":
        libs = Path(torch.__file__).parent / "lib"
        os.environ["PATH"] = str(libs) + os.pathsep + os.environ["PATH"]
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    wav_path = job["audio"]
    start_time = time.perf_counter()
    model = WhisperModel(
        job["model"],
        device="cuda",
        device_index=0,
        compute_type=job["precision"],
        local_files_only=True,
    )
    decoder = BatchedInferencePipeline(model)
    loaded = time.perf_counter()
    parts = []
    with wave.open(wav_path, "rb") as wav:
        length = wav.getnframes()
        progress = Reporter("asr", "samples")
        progress(0, length)
        for i, (lo, hi, start, end) in enumerate(chunks(wav_path)):
            checkpoint = Path(job["cache"]) / f"chunk-{i:04}.json"
            if checkpoint.exists():
                parts.append(read(checkpoint))
                progress(hi, length)
                continue
            audio = (
                np.frombuffer(pcm(wav, start, end), dtype="<i2").astype("float32")
                / 32768
            )
            words = decode_windows(
                decoder,
                audio,
                start,
                job.get("prompt"),
                lambda pos, lo=lo, hi=hi: progress(max(lo, min(hi, pos)), length),
            )
            part = {"core": [lo, hi], "words": words}
            save(checkpoint, part)
            parts.append(part)
    return {
        "words": stitch(parts),
        "load_s": loaded - start_time,
        "inference_s": time.perf_counter() - loaded,
    }


def windows(length, size):
    """50% overlapping windows; each owns one contiguous core."""
    step = size // 2
    start = 0
    while True:
        last = start + size >= length
        lo = start + (step // 2 if start else 0)
        hi = length if last else start + step + step // 2
        yield start, lo, hi
        if last:
            break
        start += step


def runs(mask):
    import numpy as np

    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(edges[::2], edges[1::2])


# Consecutive model frame centres delimit activity on the PCM timeline.
def spans(mask, lower, upper, start, resolution):
    out = []
    for i, j in runs(mask):
        left = resolution[int(i)].middle - resolution.step / 2
        right = resolution[int(j) - 1].middle + resolution.step / 2
        a = max(lower, start + round(left * RATE))
        b = min(upper, start + round(right * RATE))
        if b > a:
            out.append([a, b])
    return out


def segment(job):
    import numpy as np
    import torch
    from pyannote.audio import Inference, Model

    model = Model.from_pretrained(
        str(Path(job["segment"]) / SEGMENT), map_location="cpu"
    )
    inference = Inference(model, device=torch.device("cuda"), batch_size=1)
    window = round(model.specifications.duration * RATE)
    resolution = model.receptive_field
    records = []
    with wave.open(job["audio"], "rb") as wav:
        length = wav.getnframes()
        progress = Reporter("segment", "samples")
        for start, lo, hi in windows(length, window):
            progress(lo, length)
            audio = np.frombuffer(
                pcm(wav, start, min(length, start + window)), dtype="<i2"
            )
            if not np.any(audio):
                continue
            samples = np.zeros(window, dtype="float32")
            samples[: len(audio)] = audio / 32768.0
            activities = inference.infer(torch.from_numpy(samples)[None, None])[0] > 0.5

            single = activities.sum(axis=1) == 1
            for mask in activities.T:
                core = spans(mask, lo, hi, start, resolution)
                if core:
                    records.append(
                        {
                            "core": core,
                            "clean": spans(
                                mask & single,
                                start,
                                min(length, start + window),
                                start,
                                resolution,
                            ),
                            "overlap": spans(mask & ~single, lo, hi, start, resolution),
                            "window": start,
                        }
                    )
        progress(length, length)
    return records


def counted_labels(records, indices, vectors, count, threshold):
    """Seed speakers from recurrent voice clusters before assigning each window."""
    import numpy as np
    from scipy.cluster.hierarchy import cut_tree, fcluster, linkage
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    if not indices:
        return {}
    vec = np.asarray(vectors)
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    count = min(count, len(vec))
    groups = np.zeros(len(vec), dtype=int)
    if len(vec) > 1:
        tree = linkage(vec, method="average", metric="cosine")
        groups = fcluster(tree, t=threshold, criterion="distance")
        if len(np.unique(groups)) < count:
            groups = cut_tree(tree, n_clusters=count).ravel()
    clusters = {}
    windows = {}
    for pos, i in enumerate(indices):
        clusters.setdefault(groups[pos], []).append(pos)
        windows.setdefault(records[i]["window"], []).append(pos)
    seeds = sorted(clusters.values(), key=len, reverse=True)[:count]
    centers = np.stack([vec[group].mean(axis=0) for group in seeds])
    labels = {}
    names = {}
    for slots in windows.values():
        distance = cdist(vec[slots], centers, metric="cosine")
        # Each local speaker may abstain independently if its voice is a poor match.
        cost = np.column_stack((distance, np.full((len(slots), len(slots)), threshold)))
        rows, columns = linear_sum_assignment(cost)
        for row, column in zip(rows, columns):
            if column < len(centers) and distance[row, column] <= threshold:
                labels[indices[slots[row]]] = names.setdefault(
                    column, f"SPEAKER_{len(names) + 1:02}"
                )
    return labels


def diarize(job):
    import numpy as np
    import torch
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    from speechbrain.lobes.features import Fbank
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
    from speechbrain.processing.features import InputNormalization

    # Architecture and feature normalization from the official VoxCeleb hyperparams.
    net = ECAPA_TDNN(
        input_size=80,
        channels=[1024, 1024, 1024, 1024, 3072],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        attention_channels=128,
        lin_neurons=192,
    )
    net.load_state_dict(
        torch.load(Path(job["embed"]) / EMBED, map_location="cpu", weights_only=True)
    )
    net = net.eval().cuda()
    fbank = Fbank(n_mels=80).eval().cuda()
    normalize = InputNormalization(norm_type="sentence", std_norm=False).eval().cuda()
    records = read(job["activity"])
    idx, vectors = [], []
    progress = Reporter("diarize", "regions")
    with wave.open(job["audio"], "rb") as wav, torch.inference_mode():
        for i, record in enumerate(records):
            progress(i, len(records))
            raw = b"".join(pcm(wav, a, b) for a, b in record["clean"])
            if len(raw) < RATE:  # at least half a second of single-speaker audio
                continue
            x = (
                torch.from_numpy(
                    np.frombuffer(raw, dtype="<i2").astype("float32")
                ).cuda()[None]
                / 32768
            )
            features = normalize(fbank(x), torch.ones(1, device=x.device))
            vector = net(features).reshape(-1).cpu().numpy()
            if np.isfinite(vector).all() and np.linalg.norm(vector) > 0:
                idx.append(i)
                vectors.append(vector)
    progress(len(records), len(records))
    Reporter("diarize", "cluster")()
    labels = {}
    if job.get("speakers"):
        labels = counted_labels(
            records, idx, vectors, job["speakers"], job["threshold"]
        )
    elif vectors:
        if len(idx) == 1:
            groups = [1]
        else:
            tree = linkage(np.stack(vectors), method="average", metric="cosine")
            groups = fcluster(tree, t=job["threshold"], criterion="distance")
        names = {}
        for i, group in zip(idx, groups):
            names.setdefault(int(group), f"SPEAKER_{len(names) + 1:02}")
            labels[i] = names[int(group)]
    # Distinct local speaker slots must not collapse onto one ID in the same window.
    if vectors and not job.get("speakers"):
        names = list(dict.fromkeys(labels.values()))
        vec = np.asarray(vectors)
        centroids = np.stack(
            [vec[[labels[i] == n for i in idx]].mean(axis=0) for n in names]
        )
        per_window = {}
        for pos, i in enumerate(idx):
            per_window.setdefault(records[i]["window"], []).append(pos)
        for slots in per_window.values():
            if len({labels[idx[pos]] for pos in slots}) == len(slots):
                continue
            dist = cdist(vec[slots], centroids, metric="cosine")
            rows, cols = linear_sum_assignment(dist)
            for pos in slots:
                labels[idx[pos]] = UNKNOWN
            for r, c in zip(rows, cols):
                if dist[r, c] <= job["threshold"]:
                    labels[idx[slots[r]]] = names[c]
    turns = []
    for i, record in enumerate(records):
        for a, b in record["core"]:
            speaker = labels.get(i, UNKNOWN)
            cuts = sorted(
                {a, b} | {v for span in record["overlap"] for v in span if a < v < b}
            )
            for lo, hi in itertools.pairwise(cuts):
                turns.append(
                    {
                        "start_ms": round(lo * 1000 / RATE),
                        "end_ms": round(hi * 1000 / RATE),
                        "speaker": speaker,
                        "overlap": any(x < hi and y > lo for x, y in record["overlap"]),
                    }
                )
    return sorted(turns, key=lambda t: t["start_ms"])
