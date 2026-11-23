from collections import defaultdict

from .audio import RATE

UNKNOWN = "화자 미확정"


def stitch(parts):
    """Each core owns its word midpoints. Remove timestamp-jitter duplicates at seams."""
    result = []
    for part in parts:
        core_start, core_end = part["core"]
        owned = [
            w
            for w in part["words"]
            if core_start * 2000
            <= (w["start_ms"] + w["end_ms"]) * RATE
            < core_end * 2000
        ]
        # Match a suffix/prefix only near the seam. Repeated adjacent words
        # with non-overlapping times are valid utterances.
        for count in range(min(8, len(owned), len(result)), 0, -1):
            pairs = zip(result[-count:], owned[:count])
            if all(
                a["text"].strip() == b["text"].strip()
                and min(a["end_ms"], b["end_ms"]) > max(a["start_ms"], b["start_ms"])
                for a, b in pairs
            ):
                owned = owned[count:]
                break
        result.extend(owned)
    return result


def attribute(words, turns):
    turns = sorted(turns, key=lambda t: t["start_ms"])
    i = 0
    active = []
    utterances = []
    for word in words:
        a, b = word["start_ms"], word["end_ms"]
        while i < len(turns) and turns[i]["start_ms"] < b:
            active.append(turns[i])
            i += 1
        active = [t for t in active if t["end_ms"] > a]
        spans = defaultdict(list)
        for t in active:
            lo, hi = max(a, t["start_ms"]), min(b, t["end_ms"])
            if hi > lo:
                spans[t["speaker"]].append((lo, hi))
        scores = {}
        events = []
        for speaker, periods in spans.items():
            end = -1
            scores[speaker] = 0
            for lo, hi in sorted(periods):
                if hi > max(end, lo):
                    start = max(end, lo)
                    scores[speaker] += hi - start
                    events.extend([(start, 1), (hi, -1)])
                end = max(end, hi)
        concurrency = 0
        overlap = any(
            t.get("overlap") and t["start_ms"] < b and a < t["end_ms"] for t in active
        )
        for _, delta in sorted(events):
            concurrency += delta
            overlap |= concurrency > 1
        ranked = sorted(scores, key=scores.get, reverse=True)
        speaker = UNKNOWN
        if (
            ranked
            and not overlap
            and scores[ranked[0]] * 2 >= max(1, b - a)
            and (len(ranked) == 1 or scores[ranked[0]] > scores[ranked[1]])
        ):
            speaker = ranked[0]
        item = {
            **word,
            "speaker": speaker,
            "overlap": overlap,
            "candidates": sorted(scores),
        }
        previous = utterances[-1] if utterances else None
        if (
            not previous
            or previous["speaker"] != speaker
            or previous["overlap"] != overlap
            or a - previous["end_ms"] > 1200
            or b - previous["start_ms"] > 30000
        ):
            previous = {
                "id": f"U{len(utterances) + 1:06}",
                "start_ms": a,
                "end_ms": b,
                "speaker": speaker,
                "overlap": overlap,
                "words": [],
                "text": "",
            }
            utterances.append(previous)
        previous["words"].append(item)
        previous["end_ms"] = max(b, previous["end_ms"])
        previous["text"] += word["text"]
    for u in utterances:
        u["text"] = u["text"].strip()
    return utterances
