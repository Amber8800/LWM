from collections import defaultdict

from .audio import RATE

UNKNOWN = "화자 미확정"
BOUNDARY_MS = 250


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
    assigned = []
    boundary_candidates = []
    for word in words:
        a, b = word["start_ms"], word["end_ms"]
        while i < len(turns) and turns[i]["start_ms"] < b + BOUNDARY_MS:
            active.append(turns[i])
            i += 1
        active = [t for t in active if t["end_ms"] > a - BOUNDARY_MS]
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
        assigned.append(item)
        boundary_candidates.append(
            boundary_candidate(a, b, active, scores)
            if speaker == UNKNOWN and not overlap
            else None
        )

    # Only a directly attributed neighbouring word may anchor an inference.
    direct = [word["speaker"] for word in assigned]
    for idx, candidate in enumerate(boundary_candidates):
        if candidate is None:
            continue
        word = assigned[idx]
        neighbours = set()
        if idx and abs(word["start_ms"] - assigned[idx - 1]["end_ms"]) <= BOUNDARY_MS:
            neighbours.add(direct[idx - 1])
        if (
            idx + 1 < len(assigned)
            and abs(assigned[idx + 1]["start_ms"] - word["end_ms"]) <= BOUNDARY_MS
        ):
            neighbours.add(direct[idx + 1])
        if neighbours - {UNKNOWN} == {candidate}:
            word["speaker"] = candidate
            word["speaker_inferred"] = True

    utterances = []
    for item in assigned:
        a, b = item["start_ms"], item["end_ms"]
        speaker, overlap = item["speaker"], item["overlap"]
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
        previous["text"] += item["text"]
    for u in utterances:
        u["text"] = u["text"].strip()
    return utterances


def boundary_candidate(a, b, active, scores):
    """Allow a partial match or a short word just outside a single voice's turn."""
    duration = b - a
    if not 0 < duration <= 3000:
        return None
    nearby = [
        t for t in active
        if t["start_ms"] < b + BOUNDARY_MS and t["end_ms"] > a - BOUNDARY_MS
    ]
    speakers = {t["speaker"] for t in nearby}
    if len(speakers) != 1 or UNKNOWN in speakers or any(t.get("overlap") for t in nearby):
        return None
    speaker = next(iter(speakers))
    support = scores.get(speaker, 0)
    if support >= 80 and support * 5 >= duration:
        return speaker
    # For very short tails, require the whole word to fit the padded turn.
    if duration <= BOUNDARY_MS and any(
        t["start_ms"] - BOUNDARY_MS <= a and b <= t["end_ms"] + BOUNDARY_MS
        for t in nearby
    ):
        return speaker
    return None
