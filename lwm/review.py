"""Score human-reviewed decision and action items against a reference checklist."""

from .evaluate import ratio


def score_review(review):
    results = {}
    for kind in ("decisions", "actions"):
        section = review[kind]
        reference = set(section["reference_ids"])
        if len(reference) != len(section["reference_ids"]):
            raise ValueError("정답 ID 중복")
        seen = set()
        rows = section["predictions"]
        valid_evidence = owners_correct = assigned = unsupported = 0
        for row in rows:
            ref = row.get("matched_reference")
            if ref is not None and ref not in reference:
                raise ValueError("알 수 없는 정답 ID")
            if ref is not None:
                seen.add(ref)
            valid_evidence += row["evidence_valid"] is True
            if kind == "actions":
                assigned += row["owner_assigned"] is True
                owners_correct += row["owner_assigned"] and row["owner_correct"]
                unsupported += row["owner_assigned"] and not row["owner_supported"]
        results[kind] = {
            "precision": ratio(len(seen), len(rows)),
            "recall": ratio(len(seen), len(reference)),
            "evidence_validity": ratio(valid_evidence, len(rows)),
        }
        if kind == "actions":
            results[kind].update(
                owner_accuracy=ratio(owners_correct, assigned),
                unsupported_assignments=unsupported,
            )
    return results
