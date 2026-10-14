"""Load the Source Analysis gold set, validate predictions, and compute core metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any

from src.data.loading import iter_normalized_records


ALLOWED_DOMAINS = {
    "physics", "chemistry", "mathematics", "biology", "astronomy", "geography"
}
ALLOWED_DECISIONS = {"keep", "keep_control", "transform_candidate", "reject"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            records.append(value)
    return records


def load_gold(path: Path, dataset_root: Path) -> list[dict[str, Any]]:
    """Resolve source pointers and return self-contained gold evaluation records."""

    annotations = _read_jsonl(path)
    by_file: dict[str, dict[str, Any]] = {}
    result = []
    for annotation in annotations:
        relative = annotation["source_file"]
        if relative not in by_file:
            source_path = dataset_root / relative
            by_file[relative] = {
                record.source_id: record for record in iter_normalized_records(source_path, root=dataset_root)
            }
        source_id = str(annotation["source_id"])
        if source_id not in by_file[relative]:
            raise ValueError(f"Missing source record {relative} id={source_id}")
        source = by_file[relative][source_id]
        result.append(
            {
                **annotation,
                "source": {
                    "question": source.question,
                    "choices": source.choices or None,
                    "answer": source.answer,
                    "directory_domain": source.source_domain,
                },
            }
        )
    counts = Counter(item["category"] for item in result)
    if len(result) != 30 or any(value != 5 for value in counts.values()) or len(counts) != 6:
        raise ValueError(f"Gold set must contain six categories of five records; got {dict(counts)}")
    return result


def parse_prediction(text: str) -> dict[str, Any]:
    """Parse one raw analyzer response without silently canonicalizing domain labels."""

    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("Prediction must be one JSON object")
    required = {
        "scientifically_valid", "primary_domain", "required_domains", "background_domains",
        "already_cross_domain", "ordinary_math_only", "decision",
        "natural_extension_possible", "natural_extension_direction", "confidence",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"Missing prediction fields: {sorted(missing)}")
    if not isinstance(value["required_domains"], list) or not isinstance(value["background_domains"], list):
        raise ValueError("Domain fields must be lists")
    for item in value["required_domains"]:
        if not isinstance(item, dict) or not all(isinstance(item.get(k), str) for k in ("domain", "evidence", "necessity_reason")):
            raise ValueError("Invalid required-domain evidence item")
    for item in value["background_domains"]:
        if not isinstance(item, dict) or not all(isinstance(item.get(k), str) for k in ("domain", "evidence", "why_not_required")):
            raise ValueError("Invalid background-domain evidence item")
    return value


def compute_metrics(gold: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the small set of Stage 01A metrics, counting parse failures as wrong."""

    by_id = {item["id"]: item for item in predictions}
    total = len(gold)
    parsed = 0
    validity_correct = primary_correct = domains_exact = decision_correct = 0
    invalid_domain_outputs = 0
    tp = fp = fn = 0
    math_denominator = math_fp = 0
    subfield_denominator = subfield_fp = 0
    per_category: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for item in gold:
        output = by_id.get(item["id"], {})
        pred = output.get("prediction")
        gold_value = item["gold"]
        per_category[item["category"]][1] += 1
        gold_domains = {entry["domain"] for entry in gold_value["required_domains"]}
        gold_cross = len(gold_domains) >= 2
        if pred is None:
            if gold_cross:
                fn += 1
            continue
        parsed += 1
        pred_domains = {entry["domain"] for entry in pred["required_domains"]}
        if (pred_domains | {pred.get("primary_domain")}) - ALLOWED_DOMAINS:
            invalid_domain_outputs += 1
        validity_correct += pred["scientifically_valid"] == gold_value["scientifically_valid"]
        primary_correct += pred["primary_domain"] == gold_value["primary_domain"]
        exact = pred_domains == gold_domains
        domains_exact += exact
        per_category[item["category"]][0] += exact
        decision_correct += pred["decision"] == gold_value["decision"]
        pred_cross = len(pred_domains) >= 2
        tp += pred_cross and gold_cross
        fp += pred_cross and not gold_cross
        fn += (not pred_cross) and gold_cross
        if gold_value["ordinary_math_only"] and "mathematics" not in gold_domains:
            math_denominator += 1
            math_fp += "mathematics" in pred_domains
        if item["category"] == "subfield_not_new_domain":
            subfield_denominator += 1
            subfield_fp += not exact

    def ratio(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 4) if denominator else 0.0

    return {
        "total": total,
        "parsed": parsed,
        "parse_success_rate": ratio(parsed, total),
        "validity_accuracy": ratio(validity_correct, total),
        "primary_domain_accuracy": ratio(primary_correct, total),
        "domain_exact_match_accuracy": ratio(domains_exact, total),
        "decision_accuracy": ratio(decision_correct, total),
        "cross_domain_precision": ratio(tp, tp + fp),
        "cross_domain_recall": ratio(tp, tp + fn),
        "math_false_positive_rate": ratio(math_fp, math_denominator),
        "math_false_positive_count": f"{math_fp}/{math_denominator}",
        "subfield_false_positive_rate": ratio(subfield_fp, subfield_denominator),
        "subfield_false_positive_count": f"{subfield_fp}/{subfield_denominator}",
        "invalid_domain_output_rate": ratio(invalid_domain_outputs, parsed),
        "per_category_domain_exact_match": {
            category: ratio(values[0], values[1]) for category, values in sorted(per_category.items())
        },
    }
