#!/usr/bin/env python3
"""Batch-label unchanged native science QA and preserve every DeepSeek trace.

Purpose: classify source QA without rewriting questions or generating reasoning.
Input: a JSONL native-QA manifest, a versioned prompt, runtime config, and source-
universe summary. Output: resumable raw batch traces, all valid labels, high-priority
single/cross-domain subsets, aggregate statistics, and optional gold metrics.

Example:
    python scripts/classify_native_qa.py --input data/manifest.jsonl \
        --output-dir data/labeled --workers 3 --scope pilot
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.deepseek_api import DeepSeekJSONClient


DOMAINS = {"physics", "chemistry", "biology", "astronomy", "geography", "mathematics"}
QUESTION_TYPES = {"mcq", "numeric", "short_answer", "symbolic", "qualitative"}
DOMAIN_TYPES = {"single", "native_cross_domain"}
VERIFIABILITY = {"rule_verifiable", "rubric_verifiable", "unreliable"}
PRIORITIES = {"high", "medium", "low"}
LABEL_FIELDS = {
    "valid",
    "question_type",
    "required_domains",
    "domain_type",
    "answer_verifiability",
    "keep_priority",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def batches(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def validate_label(item: dict[str, Any], label: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = LABEL_FIELDS | {"id"}
    if set(label) != expected:
        errors.append(f"fields must be exactly {sorted(expected)}")
    if label.get("id") != item["native_qa_id"]:
        errors.append("id mismatch")
    if not isinstance(label.get("valid"), bool):
        errors.append("valid is not boolean")
    if label.get("question_type") not in QUESTION_TYPES:
        errors.append("invalid question_type")
    domains = label.get("required_domains")
    if not isinstance(domains, list):
        errors.append("required_domains must be a list")
        domains = []
    elif len(domains) != len(set(domains)) or any(domain not in DOMAINS for domain in domains):
        errors.append("required_domains contains duplicate or invalid domains")
    if not domains and label.get("valid") is not False:
        errors.append("valid item requires at least one required domain")
    if label.get("domain_type") not in DOMAIN_TYPES:
        errors.append("invalid domain_type")
    else:
        expected_type = "single" if len(domains) <= 1 else "native_cross_domain"
        if label["domain_type"] != expected_type:
            errors.append("domain_type conflicts with required_domains")
    if label.get("answer_verifiability") not in VERIFIABILITY:
        errors.append("invalid answer_verifiability")
    if label.get("keep_priority") not in PRIORITIES:
        errors.append("invalid keep_priority")
    has_choices = bool(item.get("choices"))
    if has_choices and label.get("question_type") != "mcq":
        errors.append("item with choices must be mcq")
    if not has_choices and label.get("question_type") == "mcq":
        errors.append("item without choices cannot be mcq")
    if label.get("valid") is False:
        if label.get("answer_verifiability") != "unreliable":
            errors.append("invalid item must be unreliable")
        if label.get("keep_priority") != "low":
            errors.append("invalid item must be low priority")
    if label.get("answer_verifiability") == "unreliable" and label.get("keep_priority") != "low":
        errors.append("unreliable item must be low priority")
    if label.get("keep_priority") == "high" and (
        label.get("valid") is not True or label.get("answer_verifiability") == "unreliable"
    ):
        errors.append("high priority requires valid and verifiable")
    return errors


def request_payload(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": item["native_qa_id"],
            "question": item["question"],
            "choices": item.get("choices") or None,
            "reference_answer": item["answer"],
        }
        for item in items
    ]


def classify_batch(
    batch_id: str,
    items: list[dict[str, Any]],
    prompt: dict[str, Any],
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    client = DeepSeekJSONClient(model=model)
    user = prompt["user_template"].format(
        items_json=json.dumps(request_payload(items), ensure_ascii=False, indent=2)
    )
    try:
        result, metadata = client.generate_json(
            [
                {"role": "system", "content": prompt["system_prompt"]},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            thinking_enabled=False,
            retries=0,
        )
        top_errors = [] if set(result) == {"items"} else ["top-level object must contain only items"]
        raw_labels = result.get("items")
        if not isinstance(raw_labels, list):
            top_errors.append("items is not a list")
            raw_labels = []
        by_id = {
            label.get("id"): label
            for label in raw_labels
            if isinstance(label, dict) and isinstance(label.get("id"), str)
        }
        item_results = []
        for item in items:
            label = by_id.get(item["native_qa_id"])
            errors = ["missing result"] if label is None else validate_label(item, label)
            item_results.append(
                {
                    "id": item["native_qa_id"],
                    "label": label,
                    "schema_valid": label is not None and not errors and not top_errors,
                    "validation_errors": top_errors + errors,
                }
            )
        extra_ids = sorted(set(by_id) - {item["native_qa_id"] for item in items})
        if extra_ids:
            top_errors.append(f"unexpected ids: {extra_ids}")
            for value in item_results:
                value["schema_valid"] = False
                value["validation_errors"] = top_errors + value["validation_errors"]
        return {
            "batch_id": batch_id,
            "item_ids": [item["native_qa_id"] for item in items],
            "result": result,
            "item_results": item_results,
            "api_metadata": metadata,
            "prompt_version": prompt["version"],
            "error": None,
        }
    except Exception as exc:
        return {
            "batch_id": batch_id,
            "item_ids": [item["native_qa_id"] for item in items],
            "result": None,
            "item_results": [],
            "api_metadata": None,
            "prompt_version": prompt["version"],
            "error": f"{type(exc).__name__}: {exc}",
        }


def usage_totals(traces: list[dict[str, Any]]) -> dict[str, int]:
    totals = Counter()
    for trace in traces:
        usage = (trace.get("api_metadata") or {}).get("usage") or {}
        totals["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
        totals["completion_tokens"] += int(usage.get("completion_tokens", 0))
        totals["total_tokens"] += int(usage.get("total_tokens", 0))
        totals["prompt_cache_hit_tokens"] += int(usage.get("prompt_cache_hit_tokens", 0))
        totals["prompt_cache_miss_tokens"] += int(usage.get("prompt_cache_miss_tokens", 0))
    return dict(totals)


def price(usage: dict[str, int], *, peak: bool) -> float:
    multiplier = 2.0 if peak else 1.0
    return (
        usage.get("prompt_cache_hit_tokens", 0) / 1_000_000 * 0.003 * multiplier
        + usage.get("prompt_cache_miss_tokens", 0) / 1_000_000 * 0.15 * multiplier
        + usage.get("completion_tokens", 0) / 1_000_000 * 0.60 * multiplier
    )


def gold_metrics(manifest: list[dict[str, Any]], labeled: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gold_items = [item for item in manifest if item.get("calibration_gold")]
    parsed = validity = domain_exact = domain_type = 0
    tp = fp = fn = 0
    valid_total = valid_retained = valid_domain_exact = valid_domain_type = 0
    valid_cross_total = valid_predicted_cross = valid_cross_tp = 0
    invalid_total = invalid_rejected = 0
    per_category: dict[str, list[int]] = {}
    for item in gold_items:
        label = labeled.get(item["native_qa_id"])
        gold = item["calibration_gold"]
        gold_domains = {entry["domain"] for entry in gold["required_domains"]}
        gold_cross = len(gold_domains) >= 2
        gold_valid = bool(gold["scientifically_valid"])
        if gold_valid:
            valid_total += 1
            valid_cross_total += int(gold_cross)
        else:
            invalid_total += 1
        category = gold["category"]
        per_category.setdefault(category, [0, 0])[1] += 1
        if label is None:
            fn += int(gold_cross)
            continue
        parsed += 1
        predicted = set(label["required_domains"])
        predicted_cross = label["domain_type"] == "native_cross_domain"
        validity += label["valid"] == gold["scientifically_valid"]
        exact = predicted == gold_domains
        domain_exact += exact
        per_category[category][0] += int(exact)
        domain_type += predicted_cross == gold_cross
        tp += int(predicted_cross and gold_cross)
        fp += int(predicted_cross and not gold_cross)
        fn += int(not predicted_cross and gold_cross)
        if gold_valid:
            valid_retained += int(label["valid"])
            valid_domain_exact += int(exact)
            valid_domain_type += int(predicted_cross == gold_cross)
            eligible_cross = bool(label["valid"] and predicted_cross)
            valid_predicted_cross += int(eligible_cross)
            valid_cross_tp += int(eligible_cross and gold_cross)
        else:
            invalid_rejected += int(not label["valid"])

    def ratio(value: int, denominator: int) -> float:
        return round(value / denominator, 4) if denominator else 0.0

    total = len(gold_items)
    return {
        "total": total,
        "schema_valid": parsed,
        "validity_accuracy": ratio(validity, total),
        "required_domains_exact_match": ratio(domain_exact, total),
        "domain_type_accuracy": ratio(domain_type, total),
        "cross_domain_precision": ratio(tp, tp + fp),
        "cross_domain_recall": ratio(tp, tp + fn),
        "valid_only": {
            "total": valid_total,
            "retention_rate": ratio(valid_retained, valid_total),
            "required_domains_exact_match": ratio(valid_domain_exact, valid_total),
            "domain_type_accuracy": ratio(valid_domain_type, valid_total),
            "cross_domain_precision": ratio(valid_cross_tp, valid_predicted_cross),
            "cross_domain_recall": ratio(valid_cross_tp, valid_cross_total),
        },
        "invalid_controls": {
            "total": invalid_total,
            "rejection_rate": ratio(invalid_rejected, invalid_total),
        },
        "per_category_domain_exact_match": {
            key: ratio(value[0], value[1]) for key, value in sorted(per_category.items())
        },
    }


def export(
    manifest: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    output_dir: Path,
    universe: dict[str, Any],
    scope: str,
) -> None:
    source_by_id = {item["native_qa_id"]: item for item in manifest}
    labels: dict[str, dict[str, Any]] = {}
    label_batches: dict[str, str] = {}
    for trace in traces:
        raw_items = (trace.get("result") or {}).get("items") or []
        by_id = {
            label.get("id"): label
            for label in raw_items
            if isinstance(label, dict) and isinstance(label.get("id"), str)
        }
        for item_id in trace.get("item_ids", []):
            label = by_id.get(item_id)
            source = source_by_id.get(item_id)
            if source is not None and label is not None and not validate_label(source, label):
                labels[item_id] = label
                label_batches[item_id] = trace["batch_id"]

    ordered = []
    for item in manifest:
        label = labels.get(item["native_qa_id"])
        if label is None:
            continue
        source = {key: value for key, value in item.items() if key != "calibration_gold"}
        ordered.append(
            {
                **source,
                **{field: label[field] for field in sorted(LABEL_FIELDS)},
                "labeling": {"prompt": "native_qa_classifier_v1", "batch_id": label_batches[item["native_qa_id"]]},
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "all_labeled": output_dir / "all_labeled.jsonl",
        "high_single": output_dir / "high_priority_single.jsonl",
        "high_cross": output_dir / "high_priority_cross_domain.jsonl",
    }
    selections = {
        "all_labeled": ordered,
        "high_single": [item for item in ordered if item["valid"] and item["keep_priority"] == "high" and item["domain_type"] == "single"],
        "high_cross": [item for item in ordered if item["valid"] and item["keep_priority"] == "high" and item["domain_type"] == "native_cross_domain"],
    }
    for key, path in paths.items():
        with path.open("w", encoding="utf-8") as handle:
            for item in selections[key]:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    usage = usage_totals(traces)
    labeled_count = len(ordered)
    unique_total = int(universe["exact_unique_question_answer_pairs"])
    scale = unique_total / labeled_count if labeled_count else 0.0
    projected_usage = {key: math.ceil(value * scale) for key, value in usage.items()}
    summary = {
        "scope": scope,
        "total_questions": len(manifest),
        "schema_valid_labeled": labeled_count,
        "failed_or_invalid": len(manifest) - labeled_count,
        "valid": sum(item["valid"] for item in ordered),
        "domain_type": dict(Counter(item["domain_type"] for item in ordered)),
        "required_domains": dict(Counter(domain for item in ordered for domain in item["required_domains"])),
        "question_type": dict(Counter(item["question_type"] for item in ordered)),
        "keep_priority": dict(Counter(item["keep_priority"] for item in ordered)),
        "answer_verifiability": dict(Counter(item["answer_verifiability"] for item in ordered)),
        "high_priority_single": len(selections["high_single"]),
        "high_priority_cross_domain": len(selections["high_cross"]),
        "api_batches": len(traces),
        "api_errors": sum(trace.get("error") is not None for trace in traces),
        "api_usage": usage,
        "actual_cost_usd": {"off_peak": round(price(usage, peak=False), 6), "peak": round(price(usage, peak=True), 6)},
        "full_scan_projection": {
            "raw_non_mixed_records": int(universe["raw_non_mixed_records"]),
            "exact_unique_question_answer_pairs": unique_total,
            "method": "linear scaling from this calibration after exact question-answer deduplication",
            "projected_usage": projected_usage,
            "estimated_cost_usd": {
                "off_peak": round(price(projected_usage, peak=False), 2),
                "peak": round(price(projected_usage, peak=True), 2),
            },
        },
        "pricing": {
            "as_of": "2026-10-08",
            "model": "DeepSeek-V4.1-Flash via deepseek-flash",
            "source": "https://api-docs.deepseek.com/quick_start/pricing",
            "usd_per_million_tokens": {
                "off_peak": {"input_cache_hit": 0.003, "input_cache_miss": 0.15, "output": 0.6},
                "peak": {"input_cache_hit": 0.006, "input_cache_miss": 0.3, "output": 1.2}
            }
        }
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "gold_metrics.json").write_text(json.dumps(gold_metrics(manifest, labels), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    output_dir = PROJECT_ROOT / "datasets/processed/native_qa_mining_v1/calibration_60"
    parser.add_argument("--input", type=Path, default=output_dir / "manifest.jsonl")
    parser.add_argument("--prompt", type=Path, default=PROJECT_ROOT / "src/prompts/native_qa_classifier_v1.json")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/data/native_qa_mining_v1.json")
    parser.add_argument("--universe-summary", type=Path, default=output_dir / "source_universe_summary.json")
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--batch-prefix", default="native-calibration-batch")
    parser.add_argument("--scope", default="calibration_60")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    prompt = json.loads(args.prompt.read_text(encoding="utf-8"))
    manifest = read_jsonl(args.input)
    if args.limit is not None:
        manifest = manifest[: args.limit]
    batch_size = args.batch_size or int(config["batch_size"])
    grouped = batches(manifest, batch_size)
    traces_path = args.output_dir / "batch_traces.jsonl"
    previous = read_jsonl(traces_path)
    previous_by_id = {trace["batch_id"]: trace for trace in previous}
    pending = []
    for index, group in enumerate(grouped):
        batch_id = f"{args.batch_prefix}-{index:03d}"
        if batch_id not in previous_by_id:
            pending.append((batch_id, group))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with traces_path.open("w", encoding="utf-8") as handle:
        for trace in previous_by_id.values():
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
        handle.flush()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    classify_batch,
                    batch_id,
                    group,
                    prompt,
                    config["model"],
                    int(config["max_tokens"]),
                ): batch_id
                for batch_id, group in pending
            }
            for count, future in enumerate(as_completed(futures), start=1):
                trace = future.result()
                handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"[{count}/{len(pending)}] {trace['batch_id']} error={trace['error'] is not None}", flush=True)

    selected_batches = {f"{args.batch_prefix}-{index:03d}" for index in range(len(grouped))}
    traces = [trace for trace in read_jsonl(traces_path) if trace["batch_id"] in selected_batches]
    universe = json.loads(args.universe_summary.read_text(encoding="utf-8"))
    export(manifest, traces, args.output_dir, universe, args.scope)
    print(f"summary={args.output_dir / 'summary.json'} batches={len(traces)}/{len(grouped)}")


if __name__ == "__main__":
    main()
