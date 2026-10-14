#!/usr/bin/env python3
"""Summarize Qwen baseline traces and build an arm-blinded review packet.

Input: one or more trace JSONL files and the verified QA JSONL.
Output: aggregate JSON, per-item scores, blinded review JSONL, and private arm mapping.
Example: ``python scripts/summarize_qwen_baselines.py --trace outputs/baselines/run/traces.jsonl --output-dir outputs/baselines/run/review``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import statistics
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.answer_scoring import score_prediction


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def route_summary(traces: list[dict[str, Any]], records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    routed = [trace for trace in traces if trace["arm"] == "routed_multi_agent"]
    tp = fp = fn = exact = over = under = 0
    for trace in routed:
        predicted = set(trace["routing"]["selected_domains"])
        expected = set(records[trace["item_id"]]["required_domains"])
        tp += len(predicted & expected)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
        exact += predicted == expected
        over += bool(predicted - expected)
        under += bool(expected - predicted)
    return {
        "items": len(routed),
        "exact_set_match": exact / len(routed) if routed else None,
        "micro_precision": tp / (tp + fp) if tp + fp else None,
        "micro_recall": tp / (tp + fn) if tp + fn else None,
        "over_routed_items": over,
        "under_routed_items": under,
    }


def efficiency_summary(traces: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for arm in sorted({trace["arm"] for trace in traces}):
        arm_traces = [trace for trace in traces if trace["arm"] == arm]
        calls = [len(trace["calls"]) for trace in arm_traces]
        input_tokens = [
            sum(call.get("input_tokens") or 0 for call in trace["calls"])
            for trace in arm_traces
        ]
        output_tokens = [
            sum(call.get("output_tokens") or 0 for call in trace["calls"])
            for trace in arm_traces
        ]
        latency = [
            sum(call["latency_seconds"] for call in trace["calls"])
            for trace in arm_traces
        ]
        result[arm] = {
            "items": len(arm_traces),
            "mean_calls": statistics.mean(calls) if calls else None,
            "mean_input_tokens": statistics.mean(input_tokens) if input_tokens else None,
            "mean_output_tokens": statistics.mean(output_tokens) if output_tokens else None,
            "mean_latency_seconds": statistics.mean(latency) if latency else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets/processed/deepseek_qa_curated_v2/verified_qa_curated_49.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = {item["id"]: item for item in read_jsonl(args.dataset)}
    latest = {}
    for path in args.trace:
        for trace in read_jsonl(path):
            latest[(trace["item_id"], trace["arm"])] = trace
    traces = [trace for key, trace in sorted(latest.items()) if trace["item_id"] in records]

    scores = []
    score_counts: dict[str, Counter] = defaultdict(Counter)
    for trace in traces:
        score = score_prediction(records[trace["item_id"]], trace["final_prediction"])
        row = {"item_id": trace["item_id"], "arm": trace["arm"], **score}
        scores.append(row)
        label = "manual" if score["auto_correct"] is None else str(score["auto_correct"]).lower()
        score_counts[trace["arm"]][label] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "preliminary_scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in scores:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    trace_by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        trace_by_item[trace["item_id"]].append(trace)
    rng = random.Random(args.seed)
    review_rows = []
    mapping_rows = []
    manual_keys = {
        (row["item_id"], row["arm"])
        for row in scores if row["auto_correct"] is None
    }
    for item_id in sorted(trace_by_item):
        candidates = [
            trace for trace in trace_by_item[item_id]
            if (item_id, trace["arm"]) in manual_keys
        ]
        if not candidates:
            continue
        rng.shuffle(candidates)
        coded = []
        mappings = []
        for index, trace in enumerate(candidates):
            code = chr(65 + index)
            coded.append({"candidate_code": code, "prediction": trace["final_prediction"]})
            mappings.append({"candidate_code": code, "arm": trace["arm"]})
        record = records[item_id]
        review_rows.append(
            {
                "item_id": item_id,
                "question_type": record["question_type"],
                "question": record["question"],
                "choices": record.get("choices"),
                "gold": record["gold"],
                "candidates": coded,
                "review_schema": {
                    "candidate_code": "A | B | C",
                    "correct": "true | false",
                    "rubric_notes": "brief evidence-grounded explanation",
                },
            }
        )
        mapping_rows.append({"item_id": item_id, "mapping": mappings})
    for name, rows in (("blinded_review.jsonl", review_rows), ("arm_mapping.jsonl", mapping_rows)):
        with (args.output_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "trace_items": len(traces),
        "preliminary_score_counts": {
            arm: dict(counts) for arm, counts in sorted(score_counts.items())
        },
        "routing": route_summary(traces, records),
        "efficiency": efficiency_summary(traces),
        "manual_review_items": len(review_rows),
        "warning": "Do not report overall arm accuracy until blinded manual reviews are merged.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
