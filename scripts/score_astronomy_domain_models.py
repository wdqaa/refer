#!/usr/bin/env python3
"""Score Qwen3-8B and AstroSage on the fixed astronomy benchmark.

MCQs use deterministic exact-choice scoring. Non-MCQs require a judgment JSONL
that has been audited independently of model identity. No inference or API call is
performed by this script.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.domain_model_benchmark import parse_benchmark_answer


MODEL_KEYS = ("qwen3_8b", "astrosage_8b")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def unique_traces(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        item_id = row["item_id"]
        if item_id in rows:
            raise ValueError(f"duplicate trace in {path}: {item_id}")
        rows[item_id] = row
    return rows


def parse_trace_specs(specs: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for spec in specs:
        key, separator, raw_path = spec.partition("=")
        if not separator or key not in MODEL_KEYS or key in result:
            raise ValueError(f"invalid or duplicate trace specification: {spec}")
        result[key] = Path(raw_path)
    if set(result) != set(MODEL_KEYS):
        raise ValueError(f"expected traces for {MODEL_KEYS}")
    return result


def accepted_gold(record: dict[str, Any]) -> dict[str, Any]:
    gold = record.get("gold_enrichment")
    if not isinstance(gold, dict) or gold.get("accepted") is not True:
        raise ValueError(f"{record['native_qa_id']} lacks accepted Gold")
    if sum(row["points"] for row in gold["scoring_rubric"]) != 10:
        raise ValueError(f"{record['native_qa_id']} rubric points do not sum to 10")
    if record["question_type"] != "mcq":
        minimum_criteria = 1 if record["question_type"] == "short_answer" else 3
        if not minimum_criteria <= len(gold["scoring_rubric"]) <= 5:
            raise ValueError(
                f"{record['native_qa_id']} {record['question_type']} rubric must contain "
                f"{minimum_criteria}-5 criteria"
            )
    return gold


def efficiency(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    inputs = [row.get("input_tokens") or 0 for row in rows.values()]
    outputs = [row.get("output_tokens") or 0 for row in rows.values()]
    latencies = [row.get("latency_seconds") or 0 for row in rows.values()]
    return {
        "items": len(rows),
        "input_tokens": sum(inputs),
        "output_tokens": sum(outputs),
        "summed_latency_seconds": sum(latencies),
        "mean_input_tokens": statistics.mean(inputs),
        "mean_output_tokens": statistics.mean(outputs),
        "mean_latency_seconds": statistics.mean(latencies),
        "json_parse_count": sum(row.get("parse_mode") == "json" for row in rows.values()),
        "plain_text_fallback_count": sum(
            row.get("parse_mode") == "plain_text_fallback" for row in rows.values()
        ),
        "parse_error_count": sum(row.get("status") != "ok" for row in rows.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-items",
        type=int,
        default=24,
        help="Frozen dataset-size guard. Pass the audited cohort size for a new benchmark.",
    )
    parser.add_argument(
        "--allow-unaudited",
        action="store_true",
        help="Permit a preliminary summary from DeepSeek first-pass judgments.",
    )
    args = parser.parse_args()

    records_list = read_jsonl(args.dataset)
    records = {row["native_qa_id"]: row for row in records_list}
    if len(records) != args.expected_items:
        raise ValueError(
            f"expected {args.expected_items} curated astronomy items, found {len(records)}"
        )
    for record in records.values():
        accepted_gold(record)

    trace_paths = parse_trace_specs(args.trace)
    traces: dict[str, dict[str, dict[str, Any]]] = {}
    for key, path in trace_paths.items():
        rows = unique_traces(path)
        missing = sorted(set(records) - set(rows))
        if missing:
            raise ValueError(f"{key} is missing {len(missing)} benchmark traces")
        traces[key] = {item_id: rows[item_id] for item_id in records}

    judgment_rows = read_jsonl(args.judgments)
    judgments: dict[tuple[str, str], dict[str, Any]] = {}
    for row in judgment_rows:
        key = (row["item_id"], row["model_key"])
        if key in judgments:
            raise ValueError(f"duplicate judgment: {key}")
        if not args.allow_unaudited and row.get("review_status") != "codex_audited":
            raise ValueError(
                f"{key} is not codex_audited; use --allow-unaudited only for a preliminary summary"
            )
        judgments[key] = row

    item_scores: list[dict[str, Any]] = []
    correctness: dict[str, dict[str, bool]] = defaultdict(dict)
    coverage: dict[str, list[float]] = defaultdict(list)
    material_errors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    minor_errors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item_id, record in records.items():
        gold = accepted_gold(record)
        if record["question_type"] == "mcq":
            expected = str(gold["gold_answer"]["correct_choice"]).upper()
            row = {
                "item_id": item_id,
                "question_type": "mcq",
                "expected_choice": expected,
                "models": {},
            }
            for model_key in MODEL_KEYS:
                trace = traces[model_key][item_id]
                prediction = trace.get("final_prediction") or {}
                if trace.get("raw"):
                    # Reparse the immutable raw output with the current deterministic
                    # parser rather than trusting a cached choice produced by an older
                    # fallback.  This never consults Gold.
                    prediction, _, _ = parse_benchmark_answer(
                        trace["raw"], record.get("choices") or None
                    )
                actual = str(prediction.get("choice") or "").upper()
                correct = actual == expected
                correctness[model_key][item_id] = correct
                row["models"][model_key] = {"choice": actual or None, "correct": correct}
            item_scores.append(row)
            continue

        rubric = {
            f"C{index}": criterion
            for index, criterion in enumerate(gold["scoring_rubric"], 1)
        }
        row = {
            "item_id": item_id,
            "question_type": record["question_type"],
            "models": {},
        }
        for model_key in MODEL_KEYS:
            judgment = judgments.get((item_id, model_key))
            if judgment is None:
                raise ValueError(f"missing non-MCQ judgment for {(item_id, model_key)}")
            if judgment["rubric_status"] != "usable":
                raise ValueError(f"problematic rubric requires resolution before scoring: {item_id}")
            results = {entry["criterion_id"]: entry for entry in judgment["criterion_results"]}
            if set(results) != set(rubric):
                raise ValueError(f"criterion mismatch for {(item_id, model_key)}")
            covered_points = sum(
                criterion["points"]
                for criterion_id, criterion in rubric.items()
                if results[criterion_id]["covered"]
            )
            total_points = sum(criterion["points"] for criterion in rubric.values())
            item_coverage = covered_points / total_points
            coverage[model_key].append(item_coverage)
            errors = judgment["scientific_errors"]
            material = [error for error in errors if error["severity"] == "material"]
            minor = [error for error in errors if error["severity"] == "minor"]
            material_errors[model_key].extend(
                {"item_id": item_id, **error} for error in material
            )
            minor_errors[model_key].extend({"item_id": item_id, **error} for error in minor)
            strict_correct = item_coverage == 1.0 and not material
            correctness[model_key][item_id] = strict_correct
            row["models"][model_key] = {
                "coverage": item_coverage,
                "covered_points": covered_points,
                "total_points": total_points,
                "material_error_count": len(material),
                "minor_error_count": len(minor),
                "strict_correct": strict_correct,
            }
        item_scores.append(row)

    question_types = Counter(row["question_type"] for row in records.values())
    summaries: dict[str, Any] = {}
    for model_key in MODEL_KEYS:
        mcq_ids = [item_id for item_id, row in records.items() if row["question_type"] == "mcq"]
        non_mcq_ids = [item_id for item_id in records if item_id not in mcq_ids]
        summaries[model_key] = {
            "mcq_correct": sum(correctness[model_key][item_id] for item_id in mcq_ids),
            "mcq_total": len(mcq_ids),
            "mcq_accuracy": statistics.mean(correctness[model_key][item_id] for item_id in mcq_ids),
            "non_mcq_mean_rubric_coverage": statistics.mean(coverage[model_key]),
            "non_mcq_strict_correct": sum(
                correctness[model_key][item_id] for item_id in non_mcq_ids
            ),
            "non_mcq_total": len(non_mcq_ids),
            "responses_with_material_error": len(
                {row["item_id"] for row in material_errors[model_key]}
            ),
            "material_error_claims": len(material_errors[model_key]),
            "minor_error_claims": len(minor_errors[model_key]),
            "overall_strict_correct": sum(correctness[model_key].values()),
            "efficiency": efficiency(traces[model_key]),
        }

    quadrants = Counter()
    complement_items: dict[str, list[str]] = defaultdict(list)
    for item_id in records:
        qwen = correctness["qwen3_8b"][item_id]
        astro = correctness["astrosage_8b"][item_id]
        name = (
            "both_correct" if qwen and astro else
            "qwen_only" if qwen else
            "astrosage_only" if astro else
            "both_wrong"
        )
        quadrants[name] += 1
        complement_items[name].append(item_id)

    non_mcq_deltas = {}
    for row in item_scores:
        if row["question_type"] != "mcq":
            non_mcq_deltas[row["item_id"]] = (
                row["models"]["astrosage_8b"]["coverage"]
                - row["models"]["qwen3_8b"]["coverage"]
            )
    summary = {
        "schema_version": "astronomy_domain_model_benchmark_summary.v1",
        "items": len(records),
        "question_types": dict(sorted(question_types.items())),
        "judgment_status": "preliminary_unaudited" if args.allow_unaudited else "codex_audited",
        "models": summaries,
        "strict_correctness_definition": (
            "MCQ exact choice; non-MCQ 100% weighted rubric coverage and no material error"
        ),
        "complementarity_quadrants": dict(quadrants),
        "complementarity_items": dict(complement_items),
        "non_mcq_astrosage_minus_qwen": {
            "mean_coverage_delta": statistics.mean(non_mcq_deltas.values()),
            "improved": [
                {"item_id": item_id, "delta": delta}
                for item_id, delta in non_mcq_deltas.items() if delta > 0
            ],
            "degraded": [
                {"item_id": item_id, "delta": delta}
                for item_id, delta in non_mcq_deltas.items() if delta < 0
            ],
            "tied": sum(delta == 0 for delta in non_mcq_deltas.values()),
        },
        "material_errors": dict(material_errors),
        "interpretation_boundary": (
            "AstroSage and Qwen use different released prompt interfaces. Both use greedy "
            "decoding and the same 1024-token cap; Qwen thinking is disabled. Results mix "
            "architecture, base-model generation, domain training, and interface differences."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "item_scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in item_scores:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
