#!/usr/bin/env python3
"""Classify source transformability once per source and preserve every trace.

Input: the keyword-free candidate JSONL and ``DEEPSEEK_API_KEY`` from the environment.
Output: one Selector trace JSONL and one aggregate JSON summary.
Example: ``python scripts/analyze_source_transformability.py --limit 10 --workers 2``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.deepseek_api import DeepSeekJSONClient


RELATIONS = {
    "quantitative_modeling",
    "constraint_coupling",
    "measurement_inference",
}
DOMAINS = {"physics", "chemistry", "biology", "astronomy", "geography", "mathematics"}
RISK_FLAGS = {
    "underspecified",
    "paper_specific",
    "time_sensitive",
    "external_context_required",
    "answer_already_supplied",
    "false_boundary_split",
    "too_broad",
    "fact_lookup_only",
    "no_natural_extension",
}
TRANSFORMABILITY = {"high", "medium", "low"}
REQUIRED_FIELDS = {
    "self_contained",
    "answer_grounded",
    "transformability",
    "candidate_relations",
    "possible_new_domains",
    "reason",
    "risk_flags",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _validate_unique_list(
    analysis: dict[str, Any], field: str, allowed: set[str], errors: list[str]
) -> list[str]:
    value = analysis.get(field)
    if not isinstance(value, list):
        errors.append(f"{field} is not a list")
        return []
    invalid = [item for item in value if not isinstance(item, str) or item not in allowed]
    if invalid:
        errors.append(f"invalid {field}: {invalid}")
    if len(value) != len(set(item for item in value if isinstance(item, str))):
        errors.append(f"{field} contains duplicates")
    return [item for item in value if isinstance(item, str) and item in allowed]


def validate(analysis: dict[str, Any]) -> list[str]:
    """Validate the exact schema plus semantic invariants used by the strict gate."""

    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - analysis.keys())
    unexpected = sorted(analysis.keys() - REQUIRED_FIELDS)
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
    if unexpected:
        errors.append(f"unexpected fields: {', '.join(unexpected)}")
    for field in ("self_contained", "answer_grounded"):
        if not isinstance(analysis.get(field), bool):
            errors.append(f"{field} is not boolean")
    level = analysis.get("transformability")
    if level not in TRANSFORMABILITY:
        errors.append("invalid transformability")
    relations = _validate_unique_list(analysis, "candidate_relations", RELATIONS, errors)
    domains = _validate_unique_list(analysis, "possible_new_domains", DOMAINS, errors)
    risks = _validate_unique_list(analysis, "risk_flags", RISK_FLAGS, errors)
    reason = analysis.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        errors.append("reason is empty or not a string")
    if level == "high":
        if analysis.get("self_contained") is not True:
            errors.append("high requires self_contained=true")
        if analysis.get("answer_grounded") is not True:
            errors.append("high requires answer_grounded=true")
        if not relations:
            errors.append("high requires at least one candidate relation")
        if not domains:
            errors.append("high requires at least one possible new domain")
        if risks:
            errors.append("high cannot contain risk flags")
    if "false_boundary_split" in risks and level == "high":
        errors.append("false_boundary_split cannot be high")
    if "no_natural_extension" in risks:
        if level != "low":
            errors.append("no_natural_extension requires low")
        if relations or domains:
            errors.append("no_natural_extension requires empty candidate lists")
    return errors


def analyze_one(
    item: dict[str, Any], prompt: dict[str, Any], model: str, max_tokens: int
) -> dict[str, Any]:
    client = DeepSeekJSONClient(model=model)
    user_prompt = prompt["user_template"].format(
        source_domain=item["source_domain"],
        question=item["question"],
        choices_json=json.dumps(item.get("choices", {}), ensure_ascii=False),
        reference_answer=item["answer"],
    )
    try:
        analysis, metadata = client.generate_json(
            [
                {"role": "system", "content": prompt["system_prompt"]},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            thinking_enabled=False,
            retries=0,
        )
        errors = validate(analysis)
        return {
            "selector_source_id": item["selector_source_id"],
            "source_domain": item["source_domain"],
            "sampling_stratum": item["sampling_stratum"],
            "analysis": analysis,
            "schema_valid": not errors,
            "validation_errors": errors,
            "api_metadata": metadata,
            "prompt_version": prompt["version"],
            "error": None,
        }
    except Exception as exc:  # Preserve failures; this runner never auto-repairs them.
        return {
            "selector_source_id": item["selector_source_id"],
            "source_domain": item["source_domain"],
            "sampling_stratum": item["sampling_stratum"],
            "analysis": None,
            "schema_valid": False,
            "validation_errors": [],
            "api_metadata": None,
            "prompt_version": prompt["version"],
            "error": f"{type(exc).__name__}: {exc}",
        }


def summarize(
    sources: list[dict[str, Any]], traces: list[dict[str, Any]]
) -> dict[str, Any]:
    valid = [trace for trace in traces if trace.get("schema_valid")]
    analyses = [trace["analysis"] for trace in valid]
    strict = [
        analysis
        for analysis in analyses
        if analysis["transformability"] == "high"
        and analysis["self_contained"]
        and analysis["answer_grounded"]
        and not analysis["risk_flags"]
    ]
    return {
        "initial_candidate_sources": len(sources),
        "selector_traces": len(traces),
        "schema_valid": len(valid),
        "failed_or_invalid": len(traces) - len(valid),
        "transformability": dict(Counter(a["transformability"] for a in analyses)),
        "candidate_relations": dict(
            Counter(relation for a in analyses for relation in a["candidate_relations"])
        ),
        "possible_new_domains": dict(
            Counter(domain for a in analyses for domain in a["possible_new_domains"])
        ),
        "risk_flags": dict(Counter(flag for a in analyses for flag in a["risk_flags"])),
        "strict_high_eligible": len(strict),
    }


def write_summary(
    sources: list[dict[str, Any]], traces: list[dict[str, Any]], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summarize(sources, traces), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    base = PROJECT_ROOT / "datasets/processed/transformability_selector_calibration_v1"
    parser.add_argument("--input", type=Path, default=base / "candidate_sources_150.jsonl")
    parser.add_argument(
        "--prompt",
        type=Path,
        default=PROJECT_ROOT / "src/prompts/source_transformability_selector_v1.json",
    )
    parser.add_argument("--output", type=Path, default=base / "selector_traces.jsonl")
    parser.add_argument("--summary", type=Path, default=base / "selector_summary.json")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--source-id", action="append", help="Analyze only this Selector source ID; may repeat."
    )
    args = parser.parse_args()

    items = read_jsonl(args.input)
    if args.source_id:
        requested = set(args.source_id)
        items = [item for item in items if item["selector_source_id"] in requested]
        missing = requested - {item["selector_source_id"] for item in items}
        if missing:
            parser.error(f"Unknown source IDs: {sorted(missing)}")
    if args.limit is not None:
        items = items[: args.limit]
    prompt = json.loads(args.prompt.read_text(encoding="utf-8"))
    previous = read_jsonl(args.output)
    # Every preserved trace is terminal. Failed/invalid records require an explicit new run,
    # rather than an implicit repair request that changes cost or sampling semantics.
    latest_previous = {record["selector_source_id"]: record for record in previous}
    completed = set(latest_previous)
    pending = [item for item in items if item["selector_source_id"] not in completed]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in latest_previous.values():
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(analyze_one, item, prompt, args.model, args.max_tokens): item[
                    "selector_source_id"
                ]
                for item in pending
            }
            for completed_count, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[{completed_count}/{len(pending)}] {record['selector_source_id']} "
                    f"schema_valid={record['schema_valid']} error={record['error'] is not None}",
                    flush=True,
                )

    selected_ids = {item["selector_source_id"] for item in items}
    traces = [
        record for record in read_jsonl(args.output) if record["selector_source_id"] in selected_ids
    ]
    write_summary(items, traces, args.summary)
    print(f"summary={args.summary} completed={len(traces)}/{len(items)}")


if __name__ == "__main__":
    main()
