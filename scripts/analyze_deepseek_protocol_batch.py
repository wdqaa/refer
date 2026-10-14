#!/usr/bin/env python3
"""Run the DeepSeek Source Analyzer over a protocol batch with resume support.

Input: a protocol JSONL plus ``DEEPSEEK_API_KEY`` in the process environment.
Output: one analysis JSONL and one aggregate JSON summary.
Example: ``python scripts/analyze_deepseek_protocol_batch.py --workers 6``.
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


CALLABLE_DOMAINS = {
    "physics", "chemistry", "biology", "astronomy", "geography", "mathematics"
}
QUESTION_TYPES = {"mcq", "numeric", "short_answer", "symbolic", "qualitative"}
TAXONOMIES = {"A1", "A2", "B1", "B2", "C1", "C2"}
DECISIONS = {"keep_control", "keep_cross_domain", "transform_candidate", "reject"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_prompt(path: Path) -> dict[str, Any]:
    """Load one prompt, optionally extending a versioned prompt in the same directory."""

    prompt = json.loads(path.read_text(encoding="utf-8"))
    base_name = prompt.get("base_prompt")
    if not base_name:
        return prompt
    base = json.loads(path.with_name(base_name).read_text(encoding="utf-8"))
    addendum = prompt.get("system_prompt_addendum")
    if not isinstance(addendum, str) or not addendum.strip():
        raise ValueError("An extending prompt requires a non-empty system_prompt_addendum")
    return {
        **base,
        **prompt,
        "system_prompt": base["system_prompt"] + "\n\n" + addendum,
        "user_template": base["user_template"],
    }


def validate(analysis: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_fields = {
        "scientifically_valid",
        "validity_issue",
        "question_type",
        "answerability",
        "source_taxonomy",
        "primary_domain",
        "required_domains",
        "helpful_domains",
        "background_domains",
        "already_cross_domain",
        "mathematics_level",
        "reference_answer_role",
        "reference_answer_issue",
        "decision",
        "natural_extension_possible",
        "natural_extension_direction",
        "confidence",
    }
    missing = sorted(required_fields - analysis.keys())
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
    required_items = analysis.get("required_domains")
    domains = []
    if not isinstance(required_items, list):
        errors.append("required_domains is not a list")
    else:
        for item in required_items:
            domain = item.get("domain") if isinstance(item, dict) else None
            if domain not in CALLABLE_DOMAINS:
                errors.append(f"invalid required domain: {domain}")
            else:
                domains.append(domain)
    helpful_items = analysis.get("helpful_domains")
    helpful_domains = []
    if not isinstance(helpful_items, list):
        errors.append("helpful_domains is not a list")
    else:
        for item in helpful_items:
            domain = item.get("domain") if isinstance(item, dict) else None
            if domain not in CALLABLE_DOMAINS:
                errors.append(f"invalid helpful domain: {domain}")
            else:
                helpful_domains.append(domain)
    if set(domains) & set(helpful_domains):
        errors.append("required_domains and helpful_domains must be disjoint")
    if analysis.get("question_type") not in QUESTION_TYPES:
        errors.append("invalid question_type")
    taxonomy = analysis.get("source_taxonomy")
    if taxonomy not in TAXONOMIES:
        errors.append("invalid source_taxonomy")
    if analysis.get("decision") not in DECISIONS:
        errors.append("invalid decision")
    if analysis.get("answerability") not in {"objective", "qualitative"}:
        errors.append("invalid answerability")
    if analysis.get("reference_answer_role") not in {
        "objective_gold_candidate",
        "qualitative_reference",
        "unreliable_or_insufficient",
    }:
        errors.append("invalid reference_answer_role")
    if analysis.get("primary_domain") not in CALLABLE_DOMAINS | {None}:
        errors.append("invalid primary_domain")
    math_level = analysis.get("mathematics_level")
    if math_level not in {"none", "ordinary", "independent"}:
        errors.append("invalid mathematics_level")
    if math_level == "independent" and "mathematics" not in domains:
        errors.append("independent mathematics must be required")
    if math_level != "independent" and "mathematics" in domains:
        errors.append("required mathematics must be independent")
    cross = len(set(domains)) >= 2
    if analysis.get("already_cross_domain") is not cross:
        errors.append("already_cross_domain inconsistent with required_domains")
    if taxonomy in TAXONOMIES and taxonomy.endswith("2") != cross:
        errors.append("taxonomy cross-domain suffix inconsistent")
    expected_prefix = {
        "mcq": "A",
        "numeric": "B",
        "short_answer": "B",
        "symbolic": "B",
        "qualitative": "C",
    }.get(analysis.get("question_type"))
    if expected_prefix and not taxonomy.startswith(expected_prefix):
        errors.append("taxonomy inconsistent with question_type")
    if analysis.get("decision") == "keep_cross_domain" and not cross:
        errors.append("keep_cross_domain without two required domains")
    if analysis.get("decision") == "transform_candidate" and cross:
        errors.append("transform_candidate already cross-domain")
    natural = analysis.get("natural_extension_possible")
    if natural is not (analysis.get("decision") == "transform_candidate"):
        errors.append("natural_extension_possible inconsistent with decision")
    confidence = analysis.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        errors.append("invalid confidence")
    return errors


def analyze_one(
    item: dict[str, Any], prompt: dict[str, Any], model: str, max_tokens: int
) -> dict[str, Any]:
    client = DeepSeekJSONClient(model=model)
    user_prompt = prompt["user_template"].format(
        question=item["question"],
        choices_json=json.dumps(item["choices"], ensure_ascii=False),
        reference_answer=item["answer"],
    )
    try:
        analysis, metadata = client.generate_json(
            [
                {"role": "system", "content": prompt["system_prompt"]},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            reasoning_effort="low",
        )
        errors = validate(analysis)
        return {
            "protocol_id": item["protocol_id"],
            "source_domain": item["source_domain"],
            "sample_group": item["sample_group"],
            "surface_bucket": item["surface_bucket"],
            "analysis": analysis,
            "schema_valid": not errors,
            "validation_errors": errors,
            "api_metadata": metadata,
            "prompt_version": prompt["version"],
            "error": None,
        }
    except Exception as exc:  # Preserve failed IDs for explicit inspection/resume.
        return {
            "protocol_id": item["protocol_id"],
            "source_domain": item["source_domain"],
            "sample_group": item["sample_group"],
            "surface_bucket": item["surface_bucket"],
            "analysis": None,
            "schema_valid": False,
            "validation_errors": [],
            "api_metadata": None,
            "prompt_version": prompt["version"],
            "error": f"{type(exc).__name__}: {exc}",
        }


def write_summary(records: list[dict[str, Any]], path: Path) -> None:
    valid = [record for record in records if record.get("schema_valid")]

    def summarize_subset(subset: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(subset)
        return {
            "total": total,
            "taxonomy": dict(
                Counter(record["analysis"]["source_taxonomy"] for record in subset)
            ),
            "decision": dict(
                Counter(record["analysis"]["decision"] for record in subset)
            ),
            "question_type": dict(
                Counter(record["analysis"]["question_type"] for record in subset)
            ),
            "native_cross_domain": sum(
                bool(record["analysis"]["already_cross_domain"]) for record in subset
            ),
            "source_primary_domain_mismatches": sum(
                record["source_domain"] != record["analysis"]["primary_domain"]
                for record in subset
            ),
        }

    counters = {
        "source_domain": Counter(record["source_domain"] for record in valid),
        "sample_group": Counter(record["sample_group"] for record in valid),
        "taxonomy": Counter(record["analysis"]["source_taxonomy"] for record in valid),
        "decision": Counter(record["analysis"]["decision"] for record in valid),
        "question_type": Counter(record["analysis"]["question_type"] for record in valid),
        "primary_domain": Counter(str(record["analysis"]["primary_domain"]) for record in valid),
        "required_domain_set": Counter(
            "+".join(sorted(item["domain"] for item in record["analysis"]["required_domains"]))
            or "none"
            for record in valid
        ),
        "helpful_domain_set": Counter(
            "+".join(
                sorted(item["domain"] for item in record["analysis"]["helpful_domains"])
            )
            or "none"
            for record in valid
        ),
        "reference_answer_role": Counter(
            record["analysis"]["reference_answer_role"] for record in valid
        ),
    }
    mismatches = sum(
        record["source_domain"] != record["analysis"]["primary_domain"] for record in valid
    )
    independent_math = sum(
        record["analysis"]["mathematics_level"] == "independent" for record in valid
    )
    cross_domain = sum(bool(record["analysis"]["already_cross_domain"]) for record in valid)
    summary = {
        "total_records": len(records),
        "schema_valid_records": len(valid),
        "failed_or_invalid_records": len(records) - len(valid),
        "source_primary_domain_mismatches": mismatches,
        "independent_mathematics": independent_math,
        "native_cross_domain": cross_domain,
        "counts": {name: dict(counter) for name, counter in counters.items()},
        "by_sample_group": {
            group: summarize_subset(
                [record for record in valid if record["sample_group"] == group]
            )
            for group in ("representative", "stress")
        },
        "by_source_domain": {
            domain: {
                "total": len(subset),
                "decision": dict(
                    Counter(record["analysis"]["decision"] for record in subset)
                ),
                "primary_domain": dict(
                    Counter(str(record["analysis"]["primary_domain"]) for record in subset)
                ),
            }
            for domain in sorted({record["source_domain"] for record in valid})
            for subset in [[record for record in valid if record["source_domain"] == domain]]
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    base = PROJECT_ROOT / "datasets/processed/deepseek_protocol_v1"
    parser.add_argument("--input", type=Path, default=base / "protocol_sources_150.jsonl")
    parser.add_argument(
        "--prompt",
        type=Path,
        default=PROJECT_ROOT / "src/prompts/deepseek_source_analyzer_v2_1.json",
    )
    parser.add_argument("--output", type=Path, default=base / "source_analysis_v2_1.jsonl")
    parser.add_argument("--summary", type=Path, default=base / "source_analysis_summary_v2_1.json")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--protocol-id",
        action="append",
        help="Analyze only this protocol ID; may be repeated.",
    )
    args = parser.parse_args()

    items = read_jsonl(args.input)
    if args.protocol_id:
        requested_ids = set(args.protocol_id)
        items = [item for item in items if item["protocol_id"] in requested_ids]
        missing_ids = requested_ids - {item["protocol_id"] for item in items}
        if missing_ids:
            parser.error(f"Unknown protocol IDs: {sorted(missing_ids)}")
    if args.limit is not None:
        items = items[: args.limit]
    prompt = load_prompt(args.prompt)
    previous = read_jsonl(args.output) if args.output.exists() else []
    latest_previous = {record["protocol_id"]: record for record in previous}
    successful_previous = {
        protocol_id: record
        for protocol_id, record in latest_previous.items()
        if record.get("error") is None
    }
    completed = set(successful_previous)
    pending = [item for item in items if item["protocol_id"] not in completed]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in successful_previous.values():
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    analyze_one, item, prompt, args.model, args.max_tokens
                ): item["protocol_id"]
                for item in pending
            }
            for completed_count, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[{completed_count}/{len(pending)}] {record['protocol_id']} "
                    f"schema_valid={record['schema_valid']} error={record['error'] is not None}",
                    flush=True,
                )

    records = read_jsonl(args.output)
    selected_ids = {item["protocol_id"] for item in items}
    records = [record for record in records if record["protocol_id"] in selected_ids]
    write_summary(records, args.summary)
    print(f"summary={args.summary} completed={len(records)}/{len(items)}")


if __name__ == "__main__":
    main()
