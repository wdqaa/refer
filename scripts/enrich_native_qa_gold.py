#!/usr/bin/env python3
"""Add auditable Gold solutions and rubrics to a frozen native-QA evaluation split.

Input: frozen evaluation JSONL, versioned prompt, and runtime config. Output: resumable
raw DeepSeek traces, schema-valid enriched records, rejected/failed records, and a
usage/cost summary. Source questions and answers are never modified.

Example:
    python scripts/enrich_native_qa_gold.py --workers 3
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.deepseek_api import DeepSeekJSONClient


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def keyed_choice(item: dict[str, Any]) -> str | None:
    if not item.get("choices"):
        return None
    answer = str(item["answer"]).strip().upper()
    match = re.search(r"(?:FINAL\s+ANSWER\s*:\s*)?([A-J])\s*$", answer)
    return match.group(1) if match and match.group(1) in item["choices"] else None


def request_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item["native_qa_id"],
        "question_type": item["question_type"],
        "question": item["question"],
        "choices": item.get("choices") or None,
        "reference_answer": item["answer"],
        "reviewed_required_domains": item["required_domains"],
    }


def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize harmless JSON-shape variants without changing semantic content."""
    normalized = json.loads(json.dumps(result, ensure_ascii=False))
    # Some JSON-mode responses correctly generate all semantic fields but nest the
    # solution/rubric/issues beside answer_text inside gold_answer. Lift only those
    # explicitly emitted values when the corresponding top-level field is empty.
    gold = normalized.get("gold_answer")
    if isinstance(gold, dict):
        for field in ("standard_solution", "scoring_rubric", "issues"):
            if field in gold and not normalized.get(field):
                normalized[field] = gold.pop(field)
    if isinstance(normalized.get("standard_solution"), str):
        value = normalized["standard_solution"].strip()
        normalized["standard_solution"] = [value] if value else []
    if isinstance(normalized.get("scoring_rubric"), list):
        normalized["scoring_rubric"] = [
            entry for entry in normalized["scoring_rubric"]
            if not isinstance(entry, dict) or entry.get("points") != 0
        ]
    return normalized


def validate(item: dict[str, Any], result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(result) != {"id", "accepted", "gold_answer", "standard_solution", "scoring_rubric", "issues"}:
        errors.append("wrong fields")
    if result.get("id") != item["native_qa_id"]:
        errors.append("id mismatch")
    if not isinstance(result.get("accepted"), bool):
        errors.append("accepted must be boolean")
    gold = result.get("gold_answer")
    if not isinstance(gold, dict) or set(gold) != {"correct_choice", "answer_text"}:
        errors.append("invalid gold_answer")
        gold = {}
    solution = result.get("standard_solution")
    rubric = result.get("scoring_rubric")
    issues = result.get("issues")
    if not isinstance(solution, list) or any(not isinstance(x, str) or not x.strip() for x in solution):
        errors.append("invalid standard_solution")
        solution = []
    if not isinstance(rubric, list):
        errors.append("invalid scoring_rubric")
        rubric = []
    if not isinstance(issues, list) or any(not isinstance(x, str) or not x.strip() for x in issues):
        errors.append("invalid issues")
        issues = []
    if result.get("accepted") is True:
        if not 1 <= len(solution) <= 8:
            errors.append("accepted solution needs 1-8 steps")
        if not 1 <= len(rubric) <= 6:
            errors.append("accepted rubric needs 1-6 criteria")
        points = 0
        for entry in rubric:
            if not isinstance(entry, dict) or set(entry) != {"criterion", "points"}:
                errors.append("invalid rubric entry")
                continue
            if not isinstance(entry["criterion"], str) or not entry["criterion"].strip():
                errors.append("empty rubric criterion")
            if not isinstance(entry["points"], int) or entry["points"] <= 0:
                errors.append("invalid rubric points")
            else:
                points += entry["points"]
        if points != 10:
            errors.append("rubric points must sum to 10")
        if not isinstance(gold.get("answer_text"), str) or not gold.get("answer_text", "").strip():
            errors.append("accepted answer_text required")
        expected = keyed_choice(item)
        if expected is not None and gold.get("correct_choice") != expected:
            errors.append("MCQ choice changed")
        if expected is None and gold.get("correct_choice") is not None:
            errors.append("non-MCQ correct_choice must be null")
        if issues:
            errors.append("accepted item cannot have issues")
    elif result.get("accepted") is False:
        if gold.get("correct_choice") is not None or gold.get("answer_text") != "" or solution or rubric or not issues:
            errors.append("rejected item payload is inconsistent")
    return errors


def classify_batch(batch_id: str, items: list[dict[str, Any]], prompt: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    client = DeepSeekJSONClient(model=config["model"])
    user = prompt["user_template"].format(items_json=json.dumps([request_item(x) for x in items], ensure_ascii=False, indent=2))
    try:
        result, metadata = client.generate_json(
            [{"role": "system", "content": prompt["system_prompt"]}, {"role": "user", "content": user}],
            max_tokens=int(config["max_tokens"]), thinking_enabled=False, retries=0,
        )
        if set(result) != {"items"} or not isinstance(result["items"], list):
            raise ValueError("top-level JSON must contain only items")
        by_id = {entry.get("id"): entry for entry in result["items"] if isinstance(entry, dict)}
        checks = [{"id": item["native_qa_id"], "errors": validate(item, normalize_result(by_id.get(item["native_qa_id"], {})))} for item in items]
        return {"batch_id": batch_id, "item_ids": [x["native_qa_id"] for x in items], "result": result, "validation": checks, "api_metadata": metadata, "error": None}
    except Exception as exc:
        return {"batch_id": batch_id, "item_ids": [x["native_qa_id"] for x in items], "result": None, "validation": [], "api_metadata": None, "error": f"{type(exc).__name__}: {exc}"}


def cost(usage: Counter[str], peak: bool) -> float:
    factor = 2 if peak else 1
    return factor * (usage["prompt_cache_hit_tokens"] * 0.003 + usage["prompt_cache_miss_tokens"] * 0.15 + usage["completion_tokens"] * 0.6) / 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser()
    base = PROJECT_ROOT / "datasets/processed/native_qa_mining_v1/pilot_500/eval_split_v1"
    parser.add_argument("--input", type=Path, default=base / "eval_50.jsonl")
    parser.add_argument("--output-dir", type=Path, default=base)
    parser.add_argument("--prompt", type=Path, default=PROJECT_ROOT / "src/prompts/native_qa_gold_enricher_v1.json")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/data/native_qa_gold_enrichment_v1.json")
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--artifact-prefix",
        default="eval_50",
        help="Output filename prefix; the default preserves the historical eval50 names.",
    )
    args = parser.parse_args()
    items, prompt = read_jsonl(args.input), json.loads(args.prompt.read_text())
    config = json.loads(args.config.read_text())
    size, workers = int(config["batch_size"]), args.workers or int(config["workers"])
    groups = [items[i:i+size] for i in range(0, len(items), size)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_name = (
        "gold_enrichment_traces.jsonl"
        if args.artifact_prefix == "eval_50"
        else f"{args.artifact_prefix}_gold_enrichment_traces.jsonl"
    )
    trace_path = args.output_dir / trace_name
    previous = {x["batch_id"]: x for x in read_jsonl(trace_path)}
    pending = [(f"gold-enrich-{i:03d}", group) for i, group in enumerate(groups) if f"gold-enrich-{i:03d}" not in previous]
    with trace_path.open("w", encoding="utf-8") as handle:
        for trace in previous.values(): handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
        handle.flush()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(classify_batch, batch_id, group, prompt, config): batch_id for batch_id, group in pending}
            for n, future in enumerate(as_completed(futures), 1):
                trace = future.result(); handle.write(json.dumps(trace, ensure_ascii=False) + "\n"); handle.flush()
                print(f"[{n}/{len(pending)}] {trace['batch_id']} error={trace['error'] is not None}", flush=True)
    traces = read_jsonl(trace_path)
    source = {x["native_qa_id"]: x for x in items}
    enriched, rejected, failures = [], [], []
    usage: Counter[str] = Counter()
    for trace in traces:
        for key, value in ((trace.get("api_metadata") or {}).get("usage") or {}).items():
            if isinstance(value, int): usage[key] += value
        if trace.get("error"):
            failures.extend({"id": x, "reason": trace["error"]} for x in trace["item_ids"]); continue
        for result in trace["result"]["items"]:
            item_id = result.get("id")
            normalized = normalize_result(result)
            if item_id not in source or validate(source[item_id], normalized):
                failures.append({"id": item_id, "reason": "schema_validation_failed"}); continue
            record = {**source[item_id], "gold_enrichment": normalized, "gold_prompt_version": prompt["version"]}
            (enriched if normalized["accepted"] else rejected).append(record)
    # API batches finish concurrently, but exported datasets must retain the frozen
    # input order so rerunning from the same traces is byte-stable and easy to audit.
    source_order = {item["native_qa_id"]: index for index, item in enumerate(items)}
    enriched.sort(key=lambda item: source_order[item["native_qa_id"]])
    rejected.sort(key=lambda item: source_order[item["native_qa_id"]])
    names = (
        (f"{args.artifact_prefix}_enriched.jsonl", enriched),
        (
            "gold_rejected.jsonl"
            if args.artifact_prefix == "eval_50"
            else f"{args.artifact_prefix}_gold_rejected.jsonl",
            rejected,
        ),
        (
            "gold_failures.jsonl"
            if args.artifact_prefix == "eval_50"
            else f"{args.artifact_prefix}_gold_failures.jsonl",
            failures,
        ),
    )
    for name, records in names:
        with (args.output_dir / name).open("w", encoding="utf-8") as handle:
            for record in records: handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {"input": len(items), "enriched": len(enriched), "rejected": len(rejected), "failed": len(failures), "batches": len(traces), "api_errors": sum(bool(x.get("error")) for x in traces), "usage": dict(usage), "cost_usd": {"off_peak": round(cost(usage, False), 6), "peak": round(cost(usage, True), 6)}}
    summary_name = (
        "enrichment_summary.json"
        if args.artifact_prefix == "eval_50"
        else f"{args.artifact_prefix}_enrichment_summary.json"
    )
    (args.output_dir / summary_name).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
