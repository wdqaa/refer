#!/usr/bin/env python3
"""Run fixed human bridge pairs through Transformer, Blind Solver, and Judge.

Input: eight fixed source-bridge pairs, existing source records, prompts, and API key.
Output: complete traces, machine-accepted QA, and a compact positive/negative summary.
Example: ``python scripts/generate_human_bridge_transform_smoke.py --workers 2``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.deepseek_api import DeepSeekJSONClient


DOMAINS = {"physics", "chemistry", "biology", "astronomy", "geography", "mathematics"}
QUESTION_TYPES = {"mcq", "numeric", "short_answer", "symbolic", "qualitative"}
FORBIDDEN_WORDING = re.compile(
    r"interdisciplin|cross[- ]domain|domain expert|bridge|combine (?:physics|chemistry|"
    r"mathematics|biology|astronomy|geography)",
    flags=re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_prompt(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prompt_text(prompt: dict[str, Any]) -> str:
    return prompt["system_prompt"] + "\n\nRequired JSON schema:\n" + json.dumps(
        prompt["output_schema"], ensure_ascii=False, indent=2
    )


def call_json(
    client: DeepSeekJSONClient,
    prompt: dict[str, Any],
    payload: dict[str, Any],
    max_tokens: int,
    thinking_enabled: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return client.generate_json(
        [
            {"role": "system", "content": prompt_text(prompt)},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ],
        max_tokens=max_tokens,
        reasoning_effort="low",
        thinking_enabled=thinking_enabled,
        retries=0,
    )


def build_pairs(sources: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    source_by_id = {item["bridge_source_id"]: item for item in sources}
    pairs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in config["pairs"]:
        source_id = spec["bridge_source_id"]
        if source_id in seen or source_id not in source_by_id:
            raise ValueError(f"Duplicate or unknown bridge source ID: {source_id}")
        if spec["expected_case"] not in {"positive", "negative"}:
            raise ValueError("expected_case must be positive or negative")
        source_item = source_by_id[source_id]
        primary = source_item["primary_domain"]
        target = spec["target_domain"]
        if target not in DOMAINS - {primary}:
            raise ValueError(f"Invalid target domain for {source_id}: {target}")
        seen.add(source_id)
        pairs.append(
            {
                "pair_id": source_id.replace("bridge-v1-", "human-bridge-v1-", 1),
                "expected_case": spec["expected_case"],
                "primary_domain": primary,
                "target_domain": target,
                "bridge": spec["bridge"],
                "source": source_item["source"],
                "source_analysis": source_item["source_analysis"],
            }
        )
    return pairs


def validate_transform(pair: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    if candidate.get("decision") == "reject_transform":
        return []
    issues: list[str] = []
    if candidate.get("decision") != "transform":
        return ["invalid transformer decision"]
    question = candidate.get("question")
    if not isinstance(question, str) or not question.strip():
        issues.append("missing transformed question")
    elif question.strip() == pair["source"]["question"].strip():
        issues.append("transformed question is unchanged")
    elif FORBIDDEN_WORDING.search(question):
        issues.append("question leaks bridge or domain wording")
    question_type = candidate.get("question_type")
    if question_type not in QUESTION_TYPES:
        issues.append("invalid transformed question type")
    target = pair["target_domain"]
    primary = pair["primary_domain"]
    if candidate.get("new_required_domain") != target:
        issues.append("transformer changed the fixed target domain")
    domains_after = candidate.get("required_domains_after")
    if not isinstance(domains_after, list) or set(domains_after) != {primary, target}:
        issues.append("required_domains_after must equal primary plus target domain")
    choices = candidate.get("choices")
    if question_type == "mcq":
        if not isinstance(choices, dict) or len(choices) < 3:
            issues.append("invalid transformed choices")
        elif candidate.get("provisional_correct_choice") not in choices:
            issues.append("provisional correct choice is missing")
    if not isinstance(candidate.get("derivation"), list) or not candidate["derivation"]:
        issues.append("missing provisional derivation")
    for field in (
        "source_dependency_preserved",
        "new_dependency_indispensable",
        "transformation_summary",
    ):
        if not isinstance(candidate.get(field), str) or not candidate[field].strip():
            issues.append(f"missing {field}")
    return issues


def validate_solver(
    question: str, choices: Any, solver: dict[str, Any], claimed_domains: set[str]
) -> list[str]:
    issues: list[str] = []
    if solver.get("well_posed") is not True:
        issues.append("blind solver judged item not well posed")
    if solver.get("question_type") not in QUESTION_TYPES:
        issues.append("invalid solver question type")
    if not isinstance(solver.get("answer"), str) or not solver["answer"].strip():
        issues.append("missing blind answer")
    if not isinstance(solver.get("standard_solution"), list) or not solver["standard_solution"]:
        issues.append("missing standard solution")
    observed = solver.get("required_domains_observed")
    helpful = solver.get("helpful_domains_observed")
    if not isinstance(observed, list) or set(observed) - DOMAINS:
        issues.append("invalid solver required domains")
    elif not claimed_domains.issubset(set(observed)):
        issues.append("blind solver did not observe both claimed required domains")
    if not isinstance(helpful, list) or set(helpful) - DOMAINS:
        issues.append("invalid solver helpful domains")
    elif isinstance(observed, list) and set(observed) & set(helpful):
        issues.append("solver required/helpful domains overlap")
    if solver.get("question_type") == "mcq":
        if not isinstance(choices, dict) or solver.get("correct_choice") not in choices:
            issues.append("invalid blind-solver correct choice")
    if solver.get("question_type") == "qualitative":
        rubric = solver.get("rubric")
        if not isinstance(rubric, dict) or not rubric.get("required_points") or not rubric.get("fatal_errors"):
            issues.append("qualitative answer lacks usable rubric")
    if not question.strip():
        issues.append("empty final question")
    return issues


def validate_judge(judge: dict[str, Any], claimed_domains: set[str]) -> list[str]:
    issues: list[str] = []
    for field in (
        "scientifically_valid",
        "well_posed",
        "answer_correct",
        "answer_format_valid",
        "required_domains_defensible",
        "source_dependency_preserved",
        "new_dependency_indispensable",
        "integrated_reasoning",
    ):
        if judge.get(field) is not True:
            issues.append(f"judge rejected {field}")
    if judge.get("difficulty_for_8b") == "excessive":
        issues.append("judge marked difficulty excessive for 8B")
    observed = judge.get("observed_required_domains")
    if not isinstance(observed, list) or set(observed) - DOMAINS:
        issues.append("judge returned invalid domains")
    elif not claimed_domains.issubset(set(observed)):
        issues.append("judge did not observe both claimed required domains")
    reported = judge.get("issues")
    if not isinstance(reported, list):
        issues.append("judge issues must be a list")
    elif reported:
        issues.append("judge reported material issues")
    if judge.get("accepted") is not True:
        issues.append("judge accepted=false")
    return issues


def process_pair(
    pair: dict[str, Any],
    prompts: dict[str, dict[str, Any]],
    model: str,
    max_tokens: int,
    thinking_enabled: bool,
) -> dict[str, Any]:
    client = DeepSeekJSONClient(model=model)
    candidate = solver = judge = None
    candidate_meta = solver_meta = judge_meta = None
    candidate_issues: list[str] = []
    solver_issues: list[str] = []
    judge_issues: list[str] = []
    status = "error"
    error = None
    try:
        candidate, candidate_meta = call_json(
            client,
            prompts["transformer"],
            {
                "source_question": pair["source"]["question"],
                "source_choices": pair["source"].get("choices", {}),
                "source_answer": pair["source"]["answer"],
                "source_analysis": pair["source_analysis"],
                "primary_domain": pair["primary_domain"],
                "target_domain": pair["target_domain"],
                "bridge": pair["bridge"],
            },
            max_tokens,
            thinking_enabled,
        )
        candidate_issues = validate_transform(pair, candidate)
        if candidate.get("decision") == "reject_transform":
            status = "rejected_by_transformer"
        elif candidate_issues:
            status = "rejected_by_transform_gate"
        else:
            claimed_domains = {pair["primary_domain"], pair["target_domain"]}
            solver, solver_meta = call_json(
                client,
                prompts["solver"],
                {
                    "question": candidate["question"],
                    "choices": candidate["choices"],
                    "declared_type": candidate["question_type"],
                },
                max_tokens,
                thinking_enabled,
            )
            solver_issues = validate_solver(
                candidate["question"], candidate["choices"], solver, claimed_domains
            )
            if solver_issues:
                status = "rejected_by_solver_gate"
            else:
                judge, judge_meta = call_json(
                    client,
                    prompts["judge"],
                    {
                        "source_question": pair["source"]["question"],
                        "source_answer": pair["source"]["answer"],
                        "source_analysis": pair["source_analysis"],
                        "human_bridge": {
                            "primary_domain": pair["primary_domain"],
                            "target_domain": pair["target_domain"],
                            "bridge": pair["bridge"],
                        },
                        "transformed_candidate": candidate,
                        "final_question": candidate["question"],
                        "final_choices": candidate["choices"],
                        "blind_solver_gold": solver,
                    },
                    max_tokens,
                    thinking_enabled,
                )
                judge_issues = validate_judge(judge, claimed_domains)
                status = "accepted" if not judge_issues else "rejected_by_judge"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        status = "error"
    return {
        "schema_version": "human_bridge_qa_trace.v1",
        "pair_id": pair["pair_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expected_case": pair["expected_case"],
        "primary_domain": pair["primary_domain"],
        "target_domain": pair["target_domain"],
        "bridge": pair["bridge"],
        "source": pair["source"],
        "source_analysis": pair["source_analysis"],
        "transformer": {"result": candidate, "metadata": candidate_meta},
        "blind_solver": {"result": solver, "metadata": solver_meta},
        "judge": {"result": judge, "metadata": judge_meta},
        "gate": {
            "candidate_issues": candidate_issues,
            "solver_issues": solver_issues,
            "judge_issues": judge_issues,
            "status": status,
        },
        "prompt_versions": {name: prompt["version"] for name, prompt in prompts.items()},
        "error": error,
    }


def verified_record(trace: dict[str, Any]) -> dict[str, Any]:
    candidate = trace["transformer"]["result"]
    solver = trace["blind_solver"]["result"]
    judge = trace["judge"]["result"]
    source = trace["source"]
    return {
        "schema_version": "verified_science_qa.v1",
        "id": trace["pair_id"],
        "source": {
            "source_file": source["source_file"],
            "source_id": source["source_id"],
            "original_question": source["question"],
            "original_answer": source["answer"],
        },
        "origin_type": "transformed_cross_domain",
        "question_type": solver["question_type"],
        "question": candidate["question"],
        "choices": candidate["choices"],
        "gold": {
            "answer": solver["answer"],
            "correct_choice": solver["correct_choice"],
            "normalized_answer": solver["normalized_answer"],
            "standard_solution": solver["standard_solution"],
            "rubric": solver["rubric"],
        },
        "required_domains": judge["observed_required_domains"],
        "helpful_domains": solver["helpful_domains_observed"],
        "fusion_relation": "bridge_guided",
        "bridge": trace["bridge"],
        "difficulty": judge["difficulty_for_8b"],
        "verification": {
            "scientifically_valid": judge["scientifically_valid"],
            "answer_correct": judge["answer_correct"],
            "dependency_verified": judge["required_domains_defensible"],
            "integrated_reasoning": judge["integrated_reasoning"],
            "accepted": judge["accepted"],
        },
        "provenance": {
            "model": trace["judge"]["metadata"]["model"],
            "prompt_versions": trace["prompt_versions"],
        },
    }


def write_outputs(
    traces: list[dict[str, Any]], verified_path: Path, summary_path: Path
) -> None:
    accepted = [trace for trace in traces if trace["gate"]["status"] == "accepted"]
    verified = [verified_record(trace) for trace in accepted]
    verified_path.parent.mkdir(parents=True, exist_ok=True)
    with verified_path.open("w", encoding="utf-8") as handle:
        for record in verified:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "total": len(traces),
        "transform_decision": dict(
            Counter(
                ((trace["transformer"]["result"] or {}).get("decision") or "error")
                for trace in traces
            )
        ),
        "status": dict(Counter(trace["gate"]["status"] for trace in traces)),
        "by_expected_case": {
            case: {
                "total": len(subset),
                "transform": sum(
                    (trace["transformer"]["result"] or {}).get("decision") == "transform"
                    for trace in subset
                ),
                "reject_transform": sum(
                    (trace["transformer"]["result"] or {}).get("decision")
                    == "reject_transform"
                    for trace in subset
                ),
                "machine_accepted": sum(
                    trace["gate"]["status"] == "accepted" for trace in subset
                ),
            }
            for case in ("positive", "negative")
            for subset in [[trace for trace in traces if trace["expected_case"] == case]]
        },
        "machine_accepted": len(verified),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    base = PROJECT_ROOT / "datasets/processed/human_bridge_transform_smoke_v1"
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/data/human_bridge_transform_smoke_v1.json",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=PROJECT_ROOT / "datasets/processed/bridge_guided_calibration_v1/sources_25.jsonl",
    )
    parser.add_argument("--trace", type=Path, default=base / "trace.jsonl")
    parser.add_argument("--verified", type=Path, default=base / "machine_accepted_qa.jsonl")
    parser.add_argument("--summary", type=Path, default=base / "summary.json")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument(
        "--thinking-enabled",
        action="store_true",
        help="Enable low-effort thinking; default is non-thinking JSON generation.",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    pairs = build_pairs(read_jsonl(args.sources), config)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    prompt_dir = PROJECT_ROOT / "src/prompts"
    prompts = {
        "transformer": load_prompt(prompt_dir / "human_bridge_question_transformer_v1.json"),
        "solver": load_prompt(prompt_dir / "deepseek_qa_solver_v1.json"),
        "judge": load_prompt(prompt_dir / "human_bridge_qa_judge_v1.json"),
    }
    previous = read_jsonl(args.trace)
    latest = {trace["pair_id"]: trace for trace in previous}
    pending = [pair for pair in pairs if pair["pair_id"] not in latest]
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    with args.trace.open("w", encoding="utf-8") as handle:
        for trace in latest.values():
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
        handle.flush()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_pair,
                    pair,
                    prompts,
                    args.model,
                    args.max_tokens,
                    args.thinking_enabled,
                ): pair["pair_id"]
                for pair in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                trace = future.result()
                handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[{index}/{len(pending)}] {trace['pair_id']} {trace['gate']['status']}",
                    flush=True,
                )
    selected_ids = {pair["pair_id"] for pair in pairs}
    traces = [trace for trace in read_jsonl(args.trace) if trace["pair_id"] in selected_ids]
    write_outputs(traces, args.verified, args.summary)
    print(f"summary={args.summary} completed={len(traces)}/{len(pairs)}")


if __name__ == "__main__":
    main()
