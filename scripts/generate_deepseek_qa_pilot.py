#!/usr/bin/env python3
"""Produce, blind-solve, and judge the 30-item DeepSeek QA pilot.

Input: ``manifest_30.jsonl``, four versioned prompts, and ``DEEPSEEK_API_KEY``.
Output: a resumable trace, accepted QA JSONL, and compact summary JSON.
Example: ``python scripts/generate_deepseek_qa_pilot.py --workers 5``.
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
NEW_DOMAINS = DOMAINS - {"mathematics"}
RELATIONS = {
    "mechanism_explanation", "quantitative_modeling", "constraint_coupling",
    "measurement_observation",
}
QUESTION_TYPES = {"mcq", "numeric", "short_answer", "symbolic", "qualitative"}
FORBIDDEN_WORDING = re.compile(
    r"interdisciplin|cross[- ]domain|domain expert|from a (?:physics|chemistry|"
    r"mathematics|biology|astronomy|geography) perspective",
    flags=re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    return client.generate_json(
        [
            {"role": "system", "content": prompt_text(prompt)},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ],
        max_tokens=max_tokens,
        reasoning_effort="low",
    )


def domain_list(items: Any, *, object_items: bool = False) -> set[str]:
    if not isinstance(items, list):
        return set()
    if object_items:
        return {
            item.get("domain") for item in items
            if isinstance(item, dict) and item.get("domain") in DOMAINS
        }
    return {item for item in items if item in DOMAINS}


def validate_plan(item: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    if plan.get("decision") == "reject_transform":
        return []
    issues = []
    if plan.get("decision") != "propose":
        issues.append("invalid relation-plan decision")
    source_domains = domain_list(item["source_analysis"]["required_domains"], object_items=True)
    new_domain = plan.get("new_domain")
    if new_domain not in NEW_DOMAINS - source_domains:
        issues.append("new domain is invalid, mathematics, or already required")
    if plan.get("fusion_relation") not in RELATIONS:
        issues.append("invalid fusion relation")
    preferred_relation = item.get("preferred_fusion_relation")
    if preferred_relation is not None and plan.get("fusion_relation") != preferred_relation:
        issues.append("relation planner changed the preferred fusion relation")
    for field in (
        "scientific_bridge", "new_condition", "new_dependency", "why_natural",
        "source_dependency_preserved", "expected_solution_change",
    ):
        if not isinstance(plan.get(field), str) or not plan[field].strip():
            issues.append(f"missing {field}")
    return issues


def validate_transform(
    item: dict[str, Any], plan: dict[str, Any], candidate: dict[str, Any]
) -> list[str]:
    if candidate.get("decision") == "reject_transform":
        return []
    issues = []
    if candidate.get("decision") != "transform":
        issues.append("invalid transformer decision")
        return issues
    question = candidate.get("question")
    source_question = item["source"]["question"]
    if not isinstance(question, str) or not question.strip():
        issues.append("missing transformed question")
    elif question.strip() == source_question.strip():
        issues.append("transformed question is unchanged")
    elif FORBIDDEN_WORDING.search(question):
        issues.append("question leaks routing language")
    if candidate.get("question_type") not in QUESTION_TYPES:
        issues.append("invalid transformed question type")
    source_domains = domain_list(item["source_analysis"]["required_domains"], object_items=True)
    after = domain_list(candidate.get("required_domains_after"))
    new_domain = candidate.get("new_required_domain")
    if new_domain != plan.get("new_domain"):
        issues.append("transformer changed the planned new domain")
    if new_domain not in NEW_DOMAINS - source_domains:
        issues.append("invalid new required domain")
    if after != source_domains | {new_domain}:
        issues.append("required_domains_after is not source domains plus one")
    choices = candidate.get("choices")
    if candidate.get("question_type") == "mcq":
        if not isinstance(choices, dict) or len(choices) < 3:
            issues.append("invalid transformed choices")
        elif candidate.get("provisional_correct_choice") not in choices:
            issues.append("provisional correct choice is missing")
    if not isinstance(candidate.get("derivation"), list) or not candidate["derivation"]:
        issues.append("missing provisional derivation")
    return issues


def validate_solver(question: str, choices: Any, solver: dict[str, Any]) -> list[str]:
    issues = []
    if solver.get("well_posed") is not True:
        issues.append("blind solver judged item not well posed")
    question_type = solver.get("question_type")
    if question_type not in QUESTION_TYPES:
        issues.append("invalid solver question type")
    if not isinstance(solver.get("answer"), str) or not solver["answer"].strip():
        issues.append("missing blind answer")
    if not isinstance(solver.get("standard_solution"), list) or not solver["standard_solution"]:
        issues.append("missing standard solution")
    observed = solver.get("required_domains_observed")
    helpful = solver.get("helpful_domains_observed")
    if not isinstance(observed, list) or set(observed) - DOMAINS:
        issues.append("invalid solver required domains")
    if not isinstance(helpful, list) or set(helpful) - DOMAINS:
        issues.append("invalid solver helpful domains")
    if isinstance(observed, list) and isinstance(helpful, list) and set(observed) & set(helpful):
        issues.append("solver required/helpful domains overlap")
    if question_type == "mcq":
        if not isinstance(choices, dict) or solver.get("correct_choice") not in choices:
            issues.append("invalid blind-solver correct choice")
    if question_type == "qualitative":
        rubric = solver.get("rubric")
        if not isinstance(rubric, dict) or not rubric.get("required_points") or not rubric.get("fatal_errors"):
            issues.append("qualitative answer lacks usable rubric")
    if not question.strip():
        issues.append("empty final question")
    return issues


def validate_judge(route: str, judge: dict[str, Any]) -> list[str]:
    issues = []
    required_true = (
        "scientifically_valid", "well_posed", "answer_correct",
        "answer_format_valid", "required_domains_defensible",
    )
    for field in required_true:
        if judge.get(field) is not True:
            issues.append(f"judge rejected {field}")
    if route == "transform_candidate":
        for field in (
            "source_dependency_preserved", "new_dependency_indispensable", "integrated_reasoning"
        ):
            if judge.get(field) is not True:
                issues.append(f"judge rejected {field}")
    if judge.get("difficulty_for_8b") == "excessive":
        issues.append("judge marked difficulty excessive for 8B")
    if judge.get("accepted") is not True:
        issues.append("judge accepted=false")
    if not isinstance(judge.get("observed_required_domains"), list) or set(
        judge.get("observed_required_domains") or []
    ) - DOMAINS:
        issues.append("judge returned invalid domains")
    reported_issues = judge.get("issues")
    if not isinstance(reported_issues, list):
        issues.append("judge issues must be a list")
    elif reported_issues:
        issues.append("judge reported material issues")
    return issues


def process_item(
    item: dict[str, Any], prompts: dict[str, dict[str, Any]], model: str, max_tokens: int
) -> dict[str, Any]:
    client = DeepSeekJSONClient(model=model)
    route = item["route"]
    plan = candidate = None
    plan_meta = candidate_meta = None
    plan_issues: list[str] = []
    candidate_issues: list[str] = []
    solver = judge = None
    solver_meta = judge_meta = None
    solver_issues: list[str] = []
    judge_issues: list[str] = []
    error = None
    status = "error"

    try:
        if route == "transform_candidate":
            planner_payload = {
                "source_question": item["source"]["question"],
                "source_choices": item["source"]["choices"],
                "verified_source_analysis": item["source_analysis"],
            }
            if item.get("preferred_fusion_relation") is not None:
                planner_payload["preferred_fusion_relation"] = item[
                    "preferred_fusion_relation"
                ]
            plan, plan_meta = call_json(
                client,
                prompts["planner"],
                planner_payload,
                max_tokens,
            )
            plan_issues = validate_plan(item, plan)
            if plan.get("decision") == "reject_transform":
                status = "rejected_by_relation_planner"
            elif plan_issues:
                status = "rejected_by_relation_gate"
            else:
                candidate, candidate_meta = call_json(
                    client,
                    prompts["transformer"],
                    {
                        "source_question": item["source"]["question"],
                        "source_choices": item["source"]["choices"],
                        "untrusted_source_answer": item["source"]["answer"],
                        "verified_source_analysis": item["source_analysis"],
                        "relation_plan": plan,
                    },
                    max_tokens,
                )
                candidate_issues = validate_transform(item, plan, candidate)
                if candidate.get("decision") == "reject_transform":
                    status = "rejected_by_transformer"
                elif candidate_issues:
                    status = "rejected_by_transform_gate"

        if route != "transform_candidate":
            final_question = item["source"]["question"]
            final_choices = item["source"]["choices"] or None
            final_type = item["source_analysis"]["question_type"]
            ready = True
        else:
            ready = status == "error" and candidate is not None and not candidate_issues
            final_question = candidate.get("question") if candidate else None
            final_choices = candidate.get("choices") if candidate else None
            final_type = candidate.get("question_type") if candidate else None

        if ready:
            solver, solver_meta = call_json(
                client,
                prompts["solver"],
                {"question": final_question, "choices": final_choices, "declared_type": final_type},
                max_tokens,
            )
            solver_issues = validate_solver(final_question, final_choices, solver)
            if solver_issues:
                status = "rejected_by_solver_gate"
            else:
                judge, judge_meta = call_json(
                    client,
                    prompts["judge"],
                    {
                        "route": route,
                        "source_question": item["source"]["question"],
                        "untrusted_source_answer": item["source"]["answer"],
                        "verified_source_analysis": item["source_analysis"],
                        "relation_plan": plan,
                        "transformed_candidate": candidate,
                        "final_question": final_question,
                        "final_choices": final_choices,
                        "blind_solver_gold": solver,
                    },
                    max_tokens,
                )
                judge_issues = validate_judge(route, judge)
                if route == "transform_candidate":
                    claimed = set(candidate["required_domains_after"])
                    judged = set(judge.get("observed_required_domains") or [])
                    if not claimed.issubset(judged):
                        judge_issues.append("judge did not observe every transformed required domain")
                status = "accepted" if not judge_issues else "rejected_by_judge"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        status = "error"

    return {
        "schema_version": "deepseek_qa_trace.v1",
        "qa_id": item["qa_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "route": route,
        "source": item["source"],
        "source_analysis": item["source_analysis"],
        "relation_planner": {"result": plan, "metadata": plan_meta},
        "transformer": {"result": candidate, "metadata": candidate_meta},
        "blind_solver": {"result": solver, "metadata": solver_meta},
        "judge": {"result": judge, "metadata": judge_meta},
        "gate": {
            "plan_issues": plan_issues,
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
    transformed = trace["route"] == "transform_candidate"
    source = trace["source"]
    return {
        "schema_version": "verified_science_qa.v1",
        "id": trace["qa_id"],
        "source": {
            "source_file": source["source_file"],
            "source_id": source["source_id"],
            "original_question": source["question"],
            "original_answer": source["answer"],
        },
        "origin_type": (
            "transformed_cross_domain" if transformed
            else "native_cross_domain" if trace["route"] == "keep_cross_domain"
            else "native_single"
        ),
        "question_type": solver["question_type"],
        "question": candidate["question"] if transformed else source["question"],
        "choices": candidate["choices"] if transformed else (source["choices"] or None),
        "gold": {
            "answer": solver["answer"],
            "correct_choice": solver["correct_choice"],
            "normalized_answer": solver["normalized_answer"],
            "standard_solution": solver["standard_solution"],
            "rubric": solver["rubric"],
        },
        "required_domains": judge["observed_required_domains"],
        "helpful_domains": solver["helpful_domains_observed"],
        "fusion_relation": (
            trace["relation_planner"]["result"]["fusion_relation"] if transformed else None
        ),
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


def write_outputs(traces: list[dict[str, Any]], verified_path: Path, summary_path: Path) -> None:
    accepted = [record for record in traces if record["gate"]["status"] == "accepted"]
    verified = [verified_record(record) for record in accepted]
    verified_path.parent.mkdir(parents=True, exist_ok=True)
    with verified_path.open("w", encoding="utf-8") as handle:
        for record in verified:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "total": len(traces),
        "accepted": len(verified),
        "status": dict(Counter(record["gate"]["status"] for record in traces)),
        "route": dict(Counter(record["route"] for record in traces)),
        "accepted_origin_type": dict(Counter(record["origin_type"] for record in verified)),
        "accepted_question_type": dict(Counter(record["question_type"] for record in verified)),
        "accepted_difficulty": dict(Counter(record["difficulty"] for record in verified)),
        "accepted_required_domains": dict(
            Counter("+".join(sorted(record["required_domains"])) for record in verified)
        ),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    base = PROJECT_ROOT / "datasets/processed/deepseek_qa_pilot_v1"
    parser.add_argument("--manifest", type=Path, default=base / "manifest_30.jsonl")
    parser.add_argument("--trace", type=Path, default=base / "trace.jsonl")
    parser.add_argument("--verified", type=Path, default=base / "verified_qa.jsonl")
    parser.add_argument("--summary", type=Path, default=base / "summary.json")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--qa-id", action="append", help="Run only this QA ID; may repeat.")
    parser.add_argument(
        "--planner-prompt",
        type=Path,
        default=PROJECT_ROOT / "src/prompts/deepseek_relation_planner_v1.json",
    )
    parser.add_argument(
        "--transformer-prompt",
        type=Path,
        default=PROJECT_ROOT / "src/prompts/deepseek_question_transformer_v1.json",
    )
    parser.add_argument(
        "--judge-prompt",
        type=Path,
        default=PROJECT_ROOT / "src/prompts/deepseek_qa_judge_v1.json",
    )
    args = parser.parse_args()

    prompt_dir = PROJECT_ROOT / "src/prompts"
    prompts = {
        "planner": load_prompt(args.planner_prompt),
        "transformer": load_prompt(args.transformer_prompt),
        "solver": load_prompt(prompt_dir / "deepseek_qa_solver_v1.json"),
        "judge": load_prompt(args.judge_prompt),
    }
    items = read_jsonl(args.manifest)
    if args.qa_id:
        requested_ids = set(args.qa_id)
        items = [item for item in items if item["qa_id"] in requested_ids]
        missing_ids = requested_ids - {item["qa_id"] for item in items}
        if missing_ids:
            parser.error(f"Unknown QA IDs: {sorted(missing_ids)}")
    if args.limit is not None:
        items = items[: args.limit]
    previous = read_jsonl(args.trace) if args.trace.exists() else []
    latest = {record["qa_id"]: record for record in previous}
    terminal = {
        qa_id: record for qa_id, record in latest.items() if record.get("error") is None
    }
    pending = [item for item in items if item["qa_id"] not in terminal]

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    with args.trace.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_item, item, prompts, args.model, args.max_tokens): item["qa_id"]
                for item in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[{index}/{len(pending)}] {record['qa_id']} {record['gate']['status']}",
                    flush=True,
                )

    all_attempts = read_jsonl(args.trace)
    latest = {record["qa_id"]: record for record in all_attempts}
    selected_ids = {item["qa_id"] for item in items}
    traces = [record for qa_id, record in latest.items() if qa_id in selected_ids]
    write_outputs(traces, args.verified, args.summary)
    print(f"summary={args.summary} completed={len(traces)}/{len(items)}")


if __name__ == "__main__":
    main()
