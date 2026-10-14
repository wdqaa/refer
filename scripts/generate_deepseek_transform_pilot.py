"""Generate and blind-solve a small DeepSeek transformation pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.source_analysis import ALLOWED_DOMAINS, load_gold  # noqa: E402
from src.generation.deepseek_api import DeepSeekJSONClient  # noqa: E402


DEFAULT_IDS = [
    "sa_single_02",
    "sa_surface_01",
    "sa_surface_04",
    "sa_surface_05",
    "sa_subfield_02",
]
FORBIDDEN_WORDING = re.compile(
    r"interdisciplin|cross[- ]domain|domain expert|from a (?:physics|chemistry|mathematics|biology|astronomy|geography) perspective",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--generator-prompt", required=True, type=Path)
    parser.add_argument("--solver-prompt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ids", nargs="*", default=DEFAULT_IDS)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--reasoning-effort", choices=("low", "high", "max"), default="high")
    parser.add_argument("--max-tokens", type=int, default=4096)
    return parser.parse_args()


def _prompt_text(config: dict[str, Any]) -> str:
    return config["system_prompt"] + "\n\nRequired JSON schema:\n" + json.dumps(
        config["output_schema"], ensure_ascii=False, indent=2
    )


def _required_domains(analysis: dict[str, Any]) -> set[str]:
    return {item["domain"] for item in analysis["required_domains"]}


def validate_candidate(source: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if candidate.get("decision") not in {"transform", "reject_transform"}:
        return ["invalid generator decision"]
    if candidate["decision"] == "reject_transform":
        return issues
    question = candidate.get("question")
    if not isinstance(question, str) or not question.strip():
        issues.append("missing transformed question")
    elif question.strip() == source["source"]["question"].strip():
        issues.append("transformed question is unchanged")
    elif FORBIDDEN_WORDING.search(question):
        issues.append("question contains explicit routing/fusion wording")
    before = _required_domains(source["gold"])
    after = set(candidate.get("required_domains_after") or [])
    new_domain = candidate.get("new_required_domain")
    if after - ALLOWED_DOMAINS:
        issues.append("unsupported required domain")
    if new_domain not in ALLOWED_DOMAINS - before:
        issues.append("new domain is absent, invalid, or already required")
    if after != before | {new_domain}:
        issues.append("required_domains_after must equal source domains plus one new domain")
    choices = candidate.get("choices")
    correct = candidate.get("correct_choice")
    if candidate.get("answer_type") == "multiple_choice":
        if not isinstance(choices, dict) or len(choices) < 3:
            issues.append("multiple-choice candidate has invalid choices")
        elif correct not in choices:
            issues.append("correct_choice is not a choice key")
    if not candidate.get("derivation"):
        issues.append("missing candidate derivation")
    if not candidate.get("ablation_without_new_domain"):
        issues.append("missing new-domain ablation explanation")
    if not candidate.get("source_dependency_preserved"):
        issues.append("missing source-dependency preservation explanation")
    if not candidate.get("ablation_without_source_domain"):
        issues.append("missing source-domain ablation explanation")
    return issues


def validate_solver(candidate: dict[str, Any], solver: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if solver.get("well_posed") is not True:
        issues.append("blind solver judged the question not well posed")
    observed = set(solver.get("required_domains_observed") or [])
    claimed = set(candidate.get("required_domains_after") or [])
    if observed - ALLOWED_DOMAINS:
        issues.append("blind solver returned unsupported domain")
    if candidate.get("new_required_domain") not in observed:
        issues.append("blind solver did not require the claimed new domain")
    missing_claimed = claimed - observed
    if missing_claimed:
        issues.append("blind solver did not require claimed domains: " + ", ".join(sorted(missing_claimed)))
    if candidate.get("answer_type") == "multiple_choice":
        if solver.get("correct_choice") != candidate.get("correct_choice"):
            issues.append("generator and blind solver disagree on correct choice")
    if not solver.get("derivation"):
        issues.append("blind solver returned no derivation")
    return issues


def main() -> int:
    args = parse_args()
    all_items = load_gold(args.gold, args.dataset_root)
    indexed = {item["id"]: item for item in all_items}
    selected = []
    for item_id in args.ids:
        if item_id not in indexed:
            raise ValueError(f"Unknown gold id: {item_id}")
        item = indexed[item_id]
        if item["gold"]["decision"] != "transform_candidate":
            raise ValueError(f"{item_id} is not a verified transform_candidate")
        selected.append(item)

    generator_prompt = json.loads(args.generator_prompt.read_text(encoding="utf-8"))
    solver_prompt = json.loads(args.solver_prompt.read_text(encoding="utf-8"))
    client = DeepSeekJSONClient(model=args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    if args.output.exists():
        completed = {
            json.loads(line)["id"] for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()
        }

    with args.output.open("a", encoding="utf-8") as handle:
        for item in selected:
            if item["id"] in completed:
                continue
            generator_input = {
                "source_question": item["source"]["question"],
                "source_choices": item["source"]["choices"],
                "source_answer": item["source"]["answer"],
                "verified_source_analysis": item["gold"],
            }
            candidate, generator_meta = client.generate_json(
                [
                    {"role": "system", "content": _prompt_text(generator_prompt)},
                    {"role": "user", "content": json.dumps(generator_input, ensure_ascii=False, indent=2)},
                ],
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
            )
            candidate_issues = validate_candidate(item, candidate)
            solver = None
            solver_meta = None
            solver_issues: list[str] = []
            if candidate.get("decision") == "transform" and not candidate_issues:
                blind_input = {
                    "question": candidate["question"],
                    "answer_type": candidate.get("answer_type"),
                    "choices": candidate.get("choices"),
                }
                solver, solver_meta = client.generate_json(
                    [
                        {"role": "system", "content": _prompt_text(solver_prompt)},
                        {"role": "user", "content": json.dumps(blind_input, ensure_ascii=False, indent=2)},
                    ],
                    max_tokens=args.max_tokens,
                    reasoning_effort=args.reasoning_effort,
                )
                solver_issues = validate_solver(candidate, solver)
            if candidate.get("decision") == "reject_transform":
                status = "rejected_by_generator"
            elif candidate_issues or solver_issues:
                status = "rejected_by_gate"
            else:
                status = "provisional_accept_needs_human_review"
            record = {
                "schema_version": "deepseek_transform_pilot.v1",
                "id": item["id"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": {
                    "file": item["source_file"],
                    "id": item["source_id"],
                    "question": item["source"]["question"],
                    "choices": item["source"]["choices"],
                    "answer": item["source"]["answer"],
                },
                "verified_source_analysis": item["gold"],
                "generator": {
                    "prompt_version": generator_prompt["id"],
                    "candidate": candidate,
                    "metadata": generator_meta,
                },
                "blind_solver": {
                    "prompt_version": solver_prompt["id"],
                    "result": solver,
                    "metadata": solver_meta,
                },
                "gate": {
                    "candidate_issues": candidate_issues,
                    "solver_issues": solver_issues,
                    "status": status,
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(item["id"], status, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
