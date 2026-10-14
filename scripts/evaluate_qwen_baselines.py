#!/usr/bin/env python3
"""Run Direct, routed Multi-Agent, and oracle-domain Qwen3-8B baselines.

Input: curated verified QA JSONL, versioned prompts, and one local Qwen3-8B checkpoint.
Output: append-only inference traces and a separate deterministic objective-score JSONL.
Example: ``python scripts/evaluate_qwen_baselines.py --limit 3 --device cuda:0 --output-dir outputs/baselines/qwen_smoke3``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.answer_scoring import score_prediction
from src.evaluation.baseline_runner import ARMS, run_arm
from src.generation.local_model import GenerationSettings, TransformersChatModel
from src.prompts.loading import load_prompt_registry


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "datasets/processed/deepseek_qa_curated_v2/verified_qa_curated_49.jsonl")
    parser.add_argument("--prompt", type=Path, default=PROJECT_ROOT / "src/prompts/qwen_baselines_v1.json")
    parser.add_argument(
        "--agent-registry",
        type=Path,
        default=PROJECT_ROOT / "src/prompts/agent_prompts.v3.json",
    )
    parser.add_argument("--model-path", type=Path, default=Path("/home/dqwang/Model/Qwen3-8B"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--role-max-new-tokens", type=int, default=1024)
    parser.add_argument("--final-max-new-tokens", type=int, default=2048)
    parser.add_argument("--final-thinking", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--id", action="append")
    parser.add_argument("--arm", action="append", choices=sorted(ARMS))
    args = parser.parse_args()

    all_rows = read_jsonl(args.dataset)
    rows = list(all_rows)
    if args.id:
        requested = set(args.id)
        rows = [row for row in rows if row["id"] in requested]
        missing = requested - {row["id"] for row in rows}
        if missing:
            parser.error(f"Unknown IDs: {sorted(missing)}")
    if args.limit is not None:
        rows = rows[: args.limit]
    arms = args.arm or ["direct", "routed_multi_agent", "oracle_multi_agent"]
    prompts = json.loads(args.prompt.read_text(encoding="utf-8"))
    registry = load_prompt_registry(args.agent_registry)
    model = TransformersChatModel(args.model_path, device=args.device)
    role_settings = GenerationSettings(max_new_tokens=args.role_max_new_tokens, seed=args.seed, enable_thinking=False)
    final_settings = GenerationSettings(max_new_tokens=args.final_max_new_tokens, seed=args.seed, enable_thinking=args.final_thinking)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "traces.jsonl"
    previous = read_jsonl(trace_path) if trace_path.exists() else []
    completed = {(item["item_id"], item["arm"]) for item in previous if item["status"] == "ok"}
    with trace_path.open("a", encoding="utf-8") as handle:
        total = len(rows) * len(arms)
        done = 0
        for record in rows:
            for arm in arms:
                done += 1
                if (record["id"], arm) in completed:
                    continue
                trace = run_arm(
                    model, record, arm, prompts, registry, role_settings, final_settings
                )
                trace["created_at"] = datetime.now(timezone.utc).isoformat()
                trace["config"] = {
                    "model": "Qwen/Qwen3-8B",
                    "model_path": str(args.model_path),
                    "seed": args.seed,
                    "role_max_new_tokens": args.role_max_new_tokens,
                    "final_max_new_tokens": args.final_max_new_tokens,
                    "final_thinking": args.final_thinking,
                    "baseline_prompt_id": prompts["id"],
                    "agent_registry": args.agent_registry.name,
                }
                handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"[{done}/{total}] {record['id']} {arm} {trace['status']}", flush=True)

    latest = {(item["item_id"], item["arm"]): item for item in read_jsonl(trace_path)}
    # Use the full dataset when rebuilding scores after a targeted retry so a
    # single ``--id`` does not truncate the arm-level score artifact.
    by_id = {row["id"]: row for row in all_rows}
    scores = []
    for (item_id, arm), trace in latest.items():
        if item_id not in by_id or arm not in arms:
            continue
        score = score_prediction(by_id[item_id], trace["final_prediction"])
        scores.append({"item_id": item_id, "arm": arm, **score})
    with (args.output_dir / "objective_scores.jsonl").open("w", encoding="utf-8") as handle:
        for score in scores:
            handle.write(json.dumps(score, ensure_ascii=False) + "\n")
    print(f"traces={trace_path} scores={len(scores)}")


if __name__ == "__main__":
    main()
