#!/usr/bin/env python3
"""Run fixed Expert-Critique or matched Self-Refine from cached Direct drafts.

Input: the frozen native baseline view, cached Direct and routed traces, the audited
item-exclusion file, versioned refinement prompts, and a local Qwen checkpoint.
Output: append-only full traces plus deterministic objective scores.

Example::

    CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_qwen_refinement.py \
      --arm expert_critique --device cuda:0 \
      --output-dir outputs/baselines/qwen3_8b_native_eval48_refinement_v1/expert_critique
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
from src.evaluation.refinement_runner import ARMS, run_refinement_arm
from src.generation.local_model import GenerationSettings, TransformersChatModel


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--direct-trace", type=Path, required=True)
    parser.add_argument("--routed-trace", type=Path, required=True)
    parser.add_argument("--exclusion-audit", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path("/home/dqwang/Model/Qwen3-8B"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--critic-max-new-tokens", type=int, default=512)
    parser.add_argument("--final-max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--id", action="append")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    if args.num_shards < 1:
        parser.error("--num-shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")

    exclusions = {
        row["item_id"]
        for row in read_jsonl(args.exclusion_audit)
        if row["disposition"] == "exclude_from_audited_summary"
    }
    all_rows = [row for row in read_jsonl(args.dataset) if row["id"] not in exclusions]
    if len(all_rows) != 48:
        parser.error(f"Expected audited 48-item dataset, found {len(all_rows)}")
    rows = list(all_rows)
    if args.id:
        requested = set(args.id)
        rows = [row for row in rows if row["id"] in requested]
        missing = requested - {row["id"] for row in rows}
        if missing:
            parser.error(f"Unknown audited IDs: {sorted(missing)}")
    rows = rows[args.shard_index :: args.num_shards]
    if args.limit is not None:
        rows = rows[: args.limit]

    direct = {row["item_id"]: row for row in read_jsonl(args.direct_trace)}
    routed = {row["item_id"]: row for row in read_jsonl(args.routed_trace)}
    missing_direct = sorted({row["id"] for row in all_rows} - set(direct))
    missing_routed = sorted({row["id"] for row in all_rows} - set(routed))
    if missing_direct or missing_routed:
        parser.error(f"Missing cached traces; direct={missing_direct}, routed={missing_routed}")
    prompts = json.loads(args.prompt.read_text(encoding="utf-8"))

    model = TransformersChatModel(args.model_path, device=args.device)
    critic_settings = GenerationSettings(
        max_new_tokens=args.critic_max_new_tokens, seed=args.seed, enable_thinking=False
    )
    final_settings = GenerationSettings(
        max_new_tokens=args.final_max_new_tokens, seed=args.seed, enable_thinking=False
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "traces.jsonl"
    previous = read_jsonl(trace_path) if trace_path.exists() else []
    completed = {(row["item_id"], row["arm"]) for row in previous if row["status"] == "ok"}

    with trace_path.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(rows, 1):
            key = (record["id"], args.arm)
            if key in completed:
                continue
            direct_trace = direct[record["id"]]
            routed_trace = routed[record["id"]]
            trace = run_refinement_arm(
                model,
                record,
                arm=args.arm,
                direct_prediction=direct_trace["final_prediction"],
                selected_domains=list(routed_trace["routing"]["selected_domains"]),
                routing_rationale=routed_trace["routing"].get("rationale"),
                prompts=prompts,
                critic_settings=critic_settings,
                final_settings=final_settings,
            )
            trace["created_at"] = datetime.now(timezone.utc).isoformat()
            trace["cached_sources"] = {
                "direct_trace": str(args.direct_trace),
                "routed_trace": str(args.routed_trace),
            }
            trace["config"] = {
                "model": "Qwen/Qwen3-8B",
                "model_path": str(args.model_path),
                "seed": args.seed,
                "critic_max_new_tokens": args.critic_max_new_tokens,
                "final_max_new_tokens": args.final_max_new_tokens,
                "thinking": False,
                "prompt_id": prompts["id"],
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            }
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(rows)}] {record['id']} {args.arm} {trace['status']}", flush=True)

    latest = {(row["item_id"], row["arm"]): row for row in read_jsonl(trace_path)}
    by_id = {row["id"]: row for row in all_rows}
    scores = []
    for (item_id, arm), trace in sorted(latest.items()):
        if arm != args.arm or item_id not in by_id:
            continue
        score = score_prediction(by_id[item_id], trace["final_prediction"])
        scores.append({"item_id": item_id, "arm": arm, **score})
    with (args.output_dir / "objective_scores.jsonl").open("w", encoding="utf-8") as handle:
        for score in scores:
            handle.write(json.dumps(score, ensure_ascii=False) + "\n")
    print(f"traces={trace_path} scores={len(scores)}")


if __name__ == "__main__":
    main()
