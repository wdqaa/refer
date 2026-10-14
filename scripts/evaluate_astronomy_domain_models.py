#!/usr/bin/env python3
"""Run one frozen astronomy benchmark arm with the checkpoint's proper renderer.

Examples:
  python scripts/evaluate_astronomy_domain_models.py --model qwen3_8b --device cuda:0
  python scripts/evaluate_astronomy_domain_models.py --model astrosage_8b --device cuda:1
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.domain_model_benchmark import (
    record_id,
    run_domain_model_batch,
    run_domain_model_item,
)
from src.generation.local_model import (
    GenerationSettings,
    TransformersChatModel,
    TransformersCompletionModel,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def validate_checkpoint(path: Path) -> None:
    if not path.is_dir():
        raise ValueError(f"model directory does not exist: {path}")
    index = path / "model.safetensors.index.json"
    single = path / "model.safetensors"
    if index.exists():
        data = json.loads(index.read_text(encoding="utf-8"))
        shards = sorted(set(data.get("weight_map", {}).values()))
        missing = [name for name in shards if not (path / name).is_file()]
        if missing:
            raise ValueError(
                f"checkpoint download is incomplete: {len(missing)}/{len(shards)} shards missing"
            )
    elif not single.is_file():
        raise ValueError(f"checkpoint has neither {index.name} nor {single.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/evaluation/astronomy_domain_model_benchmark_v1.json",
    )
    parser.add_argument("--model", choices=("qwen3_8b", "astrosage_8b"), required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--id", action="append")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size. Greedy decoding and per-item prompts remain unchanged.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_config = config["models"][args.model]
    dataset_path = args.dataset or resolve(config["dataset"])
    prompt_path = resolve(config["prompt"])
    model_path = args.model_path or Path(model_config["model_path"])
    output_dir = args.output_dir or (
        PROJECT_ROOT / "outputs/domain_model_benchmark_v1/astronomy_eval30" / args.model
    )
    validate_checkpoint(model_path)

    all_rows = read_jsonl(dataset_path)
    rows = list(all_rows)
    if args.id:
        requested = set(args.id)
        rows = [row for row in rows if record_id(row) in requested]
        missing = requested - {record_id(row) for row in rows}
        if missing:
            parser.error(f"unknown IDs: {sorted(missing)}")
    if args.limit is not None:
        rows = rows[: args.limit]
    prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
    generation = config["generation"]
    settings = GenerationSettings(
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        max_new_tokens=int(generation["max_new_tokens"]),
        seed=int(generation["seed"]),
        enable_thinking=bool(model_config["enable_thinking"]),
    )
    renderer = model_config["prompt_renderer"]
    model_class = (
        TransformersChatModel if renderer == "chat_template" else TransformersCompletionModel
    )
    model = model_class(model_path, device=args.device)

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "traces.jsonl"
    previous = read_jsonl(trace_path) if trace_path.exists() else []
    completed = {row["item_id"] for row in previous}
    pending = [row for row in rows if record_id(row) not in completed]
    for index, row in enumerate(rows, 1):
        if record_id(row) in completed:
            print(f"[{index}/{len(rows)}] {record_id(row)} already_recorded", flush=True)
    with trace_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            if len(batch) == 1:
                traces = [
                    run_domain_model_item(
                        model,
                        batch[0],
                        prompt=prompt,
                        settings=settings,
                        model_id=model_config["model_id"],
                        prompt_renderer=renderer,
                    )
                ]
            else:
                traces = run_domain_model_batch(
                    model,
                    batch,
                    prompt=prompt,
                    settings=settings,
                    model_id=model_config["model_id"],
                    prompt_renderer=renderer,
                )
            for within_batch, trace in enumerate(traces):
                trace["created_at"] = datetime.now(timezone.utc).isoformat()
                trace["config"] = {
                    "benchmark_config": config["config_version"],
                    "dataset": str(dataset_path),
                    "model_path": str(model_path),
                    "device": args.device,
                    "temperature": settings.temperature,
                    "top_p": settings.top_p,
                    "max_new_tokens": settings.max_new_tokens,
                    "seed": settings.seed,
                    "enable_thinking": settings.enable_thinking,
                    "batch_size": args.batch_size,
                }
                handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
                handle.flush()
                completed_count = len(completed) + start + within_batch + 1
                print(
                    f"[{completed_count}/{len(rows)}] {trace['item_id']} "
                    f"{trace['status']} {trace['parse_mode']}",
                    flush=True,
                )

    print(f"traces={trace_path}")


if __name__ == "__main__":
    main()
