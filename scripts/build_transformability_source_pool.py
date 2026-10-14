#!/usr/bin/env python3
"""Build a deterministic keyword-free pool for transformability calibration.

Input: five normalized train files, a versioned quota config, and prior source JSONL.
Output: 150 unused source records sampled only by domain and MCQ/open surface form.
Example: ``python scripts/build_transformability_source_pool.py``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_deepseek_protocol_batch import DATA_ROOT, SOURCE_FILES
from src.data.loading import iter_normalized_records


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSONL file, treating a missing optional exclusion as empty."""

    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def source_identity(item: dict[str, Any]) -> tuple[str, int] | None:
    """Extract source identity from a flat source, manifest, or verified record."""

    source = item.get("source") if isinstance(item.get("source"), dict) else item
    source_file = source.get("source_file")
    source_index = source.get("source_index")
    if source_index is None:
        source_id = source.get("source_id")
        if isinstance(source_id, str) and source_id.startswith("index:"):
            source_index = source_id.split(":", 1)[1]
    if not isinstance(source_file, str):
        return None
    try:
        return source_file, int(source_index)
    except (TypeError, ValueError):
        return None


def excluded_identities(paths: Iterable[Path]) -> set[tuple[str, int]]:
    identities: set[tuple[str, int]] = set()
    for path in paths:
        for item in read_jsonl(path):
            identity = source_identity(item)
            if identity is not None:
                identities.add(identity)
    return identities


def broad_surface(item: dict[str, Any]) -> str:
    """Use format only; do not infer transformability from question words."""

    return "mcq" if item.get("choices") else "open"


def build_pool(
    records_by_domain: dict[str, list[Any]],
    quotas: dict[str, dict[str, int]],
    excluded: set[tuple[str, int]],
    *,
    pool_id: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample a unique pool without relation keywords or affordance hints."""

    rng = random.Random(seed)
    output: list[dict[str, Any]] = []
    selected: set[tuple[str, int]] = set()
    for domain, surface_quotas in quotas.items():
        serialized = [
            record.to_dict()
            for record in records_by_domain[domain]
            if (record.source_file, record.source_index) not in excluded
        ]
        pools = {
            surface: [item for item in serialized if broad_surface(item) == surface]
            for surface in ("mcq", "open")
        }
        for pool in pools.values():
            rng.shuffle(pool)
        for surface, raw_quota in surface_quotas.items():
            if surface not in pools:
                raise ValueError(f"Unknown surface quota {domain}:{surface}")
            quota = int(raw_quota)
            available = [
                item
                for item in pools[surface]
                if (item["source_file"], item["source_index"]) not in selected
            ]
            if len(available) < quota:
                raise RuntimeError(
                    f"Not enough {domain}:{surface} records: need {quota}, found {len(available)}"
                )
            for item in available[:quota]:
                identity = (item["source_file"], item["source_index"])
                selected.add(identity)
                output.append(
                    {
                        "selector_source_id": (
                            f"selector-{pool_id}-{domain}-{item['source_index']}"
                        ),
                        "pool_id": pool_id,
                        "sample_group": "transformability_candidate",
                        "sampling_stratum": surface,
                        **item,
                    }
                )
    expected = sum(sum(int(value) for value in domain.values()) for domain in quotas.values())
    if len(output) != expected:
        raise RuntimeError(f"Expected {expected} records, built {len(output)}")
    if len({item["selector_source_id"] for item in output}) != expected:
        raise RuntimeError("Selector pool IDs are not unique")
    rng.shuffle(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    base = PROJECT_ROOT / "datasets/processed/transformability_selector_calibration_v1"
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/data/transformability_selector_calibration_v1.json",
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        action="append",
        default=[
            PROJECT_ROOT / "datasets/processed/deepseek_protocol_v1/protocol_sources_150.jsonl",
            PROJECT_ROOT / "datasets/processed/stage015_calibration_v1/sources_50.jsonl",
        ],
        help="Prior JSONL whose source identities must be excluded; may repeat.",
    )
    parser.add_argument("--output", type=Path, default=base / "candidate_sources_150.jsonl")
    parser.add_argument("--summary", type=Path, default=base / "candidate_pool_summary.json")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    excluded = excluded_identities(args.exclude)
    records_by_domain = {
        domain: list(iter_normalized_records(SOURCE_FILES[domain], root=DATA_ROOT))
        for domain in config["domain_quotas"]
    }
    records = build_pool(
        records_by_domain,
        config["domain_quotas"],
        excluded,
        pool_id=config["pool_id"],
        seed=int(config["seed"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "config_id": config["id"],
        "seed": config["seed"],
        "total": len(records),
        "domain": dict(Counter(item["source_domain"] for item in records)),
        "sampling_stratum": dict(Counter(item["sampling_stratum"] for item in records)),
        "excluded_source_identities": len(excluded),
        "uses_relation_keywords": False,
        "contains_affordance_hints": False,
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
