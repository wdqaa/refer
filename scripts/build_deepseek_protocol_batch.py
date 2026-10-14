#!/usr/bin/env python3
"""Build the fixed 150-source DeepSeek protocol batch without editing source data.

Input: five domain ``*_train.jsonl`` files below ``mixed_dataset``.
Output: ``datasets/processed/deepseek_protocol_v1/protocol_sources_150.jsonl``.
Example: ``python scripts/build_deepseek_protocol_batch.py --seed 42``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loading import iter_normalized_records


DATA_ROOT = PROJECT_ROOT / "datasets/science_dataset/mixed_dataset"
SOURCE_FILES = {
    "astronomy": DATA_ROOT / "astronomy/astronomy_train.jsonl",
    "biology": DATA_ROOT / "biology/biology_train.jsonl",
    "chemistry": DATA_ROOT / "chemistry/chemistry_train.jsonl",
    "physics": DATA_ROOT / "physics/physics_train.jsonl",
    "geography": DATA_ROOT / "geography/geography_train.jsonl",
}
REPRESENTATIVE_PER_DOMAIN = 20
STRESS_QUOTAS = {
    "astronomy": 5,
    "biology": 10,
    "chemistry": 10,
    "physics": 10,
    "geography": 15,
}


def surface_bucket(record: dict[str, Any]) -> str:
    """Give sampling-only surface hints; DeepSeek supplies the real taxonomy."""

    if record["choices"]:
        return "mcq"
    question = record["question"].lower()
    answer_words = len(record["answer"].split())
    objective_signals = re.compile(
        r"\b(calculate|compute|determine|find|derive|solve|what is the (?:value|mass|"
        r"energy|force|speed|velocity|temperature|pressure|wavelength|frequency|"
        r"concentration|probability|distance|time|number))\b"
    )
    qualitative_signals = re.compile(
        r"\b(explain|discuss|describe|why|how does|how do|mechanism|effect of|"
        r"impact of|compare|evaluate)\b"
    )
    if objective_signals.search(question) or answer_words <= 12:
        return "likely_objective_open"
    if qualitative_signals.search(question) or answer_words >= 45:
        return "likely_qualitative_open"
    return "uncertain_open"


def serialized(record: Any, *, group: str) -> dict[str, Any]:
    item = record.to_dict()
    return {
        "protocol_id": f"protocol-v1-{item['source_domain']}-{item['source_index']}",
        "sample_group": group,
        "surface_bucket": surface_bucket(item),
        **item,
    }


def choose_stress(records: list[Any], quota: int, rng: random.Random) -> list[Any]:
    pools: dict[str, list[Any]] = {
        "mcq": [],
        "likely_objective_open": [],
        "likely_qualitative_open": [],
        "uncertain_open": [],
    }
    for record in records:
        pools[surface_bucket(record.to_dict())].append(record)
    for pool in pools.values():
        rng.shuffle(pool)

    order = ["mcq", "likely_objective_open", "likely_qualitative_open", "uncertain_open"]
    chosen: list[Any] = []
    while len(chosen) < quota:
        progressed = False
        for bucket in order:
            if pools[bucket] and len(chosen) < quota:
                chosen.append(pools[bucket].pop())
                progressed = True
        if not progressed:
            raise RuntimeError("Not enough records to fill stress quota")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "datasets/processed/deepseek_protocol_v1/protocol_sources_150.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output: list[dict[str, Any]] = []
    for domain, path in SOURCE_FILES.items():
        records = list(iter_normalized_records(path, root=DATA_ROOT))
        representative = rng.sample(records, REPRESENTATIVE_PER_DOMAIN)
        selected_indices = {record.source_index for record in representative}
        remaining = [record for record in records if record.source_index not in selected_indices]
        stress = choose_stress(remaining, STRESS_QUOTAS[domain], rng)
        output.extend(serialized(record, group="representative") for record in representative)
        output.extend(serialized(record, group="stress") for record in stress)

    if len(output) != 150 or len({item["protocol_id"] for item in output}) != 150:
        raise RuntimeError("Protocol batch must contain 150 unique records")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in output:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for item in output:
        key = f"{item['source_domain']}:{item['sample_group']}:{item['surface_bucket']}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({"output": str(args.output), "total": len(output), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
