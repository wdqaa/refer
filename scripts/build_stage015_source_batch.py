#!/usr/bin/env python3
"""Build one deterministic Stage 01.5 source-calibration wave.

Input: five read-only train files, a versioned quota config, and prior source JSONL.
Output: 50 fresh normalized source records plus a compact sampling summary.
Example: ``python scripts/build_stage015_source_batch.py``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_deepseek_protocol_batch import DATA_ROOT, SOURCE_FILES, surface_bucket
from src.data.loading import iter_normalized_records


RELATION_PATTERNS = {
    "quantitative_modeling": re.compile(
        r"\b(calculate|compute|estimate|predict|model|rate|ratio|equation|scal(?:e|ing)|"
        r"proportional|how much|how many)\b",
        re.IGNORECASE,
    ),
    "constraint_coupling": re.compile(
        r"\b(constraint|limit|threshold|maximum|minimum|boundary|range|capacity|"
        r"tolerance|condition|required|equilibrium|stability)\b",
        re.IGNORECASE,
    ),
    "measurement_observation": re.compile(
        r"\b(measur|observ|spectrum|spectral|signal|sensor|survey|remote sensing|"
        r"wavelength|redshift|isotope|concentration|temperature|pressure)\b",
        re.IGNORECASE,
    ),
}
HIGH_RISK_SOURCE = re.compile(
    r"\b(according to|case study|projected|projection|specific examples?|best method|"
    r"develop|implications?|correlation between|researchers?|the study|this study)\b",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def source_identity(item: dict[str, Any]) -> tuple[str, int] | None:
    """Extract a source identity from a flat source, manifest, or verified record."""

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


def affordance_relations(record: dict[str, Any]) -> list[str]:
    """Return sampling hints, not claims that valid transformations exist."""

    if record.get("choices"):
        return []
    text = record.get("question", "")
    if HIGH_RISK_SOURCE.search(text):
        return []
    return [name for name, pattern in RELATION_PATTERNS.items() if pattern.search(text)]


def build_batch(
    records_by_domain: dict[str, list[Any]],
    quotas: dict[str, dict[str, int]],
    excluded: set[tuple[str, int]],
    *,
    batch_id: str,
    seed: int,
    relation_quotas: dict[str, int],
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    output: list[dict[str, Any]] = []
    selected: set[tuple[str, int]] = set()
    relation_remaining = dict(relation_quotas)
    for domain, domain_quotas in quotas.items():
        records = [
            record for record in records_by_domain[domain]
            if (record.source_file, record.source_index) not in excluded
        ]
        serialized = [record.to_dict() for record in records]
        broad_pools = {
            "mcq": [item for item in serialized if surface_bucket(item) == "mcq"],
            "likely_objective_open": [
                item for item in serialized if surface_bucket(item) == "likely_objective_open"
            ],
        }
        relation_pools = {
            relation: [item for item in serialized if relation in affordance_relations(item)]
            for relation in relation_quotas
        }
        for pool in [*broad_pools.values(), *relation_pools.values()]:
            rng.shuffle(pool)

        # Reserve fusion-affordance records with an exact global relation balance.
        fusion_quota = int(domain_quotas["non_mechanism_affordance"])
        for _ in range(fusion_quota):
            chosen_relation = None
            chosen_item = None
            for relation in sorted(
                relation_remaining, key=lambda name: (-relation_remaining[name], name)
            ):
                if relation_remaining[relation] <= 0:
                    continue
                candidate = next(
                    (
                        item for item in relation_pools[relation]
                        if (item["source_file"], item["source_index"]) not in selected
                    ),
                    None,
                )
                if candidate is not None:
                    chosen_relation = relation
                    chosen_item = candidate
                    break
            if chosen_item is None or chosen_relation is None:
                raise RuntimeError(f"Cannot fill balanced non-mechanism quota in {domain}")
            identity = (chosen_item["source_file"], chosen_item["source_index"])
            selected.add(identity)
            relation_remaining[chosen_relation] -= 1
            output.append(
                {
                    "protocol_id": f"protocol-{batch_id}-{domain}-{chosen_item['source_index']}",
                    "batch_id": batch_id,
                    "sample_group": "stage015_calibration",
                    "sampling_stratum": "non_mechanism_affordance",
                    "affordance_hint": chosen_relation,
                    "surface_bucket": surface_bucket(chosen_item),
                    **chosen_item,
                }
            )

        for stratum in ("mcq", "likely_objective_open"):
            quota = int(domain_quotas[stratum])
            available = [
                item for item in broad_pools[stratum]
                if (item["source_file"], item["source_index"]) not in selected
            ]
            if len(available) < quota:
                raise RuntimeError(
                    f"Not enough {domain}:{stratum} records: need {quota}, found {len(available)}"
                )
            for item in available[:quota]:
                identity = (item["source_file"], item["source_index"])
                selected.add(identity)
                output.append(
                    {
                        "protocol_id": f"protocol-{batch_id}-{domain}-{item['source_index']}",
                        "batch_id": batch_id,
                        "sample_group": "stage015_calibration",
                        "sampling_stratum": stratum,
                        "affordance_hint": None,
                        "surface_bucket": surface_bucket(item),
                        **item,
                    }
                )
    expected = sum(sum(domain.values()) for domain in quotas.values())
    if len(output) != expected or len({item["protocol_id"] for item in output}) != expected:
        raise RuntimeError("Stage 01.5 source batch is not complete and unique")
    if any(relation_remaining.values()):
        raise RuntimeError(f"Unfilled relation quotas: {relation_remaining}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    base = PROJECT_ROOT / "datasets/processed/stage015_calibration_v1"
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/data/stage015_calibration_sources_v1.json",
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        action="append",
        default=[PROJECT_ROOT / "datasets/processed/deepseek_protocol_v1/protocol_sources_150.jsonl"],
        help="Prior JSONL whose source identities must be excluded; may repeat.",
    )
    parser.add_argument("--output", type=Path, default=base / "sources_50.jsonl")
    parser.add_argument("--summary", type=Path, default=base / "source_sampling_summary.json")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    records_by_domain = {
        domain: list(iter_normalized_records(SOURCE_FILES[domain], root=DATA_ROOT))
        for domain in config["domain_quotas"]
    }
    records = build_batch(
        records_by_domain,
        config["domain_quotas"],
        excluded_identities(args.exclude),
        batch_id=config["batch_id"],
        seed=int(config["seed"]),
        relation_quotas=config["relation_quotas"],
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
        "affordance_hint": dict(
            Counter(item["affordance_hint"] for item in records if item["affordance_hint"])
        ),
        "excluded_source_identities": len(excluded_identities(args.exclude)),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
