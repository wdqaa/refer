#!/usr/bin/env python3
"""Build a deterministic, unchanged-source native-QA mining pilot.

Purpose: sample a bounded domain/surface-balanced pilot after classifier calibration.
Input: the read-only science dataset, pilot config, and prior calibration manifests.
Output: a JSONL manifest and a JSON summary; no question or answer is modified.

Example:
    python scripts/build_native_qa_pilot.py --output-dir \
        datasets/processed/native_qa_mining_v1/pilot_500
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_native_qa_calibration import (
    DATA_ROOT,
    canonical_domain,
    data_files,
    exact_qa_digest,
    serialize_source,
)
from src.data.loading import iter_normalized_records


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_exclusions(paths: list[Path]) -> tuple[set[tuple[str, int]], set[str]]:
    identities: set[tuple[str, int]] = set()
    digests: set[str] = set()
    for path in paths:
        for item in read_jsonl(path):
            identities.add((item["source_file"], int(item["source_index"])))
            digests.add(exact_qa_digest(item["question"], item["answer"]))
    return identities, digests


def build(
    root: Path,
    config: dict[str, Any],
    exclusion_paths: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    quotas = {
        (domain, surface): int(quota)
        for domain, surfaces in config["quota_by_domain_surface"].items()
        for surface, quota in surfaces.items()
    }
    if sum(quotas.values()) != int(config["total_records"]):
        raise ValueError("Pilot quotas do not sum to total_records")

    excluded_identities, excluded_digests = load_exclusions(exclusion_paths)
    rng = random.Random(int(config["seed"]))
    eligible_seen: Counter[tuple[str, str]] = Counter()
    selected: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    exact_seen: set[str] = set()
    duplicate_records = 0
    excluded_by_identity = 0
    excluded_by_exact_qa = 0

    for path in data_files(root):
        for record in iter_normalized_records(path, root=root):
            digest = exact_qa_digest(record.question, record.answer)
            if digest in exact_seen:
                duplicate_records += 1
                continue
            exact_seen.add(digest)
            identity = (record.source_file, int(record.source_index))
            if identity in excluded_identities:
                excluded_by_identity += 1
                continue
            if digest in excluded_digests:
                excluded_by_exact_qa += 1
                continue

            stratum = (canonical_domain(record.source_domain), "mcq" if record.choices else "open")
            quota = quotas.get(stratum, 0)
            if quota <= 0:
                continue
            eligible_seen[stratum] += 1
            candidate = serialize_source(record, group="pilot_500")
            sample = selected[stratum]
            if len(sample) < quota:
                sample.append(candidate)
            else:
                replacement = rng.randrange(eligible_seen[stratum])
                if replacement < quota:
                    sample[replacement] = candidate

    manifest: list[dict[str, Any]] = []
    for stratum, quota in sorted(quotas.items()):
        sample = selected[stratum]
        if len(sample) != quota:
            raise RuntimeError(f"Need {quota} records for {stratum}, found {len(sample)}")
        manifest.extend(sorted(sample, key=lambda item: (item["source_file"], item["source_index"])))

    identities = {(item["source_file"], int(item["source_index"])) for item in manifest}
    selected_digests = {exact_qa_digest(item["question"], item["answer"]) for item in manifest}
    expected = int(config["total_records"])
    if len(manifest) != expected or len(identities) != expected or len(selected_digests) != expected:
        raise RuntimeError("Pilot manifest is not unique or does not match requested size")
    if identities & excluded_identities or selected_digests & excluded_digests:
        raise RuntimeError("Pilot overlaps a calibration or held-out record")

    summary = {
        "config_id": config["id"],
        "seed": int(config["seed"]),
        "total": len(manifest),
        "source_files": len(data_files(root)),
        "directory_domain": dict(Counter(item["directory_domain"] for item in manifest)),
        "surface": dict(Counter("mcq" if item["choices"] else "open" for item in manifest)),
        "stratum": {
            f"{domain}/{surface}": sum(
                item["directory_domain"] == domain
                and ("mcq" if item["choices"] else "open") == surface
                for item in manifest
            )
            for domain, surface in sorted(quotas)
        },
        "excluded_manifest_records": sum(len(read_jsonl(path)) for path in exclusion_paths),
        "excluded_unique_identities": len(excluded_identities),
        "excluded_unique_exact_qa": len(excluded_digests),
        "excluded_by_identity": excluded_by_identity,
        "excluded_by_exact_qa": excluded_by_exact_qa,
        "source_exact_duplicates_skipped": duplicate_records,
        "overlap_with_prior_by_identity": 0,
        "overlap_with_prior_by_exact_qa": 0,
        "eligible_seen_by_stratum": {
            f"{domain}/{surface}": eligible_seen[(domain, surface)]
            for domain, surface in sorted(quotas)
        },
    }
    return manifest, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    base = PROJECT_ROOT / "datasets/processed/native_qa_mining_v1"
    parser.add_argument("--dataset-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/data/native_qa_mining_pilot_500.json",
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        nargs="+",
        default=[base / "calibration_60/manifest.jsonl", base / "holdout_30/manifest.jsonl"],
    )
    parser.add_argument("--output-dir", type=Path, default=base / "pilot_500")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest, summary = build(args.dataset_root, config, args.exclude)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary_path = args.output_dir / "manifest_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
