#!/usr/bin/env python3
"""Build a fresh, model-blind astronomy benchmark candidate manifest."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
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
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []


def candidate_views(row: dict[str, Any]) -> list[dict[str, Any]]:
    views = [row]
    if isinstance(row.get("source"), dict):
        views.append(row["source"])
    return views


def collect_exclusions(processed_root: Path, output_dir: Path) -> tuple[set[tuple[str, int]], set[str], int]:
    identities: set[tuple[str, int]] = set()
    digests: set[str] = set()
    scanned = 0
    output_resolved = output_dir.resolve()
    for path in processed_root.rglob("*.jsonl"):
        resolved = path.resolve()
        if resolved == output_resolved or output_resolved in resolved.parents:
            continue
        rows = read_jsonl(path)
        scanned += len(rows)
        for row in rows:
            for view in candidate_views(row):
                source_file = view.get("source_file")
                source_index = view.get("source_index")
                if isinstance(source_file, str) and isinstance(source_index, int):
                    identities.add((source_file, source_index))
                question = view.get("question", view.get("original_question"))
                answer = view.get("answer", view.get("original_answer"))
                if isinstance(question, str) and isinstance(answer, str):
                    digests.add(exact_qa_digest(question, answer))
    return identities, digests, scanned


def build(
    dataset_root: Path,
    processed_root: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    quotas = {
        (source_file, surface): int(quota)
        for source_file, surfaces in config["quota_by_source_surface"].items()
        for surface, quota in surfaces.items()
        if int(quota) > 0
    }
    if sum(quotas.values()) != int(config["candidate_records"]):
        raise ValueError("candidate quotas do not sum to candidate_records")
    excluded_ids, excluded_digests, processed_rows = collect_exclusions(
        processed_root, output_dir
    )
    rng = random.Random(int(config["seed"]))
    eligible_seen: Counter[tuple[str, str]] = Counter()
    selected: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    exact_seen: set[str] = set()
    raw_duplicates = excluded_identity_hits = excluded_digest_hits = 0

    for path in data_files(dataset_root):
        for record in iter_normalized_records(path, root=dataset_root):
            if canonical_domain(record.source_domain) != "astronomy":
                continue
            digest = exact_qa_digest(record.question, record.answer)
            if digest in exact_seen:
                raw_duplicates += 1
                continue
            exact_seen.add(digest)
            identity = (record.source_file, int(record.source_index))
            if identity in excluded_ids:
                excluded_identity_hits += 1
                continue
            if digest in excluded_digests:
                excluded_digest_hits += 1
                continue
            surface = "mcq" if record.choices else "open"
            stratum = (record.source_file, surface)
            quota = quotas.get(stratum, 0)
            if not quota:
                continue
            eligible_seen[stratum] += 1
            item = serialize_source(record, group=config["id"] + "_candidates")
            sample = selected[stratum]
            if len(sample) < quota:
                sample.append(item)
            else:
                replacement = rng.randrange(eligible_seen[stratum])
                if replacement < quota:
                    sample[replacement] = item

    manifest: list[dict[str, Any]] = []
    for stratum, quota in sorted(quotas.items()):
        sample = selected[stratum]
        if len(sample) != quota:
            raise RuntimeError(f"need {quota} records for {stratum}, found {len(sample)}")
        manifest.extend(sorted(sample, key=lambda row: (row["source_file"], row["source_index"])))
    identities = {(row["source_file"], int(row["source_index"])) for row in manifest}
    manifest_digests = {exact_qa_digest(row["question"], row["answer"]) for row in manifest}
    expected = int(config["candidate_records"])
    if len(manifest) != expected or len(identities) != expected or len(manifest_digests) != expected:
        raise RuntimeError("candidate manifest is not unique or does not match the configured size")
    if identities & excluded_ids or manifest_digests & excluded_digests:
        raise RuntimeError("candidate manifest overlaps prior processed records")

    source_bytes = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in manifest
    ).encode("utf-8")
    summary = {
        "schema_version": "astronomy_eval60_expansion_candidates.v1",
        "config_id": config["id"],
        "seed": int(config["seed"]),
        "records": len(manifest),
        "target_curated_records": int(config["target_curated_records"]),
        "surface": dict(sorted(Counter("mcq" if row["choices"] else "open" for row in manifest).items())),
        "source_file": dict(sorted(Counter(row["source_file"] for row in manifest).items())),
        "stratum": {
            f"{source_file}:{surface}": sum(
                row["source_file"] == source_file
                and ("mcq" if row["choices"] else "open") == surface
                for row in manifest
            )
            for source_file, surface in sorted(quotas)
        },
        "eligible_seen": {
            f"{source_file}:{surface}": eligible_seen[(source_file, surface)]
            for source_file, surface in sorted(quotas)
        },
        "prior_processed_rows_scanned": processed_rows,
        "excluded_unique_identities": len(excluded_ids),
        "excluded_unique_exact_qa": len(excluded_digests),
        "excluded_by_identity": excluded_identity_hits,
        "excluded_by_exact_qa": excluded_digest_hits,
        "raw_exact_duplicates_skipped": raw_duplicates,
        "overlap_with_prior_by_identity": 0,
        "overlap_with_prior_by_exact_qa": 0,
        "manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "model_outputs_seen_during_selection": False,
    }
    return manifest, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--processed-root", type=Path, default=PROJECT_ROOT / "datasets/processed"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/evaluation/astronomy_eval60_expansion_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "datasets/processed/domain_model_benchmark_v2/astronomy_eval60_candidates",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest, summary = build(args.dataset_root, args.processed_root, args.output_dir, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "candidate_manifest_120.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "candidate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(manifest_path), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
