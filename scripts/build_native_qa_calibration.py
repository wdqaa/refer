#!/usr/bin/env python3
"""Build a 60-item native-QA mining calibration and scan the source universe.

The immutable mixed-domain duplicate is excluded. Thirty existing human gold boundary
records are combined with thirty fresh records sampled across all six source domains.
No question or answer is modified.
"""

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

from src.data.loading import iter_normalized_records


DATA_ROOT = PROJECT_ROOT / "datasets/science_dataset/mixed_dataset"
DOMAIN_MAP = {"math": "mathematics"}
SOURCE_DIRECTORIES = {"astronomy", "biology", "chemistry", "geography", "math", "physics"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def data_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.glob("*/*")
        if path.suffix in {".json", ".jsonl"}
        and path.parts[-2] in SOURCE_DIRECTORIES
        and not path.name.startswith(".")
    )


def canonical_domain(value: str) -> str:
    return DOMAIN_MAP.get(value, value)


def exact_qa_digest(question: str, answer: str) -> str:
    normalized = " ".join(question.split()).casefold() + "\0" + " ".join(answer.split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def native_id(source_file: str, source_index: int) -> str:
    digest = hashlib.sha1(f"{source_file}:{source_index}".encode("utf-8")).hexdigest()[:14]
    return f"native-{digest}"


def serialize_source(record: Any, *, group: str, gold: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "native_qa_id": native_id(record.source_file, record.source_index),
        "calibration_group": group,
        "source_file": record.source_file,
        "source_index": record.source_index,
        "source_id": record.source_id,
        "directory_domain": canonical_domain(record.source_domain),
        "question": record.question,
        "answer": record.answer,
        "choices": record.choices,
        "calibration_gold": gold,
    }


def build(
    root: Path, gold_path: Path, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    annotations = read_jsonl(gold_path)
    gold_lookup = {(item["source_file"], str(item["source_id"])): item for item in annotations}
    found_gold: dict[str, dict[str, Any]] = {}
    fresh_quotas = {
        (domain, surface): int(quota)
        for domain, surfaces in config["fresh_per_domain"].items()
        for surface, quota in surfaces.items()
    }
    fresh_seen: Counter[tuple[str, str]] = Counter()
    fresh_samples: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    rng = random.Random(int(config["seed"]))

    files = data_files(root)
    counts_by_domain: Counter[str] = Counter()
    counts_by_file: Counter[str] = Counter()
    counts_by_surface: Counter[str] = Counter()
    exact_seen: set[str] = set()
    exact_duplicates = 0

    for path in files:
        for record in iter_normalized_records(path, root=root):
            domain = canonical_domain(record.source_domain)
            surface = "mcq" if record.choices else "open"
            counts_by_domain[domain] += 1
            counts_by_file[record.source_file] += 1
            counts_by_surface[surface] += 1
            digest = exact_qa_digest(record.question, record.answer)
            if digest in exact_seen:
                exact_duplicates += 1
            else:
                exact_seen.add(digest)

            gold_annotation = gold_lookup.get((record.source_file, record.source_id))
            if gold_annotation is not None:
                found_gold[gold_annotation["id"]] = serialize_source(
                    record,
                    group="existing_gold",
                    gold={"id": gold_annotation["id"], "category": gold_annotation["category"], **gold_annotation["gold"]},
                )
                continue

            stratum = (domain, surface)
            quota = fresh_quotas.get(stratum, 0)
            if quota <= 0:
                continue
            fresh_seen[stratum] += 1
            candidate = serialize_source(record, group="fresh_stratified")
            sample = fresh_samples[stratum]
            if len(sample) < quota:
                sample.append(candidate)
            else:
                replacement = rng.randrange(fresh_seen[stratum])
                if replacement < quota:
                    sample[replacement] = candidate

    missing_gold = sorted(set(item["id"] for item in annotations) - set(found_gold))
    if missing_gold:
        raise RuntimeError(f"Could not resolve gold records: {missing_gold}")
    fresh = []
    for stratum, quota in sorted(fresh_quotas.items()):
        selected = fresh_samples[stratum]
        if len(selected) != quota:
            raise RuntimeError(f"Need {quota} records for {stratum}, found {len(selected)}")
        fresh.extend(selected)
    manifest = [found_gold[item["id"]] for item in annotations] + fresh
    expected = int(config["gold_records"]) + sum(fresh_quotas.values())
    if len(manifest) != expected or len({item["native_qa_id"] for item in manifest}) != expected:
        raise RuntimeError(f"Expected {expected} unique calibration records, got {len(manifest)}")

    summary = {
        "source_files": len(files),
        "raw_non_mixed_records": sum(counts_by_domain.values()),
        "exact_unique_question_answer_pairs": len(exact_seen),
        "exact_duplicate_records": exact_duplicates,
        "domain": dict(sorted(counts_by_domain.items())),
        "surface": dict(sorted(counts_by_surface.items())),
        "file": dict(sorted(counts_by_file.items())),
        "mixed_train_excluded_as_known_duplicate": True,
        "calibration_records": len(manifest),
        "calibration_group": dict(Counter(item["calibration_group"] for item in manifest)),
        "calibration_directory_domain": dict(Counter(item["directory_domain"] for item in manifest)),
    }
    return manifest, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    base = PROJECT_ROOT / "datasets/processed/native_qa_mining_v1/calibration_60"
    parser.add_argument("--dataset-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--gold", type=Path, default=PROJECT_ROOT / "datasets/metadata/source_analysis_gold_v1.jsonl"
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/data/native_qa_mining_v1.json"
    )
    parser.add_argument("--output", type=Path, default=base / "manifest.jsonl")
    parser.add_argument("--summary", type=Path, default=base / "source_universe_summary.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest, summary = build(args.dataset_root, args.gold, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
