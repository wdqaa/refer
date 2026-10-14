"""Select a small deterministic Stage 01 anchor manifest from normalized source QA."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
import random
from typing import Any

from .analysis import word_count
from .loading import iter_normalized_records


DEFAULT_PLAN: dict[str, dict[str, int]] = {
    "astronomy": {"real_anchored": 3, "synthetic": 1},
    "biology": {"real_anchored": 4, "synthetic": 0},
    "chemistry": {"real_anchored": 3, "synthetic": 1},
    "mathematics": {"real_anchored": 3, "synthetic": 1},
    "physics": {"real_anchored": 3, "synthetic": 1},
}
SOURCE_TO_CANONICAL_DOMAIN = {"math": "mathematics"}


def _candidate_paths(root: Path, source_domain: str) -> list[Path]:
    paths = []
    for path in sorted((root / source_domain).glob("*")):
        if path.suffix not in {".json", ".jsonl"}:
            continue
        if "_train" in path.name:
            continue
        paths.append(path)
    return paths


def select_pilot_anchors(
    root: Path, *, seed: int = 42, plan: dict[str, dict[str, int]] | None = None
) -> list[dict[str, Any]]:
    """Select unique, moderate-length evaluation records according to a small plan."""

    plan = plan or DEFAULT_PLAN
    rng = random.Random(seed)
    candidates: dict[str, list[Any]] = defaultdict(list)
    seen_questions: set[str] = set()

    for canonical_domain in sorted(plan):
        source_domain = "math" if canonical_domain == "mathematics" else canonical_domain
        for path in _candidate_paths(root, source_domain):
            for record in iter_normalized_records(path, root=root):
                normalized_question = " ".join(record.question.lower().split())
                length = word_count(record.question)
                if not 8 <= length <= 90 or normalized_question in seen_questions:
                    continue
                seen_questions.add(normalized_question)
                candidates[canonical_domain].append(record)

    anchors: list[dict[str, Any]] = []
    for domain in sorted(plan):
        domain_records = candidates[domain]
        rng.shuffle(domain_records)
        required = sum(plan[domain].values())
        if len(domain_records) < required:
            raise ValueError(
                f"Not enough eligible {domain} records: need {required}, found {len(domain_records)}"
            )
        cursor = 0
        for mode in ("real_anchored", "synthetic"):
            for _ in range(plan[domain].get(mode, 0)):
                record = domain_records[cursor]
                cursor += 1
                digest = hashlib.sha1(
                    f"{record.source_file}:{record.source_id}:{mode}".encode("utf-8")
                ).hexdigest()[:10]
                anchors.append(
                    {
                        "anchor_version": "pilot_anchor.v1",
                        "id": f"anchor_{digest}",
                        "requested_mode": mode,
                        "source": {
                            "dataset": record.source_file,
                            "original_id": record.source_id,
                            "source_domain": domain,
                            "original_question": record.question,
                            "original_answer": record.answer,
                            "original_choices": record.choices or None,
                        },
                    }
                )

    rng.shuffle(anchors)
    return anchors
