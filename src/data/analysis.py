"""Compute reproducible descriptive statistics over normalized source QA records."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import math
from pathlib import Path
import re
from typing import Any, Iterable

from .loading import NormalizedRecord, iter_normalized_records


_QUANTITATIVE = re.compile(
    r"(?:\bcalculate\b|\bcompute\b|\bhow (?:many|much|far|long)\b|"
    r"\bwhat (?:is|are) the (?:value|magnitude|probability|ratio|rate)\b|"
    r"[=+*/^]|\\(?:frac|sqrt)|\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_MULTISTEP = re.compile(
    r"(?:\bthen\b|\bafter\b|\bcombined\b|\brespectively\b|\bfirst\b.+\bnext\b)",
    re.IGNORECASE,
)


def word_count(text: str) -> int:
    """Approximate language-agnostic length using words or individual CJK chars."""

    return len(re.findall(r"[A-Za-z0-9]+(?:['.-][A-Za-z0-9]+)*|[\u3400-\u9fff]", text))


def answer_format(record: NormalizedRecord) -> str:
    return "multiple_choice" if record.choices else "open_ended"


def reasoning_proxy(record: NormalizedRecord) -> str:
    return "quantitative" if _QUANTITATIVE.search(record.question) else "conceptual"


def difficulty_proxy(record: NormalizedRecord) -> str:
    """Return a transparent text-only proxy, not a ground-truth difficulty label."""

    length = word_count(record.question)
    quantitative = reasoning_proxy(record) == "quantitative"
    multistep = bool(_MULTISTEP.search(record.question))
    if length > 80 or (length > 45 and quantitative and multistep):
        return "hard"
    if length > 35 or quantitative or multistep:
        return "medium"
    return "easy"


def _percentile(sorted_values: list[int], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def length_summary(values: Iterable[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0}
    return {
        "count": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 2),
        "median": round(_percentile(ordered, 0.5), 2),
        "p25": round(_percentile(ordered, 0.25), 2),
        "p75": round(_percentile(ordered, 0.75), 2),
        "p90": round(_percentile(ordered, 0.9), 2),
    }


def analyze_dataset(root: Path, *, examples_per_file: int = 1) -> dict[str, Any]:
    """Analyze all JSON/JSONL files below `root` with bounded text retention."""

    if not root.is_dir():
        raise ValueError(f"Input dataset directory does not exist: {root}")
    if examples_per_file < 0:
        raise ValueError("examples_per_file must be non-negative")

    paths = sorted(
        path
        for path in root.glob("*/*")
        if path.suffix in {".json", ".jsonl"}
        and not path.relative_to(root).parts[0].startswith(".")
    )
    if not paths:
        raise ValueError(f"No JSON or JSONL files found below {root}")

    domain_counts: Counter[str] = Counter()
    domain_question_lengths: dict[str, list[int]] = defaultdict(list)
    domain_answer_lengths: dict[str, list[int]] = defaultdict(list)
    format_counts: Counter[str] = Counter()
    reasoning_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    file_reports: dict[str, Any] = {}
    examples: list[dict[str, Any]] = []

    for path in paths:
        records = 0
        schemas: Counter[str] = Counter()
        local_question_lengths: list[int] = []
        local_answer_lengths: list[int] = []
        local_formats: Counter[str] = Counter()
        local_reasoning: Counter[str] = Counter()
        local_difficulty: Counter[str] = Counter()
        relative = path.relative_to(root).as_posix()

        for record in iter_normalized_records(path, root=root):
            records += 1
            q_length = word_count(record.question)
            a_length = word_count(record.answer)
            fmt = answer_format(record)
            reasoning = reasoning_proxy(record)
            difficulty = difficulty_proxy(record)

            schemas["|".join(record.schema_fields)] += 1
            local_question_lengths.append(q_length)
            local_answer_lengths.append(a_length)
            local_formats[fmt] += 1
            local_reasoning[reasoning] += 1
            local_difficulty[difficulty] += 1

            domain_counts[record.source_domain] += 1
            domain_question_lengths[record.source_domain].append(q_length)
            domain_answer_lengths[record.source_domain].append(a_length)
            format_counts[fmt] += 1
            reasoning_counts[reasoning] += 1
            difficulty_counts[difficulty] += 1

            if len([e for e in examples if e["source_file"] == relative]) < examples_per_file:
                examples.append(
                    {
                        "source_file": relative,
                        "source_id": record.source_id,
                        "question": record.question[:500],
                        "answer": record.answer[:300],
                        "choice_count": len(record.choices),
                    }
                )

        file_reports[relative] = {
            "records": records,
            "schemas": dict(sorted(schemas.items())),
            "question_words": length_summary(local_question_lengths),
            "answer_words": length_summary(local_answer_lengths),
            "answer_format": dict(sorted(local_formats.items())),
            "reasoning_proxy": dict(sorted(local_reasoning.items())),
            "difficulty_proxy": dict(sorted(local_difficulty.items())),
        }

    non_mixed_count = sum(count for domain, count in domain_counts.items() if domain != "mixed")
    return {
        "analysis_version": "source_analysis.v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_root": str(root.resolve()),
        "notes": [
            "Counts are raw records; cross-file deduplication has not been applied.",
            "The mixed domain is reported separately and excluded from non_mixed_total.",
            "Lengths are approximate word/CJK-character counts of normalized stems.",
            "Difficulty and reasoning labels are rule-based proxies, not gold labels.",
        ],
        "totals": {
            "all_records_including_mixed": sum(domain_counts.values()),
            "non_mixed_total": non_mixed_count,
            "mixed_total": domain_counts.get("mixed", 0),
            "files": len(paths),
        },
        "domain_counts": dict(sorted(domain_counts.items())),
        "domain_lengths": {
            domain: {
                "question_words": length_summary(domain_question_lengths[domain]),
                "answer_words": length_summary(domain_answer_lengths[domain]),
            }
            for domain in sorted(domain_counts)
        },
        "answer_format": dict(sorted(format_counts.items())),
        "reasoning_proxy": dict(sorted(reasoning_counts.items())),
        "difficulty_proxy": dict(sorted(difficulty_counts.items())),
        "files": file_reports,
        "examples": examples,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the compact human-readable portion of a JSON analysis report."""

    totals = report["totals"]
    lines = [
        "# Source Dataset Analysis",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Scope",
        "",
        f"- Files: {totals['files']}",
        f"- Non-mixed raw records: {totals['non_mixed_total']:,}",
        f"- Mixed aggregate records (reported separately): {totals['mixed_total']:,}",
        f"- All records including mixed: {totals['all_records_including_mixed']:,}",
        "",
        "## Domain counts",
        "",
        "| Domain | Raw records | Question words (median / P90) | Answer words (median / P90) |",
        "|---|---:|---:|---:|",
    ]
    for domain, count in report["domain_counts"].items():
        q = report["domain_lengths"][domain]["question_words"]
        a = report["domain_lengths"][domain]["answer_words"]
        lines.append(
            f"| {domain} | {count:,} | {q['median']:.1f} / {q['p90']:.1f} | "
            f"{a['median']:.1f} / {a['p90']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Coarse distributions (including mixed)",
            "",
            f"- Answer format: `{report['answer_format']}`",
            f"- Reasoning proxy: `{report['reasoning_proxy']}`",
            f"- Difficulty proxy: `{report['difficulty_proxy']}`",
            "",
            "## Interpretation limits",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in report["notes"])
    lines.extend(
        [
            "- Directory domain is source provenance, not a verified required-domain label.",
            "- The JSON report contains per-file schemas, distributions, and bounded examples.",
            "",
        ]
    )
    return "\n".join(lines)
