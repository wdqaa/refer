"""Read heterogeneous source QA files without modifying them.

The loader supports line-delimited JSON and JSON arrays, then maps the known source
schemas to a small normalized view used by analysis and future anchor selection.
It intentionally retains source-relative identity rather than inventing global IDs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterator, Mapping


class DatasetFormatError(ValueError):
    """Raised when a source file or record cannot be interpreted safely."""


@dataclass(frozen=True)
class NormalizedRecord:
    """A loss-minimizing analysis view over one heterogeneous QA record."""

    source_file: str
    source_index: int
    source_domain: str
    source_id: str
    question: str
    answer: str
    choices: dict[str, str]
    schema_fields: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        value = asdict(self)
        value["schema_fields"] = list(self.schema_fields)
        return value


def iter_raw_records(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    """Yield `(zero_based_index, record)` pairs from a JSON or JSONL file."""

    if not path.is_file():
        raise DatasetFormatError(f"Dataset file does not exist: {path}")

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetFormatError(
                        f"Invalid JSON at {path}:{line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(record, Mapping):
                    raise DatasetFormatError(
                        f"Expected object at {path}:{line_number}, got "
                        f"{type(record).__name__}"
                    )
                yield line_number - 1, record
        return

    if path.suffix == ".json":
        try:
            with path.open("r", encoding="utf-8-sig") as handle:
                records = json.load(handle)
        except json.JSONDecodeError as exc:
            raise DatasetFormatError(f"Invalid JSON in {path}: {exc.msg}") from exc
        if not isinstance(records, list):
            raise DatasetFormatError(
                f"Expected a top-level array in {path}, got {type(records).__name__}"
            )
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise DatasetFormatError(
                    f"Expected object at {path}[{index}], got {type(record).__name__}"
                )
            yield index, record
        return

    raise DatasetFormatError(f"Unsupported dataset extension: {path}")


def _extract_prompt_fields(prompt: str) -> tuple[str, dict[str, str]]:
    """Remove chat wrappers and split a prompt into stem and inline choices."""

    text = prompt
    user_marker = "<|start_header_id|>user<|end_header_id|>"
    if user_marker in text:
        text = text.split(user_marker, maxsplit=1)[1]
    text = text.split("<|eot_id|>", maxsplit=1)[0].strip()

    paragraphs = text.split("\n\n")
    if len(paragraphs) > 1:
        first = paragraphs[0].lower()
        instruction_signals = (
            "solve the following",
            "given a question",
            "please select",
            "please answer",
        )
        if any(signal in first for signal in instruction_signals):
            text = "\n\n".join(paragraphs[1:]).strip()

    choice_pattern = re.compile(r"(?m)^\s*([A-J])[.)]\s+(.+?)\s*$")
    matches = list(choice_pattern.finditer(text))
    choices = {match.group(1): match.group(2) for match in matches}
    if matches:
        text = text[: matches[0].start()].strip()
        text = re.sub(r"(?:\n\s*)?Choices:\s*$", "", text, flags=re.IGNORECASE).strip()
    return text, choices


def _normalize_choices(value: Any) -> dict[str, str]:
    if isinstance(value, Mapping):
        return {str(key): str(item) for key, item in value.items()}
    if isinstance(value, list):
        return {chr(65 + index): str(item) for index, item in enumerate(value)}
    return {}


def normalize_record(
    raw: Mapping[str, Any], *, source_file: str, source_index: int, source_domain: str
) -> NormalizedRecord:
    """Normalize one supported record or raise a source-specific error."""

    question_value = raw.get("question", raw.get("stem"))
    prompt_choices: dict[str, str] = {}
    if question_value is None and "prompt" in raw:
        question_value, prompt_choices = _extract_prompt_fields(str(raw["prompt"]))
    if question_value is None:
        raise DatasetFormatError(
            f"No question/stem/prompt field in {source_file} record {source_index}"
        )

    answer_value = raw.get("answer", raw.get("mcq_answer", raw.get("response")))
    if answer_value is None:
        raise DatasetFormatError(
            f"No answer/mcq_answer/response field in {source_file} record {source_index}"
        )

    source_id = raw.get("id", f"index:{source_index}")
    return NormalizedRecord(
        source_file=source_file,
        source_index=source_index,
        source_domain=source_domain,
        source_id=str(source_id),
        question=str(question_value).strip(),
        answer=str(answer_value).strip(),
        choices=_normalize_choices(raw.get("choices")) or prompt_choices,
        schema_fields=tuple(sorted(str(key) for key in raw)),
    )


def iter_normalized_records(path: Path, *, root: Path) -> Iterator[NormalizedRecord]:
    """Yield normalized records while preserving paths relative to `root`."""

    try:
        relative_path = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise DatasetFormatError(f"{path} is not below dataset root {root}") from exc
    if len(path.relative_to(root).parts) < 2:
        raise DatasetFormatError(f"Expected domain/file layout below {root}: {path}")
    domain = path.relative_to(root).parts[0]
    for index, raw in iter_raw_records(path):
        yield normalize_record(
            raw,
            source_file=relative_path,
            source_index=index,
            source_domain=domain,
        )
