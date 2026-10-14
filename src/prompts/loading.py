"""Load and validate the central Stage 01 prompt registry."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


DEFAULT_REGISTRY = Path(__file__).with_name("agent_prompts.v3.json")
EXPECTED_ROLES = {
    "planner",
    "physics_expert",
    "chemistry_expert",
    "mathematics_expert",
    "biology_expert",
    "astronomy_expert",
    "geography_expert",
    "critic",
    "question_generator",
    "question_verifier",
}
LEGACY_EXPECTED_ROLES = EXPECTED_ROLES - {"geography_expert"}


class PromptRegistryError(ValueError):
    """Raised when prompt configuration is missing or internally inconsistent."""


def _read_registry(path: Path) -> tuple[str, list[dict[str, Any]], Path | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PromptRegistryError(f"Cannot read prompt registry {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PromptRegistryError(f"Invalid prompt registry JSON {path}: {exc.msg}") from exc

    version = payload.get("registry_version")
    if version not in {
        "agent_prompts.v1", "agent_prompts.v2", "agent_prompts.v3", "agent_prompts.v4"
    }:
        raise PromptRegistryError(f"Unsupported or missing registry_version: {version!r}")
    entries = payload.get("prompts")
    if not isinstance(entries, list):
        raise PromptRegistryError("prompts must be a list")
    base_value = payload.get("base_registry")
    base_path = path.with_name(base_value) if isinstance(base_value, str) else None
    return version, entries, base_path


def load_prompt_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, dict[str, Any]]:
    """Return prompts keyed by role, applying an optional versioned base registry."""

    version, entries, base_path = _read_registry(path)
    by_role: dict[str, dict[str, Any]] = (
        load_prompt_registry(base_path) if base_path is not None else {}
    )

    ids: set[str] = {entry["id"] for entry in by_role.values()}
    for entry in entries:
        if not isinstance(entry, dict):
            raise PromptRegistryError("Every prompt entry must be an object")
        prompt_id = entry.get("id")
        role = entry.get("role")
        if not isinstance(prompt_id, str) or not re.search(r"_v\d+$", prompt_id):
            raise PromptRegistryError(f"Prompt has invalid versioned id: {prompt_id!r}")
        if prompt_id in ids:
            raise PromptRegistryError(f"Duplicate prompt id: {prompt_id}")
        if not isinstance(entry.get("system_prompt"), str) or not entry["system_prompt"].strip():
            raise PromptRegistryError(f"Prompt {prompt_id} has empty system_prompt")
        if not isinstance(entry.get("output_schema"), dict):
            raise PromptRegistryError(f"Prompt {prompt_id} has no output_schema object")
        ids.add(prompt_id)
        by_role[str(role)] = entry

    expected_roles = (
        EXPECTED_ROLES
        if version in {"agent_prompts.v3", "agent_prompts.v4"}
        else LEGACY_EXPECTED_ROLES
    )
    missing = expected_roles - set(by_role)
    extra = set(by_role) - expected_roles
    if missing or extra:
        raise PromptRegistryError(
            f"Prompt roles mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return by_role
