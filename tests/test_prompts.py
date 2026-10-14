"""Tests for centralized, versioned prompt definitions."""

from __future__ import annotations

from pathlib import Path
import json
import unittest

from src.prompts.loading import EXPECTED_ROLES, load_prompt_registry


class PromptRegistryTest(unittest.TestCase):
    def test_all_expected_roles_load(self) -> None:
        prompts = load_prompt_registry()
        self.assertEqual(set(prompts), EXPECTED_ROLES)
        self.assertEqual(prompts["question_generator"]["id"], "question_generator_v3")
        self.assertEqual(prompts["planner"]["id"], "planner_v2")
        self.assertEqual(prompts["geography_expert"]["id"], "geography_expert_v1")

    def test_v1_registry_remains_loadable(self) -> None:
        path = Path(__file__).resolve().parents[1] / "src" / "prompts" / "agent_prompts.v1.json"
        prompts = load_prompt_registry(path)
        self.assertEqual(prompts["question_generator"]["id"], "question_generator_v1")
        self.assertNotIn("geography_expert", prompts)

    def test_planner_explicitly_hides_ground_truth_routing(self) -> None:
        planner = load_prompt_registry()["planner"]["system_prompt"]
        self.assertIn("required_domains", planner)
        self.assertIn("Never assume", planner)

    def test_v4_registry_overrides_all_six_experts(self) -> None:
        root = Path(__file__).resolve().parents[1]
        prompts = load_prompt_registry(root / "src/prompts/agent_prompts.v4.json")
        for domain in (
            "physics", "chemistry", "mathematics", "biology", "astronomy", "geography"
        ):
            prompt = prompts[f"{domain}_expert"]
            self.assertTrue(prompt["id"].endswith("_v2"))
            self.assertIn("complete", prompt["system_prompt"].lower())

    def test_baseline_v2_keeps_router_prompt_identical(self) -> None:
        root = Path(__file__).resolve().parents[1]
        v1 = json.loads((root / "src/prompts/qwen_baselines_v1.json").read_text())
        v2 = json.loads((root / "src/prompts/qwen_baselines_v2.json").read_text())
        self.assertEqual(v1["router"], v2["router"])
        self.assertIn("every part", v2["direct_solver"]["system_prompt"])
        self.assertIn("final problem solver", v2["synthesizer"]["system_prompt"])


if __name__ == "__main__":
    unittest.main()
