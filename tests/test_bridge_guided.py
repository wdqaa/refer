import importlib.util
import json
from pathlib import Path
import unittest


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = load_script("build_bridge_calibration_manifest")
selector = load_script("select_interdisciplinary_bridge")
transform_smoke = load_script("generate_human_bridge_transform_smoke")
ROOT = Path(__file__).resolve().parents[1]


class BridgeGuidedTest(unittest.TestCase):
    def test_library_is_small_and_contains_only_named_bridges(self):
        library = selector.load_library(
            ROOT / "configs/data/interdisciplinary_bridge_library.yaml"
        )
        self.assertEqual(set(library), {"physics", "chemistry", "biology", "astronomy", "geography"})
        self.assertIn("nonlinear_dynamics", library["biology"]["mathematics"])
        self.assertLessEqual(
            sum(len(bridges) for targets in library.values() for bridges in targets.values()),
            60,
        )

    def test_v1_1_extends_v1_without_changing_user_template(self):
        prompt = selector.load_prompt(ROOT / "src/prompts/bridge_selector_v1_1.json")
        base = selector.load_prompt(ROOT / "src/prompts/bridge_selector_v1.json")
        self.assertEqual(prompt["version"], "1.1")
        self.assertEqual(prompt["user_template"], base["user_template"])
        self.assertIn("Lack of numbers alone", prompt["system_prompt"])

    def test_valid_select_must_use_supplied_bridge(self):
        candidates = {"mathematics": ["nonlinear_dynamics"]}
        valid = {
            "decision": "select",
            "target_domain": "mathematics",
            "bridge": "nonlinear_dynamics",
            "reason": "Feedback stability changes the biological conclusion.",
            "expected_new_dependency": "Analyze nonlinear feedback stability.",
            "confidence": 0.9,
            "risk_flags": [],
        }
        self.assertEqual(selector.validate(valid, candidates), [])
        invalid = {**valid, "bridge": "statistics"}
        self.assertIn(
            "bridge is not allowed for the selected target domain",
            selector.validate(invalid, candidates),
        )

    def test_hard_risk_forces_none(self):
        candidates = {"physics": ["diffusion"]}
        invalid = {
            "decision": "select",
            "target_domain": "physics",
            "bridge": "diffusion",
            "reason": "A diffusion model could be added.",
            "expected_new_dependency": "Use Fick's law.",
            "confidence": 0.7,
            "risk_flags": ["false_boundary_split"],
        }
        self.assertIn("select cannot contain a hard risk", selector.validate(invalid, candidates))

    def test_none_requires_null_bridge_fields(self):
        candidates = {"physics": ["diffusion"]}
        valid = {
            "decision": "none",
            "target_domain": None,
            "bridge": None,
            "reason": "No natural extension.",
            "expected_new_dependency": None,
            "confidence": 0.95,
            "risk_flags": ["no_natural_extension"],
        }
        self.assertEqual(selector.validate(valid, candidates), [])

    def test_fixed_manifest_has_25_unique_unused_records(self):
        config = json.loads(
            (ROOT / "configs/data/bridge_guided_calibration_v1.json").read_text()
        )
        sources = builder.read_jsonl(
            ROOT
            / "datasets/processed/transformability_selector_calibration_v1/candidate_sources_150.jsonl"
        )
        manifest = builder.build_manifest(sources, config)
        self.assertEqual(len(manifest), 25)
        self.assertEqual(len({item["bridge_source_id"] for item in manifest}), 25)
        self.assertTrue(all(len(item["source_analysis"]["required_domains"]) == 1 for item in manifest))

    def test_human_bridge_smoke_has_four_positive_and_four_negative_pairs(self):
        config = json.loads(
            (ROOT / "configs/data/human_bridge_transform_smoke_v1.json").read_text()
        )
        sources = builder.read_jsonl(
            ROOT / "datasets/processed/bridge_guided_calibration_v1/sources_25.jsonl"
        )
        pairs = transform_smoke.build_pairs(sources, config)
        self.assertEqual(len(pairs), 8)
        self.assertEqual(
            {case: sum(pair["expected_case"] == case for pair in pairs) for case in ("positive", "negative")},
            {"positive": 4, "negative": 4},
        )

    def test_transform_gate_enforces_fixed_target_and_exact_domains(self):
        pair = {
            "primary_domain": "biology",
            "target_domain": "mathematics",
            "source": {"question": "How does feedback regulate glucose?"},
        }
        candidate = {
            "decision": "transform",
            "question": "For a glucose-feedback model, determine whether the equilibrium is stable.",
            "question_type": "short_answer",
            "choices": None,
            "provisional_correct_choice": None,
            "required_domains_after": ["biology", "mathematics"],
            "new_required_domain": "mathematics",
            "derivation": ["Identify biological feedback.", "Analyze stability."],
            "source_dependency_preserved": "Biological feedback fixes the model interpretation.",
            "new_dependency_indispensable": "Stability analysis determines the result.",
            "transformation_summary": "Adds nonlinear stability analysis.",
        }
        self.assertEqual(transform_smoke.validate_transform(pair, candidate), [])
        changed = {**candidate, "new_required_domain": "physics"}
        self.assertIn(
            "transformer changed the fixed target domain",
            transform_smoke.validate_transform(pair, changed),
        )

    def test_reject_transform_passes_transform_gate(self):
        pair = {
            "primary_domain": "astronomy",
            "target_domain": "physics",
            "source": {"question": "A standard astrophysics question."},
        }
        self.assertEqual(
            transform_smoke.validate_transform(pair, {"decision": "reject_transform"}), []
        )


if __name__ == "__main__":
    unittest.main()
