import importlib.util
from pathlib import Path
import unittest


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


validate = load_script("analyze_deepseek_protocol_batch").validate
load_prompt = load_script("analyze_deepseek_protocol_batch").load_prompt
surface_bucket = load_script("build_deepseek_protocol_batch").surface_bucket


class DeepSeekProtocolTest(unittest.TestCase):
    def valid_analysis(self):
        return {
            "scientifically_valid": True,
            "validity_issue": None,
            "question_type": "mcq",
            "answerability": "objective",
            "source_taxonomy": "A1",
            "primary_domain": "physics",
            "required_domains": [
                {"domain": "physics", "evidence": "force", "necessity_reason": "law"}
            ],
            "helpful_domains": [],
            "background_domains": [],
            "already_cross_domain": False,
            "mathematics_level": "none",
            "reference_answer_role": "objective_gold_candidate",
            "reference_answer_issue": None,
            "decision": "keep_control",
            "natural_extension_possible": False,
            "natural_extension_direction": None,
            "confidence": 0.9,
        }

    def test_valid_analysis_contract(self):
        self.assertEqual(validate(self.valid_analysis()), [])

    def test_math_label_requires_independent_level(self):
        analysis = self.valid_analysis()
        analysis["required_domains"] = [
            {"domain": "mathematics", "evidence": "x", "necessity_reason": "x"}
        ]
        errors = validate(analysis)
        self.assertTrue(any("required mathematics" in error for error in errors))

        analysis["mathematics_level"] = "independent"
        analysis["primary_domain"] = "mathematics"
        self.assertEqual(validate(analysis), [])

    def test_surface_bucket_preserves_mcq(self):
        record = {"question": "Calculate the force.", "answer": "B", "choices": {"A": "1", "B": "2"}}
        self.assertEqual(surface_bucket(record), "mcq")

    def test_v2_1_prompt_extends_v2(self):
        root = Path(__file__).resolve().parents[1]
        prompt = load_prompt(root / "src/prompts/deepseek_source_analyzer_v2_1.json")
        self.assertEqual(prompt["version"], "2.1")
        self.assertIn("minimum-required", prompt["system_prompt"])
        self.assertIn("ANOVA", prompt["system_prompt"])
        self.assertIn("{question}", prompt["user_template"])


if __name__ == "__main__":
    unittest.main()
