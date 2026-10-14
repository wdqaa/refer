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


pipeline = load_script("generate_deepseek_qa_pilot")


class DeepSeekQAPilotTest(unittest.TestCase):
    def source_item(self):
        return {
            "source": {"question": "Why do leaves appear green?"},
            "source_analysis": {
                "required_domains": [
                    {"domain": "biology", "evidence": "leaf", "necessity_reason": "pigment"}
                ]
            },
        }

    def test_relation_plan_rejects_math_as_new_domain(self):
        plan = {
            "decision": "propose",
            "new_domain": "mathematics",
            "fusion_relation": "quantitative_modeling",
            "scientific_bridge": "x",
            "new_condition": "x",
            "new_dependency": "x",
            "why_natural": "x",
            "source_dependency_preserved": "x",
            "expected_solution_change": "x",
        }
        issues = pipeline.validate_plan(self.source_item(), plan)
        self.assertTrue(any("mathematics" in issue for issue in issues))

    def test_relation_plan_cannot_switch_preferred_relation(self):
        item = self.source_item()
        item["preferred_fusion_relation"] = "measurement_observation"
        plan = {
            "decision": "propose",
            "new_domain": "chemistry",
            "fusion_relation": "mechanism_explanation",
            "scientific_bridge": "x",
            "new_condition": "x",
            "new_dependency": "x",
            "why_natural": "x",
            "source_dependency_preserved": "x",
            "expected_solution_change": "x",
        }
        issues = pipeline.validate_plan(item, plan)
        self.assertIn("relation planner changed the preferred fusion relation", issues)

    def test_transform_must_add_exactly_planned_domain(self):
        plan = {"new_domain": "chemistry"}
        candidate = {
            "decision": "transform",
            "question": "How does pigment chemistry determine why leaves appear green?",
            "question_type": "qualitative",
            "choices": None,
            "provisional_correct_choice": None,
            "required_domains_after": ["biology", "physics"],
            "new_required_domain": "physics",
            "derivation": ["step"],
        }
        issues = pipeline.validate_transform(self.source_item(), plan, candidate)
        self.assertTrue(any("planned new domain" in issue for issue in issues))

    def test_qualitative_solver_requires_rubric(self):
        solver = {
            "well_posed": True,
            "question_type": "qualitative",
            "answer": "Chlorophyll absorbs red and blue light.",
            "correct_choice": None,
            "standard_solution": ["Identify selective absorption."],
            "required_domains_observed": ["biology"],
            "helpful_domains_observed": [],
            "rubric": None,
        }
        issues = pipeline.validate_solver("Why green?", None, solver)
        self.assertTrue(any("rubric" in issue for issue in issues))

    def test_judge_material_issue_blocks_acceptance(self):
        judge = {
            "scientifically_valid": True,
            "well_posed": True,
            "answer_correct": True,
            "answer_format_valid": True,
            "required_domains_defensible": True,
            "source_dependency_preserved": True,
            "new_dependency_indispensable": True,
            "integrated_reasoning": True,
            "difficulty_for_8b": "medium",
            "observed_required_domains": ["biology", "chemistry"],
            "issues": ["The added chemistry is decorative."],
            "accepted": True,
        }
        issues = pipeline.validate_judge("transform_candidate", judge)
        self.assertIn("judge reported material issues", issues)

    def test_judge_requires_issues_list(self):
        judge = {
            "scientifically_valid": True,
            "well_posed": True,
            "answer_correct": True,
            "answer_format_valid": True,
            "required_domains_defensible": True,
            "source_dependency_preserved": None,
            "new_dependency_indispensable": None,
            "integrated_reasoning": None,
            "difficulty_for_8b": "easy",
            "observed_required_domains": ["biology"],
            "accepted": True,
        }
        issues = pipeline.validate_judge("keep_control", judge)
        self.assertIn("judge issues must be a list", issues)


if __name__ == "__main__":
    unittest.main()
