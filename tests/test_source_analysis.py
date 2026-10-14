"""Tests for Source Analysis gold resolution and core metrics."""

from __future__ import annotations

import unittest

from src.evaluation.source_analysis import compute_metrics, parse_prediction


class SourceAnalysisTest(unittest.TestCase):
    def test_parse_prediction(self) -> None:
        text = '{"scientifically_valid":true,"primary_domain":"physics","required_domains":[{"domain":"physics","evidence":"x","necessity_reason":"y"}],"background_domains":[],"already_cross_domain":false,"ordinary_math_only":true,"decision":"keep_control","natural_extension_possible":false,"natural_extension_direction":null,"confidence":0.9}'
        self.assertEqual(parse_prediction(text)["primary_domain"], "physics")

    def test_perfect_prediction_metrics(self) -> None:
        value = {
            "scientifically_valid": True,
            "primary_domain": "physics",
            "required_domains": [{"domain": "physics", "evidence": "x", "necessity_reason": "y"}],
            "background_domains": [],
            "already_cross_domain": False,
            "ordinary_math_only": True,
            "decision": "keep_control",
            "natural_extension_possible": False,
            "natural_extension_direction": None,
            "confidence": 1.0,
        }
        gold = [{"id": "x", "category": "ordinary_math_not_domain", "gold": value}]
        metrics = compute_metrics(gold, [{"id": "x", "prediction": value}])
        self.assertEqual(metrics["domain_exact_match_accuracy"], 1.0)
        self.assertEqual(metrics["math_false_positive_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
