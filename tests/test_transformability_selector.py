import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from src.data.loading import NormalizedRecord
from src.generation.deepseek_api import DeepSeekJSONClient


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


analyzer = load_script("analyze_source_transformability")
pool_builder = load_script("build_transformability_source_pool")
shortlister = load_script("select_transformability_shortlist")


def record(index, *, choices=None):
    return NormalizedRecord(
        source_file="astronomy/astronomy_train.jsonl",
        source_index=index,
        source_domain="astronomy",
        source_id=f"index:{index}",
        question="A wording that must not influence relation sampling.",
        answer="A grounded answer.",
        choices=choices or {},
        schema_fields=("question", "answer"),
    )


def valid_high():
    return {
        "self_contained": True,
        "answer_grounded": True,
        "transformability": "high",
        "candidate_relations": ["measurement_inference"],
        "possible_new_domains": ["mathematics"],
        "reason": "A measured observable can support an independent inverse model.",
        "risk_flags": [],
    }


class TransformabilitySelectorTest(unittest.TestCase):
    def test_high_schema_and_gate(self):
        analysis = valid_high()
        self.assertEqual(analyzer.validate(analysis), [])
        trace = {
            "selector_source_id": "selector-test-0",
            "analysis": analysis,
            "schema_valid": True,
            "error": None,
        }
        self.assertTrue(shortlister.eligible(trace))

    def test_high_with_risk_is_rejected(self):
        analysis = valid_high()
        analysis["risk_flags"] = ["answer_already_supplied"]
        self.assertIn("high cannot contain risk flags", analyzer.validate(analysis))

    def test_false_boundary_cannot_be_high(self):
        analysis = valid_high()
        analysis["risk_flags"] = ["false_boundary_split"]
        errors = analyzer.validate(analysis)
        self.assertIn("false_boundary_split cannot be high", errors)

    def test_no_natural_extension_has_empty_candidates(self):
        analysis = valid_high()
        analysis.update(
            {
                "transformability": "low",
                "candidate_relations": [],
                "possible_new_domains": [],
                "risk_flags": ["no_natural_extension"],
            }
        )
        self.assertEqual(analyzer.validate(analysis), [])

    def test_pool_is_deterministic_excludes_prior_and_has_no_affordance_hint(self):
        records = [
            record(0, choices={"A": "x", "B": "y"}),
            record(1, choices={"A": "x", "B": "y"}),
            record(2),
            record(3),
            record(4),
        ]
        kwargs = {
            "records_by_domain": {"astronomy": records},
            "quotas": {"astronomy": {"mcq": 1, "open": 2}},
            "excluded": {(records[4].source_file, records[4].source_index)},
            "pool_id": "test",
            "seed": 3,
        }
        first = pool_builder.build_pool(**kwargs)
        second = pool_builder.build_pool(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertNotIn(4, {item["source_index"] for item in first})
        self.assertFalse(any("affordance_hint" in item for item in first))
        self.assertEqual({item["sampling_stratum"] for item in first}, {"mcq", "open"})

    def test_shortlist_keeps_only_strict_high(self):
        sources = [
            {"selector_source_id": "a", "source_domain": "astronomy"},
            {"selector_source_id": "b", "source_domain": "biology"},
        ]
        traces = [
            {
                "selector_source_id": "a",
                "analysis": valid_high(),
                "schema_valid": True,
                "error": None,
                "prompt_version": "1.0",
            },
            {
                "selector_source_id": "b",
                "analysis": {**valid_high(), "transformability": "medium"},
                "schema_valid": True,
                "error": None,
                "prompt_version": "1.0",
            },
        ]
        selected = shortlister.select_shortlist(sources, traces, limit=30)
        self.assertEqual([item["selector_source_id"] for item in selected], ["a"])

    def test_selector_request_disables_thinking_and_has_no_retry(self):
        envelope = {
            "model": "deepseek-flash",
            "choices": [
                {
                    "message": {"content": json.dumps(valid_high())},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(envelope).encode("utf-8")

        client = DeepSeekJSONClient(api_key="test-only")
        with patch("src.generation.deepseek_api.urlopen", return_value=Response()) as mocked:
            parsed, _ = client.generate_json(
                [{"role": "user", "content": "classify"}],
                max_tokens=600,
                thinking_enabled=False,
                retries=0,
            )
        request = mocked.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(parsed, valid_high())
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", payload)
        self.assertEqual(payload["max_tokens"], 600)


if __name__ == "__main__":
    unittest.main()
