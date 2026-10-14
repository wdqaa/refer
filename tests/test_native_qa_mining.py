import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = load_script("build_native_qa_calibration")
holdout_builder = load_script("build_native_qa_holdout")
pilot_builder = load_script("build_native_qa_pilot")
pilot_review = load_script("export_native_qa_pilot_review")
eval_split = load_script("build_native_qa_eval_split")
gold_enricher = load_script("enrich_native_qa_gold")
gold_finalizer = load_script("finalize_native_qa_gold")
baseline_view = load_script("build_native_qa_baseline_view")
classifier = load_script("classify_native_qa")


class NativeQAMiningTest(unittest.TestCase):
    def test_prompt_uses_requested_label_schema(self):
        prompt = json.loads((ROOT / "src/prompts/native_qa_classifier_v1.json").read_text())
        for field in classifier.LABEL_FIELDS:
            self.assertIn(field, prompt["system_prompt"])
        self.assertIn("mature field", prompt["system_prompt"])
        self.assertIn("Do not lower priority", prompt["system_prompt"])

    def test_valid_label_and_boundary_invariants(self):
        item = {"native_qa_id": "x", "choices": {"A": "one", "B": "two"}}
        label = {
            "id": "x",
            "valid": True,
            "question_type": "mcq",
            "required_domains": ["physics"],
            "domain_type": "single",
            "answer_verifiability": "rule_verifiable",
            "keep_priority": "high",
        }
        self.assertEqual(classifier.validate_label(item, label), [])
        bad = {**label, "required_domains": ["physics", "mathematics"]}
        self.assertIn("domain_type conflicts with required_domains", classifier.validate_label(item, bad))
        invalid_high = {**label, "valid": False}
        self.assertTrue(classifier.validate_label(item, invalid_high))

    def test_invalid_item_may_have_no_required_domain(self):
        item = {"native_qa_id": "x", "choices": {}}
        label = {
            "id": "x",
            "valid": False,
            "question_type": "qualitative",
            "required_domains": [],
            "domain_type": "single",
            "answer_verifiability": "unreliable",
            "keep_priority": "low",
        }
        self.assertEqual(classifier.validate_label(item, label), [])

    def test_builder_excludes_mixed_duplicate_file(self):
        files = builder.data_files(ROOT / "datasets/science_dataset/mixed_dataset")
        self.assertTrue(files)
        self.assertFalse(any("/mixed/" in path.as_posix() for path in files))
        self.assertEqual({path.parts[-2] for path in files}, {"astronomy", "biology", "chemistry", "geography", "math", "physics"})

    def test_batching_is_lossless(self):
        items = [{"id": index} for index in range(12)]
        grouped = classifier.batches(items, 5)
        self.assertEqual([len(group) for group in grouped], [5, 5, 2])
        self.assertEqual([item for group in grouped for item in group], items)

    def test_holdout_definition_is_fixed_and_balanced(self):
        self.assertEqual(len(holdout_builder.VERIFIED_SINGLE_IDS), 15)
        self.assertEqual(len(holdout_builder.VERIFIED_CROSS_IDS), 9)
        self.assertEqual(len(holdout_builder.INVALID_PROTOCOL_IDS), 6)
        self.assertEqual(
            len(set(holdout_builder.VERIFIED_SINGLE_IDS + holdout_builder.VERIFIED_CROSS_IDS)),
            24,
        )

    def test_gold_metrics_separate_valid_items_from_invalid_controls(self):
        def gold(item_id, valid, domains, category):
            return {
                "native_qa_id": item_id,
                "calibration_gold": {
                    "scientifically_valid": valid,
                    "required_domains": [{"domain": domain} for domain in domains],
                    "category": category,
                },
            }

        def label(item_id, valid, domains):
            return {
                "id": item_id,
                "valid": valid,
                "required_domains": domains,
                "domain_type": "native_cross_domain" if len(domains) >= 2 else "single",
            }

        manifest = [
            gold("cross", True, ["physics", "biology"], "valid_cross"),
            gold("single", True, ["chemistry"], "valid_single"),
            gold("invalid", False, ["astronomy"], "invalid_control"),
        ]
        labeled = {
            "cross": label("cross", True, ["physics", "biology"]),
            "single": label("single", True, ["chemistry"]),
            "invalid": label("invalid", True, ["astronomy", "physics"]),
        }
        metrics = classifier.gold_metrics(manifest, labeled)
        self.assertEqual(metrics["valid_only"]["cross_domain_precision"], 1.0)
        self.assertEqual(metrics["valid_only"]["cross_domain_recall"], 1.0)
        self.assertEqual(metrics["invalid_controls"]["rejection_rate"], 0.0)
        self.assertEqual(metrics["cross_domain_precision"], 0.5)

    def test_pilot_config_has_exact_balanced_quota(self):
        config = json.loads(
            (ROOT / "configs/data/native_qa_mining_pilot_500.json").read_text()
        )
        quotas = config["quota_by_domain_surface"]
        self.assertEqual(set(quotas), classifier.DOMAINS)
        self.assertTrue(all(set(surfaces) == {"mcq", "open"} for surfaces in quotas.values()))
        self.assertEqual(sum(sum(surfaces.values()) for surfaces in quotas.values()), 500)

    def test_pilot_review_covers_all_high_cross_candidates(self):
        pilot = ROOT / "datasets/processed/native_qa_mining_v1/pilot_500"
        candidates = {item["native_qa_id"] for item in pilot_review.read_jsonl(pilot / "high_priority_cross_domain.jsonl")}
        reviews = pilot_review.read_jsonl(pilot / "high_cross_human_review.jsonl")
        self.assertEqual({item["native_qa_id"] for item in reviews}, candidates)
        self.assertEqual(len(reviews), len(candidates))
        self.assertEqual(sum(item["disposition"] == "accept_cross" for item in reviews), 17)

    def test_high_single_review_policy_covers_only_candidates(self):
        pilot = ROOT / "datasets/processed/native_qa_mining_v1/pilot_500"
        candidates = {
            item["native_qa_id"]
            for item in pilot_review.read_jsonl(pilot / "high_priority_single.jsonl")
        }
        policy = json.loads((pilot / "high_single_review_policy.json").read_text())
        overrides = set(policy["downgrade_medium"]) | set(policy["reject"])
        self.assertTrue(policy["reviewed_all"])
        self.assertTrue(overrides <= candidates)
        self.assertFalse(set(policy["downgrade_medium"]) & set(policy["reject"]))

    def test_native_eval_split_is_deterministic_and_balanced(self):
        records = pilot_review.read_jsonl(
            ROOT / "datasets/processed/native_qa_mining_v1/pilot_500/reviewed_native_qa.jsonl"
        )
        first, remaining = eval_split.build(records, 131)
        second, _ = eval_split.build(records, 131)
        self.assertEqual([item["native_qa_id"] for item in first], [item["native_qa_id"] for item in second])
        self.assertEqual(len(first), 50)
        self.assertEqual(len(remaining), 164)
        self.assertEqual(sum(item["domain_type"] == "native_cross_domain" for item in first), 25)

    def test_gold_enrichment_schema_and_mcq_choice_gate(self):
        item = {"native_qa_id": "x", "choices": {"A": "one", "B": "two"}, "answer": "B"}
        result = {
            "id": "x", "accepted": True,
            "gold_answer": {"correct_choice": "B", "answer_text": "two"},
            "standard_solution": ["Apply the relevant rule.", "Therefore choose B."],
            "scoring_rubric": [
                {"criterion": "Chooses B", "points": 6},
                {"criterion": "Uses the decisive rule", "points": 4},
            ],
            "issues": [],
        }
        self.assertEqual(gold_enricher.validate(item, result), [])
        result["gold_answer"]["correct_choice"] = "A"
        self.assertIn("MCQ choice changed", gold_enricher.validate(item, result))

    def test_gold_enrichment_normalizes_shape_without_semantic_repair(self):
        result = {
            "standard_solution": "One complete solution paragraph.",
            "scoring_rubric": [
                {"criterion": "Correct answer", "points": 10},
                {"criterion": "Explanation only", "points": 0},
            ],
        }
        normalized = gold_enricher.normalize_result(result)
        self.assertEqual(normalized["standard_solution"], ["One complete solution paragraph."])
        self.assertEqual(normalized["scoring_rubric"], [{"criterion": "Correct answer", "points": 10}])

        nested = {
            "gold_answer": {
                "correct_choice": None,
                "answer_text": "answer",
                "standard_solution": "Nested solution.",
                "scoring_rubric": [{"criterion": "Core point", "points": 10}],
                "issues": [],
            },
            "standard_solution": [],
            "scoring_rubric": [],
            "issues": [],
        }
        repaired = gold_enricher.normalize_result(nested)
        self.assertEqual(set(repaired["gold_answer"]), {"correct_choice", "answer_text"})
        self.assertEqual(repaired["standard_solution"], ["Nested solution."])
        self.assertEqual(repaired["scoring_rubric"], [{"criterion": "Core point", "points": 10}])

        rejected = gold_enricher.normalize_result({"standard_solution": ""})
        self.assertEqual(rejected["standard_solution"], [])

    def test_gold_human_review_preserves_raw_and_applies_explicit_edit(self):
        record = {
            "native_qa_id": "x",
            "choices": {},
            "answer": "reference",
            "gold_enrichment": {
                "id": "x", "accepted": True,
                "gold_answer": {"correct_choice": None, "answer_text": "original"},
                "standard_solution": ["Original solution."],
                "scoring_rubric": [{"criterion": "Original criterion", "points": 10}],
                "issues": [],
            },
        }
        policy = {
            "version": "test-v1",
            "default_decision": "accept",
            "default_reason": "reviewed",
            "overrides": {
                "x": {
                    "decision": "accept_with_edit",
                    "reason": "science correction",
                    "gold_enrichment": {
                        "gold_answer": {"correct_choice": None, "answer_text": "curated"}
                    },
                }
            },
        }
        review, curated = gold_finalizer.apply_review(record, policy)
        self.assertEqual(review["decision"], "accept_with_edit")
        self.assertEqual(curated["gold_enrichment"]["gold_answer"]["answer_text"], "curated")
        self.assertEqual(record["gold_enrichment"]["gold_answer"]["answer_text"], "original")
        self.assertTrue(curated["gold_source_trace_preserved"])

    def test_native_baseline_view_is_lossless_and_evaluator_only(self):
        source = {
            "native_qa_id": "native-x",
            "source_file": "biology/test.jsonl",
            "source_id": "index:1",
            "source_index": 1,
            "domain_type": "native_cross_domain",
            "question_type": "qualitative",
            "question": "Why?",
            "choices": {},
            "required_domains": ["biology", "chemistry"],
            "gold_prompt_version": "gold-v1",
            "gold_human_review": {
                "decision": "accept",
                "review_policy_version": "review-v1",
            },
            "gold_enrichment": {
                "accepted": True,
                "gold_answer": {"correct_choice": None, "answer_text": "Because."},
                "standard_solution": ["Explain the mechanism."],
                "scoring_rubric": [{"criterion": "Mechanism", "points": 10}],
            },
        }
        view = baseline_view.convert(source)
        self.assertEqual(view["id"], source["native_qa_id"])
        self.assertEqual(view["question"], source["question"])
        self.assertIsNone(view["choices"])
        self.assertEqual(view["required_domains"], source["required_domains"])
        self.assertEqual(view["gold"]["standard_solution"], source["gold_enrichment"]["standard_solution"])
        self.assertTrue(view["verification"]["evaluation_only"])


if __name__ == "__main__":
    unittest.main()
