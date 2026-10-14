import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/score_native_eval50_qualitative.py"
SPEC = importlib.util.spec_from_file_location("score_native_eval50_qualitative", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class QualitativeCoverageTest(unittest.TestCase):
    def test_audited_rubric_matches_frozen_qualitative_ids(self) -> None:
        dataset_path = (
            ROOT
            / "datasets/processed/native_qa_mining_v1/pilot_500/eval_split_v1/eval_50_gold_curated.jsonl"
        )
        dataset = MODULE.read_jsonl(dataset_path)
        qualitative_ids = {
            row["native_qa_id"] for row in dataset if row["question_type"] == "qualitative"
        }
        rubric = json.loads(
            (ROOT / "configs/evaluation/native_eval50_qualitative_rubric_v2.json").read_text(
                encoding="utf-8"
            )
        )
        MODULE.validate_rubric(rubric, qualitative_ids)
        self.assertEqual(len(qualitative_ids), 32)
        self.assertTrue(all(3 <= len(item["criteria"]) <= 5 for item in rubric["items"]))

    def test_anchor_groups_require_all_groups_by_default(self) -> None:
        criterion = {
            "anchor_groups": [["cosmic ray"], ["natural laboratory", "beyond.*accelerator"]]
        }
        partial = MODULE.match_criterion("cosmic rays are energetic", criterion)
        complete = MODULE.match_criterion(
            "Cosmic rays are a natural laboratory beyond an accelerator.", criterion
        )
        self.assertFalse(partial["covered"])
        self.assertTrue(partial["review_required"])
        self.assertTrue(complete["covered"])

    def test_adjudications_are_unique_and_well_formed(self) -> None:
        path = ROOT / "configs/evaluation/native_eval50_qualitative_adjudications_v2.json"
        overrides, errors = MODULE.load_adjudications(path)
        self.assertEqual(len(overrides), len(set(overrides)))
        self.assertTrue(all(row["arm"] in MODULE.ARMS for row in errors))
        self.assertTrue(all(row["reason"] for row in errors))


if __name__ == "__main__":
    unittest.main()
