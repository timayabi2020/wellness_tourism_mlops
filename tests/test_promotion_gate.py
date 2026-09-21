import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.check_promotion import evaluate_promotion


class PromotionGateTests(unittest.TestCase):
    model_path = Path("model/wellness_tourism_model.skops")
    test_data_path = Path("data/test.csv")
    criteria_path = Path("config/promotion_criteria.json")

    def test_current_model_passes_policy(self):
        report = evaluate_promotion(
            self.model_path,
            self.test_data_path,
            self.criteria_path,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["failures"], [])
        self.assertAlmostEqual(report["metrics"]["f1"], 0.7841726619)

    def test_model_fails_impossible_metric_threshold(self):
        criteria = json.loads(self.criteria_path.read_text(encoding="utf-8"))
        criteria["minimum_metrics"]["f1"] = 0.99

        with tempfile.TemporaryDirectory() as directory:
            criteria_path = Path(directory) / "criteria.json"
            criteria_path.write_text(json.dumps(criteria), encoding="utf-8")
            report = evaluate_promotion(
                self.model_path,
                self.test_data_path,
                criteria_path,
            )

        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["minimum_f1"]["passed"])
        self.assertTrue(any("below minimum" in item for item in report["failures"]))

    def test_model_fails_mismatched_feature_schema(self):
        test_data = pd.read_csv(self.test_data_path).drop(columns=["Age"])

        with tempfile.TemporaryDirectory() as directory:
            test_data_path = Path(directory) / "test.csv"
            test_data.to_csv(test_data_path, index=False)
            report = evaluate_promotion(
                self.model_path,
                test_data_path,
                self.criteria_path,
            )

        self.assertFalse(report["passed"])
        self.assertIn(
            "Test-data features do not match the required ordered schema.",
            report["failures"],
        )


if __name__ == "__main__":
    unittest.main()