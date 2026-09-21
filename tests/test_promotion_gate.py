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
    schema_path = Path("schema/inference_schema.v1.json")

    def test_current_model_passes_policy(self):
        report = evaluate_promotion(
            self.model_path,
            self.test_data_path,
            self.criteria_path,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["failures"], [])
        self.assertEqual(report["schema_version"], "1.0.0")
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

    def test_model_fails_schema_version_mismatch(self):
        schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
        schema["x-schema-version"] = "2.0.0"

        with tempfile.TemporaryDirectory() as directory:
            schema_path = Path(directory) / "inference_schema.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            report = evaluate_promotion(
                self.model_path,
                self.test_data_path,
                self.criteria_path,
                schema_path,
            )

        self.assertFalse(report["passed"])
        self.assertIn(
            "Inference schema version does not match the promotion policy.",
            report["failures"],
        )

    def test_model_fails_value_outside_inference_schema(self):
        test_data = pd.read_csv(self.test_data_path)
        test_data.loc[0, "Age"] = 17

        with tempfile.TemporaryDirectory() as directory:
            test_data_path = Path(directory) / "test.csv"
            test_data.to_csv(test_data_path, index=False)
            report = evaluate_promotion(
                self.model_path,
                test_data_path,
                self.criteria_path,
            )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any("field 'Age' violates inference schema" in item for item in report["failures"])
        )


if __name__ == "__main__":
    unittest.main()