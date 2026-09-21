import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import skops.io as sio
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


TRUSTED_MODEL_TYPES = ["numpy.dtype", "sklearn.tree._tree.Tree"]


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def validate_contract(
    model: Any,
    test_data: pd.DataFrame,
    criteria: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    target_column = criteria["target_column"]
    required_features = criteria["required_features"]

    if target_column not in test_data.columns:
        return [f"Missing target column: {target_column}"]

    actual_features = test_data.drop(columns=[target_column]).columns.tolist()
    if actual_features != required_features:
        failures.append(
            "Test-data features do not match the required ordered schema."
        )

    model_features = getattr(model, "feature_names_in_", None)
    if model_features is None or list(model_features) != required_features:
        failures.append("Model features do not match the required ordered schema.")

    expected_classes = {0, criteria["positive_class"]}
    actual_classes = set(test_data[target_column].dropna().unique().tolist())
    if actual_classes != expected_classes:
        failures.append(
            f"Target classes must be {sorted(expected_classes)}, got {sorted(actual_classes)}."
        )

    return failures


def calculate_metrics(
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
    positive_class: int,
) -> dict[str, float]:
    predictions = model.predict(features)
    positive_index = list(model.classes_).index(positive_class)
    probabilities = model.predict_proba(features)[:, positive_index]

    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(
            precision_score(target, predictions, pos_label=positive_class, zero_division=0)
        ),
        "recall": float(
            recall_score(target, predictions, pos_label=positive_class, zero_division=0)
        ),
        "f1": float(
            f1_score(target, predictions, pos_label=positive_class, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(target, probabilities)),
    }


def evaluate_promotion(
    model_path: Path,
    test_data_path: Path,
    criteria_path: Path,
) -> dict[str, Any]:
    criteria = load_json(criteria_path)
    unknown_types = sio.get_untrusted_types(file=model_path)
    unexpected_types = sorted(set(unknown_types) - set(TRUSTED_MODEL_TYPES))
    failures = (
        [f"Model contains unexpected serialized types: {unexpected_types}"]
        if unexpected_types
        else []
    )

    model = sio.load(model_path, trusted=TRUSTED_MODEL_TYPES)
    test_data = pd.read_csv(test_data_path)
    failures.extend(validate_contract(model, test_data, criteria))

    metrics: dict[str, float] = {}
    checks: dict[str, dict[str, Any]] = {}
    if not failures:
        target_column = criteria["target_column"]
        metrics = calculate_metrics(
            model,
            test_data.drop(columns=[target_column]),
            test_data[target_column],
            criteria["positive_class"],
        )

        for metric_name, minimum in criteria["minimum_metrics"].items():
            actual = metrics[metric_name]
            passed = actual >= minimum
            checks[f"minimum_{metric_name}"] = {
                "actual": actual,
                "required": minimum,
                "passed": passed,
            }
            if not passed:
                failures.append(
                    f"{metric_name}={actual:.4f} is below minimum {minimum:.4f}."
                )

        for metric_name, maximum_drop in criteria["maximum_regression"].items():
            incumbent = criteria["incumbent_metrics"][metric_name]
            floor = incumbent - maximum_drop
            actual = metrics[metric_name]
            passed = actual >= floor
            checks[f"regression_{metric_name}"] = {
                "actual": actual,
                "incumbent": incumbent,
                "maximum_drop": maximum_drop,
                "required": floor,
                "passed": passed,
            }
            if not passed:
                failures.append(
                    f"{metric_name}={actual:.4f} exceeds the allowed regression "
                    f"from incumbent {incumbent:.4f}."
                )

    return {
        "policy_version": criteria["policy_version"],
        "passed": not failures,
        "metrics": metrics,
        "checks": checks,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether a candidate model satisfies promotion policy."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("model/wellness_tourism_model.skops"),
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=Path("data/test.csv"),
    )
    parser.add_argument(
        "--criteria",
        type=Path,
        default=Path("config/promotion_criteria.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("promotion_report.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate_promotion(args.model, args.test_data, args.criteria)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())