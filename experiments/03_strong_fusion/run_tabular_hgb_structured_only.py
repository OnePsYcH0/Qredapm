from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "faithful_redapm_cloud" / "妯″瀷浠ｇ爜" / "data_0"
TRAIN_PATH = DATA_DIR / "train_0.json"
TEST_PATH = DATA_DIR / "test_0.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tabular_baseline_outputs" / "hgb_structured_only"
LABEL_COLUMN = "y2"

EXCLUDE_COLUMNS = {
    "jmkh",
    "visit_sn",
    "disease_names",
    "drug_names",
    "drug_code",
    "sex",
    "y1",
    "y2",
}


def read_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Data file does not exist: {path}")
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [row for row in parsed if isinstance(row, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    rows: List[Dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} line {line_no} is not valid JSON.") from exc
        if isinstance(parsed, dict):
            rows.append(parsed)
        elif isinstance(parsed, list):
            rows.extend(row for row in parsed if isinstance(row, dict))
        else:
            raise ValueError(f"{path} line {line_no} is neither a JSON object nor list.")
    return rows


def to_numeric_or_nan(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else float("nan")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return float("nan")
        try:
            number = float(stripped)
            return number if math.isfinite(number) else float("nan")
        except ValueError:
            return float("nan")
    return float("nan")


def infer_structured_feature_columns(train_rows: Sequence[Dict[str, Any]], test_rows: Sequence[Dict[str, Any]]) -> List[str]:
    all_fields = sorted({key for row in list(train_rows) + list(test_rows) for key in row})
    feature_columns: List[str] = []
    combined = list(train_rows) + list(test_rows)
    for column in all_fields:
        if column in EXCLUDE_COLUMNS:
            continue
        non_missing_values = [
            row.get(column)
            for row in combined
            if row.get(column) is not None and row.get(column) != "" and row.get(column) != []
        ]
        if not non_missing_values:
            continue
        if all(not isinstance(value, (list, dict)) and not math.isnan(to_numeric_or_nan(value)) for value in non_missing_values):
            feature_columns.append(column)
    return feature_columns


def labels_from_rows(rows: Sequence[Dict[str, Any]], split_name: str) -> np.ndarray:
    labels: List[int] = []
    invalid_indices: List[int] = []
    for index, row in enumerate(rows):
        value = to_numeric_or_nan(row.get(LABEL_COLUMN))
        if value not in (0.0, 1.0):
            invalid_indices.append(index)
        else:
            labels.append(int(value))
    if invalid_indices:
        examples = ", ".join(str(index) for index in invalid_indices[:10])
        raise ValueError(f"{split_name} has missing or non-binary y2 labels at indices: {examples}")
    return np.asarray(labels, dtype=np.int64)


def dataframe_from_rows(rows: Sequence[Dict[str, Any]], feature_columns: Sequence[str]) -> pd.DataFrame:
    records = []
    for row in rows:
        records.append({column: to_numeric_or_nan(row.get(column)) for column in feature_columns})
    return pd.DataFrame.from_records(records, columns=feature_columns)


def positive_ratio(labels: np.ndarray) -> float:
    return float(labels.mean()) if labels.size else float("nan")


def get_positive_scores(model: HistGradientBoostingClassifier, x_test: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return probabilities[:, 1]
        return probabilities.ravel()
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_test)
        return 1.0 / (1.0 + np.exp(-scores))
    raise RuntimeError("Model exposes neither predict_proba nor decision_function.")


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_predictions(path: Path, test_rows: Sequence[Dict[str, Any]], y_true: np.ndarray, y_prob: np.ndarray) -> None:
    y_pred = (y_prob >= 0.5).astype(np.int64)
    include_jmkh = any("jmkh" in row for row in test_rows)
    fieldnames = ["index"]
    if include_jmkh:
        fieldnames.append("jmkh")
    fieldnames.extend(["y_true", "y_prob", "y_pred"])
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index, (row, label, probability, prediction) in enumerate(zip(test_rows, y_true, y_prob, y_pred)):
            output_row = {
                "index": index,
                "y_true": int(label),
                "y_prob": f"{float(probability):.10f}",
                "y_pred": int(prediction),
            }
            if include_jmkh:
                output_row["jmkh"] = row.get("jmkh", "")
            writer.writerow(output_row)


def write_summary_csv(path: Path, metrics_payload: Dict[str, Any]) -> None:
    fieldnames = [
        "timestamp",
        "model_name",
        "input_type",
        "train_count",
        "test_count",
        "positive_ratio_train",
        "positive_ratio_test",
        "feature_count",
        "roc_auc",
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({field: metrics_payload.get(field) for field in fieldnames})


def append_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def log_and_print(path: Path, message: str) -> None:
    print(message, flush=True)
    append_log(path, message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a structured-only HistGradientBoosting baseline for REDAPM data_0.")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.txt"
    if log_path.exists():
        log_path.unlink()

    try:
        log_and_print(log_path, f"data_dir={DATA_DIR}")
        train_rows = read_rows(TRAIN_PATH)
        test_rows = read_rows(TEST_PATH)
        if not train_rows or not test_rows:
            raise ValueError("Train/test data must both be non-empty.")

        y_train = labels_from_rows(train_rows, "train")
        y_test = labels_from_rows(test_rows, "test")
        ratio_train = positive_ratio(y_train)
        ratio_test = positive_ratio(y_test)

        feature_columns = infer_structured_feature_columns(train_rows, test_rows)
        if not feature_columns:
            raise ValueError("No structured numeric scalar features were detected.")

        log_and_print(log_path, f"train_count={len(train_rows)}")
        log_and_print(log_path, f"test_count={len(test_rows)}")
        log_and_print(log_path, f"positive_ratio_train={ratio_train}")
        log_and_print(log_path, f"positive_ratio_test={ratio_test}")
        log_and_print(log_path, f"feature_count={len(feature_columns)}")
        log_and_print(log_path, "feature_columns=" + ", ".join(feature_columns))

        x_train_df = dataframe_from_rows(train_rows, feature_columns)
        x_test_df = dataframe_from_rows(test_rows, feature_columns)
        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(x_train_df)
        x_test = imputer.transform(x_test_df)

        model_params = {
            "loss": "log_loss",
            "learning_rate": args.learning_rate,
            "max_iter": args.max_iter,
            "max_leaf_nodes": args.max_leaf_nodes,
            "l2_regularization": args.l2_regularization,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 20,
            "random_state": args.random_state,
        }
        log_and_print(log_path, "model_params=" + json.dumps(model_params, ensure_ascii=False))

        model = HistGradientBoostingClassifier(**model_params)
        model.fit(x_train, y_train)
        y_prob = get_positive_scores(model, x_test)
        metrics = compute_metrics(y_test, y_prob, threshold=0.5)

        timestamp = datetime.now().astimezone().isoformat()
        metrics_payload = {
            "timestamp": timestamp,
            "model_name": "HistGradientBoostingClassifier",
            "input_type": "structured_only",
            "train_count": len(train_rows),
            "test_count": len(test_rows),
            "positive_ratio_train": ratio_train,
            "positive_ratio_test": ratio_test,
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tp": metrics["tp"],
            "threshold": metrics["threshold"],
            "model_params": model_params,
            "n_iter_": int(getattr(model, "n_iter_", -1)),
            "data_paths": {
                "train": str(TRAIN_PATH),
                "test": str(TEST_PATH),
            },
            "imputer": {
                "strategy": "median",
                "statistics": {column: float(value) for column, value in zip(feature_columns, imputer.statistics_)},
            },
        }

        write_json(output_dir / "metrics.json", metrics_payload)
        write_predictions(output_dir / "predictions.csv", test_rows, y_test, y_prob)
        write_json(output_dir / "feature_columns.json", feature_columns)
        write_summary_csv(output_dir / "experiment_summary.csv", metrics_payload)

        log_and_print(log_path, f"roc_auc={metrics['roc_auc']}")
        log_and_print(log_path, f"pr_auc={metrics['pr_auc']}")
        log_and_print(log_path, f"accuracy={metrics['accuracy']}")
        log_and_print(log_path, f"precision={metrics['precision']}")
        log_and_print(log_path, f"recall={metrics['recall']}")
        log_and_print(log_path, f"f1={metrics['f1']}")
        log_and_print(log_path, f"confusion_matrix tn={metrics['tn']} fp={metrics['fp']} fn={metrics['fn']} tp={metrics['tp']}")
        log_and_print(log_path, f"output_dir={output_dir.resolve()}")

        print("\nFinal metrics")
        print(f"train samples: {len(train_rows)}")
        print(f"test samples: {len(test_rows)}")
        print(f"positive ratio train: {ratio_train}")
        print(f"positive ratio test: {ratio_test}")
        print(f"structured feature count: {len(feature_columns)}")
        print(f"ROC-AUC: {metrics['roc_auc']}")
        print(f"PR-AUC: {metrics['pr_auc']}")
        print(f"Accuracy: {metrics['accuracy']}")
        print(f"Precision: {metrics['precision']}")
        print(f"Recall: {metrics['recall']}")
        print(f"F1: {metrics['f1']}")
        print(f"output directory: {output_dir.resolve()}")
    except Exception as exc:
        message = f"ERROR: {exc}"
        print(message, file=sys.stderr, flush=True)
        try:
            append_log(log_path, message)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()

