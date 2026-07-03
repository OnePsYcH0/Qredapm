from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "faithful_redapm_cloud" / "妯″瀷浠ｇ爜" / "data_0"
TRAIN_PATH = DATA_DIR / "train_0.json"
TEST_PATH = DATA_DIR / "test_0.json"
OUTPUT_DIR = PROJECT_ROOT / "tabular_baseline_outputs" / "structured_drug"
STRUCTURED_ONLY_METRICS = PROJECT_ROOT / "tabular_baseline_outputs" / "hgb_structured_only" / "metrics.json"
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


def infer_structured_feature_columns(train_rows: Sequence[Dict[str, Any]], test_rows: Sequence[Dict[str, Any]]) -> List[str]:
    combined = list(train_rows) + list(test_rows)
    all_fields = sorted({key for row in combined for key in row})
    feature_columns: List[str] = []
    for column in all_fields:
        if column in EXCLUDE_COLUMNS:
            continue
        values = [
            row.get(column)
            for row in combined
            if row.get(column) is not None and row.get(column) != "" and row.get(column) != []
        ]
        if not values:
            continue
        if all(not isinstance(value, (list, dict)) and not math.isnan(to_numeric_or_nan(value)) for value in values):
            feature_columns.append(column)
    return feature_columns


def structured_dataframe(rows: Sequence[Dict[str, Any]], feature_columns: Sequence[str]) -> pd.DataFrame:
    records = []
    for row in rows:
        records.append({column: to_numeric_or_nan(row.get(column)) for column in feature_columns})
    return pd.DataFrame.from_records(records, columns=feature_columns)


def raw_drug_vector(value: Any) -> List[float]:
    if not isinstance(value, list):
        return []
    vector: List[float] = []
    for item in value:
        number = to_numeric_or_nan(item)
        vector.append(0.0 if math.isnan(number) else number)
    return vector


def infer_drug_dimension(rows: Sequence[Dict[str, Any]]) -> tuple[int, Dict[str, Any]]:
    lengths = [len(raw_drug_vector(row.get("drug_code"))) for row in rows]
    length_counts = Counter(lengths)
    nonzero_lengths = {length: count for length, count in length_counts.items() if length > 0}
    if not nonzero_lengths:
        raise ValueError("No non-empty drug_code vectors were found.")
    dimension = max(nonzero_lengths.items(), key=lambda item: (item[1], item[0]))[0]
    warning = {
        "drug_code_length_distribution": dict(sorted(length_counts.items())),
        "selected_drug_code_dimension": dimension,
        "length_inconsistent": len(nonzero_lengths) > 1 or length_counts.get(0, 0) > 0,
    }
    return dimension, warning


def drug_matrix(rows: Sequence[Dict[str, Any]], dimension: int) -> np.ndarray:
    matrix = np.zeros((len(rows), dimension), dtype=np.float32)
    for row_index, row in enumerate(rows):
        vector = raw_drug_vector(row.get("drug_code"))
        if not vector:
            continue
        clipped = vector[:dimension]
        matrix[row_index, : len(clipped)] = np.asarray(clipped, dtype=np.float32)
    return matrix


def positive_ratio(labels: np.ndarray) -> float:
    return float(labels.mean()) if labels.size else float("nan")


def get_positive_scores(model: Any, x_test: np.ndarray) -> np.ndarray:
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
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "jmkh", "y_true", "y_prob", "y_pred"])
        writer.writeheader()
        for index, (row, label, probability, prediction) in enumerate(zip(test_rows, y_true, y_prob, y_pred)):
            writer.writerow(
                {
                    "index": index,
                    "jmkh": row.get("jmkh", ""),
                    "y_true": int(label),
                    "y_prob": f"{float(probability):.10f}",
                    "y_pred": int(prediction),
                }
            )


def append_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def log_and_print(path: Path, message: str) -> None:
    print(message, flush=True)
    append_log(path, message)


def train_and_evaluate_model(
    model_key: str,
    model_name: str,
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_rows: Sequence[Dict[str, Any]],
    feature_columns: Sequence[str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    model_dir = OUTPUT_DIR / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_dir / "train_log.txt"
    if log_path.exists():
        log_path.unlink()

    log_and_print(log_path, f"model_name={model_name}")
    log_and_print(log_path, f"input_type={metadata['input_type']}")
    log_and_print(log_path, f"train_count={metadata['train_count']}")
    log_and_print(log_path, f"test_count={metadata['test_count']}")
    log_and_print(log_path, f"feature_count={len(feature_columns)}")
    log_and_print(log_path, "Fitting model")
    model.fit(x_train, y_train)
    y_prob = get_positive_scores(model, x_test)
    metrics = compute_metrics(y_test, y_prob, threshold=0.5)

    if isinstance(model, Pipeline):
        params = model.named_steps["model"].get_params()
    else:
        params = model.get_params()
    metrics_payload = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "model_name": model_name,
        "input_type": metadata["input_type"],
        "train_count": metadata["train_count"],
        "test_count": metadata["test_count"],
        "positive_ratio_train": metadata["positive_ratio_train"],
        "positive_ratio_test": metadata["positive_ratio_test"],
        "structured_feature_count": metadata["structured_feature_count"],
        "drug_code_dimension": metadata["drug_code_dimension"],
        "total_feature_count": metadata["total_feature_count"],
        "feature_columns": list(feature_columns),
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
        "model_params": params,
        "warnings": metadata["warnings"],
    }
    write_json(model_dir / "metrics.json", metrics_payload)
    write_predictions(model_dir / "predictions.csv", test_rows, y_test, y_prob)
    write_json(model_dir / "feature_columns.json", list(feature_columns))
    log_and_print(log_path, f"roc_auc={metrics['roc_auc']}")
    log_and_print(log_path, f"pr_auc={metrics['pr_auc']}")
    log_and_print(log_path, f"accuracy={metrics['accuracy']}")
    log_and_print(log_path, f"precision={metrics['precision']}")
    log_and_print(log_path, f"recall={metrics['recall']}")
    log_and_print(log_path, f"f1={metrics['f1']}")
    log_and_print(log_path, f"confusion_matrix tn={metrics['tn']} fp={metrics['fp']} fn={metrics['fn']} tp={metrics['tp']}")
    return metrics_payload


def load_structured_only_metrics() -> Dict[str, Any] | None:
    if not STRUCTURED_ONLY_METRICS.exists():
        return None
    with STRUCTURED_ONLY_METRICS.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_experiment_summary(rows: Sequence[Dict[str, Any]], comparison: Dict[str, Any] | None) -> None:
    fieldnames = [
        "model_key",
        "model_name",
        "input_type",
        "train_count",
        "test_count",
        "positive_ratio_train",
        "positive_ratio_test",
        "structured_feature_count",
        "drug_code_dimension",
        "total_feature_count",
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
        "structured_only_roc_auc",
        "structured_only_pr_auc",
        "structured_only_f1",
        "delta_roc_auc",
        "delta_pr_auc",
        "delta_f1",
    ]
    with (OUTPUT_DIR / "experiment_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output_row = {field: row.get(field) for field in fieldnames}
            output_row["model_key"] = row["model_key"]
            if comparison and row["model_key"] == "hgb":
                output_row.update(comparison)
            writer.writerow(output_row)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        train_rows = read_rows(TRAIN_PATH)
        test_rows = read_rows(TEST_PATH)
        if not train_rows or not test_rows:
            raise ValueError("Train/test data must both be non-empty.")

        y_train = labels_from_rows(train_rows, "train")
        y_test = labels_from_rows(test_rows, "test")
        ratio_train = positive_ratio(y_train)
        ratio_test = positive_ratio(y_test)

        structured_columns = infer_structured_feature_columns(train_rows, test_rows)
        structured_train_df = structured_dataframe(train_rows, structured_columns)
        structured_test_df = structured_dataframe(test_rows, structured_columns)
        imputer = SimpleImputer(strategy="median")
        structured_train = imputer.fit_transform(structured_train_df)
        structured_test = imputer.transform(structured_test_df)

        drug_dimension, drug_warning = infer_drug_dimension(list(train_rows) + list(test_rows))
        drug_train = drug_matrix(train_rows, drug_dimension)
        drug_test = drug_matrix(test_rows, drug_dimension)
        drug_columns = [f"drug_code_{index}" for index in range(drug_dimension)]
        feature_columns = list(structured_columns) + drug_columns
        x_train = np.hstack([structured_train, drug_train])
        x_test = np.hstack([structured_test, drug_test])

        metadata = {
            "input_type": f"structured{len(structured_columns)}+drug_code{drug_dimension}",
            "train_count": len(train_rows),
            "test_count": len(test_rows),
            "positive_ratio_train": ratio_train,
            "positive_ratio_test": ratio_test,
            "structured_feature_count": len(structured_columns),
            "drug_code_dimension": drug_dimension,
            "total_feature_count": len(feature_columns),
            "warnings": [drug_warning] if drug_warning["length_inconsistent"] else [],
        }

        print(f"train samples: {len(train_rows)}")
        print(f"test samples: {len(test_rows)}")
        print(f"positive ratio train: {ratio_train}")
        print(f"positive ratio test: {ratio_test}")
        print(f"structured feature count: {len(structured_columns)}")
        print(f"drug_code dimension: {drug_dimension}")
        print(f"total feature count: {len(feature_columns)}")
        if metadata["warnings"]:
            print("WARNING: drug_code length inconsistency detected; vectors were padded/truncated.")

        logistic = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        penalty="l2",
                        solver="lbfgs",
                        class_weight=None,
                        random_state=42,
                    ),
                ),
            ]
        )
        hgb = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_iter=500,
            max_leaf_nodes=31,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
        )

        logistic_metrics = train_and_evaluate_model(
            "logistic_regression",
            "LogisticRegression",
            logistic,
            x_train,
            y_train,
            x_test,
            y_test,
            test_rows,
            feature_columns,
            metadata,
        )
        hgb_metrics = train_and_evaluate_model(
            "hgb",
            "HistGradientBoostingClassifier",
            hgb,
            x_train,
            y_train,
            x_test,
            y_test,
            test_rows,
            feature_columns,
            metadata,
        )

        comparison = None
        structured_only = load_structured_only_metrics()
        if structured_only is not None:
            comparison = {
                "structured_only_roc_auc": structured_only.get("roc_auc"),
                "structured_only_pr_auc": structured_only.get("pr_auc"),
                "structured_only_f1": structured_only.get("f1"),
                "delta_roc_auc": hgb_metrics["roc_auc"] - structured_only.get("roc_auc"),
                "delta_pr_auc": hgb_metrics["pr_auc"] - structured_only.get("pr_auc"),
                "delta_f1": hgb_metrics["f1"] - structured_only.get("f1"),
            }
            print("Structured-only HGB comparison:")
            print(f"  structured-only ROC-AUC: {comparison['structured_only_roc_auc']}")
            print(f"  structured-only PR-AUC: {comparison['structured_only_pr_auc']}")
            print(f"  structured-only F1: {comparison['structured_only_f1']}")
            print(f"  structured+drug ROC-AUC: {hgb_metrics['roc_auc']}")
            print(f"  structured+drug PR-AUC: {hgb_metrics['pr_auc']}")
            print(f"  structured+drug F1: {hgb_metrics['f1']}")
            print(f"  delta ROC-AUC: {comparison['delta_roc_auc']}")
            print(f"  delta PR-AUC: {comparison['delta_pr_auc']}")
            print(f"  delta F1: {comparison['delta_f1']}")

        summary_rows = [
            {"model_key": "logistic_regression", **logistic_metrics},
            {"model_key": "hgb", **hgb_metrics},
        ]
        write_experiment_summary(summary_rows, comparison)

        print("Logistic Regression metrics:")
        print(json.dumps({key: logistic_metrics[key] for key in ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"]}, indent=2))
        print("HGB metrics:")
        print(json.dumps({key: hgb_metrics[key] for key in ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"]}, indent=2))
        print(f"output directory: {OUTPUT_DIR.resolve()}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()

