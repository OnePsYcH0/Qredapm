from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
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
from sklearn.model_selection import RandomizedSearchCV, train_test_split


OUTPUT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = OUTPUT_DIR.parent
SN1_PACKAGE_DIR = PROJECT_ROOT / "Q-redapm" / "sn1_text_4090_package_20260617_005006"
DATA_DIR = SN1_PACKAGE_DIR / "data_0"
SN1_OUTPUT_DIR = SN1_PACKAGE_DIR / "output"
TRAIN_PATH = DATA_DIR / "train_0.json"
TEST_PATH = DATA_DIR / "test_0.json"
SN1_METRICS_PATH = SN1_OUTPUT_DIR / "metrics.json"
SN1_VAL_PRED_PATH = SN1_OUTPUT_DIR / "validation_predictions.csv"
SN1_TEST_PRED_PATH = SN1_OUTPUT_DIR / "predictions.csv"
LOG_PATH = OUTPUT_DIR / "train_log.txt"
LABEL_COLUMN = "y2"
RANDOM_STATE = 42
PREVIOUS_HGB_STRUCTURED_DRUG = {"roc_auc": 0.8229, "pr_auc": 0.6667, "f1": 0.5248}
PREVIOUS_LOGISTIC_FUSION = {"roc_auc": 0.8421, "pr_auc": 0.6932, "f1": 0.5809}

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

HGB_PARAM_DISTRIBUTIONS = {
    "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
    "max_iter": [200, 400, 600, 800],
    "max_leaf_nodes": [15, 31, 63],
    "max_depth": [3, 5, 7, None],
    "l2_regularization": [0, 0.01, 0.1, 1],
    "min_samples_leaf": [10, 20, 50],
}


def log(message: str) -> None:
    print(message, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_rows(path: Path) -> List[Dict[str, Any]]:
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
        if not isinstance(parsed, dict):
            raise ValueError(f"{path} line {line_no} is not a JSON object.")
        rows.append(parsed)
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
    bad: List[int] = []
    for index, row in enumerate(rows):
        value = to_numeric_or_nan(row.get(LABEL_COLUMN))
        if value not in (0.0, 1.0):
            bad.append(index)
        else:
            labels.append(int(value))
    if bad:
        raise ValueError(f"{split_name} has non-binary y2 at indices: {bad[:10]}")
    return np.asarray(labels, dtype=np.int64)


def required_files() -> Dict[str, Path]:
    return {
        "train_0.json": TRAIN_PATH,
        "test_0.json": TEST_PATH,
        "SN1 metrics.json": SN1_METRICS_PATH,
        "SN1 validation_predictions.csv": SN1_VAL_PRED_PATH,
        "SN1 predictions.csv": SN1_TEST_PRED_PATH,
    }


def ensure_required_files() -> None:
    missing = {name: path for name, path in required_files().items() if not path.exists()}
    if missing:
        checked = "\n".join(f"- {name}: {path}" for name, path in required_files().items())
        missing_text = "\n".join(f"- {name}: {path}" for name, path in missing.items())
        raise FileNotFoundError(f"Missing required files.\nChecked:\n{checked}\nMissing:\n{missing_text}")


def infer_structured_feature_columns(train_rows: Sequence[Dict[str, Any]], test_rows: Sequence[Dict[str, Any]]) -> List[str]:
    combined = list(train_rows) + list(test_rows)
    columns: List[str] = []
    for column in sorted({key for row in combined for key in row}):
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
            columns.append(column)
    return columns


def structured_frame(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [{column: to_numeric_or_nan(row.get(column)) for column in columns} for row in rows],
        columns=columns,
    )


def raw_drug_vector(value: Any) -> List[float]:
    if not isinstance(value, list):
        return []
    vector = []
    for item in value:
        number = to_numeric_or_nan(item)
        vector.append(0.0 if math.isnan(number) else number)
    return vector


def infer_drug_dimension(rows: Sequence[Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
    lengths = [len(raw_drug_vector(row.get("drug_code"))) for row in rows]
    counts = Counter(lengths)
    nonzero_counts = {length: count for length, count in counts.items() if length > 0}
    if not nonzero_counts:
        raise ValueError("No non-empty drug_code vectors were found.")
    dimension = max(nonzero_counts.items(), key=lambda item: (item[1], item[0]))[0]
    return dimension, {
        "drug_code_length_distribution": dict(sorted(counts.items())),
        "selected_drug_code_dimension": dimension,
        "length_inconsistent": len(nonzero_counts) > 1 or counts.get(0, 0) > 0,
    }


def drug_matrix(rows: Sequence[Dict[str, Any]], dimension: int) -> np.ndarray:
    matrix = np.zeros((len(rows), dimension), dtype=np.float32)
    for row_index, row in enumerate(rows):
        vector = raw_drug_vector(row.get("drug_code"))[:dimension]
        if vector:
            matrix[row_index, : len(vector)] = np.asarray(vector, dtype=np.float32)
    return matrix


def read_prediction_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"index", "y_true", "y_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    return df


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def positive_scores(model: Any, x: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(x)
    if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
        return probabilities[:, 1]
    return probabilities.ravel()


def write_predictions(
    path: Path,
    indices: Iterable[int],
    rows: Sequence[Dict[str, Any]],
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    y_pred = (y_prob >= 0.5).astype(np.int64)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "jmkh", "y_true", "y_prob", "y_pred"])
        writer.writeheader()
        for out_index, source_index, label, probability, prediction in zip(indices, indices, y_true, y_prob, y_pred):
            row = rows[int(source_index)]
            writer.writerow(
                {
                    "index": int(out_index),
                    "jmkh": row.get("jmkh", ""),
                    "y_true": int(label),
                    "y_prob": f"{float(probability):.10f}",
                    "y_pred": int(prediction),
                }
            )


def train_hgb_search(
    name: str,
    x_train_base: np.ndarray,
    y_train_base: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    val_indices: np.ndarray,
    train_rows: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    log(f"[{name}] RandomizedSearchCV start")
    base = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=HGB_PARAM_DISTRIBUTIONS,
        n_iter=40,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=1,
        return_train_score=True,
    )
    search.fit(x_train_base, y_train_base)
    cv_path = OUTPUT_DIR / f"{name}_cv_results.csv"
    pd.DataFrame(search.cv_results_).to_csv(cv_path, index=False, encoding="utf-8-sig")
    model = search.best_estimator_
    p_val = positive_scores(model, x_val)
    p_test = positive_scores(model, x_test)
    val_metrics = compute_metrics(y_val, p_val)
    test_metrics = compute_metrics(y_test, p_test)
    write_predictions(OUTPUT_DIR / f"{name}_validation_predictions.csv", val_indices, train_rows, y_val, p_val)
    write_predictions(OUTPUT_DIR / f"{name}_test_predictions.csv", np.arange(len(test_rows)), test_rows, y_test, p_test)
    payload = {
        "model_name": name,
        "input_feature_count": int(x_train_base.shape[1]),
        "feature_names": list(feature_names),
        "best_params": search.best_params_,
        "best_cv_roc_auc": float(search.best_score_),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "cv_results_csv": str(cv_path),
    }
    log(f"[{name}] best_cv_roc_auc={search.best_score_:.6f}")
    log(f"[{name}] test roc_auc={test_metrics['roc_auc']:.6f} pr_auc={test_metrics['pr_auc']:.6f} f1={test_metrics['f1']:.6f}")
    return {"payload": payload, "p_val": p_val, "p_test": p_test}


def fit_logistic_fusion(
    p_text_val: np.ndarray,
    p_tabular_val: np.ndarray,
    y_val: np.ndarray,
    p_text_test: np.ndarray,
    p_tabular_test: np.ndarray,
    y_test: np.ndarray,
    val_indices: np.ndarray,
    train_rows: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    log("[tuned_logistic_fusion] C search start")
    x_val = np.column_stack([p_text_val, p_tabular_val])
    x_test = np.column_stack([p_text_test, p_tabular_test])
    candidates = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    rows: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None
    for c_value in candidates:
        model = LogisticRegression(C=c_value, solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE)
        model.fit(x_val, y_val)
        p_val = positive_scores(model, x_val)
        metrics = compute_metrics(y_val, p_val)
        row = {"C": c_value, **{key: value for key, value in metrics.items() if key != "confusion_matrix"}}
        row.update(metrics["confusion_matrix"])
        rows.append(row)
        rank_key = (metrics["roc_auc"], metrics["pr_auc"], metrics["f1"])
        if best is None or rank_key > best["rank_key"]:
            best = {"C": c_value, "model": model, "validation_metrics": metrics, "rank_key": rank_key}
    assert best is not None
    with (OUTPUT_DIR / "tuned_logistic_fusion_search_results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["C", "threshold", "roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    p_val_best = positive_scores(best["model"], x_val)
    p_test = positive_scores(best["model"], x_test)
    test_metrics = compute_metrics(y_test, p_test)
    write_predictions(OUTPUT_DIR / "tuned_logistic_fusion_validation_predictions.csv", val_indices, train_rows, y_val, p_val_best)
    write_predictions(OUTPUT_DIR / "tuned_logistic_fusion_test_predictions.csv", np.arange(len(test_rows)), test_rows, y_test, p_test)
    payload = {
        "model_name": "tuned_logistic_fusion",
        "input_features": ["SN1 p_text", "tuned_hgb_structured_drug p_tabular"],
        "best_C": best["C"],
        "validation_metrics": best["validation_metrics"],
        "test_metrics": test_metrics,
    }
    log(f"[tuned_logistic_fusion] best_C={best['C']}")
    log(f"[tuned_logistic_fusion] test roc_auc={test_metrics['roc_auc']:.6f} pr_auc={test_metrics['pr_auc']:.6f} f1={test_metrics['f1']:.6f}")
    return {"payload": payload, "p_val": p_val_best, "p_test": p_test}


def fit_weighted_average(
    p_text_val: np.ndarray,
    p_tabular_val: np.ndarray,
    y_val: np.ndarray,
    p_text_test: np.ndarray,
    p_tabular_test: np.ndarray,
    y_test: np.ndarray,
    val_indices: np.ndarray,
    train_rows: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    log("[weighted_average_fusion] alpha search start")
    rows: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None
    for alpha in np.round(np.arange(0.0, 1.0001, 0.05), 2):
        p_val = alpha * p_text_val + (1.0 - alpha) * p_tabular_val
        metrics = compute_metrics(y_val, p_val)
        row = {"alpha": float(alpha), **{key: value for key, value in metrics.items() if key != "confusion_matrix"}}
        row.update(metrics["confusion_matrix"])
        rows.append(row)
        rank_key = (metrics["roc_auc"], metrics["pr_auc"], metrics["f1"])
        if best is None or rank_key > best["rank_key"]:
            best = {"alpha": float(alpha), "validation_metrics": metrics, "rank_key": rank_key}
    assert best is not None
    with (OUTPUT_DIR / "weighted_average_fusion_search_results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["alpha", "threshold", "roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    p_val_best = best["alpha"] * p_text_val + (1.0 - best["alpha"]) * p_tabular_val
    p_test = best["alpha"] * p_text_test + (1.0 - best["alpha"]) * p_tabular_test
    test_metrics = compute_metrics(y_test, p_test)
    write_predictions(OUTPUT_DIR / "weighted_average_fusion_validation_predictions.csv", val_indices, train_rows, y_val, p_val_best)
    write_predictions(OUTPUT_DIR / "weighted_average_fusion_test_predictions.csv", np.arange(len(test_rows)), test_rows, y_test, p_test)
    payload = {
        "model_name": "weighted_average_fusion",
        "formula": "p = alpha*p_text + (1-alpha)*p_tabular",
        "best_alpha": best["alpha"],
        "validation_metrics": best["validation_metrics"],
        "test_metrics": test_metrics,
    }
    log(f"[weighted_average_fusion] best_alpha={best['alpha']}")
    log(f"[weighted_average_fusion] test roc_auc={test_metrics['roc_auc']:.6f} pr_auc={test_metrics['pr_auc']:.6f} f1={test_metrics['f1']:.6f}")
    return {"payload": payload, "p_val": p_val_best, "p_test": p_test}


def metric_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    test_metrics = payload["test_metrics"]
    cm = test_metrics["confusion_matrix"]
    row = {
        "model_name": payload["model_name"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "tn": cm["tn"],
        "fp": cm["fp"],
        "fn": cm["fn"],
        "tp": cm["tp"],
    }
    if "best_params" in payload:
        row["best_params"] = json.dumps(payload["best_params"], ensure_ascii=False, sort_keys=True)
        row["best_cv_roc_auc"] = payload["best_cv_roc_auc"]
    if "best_C" in payload:
        row["best_C"] = payload["best_C"]
    if "best_alpha" in payload:
        row["best_alpha"] = payload["best_alpha"]
    return row


def main() -> None:
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    started = datetime.now().astimezone()
    log(f"started_at={started.isoformat()}")
    log("cpu_only=true")
    log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}")
    ensure_required_files()
    for name, path in required_files().items():
        log(f"{name}: {path}")

    train_rows = read_rows(TRAIN_PATH)
    test_rows = read_rows(TEST_PATH)
    if not train_rows or not test_rows:
        raise ValueError("train_0.json and test_0.json must both be non-empty.")

    y_train_full = labels_from_rows(train_rows, "train_0")
    y_test = labels_from_rows(test_rows, "test_0")
    all_train_indices = np.arange(len(train_rows))
    train_base_idx, val_idx = train_test_split(
        all_train_indices,
        test_size=0.1,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )
    val_idx_sorted = np.sort(val_idx)
    log(f"train_full_count={len(train_rows)}")
    log(f"train_base_count={len(train_base_idx)}")
    log(f"val_fusion_count={len(val_idx_sorted)}")
    log(f"test_count={len(test_rows)}")

    sn1_val = read_prediction_csv(SN1_VAL_PRED_PATH).sort_values("index").reset_index(drop=True)
    sn1_test = read_prediction_csv(SN1_TEST_PRED_PATH).sort_values("index").reset_index(drop=True)
    if len(sn1_val) != len(val_idx_sorted):
        raise ValueError(f"SN1 validation row count {len(sn1_val)} != expected {len(val_idx_sorted)}.")
    if len(sn1_test) != len(test_rows):
        raise ValueError(f"SN1 test row count {len(sn1_test)} != expected {len(test_rows)}.")
    if not np.array_equal(sn1_val["index"].to_numpy(dtype=np.int64), val_idx_sorted):
        raise ValueError("SN1 validation indices do not match train_0 stratified split with random_state=42.")
    if not np.array_equal(sn1_val["y_true"].to_numpy(dtype=np.int64), y_train_full[val_idx_sorted]):
        raise ValueError("SN1 validation y_true does not match train_0 labels.")
    if not np.array_equal(sn1_test["index"].to_numpy(dtype=np.int64), np.arange(len(test_rows))):
        raise ValueError("SN1 test indices do not match test_0 row order.")
    if not np.array_equal(sn1_test["y_true"].to_numpy(dtype=np.int64), y_test):
        raise ValueError("SN1 test y_true does not match test_0 labels.")
    p_text_val = sn1_val["y_prob"].astype(float).to_numpy()
    p_text_test = sn1_test["y_prob"].astype(float).to_numpy()
    log("SN1 prediction files validated against split and labels")

    structured_columns = infer_structured_feature_columns(train_rows, test_rows)
    if len(structured_columns) != 46:
        raise ValueError(f"Expected 46 structured features, inferred {len(structured_columns)}: {structured_columns}")
    write_json(OUTPUT_DIR / "structured_feature_columns.json", structured_columns)
    log(f"structured_feature_count={len(structured_columns)}")

    x_structured_full = structured_frame(train_rows, structured_columns).to_numpy(dtype=np.float32)
    x_structured_test = structured_frame(test_rows, structured_columns).to_numpy(dtype=np.float32)
    drug_dim, drug_warning = infer_drug_dimension(list(train_rows) + list(test_rows))
    x_drug_full = drug_matrix(train_rows, drug_dim)
    x_drug_test = drug_matrix(test_rows, drug_dim)
    drug_columns = [f"drug_code_{index}" for index in range(drug_dim)]
    write_json(OUTPUT_DIR / "structured_drug_feature_columns.json", structured_columns + drug_columns)
    log(f"drug_code_dimension={drug_dim}")
    if drug_warning["length_inconsistent"]:
        log(f"WARNING drug_code length inconsistency: {drug_warning}")

    x_structured_train_base = x_structured_full[train_base_idx]
    y_train_base = y_train_full[train_base_idx]
    x_structured_val = x_structured_full[val_idx_sorted]
    y_val = y_train_full[val_idx_sorted]
    x_structured_drug_full = np.hstack([x_structured_full, x_drug_full])
    x_structured_drug_test = np.hstack([x_structured_test, x_drug_test])
    x_structured_drug_train_base = x_structured_drug_full[train_base_idx]
    x_structured_drug_val = x_structured_drug_full[val_idx_sorted]

    results: Dict[str, Dict[str, Any]] = {}
    hgb_structured = train_hgb_search(
        "tuned_hgb_structured",
        x_structured_train_base,
        y_train_base,
        x_structured_val,
        y_val,
        x_structured_test,
        y_test,
        val_idx_sorted,
        train_rows,
        test_rows,
        structured_columns,
    )
    results["tuned_hgb_structured"] = hgb_structured["payload"]

    hgb_structured_drug = train_hgb_search(
        "tuned_hgb_structured_drug",
        x_structured_drug_train_base,
        y_train_base,
        x_structured_drug_val,
        y_val,
        x_structured_drug_test,
        y_test,
        val_idx_sorted,
        train_rows,
        test_rows,
        structured_columns + drug_columns,
    )
    results["tuned_hgb_structured_drug"] = hgb_structured_drug["payload"]

    logistic_fusion = fit_logistic_fusion(
        p_text_val,
        hgb_structured_drug["p_val"],
        y_val,
        p_text_test,
        hgb_structured_drug["p_test"],
        y_test,
        val_idx_sorted,
        train_rows,
        test_rows,
    )
    results["tuned_logistic_fusion"] = logistic_fusion["payload"]

    weighted_average = fit_weighted_average(
        p_text_val,
        hgb_structured_drug["p_val"],
        y_val,
        p_text_test,
        hgb_structured_drug["p_test"],
        y_test,
        val_idx_sorted,
        train_rows,
        test_rows,
    )
    results["weighted_average_fusion"] = weighted_average["payload"]

    rows = [metric_row(payload) for payload in results.values()]
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "cpu_tuning_summary.csv", index=False, encoding="utf-8-sig")
    best_by = {
        metric: max(results.values(), key=lambda payload: payload["test_metrics"][metric])["model_name"]
        for metric in ("roc_auc", "pr_auc", "f1")
    }
    exceeds_previous_logistic = {
        metric: any(payload["test_metrics"][metric] > PREVIOUS_LOGISTIC_FUSION[metric] for payload in results.values())
        for metric in ("roc_auc", "pr_auc", "f1")
    }
    strongest_changed = any(
        payload["test_metrics"]["roc_auc"] > PREVIOUS_LOGISTIC_FUSION["roc_auc"] for payload in results.values()
    )
    metrics_payload = {
        "started_at": started.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "output_dir": str(OUTPUT_DIR),
        "project_root": str(PROJECT_ROOT),
        "paths": {name: str(path) for name, path in required_files().items()},
        "split": {
            "random_state": RANDOM_STATE,
            "label": LABEL_COLUMN,
            "train_full_count": int(len(train_rows)),
            "train_base_count": int(len(train_base_idx)),
            "val_fusion_count": int(len(val_idx_sorted)),
            "test_count": int(len(test_rows)),
            "positive_ratio_train_base": float(y_train_base.mean()),
            "positive_ratio_val_fusion": float(y_val.mean()),
            "positive_ratio_test": float(y_test.mean()),
        },
        "drug_code": drug_warning,
        "previous_hgb_structured_drug": PREVIOUS_HGB_STRUCTURED_DRUG,
        "previous_logistic_fusion": PREVIOUS_LOGISTIC_FUSION,
        "results": results,
        "best_model_by": best_by,
        "exceeds_previous_logistic_fusion": exceeds_previous_logistic,
        "strongest_classical_baseline_changed_by_roc_auc": strongest_changed,
    }
    write_json(OUTPUT_DIR / "cpu_tuning_metrics.json", metrics_payload)

    log("")
    log("FINAL SUMMARY")
    log(f"输出文件夹路径: {OUTPUT_DIR}")
    log(f"使用的数据路径 train: {TRAIN_PATH}")
    log(f"使用的数据路径 test: {TEST_PATH}")
    log(f"使用的 SN1 路径: {SN1_OUTPUT_DIR}")
    for row in rows:
        log(
            f"{row['model_name']}: test ROC-AUC={row['test_roc_auc']:.6f} "
            f"PR-AUC={row['test_pr_auc']:.6f} F1={row['test_f1']:.6f}"
        )
    log(f"best model by ROC-AUC: {best_by['roc_auc']}")
    log(f"best model by PR-AUC: {best_by['pr_auc']}")
    log(f"best model by F1: {best_by['f1']}")
    log(f"是否超过 previous_logistic_fusion ROC-AUC: {exceeds_previous_logistic['roc_auc']}")
    log(f"是否超过 previous_logistic_fusion PR-AUC: {exceeds_previous_logistic['pr_auc']}")
    log(f"是否超过 previous_logistic_fusion F1: {exceeds_previous_logistic['f1']}")
    log(f"strongest classical baseline 是否改变: {strongest_changed}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"FAILED: {type(exc).__name__}: {exc}")
        raise
