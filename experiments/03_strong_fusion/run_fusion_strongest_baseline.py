from __future__ import annotations

import csv
import json
import math
import shutil
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
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent
SN1_OUTPUT_DIR = PROJECT_ROOT / "sn1_text_4090_package_20260617_005006" / "output"
SN1_METRICS_PATH = SN1_OUTPUT_DIR / "metrics.json"
SN1_VAL_PRED_PATH = SN1_OUTPUT_DIR / "validation_predictions.csv"
SN1_TEST_PRED_PATH = SN1_OUTPUT_DIR / "predictions.csv"
DATA_DIR = PROJECT_ROOT / "faithful_redapm_cloud" / "妯″瀷浠ｇ爜" / "data_0"
TRAIN_PATH = DATA_DIR / "train_0.json"
TEST_PATH = DATA_DIR / "test_0.json"
REFERENCE_HGB_DIR = PROJECT_ROOT / "tabular_baseline_outputs" / "structured_drug" / "hgb"
REFERENCE_HGB_METRICS = REFERENCE_HGB_DIR / "metrics.json"
REFERENCE_HGB_PRED = REFERENCE_HGB_DIR / "predictions.csv"
OUTPUT_BASE = PROJECT_ROOT / "fusion_strongest_baseline_run"
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


def ensure_required_files() -> Dict[str, Path]:
    required = {
        "sn1_metrics": SN1_METRICS_PATH,
        "sn1_validation_predictions": SN1_VAL_PRED_PATH,
        "sn1_test_predictions": SN1_TEST_PRED_PATH,
        "train_0": TRAIN_PATH,
        "test_0": TEST_PATH,
    }
    missing = {name: path for name, path in required.items() if not path.exists()}
    if missing:
        checked = "\n".join(f"- {name}: {path}" for name, path in required.items())
        missing_text = "\n".join(f"- {name}: {path}" for name, path in missing.items())
        raise FileNotFoundError(
            "Missing required files.\nChecked paths:\n"
            f"{checked}\nMissing:\n{missing_text}"
        )
    return required


def unique_output_dir(base: Path) -> Path:
    if not base.exists():
        return base
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base.parent / f"{base.name}_{stamp}"


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


def label_from_row(row: Dict[str, Any], index: int) -> int:
    value = to_numeric_or_nan(row.get(LABEL_COLUMN))
    if value not in (0.0, 1.0):
        raise ValueError(f"Non-binary y2 at row {index}: {row.get(LABEL_COLUMN)!r}")
    return int(value)


def infer_structured_feature_columns(train_rows: Sequence[Dict[str, Any]], test_rows: Sequence[Dict[str, Any]]) -> List[str]:
    combined = list(train_rows) + list(test_rows)
    all_fields = sorted({key for row in combined for key in row})
    columns: List[str] = []
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
            columns.append(column)
    return columns


def build_structured_frame(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    records = []
    for row in rows:
        records.append({column: to_numeric_or_nan(row.get(column)) for column in columns})
    return pd.DataFrame.from_records(records, columns=columns)


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


def build_drug_matrix(rows: Sequence[Dict[str, Any]], dimension: int) -> np.ndarray:
    matrix = np.zeros((len(rows), dimension), dtype=np.float32)
    for row_index, row in enumerate(rows):
        vector = raw_drug_vector(row.get("drug_code"))
        clipped = vector[:dimension]
        if clipped:
            matrix[row_index, : len(clipped)] = np.asarray(clipped, dtype=np.float32)
    return matrix


def read_prediction_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"index", "y_true", "y_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df


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


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def log(path: Path, message: str) -> None:
    print(message, flush=True)
    append_log(path, message)


def main() -> None:
    ensure_required_files()
    output_dir = unique_output_dir(OUTPUT_BASE)
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "train_log.txt"

    train_rows = read_rows(TRAIN_PATH)
    test_rows = read_rows(TEST_PATH)
    if not train_rows or not test_rows:
        raise ValueError("train_0.json and test_0.json must both be non-empty.")

    y_train_full = np.asarray([label_from_row(row, i) for i, row in enumerate(train_rows)], dtype=np.int64)
    y_test = np.asarray([label_from_row(row, i) for i, row in enumerate(test_rows)], dtype=np.int64)
    train_indices = np.arange(len(train_rows))
    train_fit_idx, val_idx = train_test_split(
        train_indices,
        test_size=0.1,
        random_state=42,
        stratify=y_train_full,
    )
    val_idx_sorted = np.sort(val_idx)

    log(log_path, f"train_count={len(train_rows)}")
    log(log_path, f"validation_count={len(val_idx_sorted)}")
    log(log_path, f"test_count={len(test_rows)}")
    log(log_path, f"positive_ratio_train_full={float(y_train_full.mean())}")
    log(log_path, f"positive_ratio_validation={float(y_train_full[val_idx_sorted].mean())}")
    log(log_path, f"positive_ratio_test={float(y_test.mean())}")

    sn1_val = read_prediction_csv(SN1_VAL_PRED_PATH).sort_values("index").reset_index(drop=True)
    sn1_test = read_prediction_csv(SN1_TEST_PRED_PATH).sort_values("index").reset_index(drop=True)
    if len(sn1_val) != len(val_idx_sorted):
        raise ValueError(
            f"SN1 validation_predictions row count {len(sn1_val)} does not match expected validation size {len(val_idx_sorted)}."
        )
    if len(sn1_test) != len(test_rows):
        raise ValueError(
            f"SN1 predictions row count {len(sn1_test)} does not match expected test size {len(test_rows)}."
        )
    if not np.array_equal(sn1_val["index"].to_numpy(dtype=np.int64), val_idx_sorted):
        raise ValueError("SN1 validation prediction indices do not match the required stratified split (seed=42, validation_ratio=0.1).")
    if not np.array_equal(sn1_val["y_true"].to_numpy(dtype=np.int64), y_train_full[val_idx_sorted]):
        raise ValueError("SN1 validation y_true does not match train_0.json labels under the required split.")
    if not np.array_equal(sn1_test["y_true"].to_numpy(dtype=np.int64), y_test):
        raise ValueError("SN1 test y_true does not match test_0.json labels.")
    log(log_path, "SN1 validation/test prediction files validated")

    structured_columns = infer_structured_feature_columns(train_rows, test_rows)
    structured_train_df = build_structured_frame(train_rows, structured_columns)
    structured_test_df = build_structured_frame(test_rows, structured_columns)
    imputer = SimpleImputer(strategy="median")
    structured_train = imputer.fit_transform(structured_train_df)
    structured_test = imputer.transform(structured_test_df)

    drug_dimension, drug_warning = infer_drug_dimension(list(train_rows) + list(test_rows))
    drug_train = build_drug_matrix(train_rows, drug_dimension)
    drug_test = build_drug_matrix(test_rows, drug_dimension)
    x_train_full = np.hstack([structured_train, drug_train])
    x_test = np.hstack([structured_test, drug_test])
    x_train_fit = x_train_full[train_fit_idx]
    y_train_fit = y_train_full[train_fit_idx]
    x_val = x_train_full[val_idx_sorted]
    y_val = y_train_full[val_idx_sorted]
    log(log_path, f"structured_feature_count={len(structured_columns)}")
    log(log_path, f"drug_code_dimension={drug_dimension}")
    log(log_path, f"total_feature_count={x_train_full.shape[1]}")
    if drug_warning["length_inconsistent"]:
        log(log_path, "WARNING: drug_code length inconsistency detected; vectors padded/truncated to the most common dimension")

    hgb_params = {
        "loss": "log_loss",
        "learning_rate": 0.05,
        "max_iter": 500,
        "max_leaf_nodes": 31,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 20,
        "random_state": 42,
    }
    hgb_val_model = HistGradientBoostingClassifier(**hgb_params)
    hgb_val_model.fit(x_train_fit, y_train_fit)
    p_tabular_val = hgb_val_model.predict_proba(x_val)[:, 1]
    hgb_full_model = HistGradientBoostingClassifier(**hgb_params)
    hgb_full_model.fit(x_train_full, y_train_full)
    p_tabular_test = hgb_full_model.predict_proba(x_test)[:, 1]
    hgb_val_metrics = compute_metrics(y_val, p_tabular_val)
    hgb_test_metrics = compute_metrics(y_test, p_tabular_test)
    log(log_path, f"HGB validation roc_auc={hgb_val_metrics['roc_auc']}")
    log(log_path, f"HGB test roc_auc={hgb_test_metrics['roc_auc']}")

    p_text_val = sn1_val["y_prob"].astype(float).to_numpy()
    p_text_test = sn1_test["y_prob"].astype(float).to_numpy()
    jmkh_test = sn1_test["jmkh"].astype(str).to_numpy() if "jmkh" in sn1_test.columns else np.asarray([row.get("jmkh", "") for row in test_rows], dtype=object)

    fusion_x_val = np.column_stack([p_text_val, p_tabular_val])
    fusion_x_test = np.column_stack([p_text_test, p_tabular_test])
    fusion_model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    fusion_model.fit(fusion_x_val, y_val)
    p_fusion = fusion_model.predict_proba(fusion_x_test)[:, 1]
    p_avg = 0.5 * p_text_test + 0.5 * p_tabular_test

    sn1_metrics_payload = json.loads(SN1_METRICS_PATH.read_text(encoding="utf-8"))
    sn1_test_metrics = sn1_metrics_payload.get("test_metrics", {})
    avg_metrics = compute_metrics(y_test, p_avg)
    fusion_metrics = compute_metrics(y_test, p_fusion)

    reference_hgb_metrics = None
    if REFERENCE_HGB_METRICS.exists() and REFERENCE_HGB_PRED.exists():
        reference_hgb_metrics = json.loads(REFERENCE_HGB_METRICS.read_text(encoding="utf-8"))
        log(log_path, "reference HGB outputs found")
    else:
        log(log_path, "reference HGB outputs not found")

    val_pred_rows = []
    for idx, y_true, p_text, p_tab in zip(val_idx_sorted, y_val, p_text_val, p_tabular_val):
        val_pred_rows.append(
            {
                "index": int(idx),
                "jmkh": train_rows[int(idx)].get("jmkh", ""),
                "y_true": int(y_true),
                "p_text": f"{float(p_text):.10f}",
                "p_tabular": f"{float(p_tab):.10f}",
            }
        )
    write_csv(output_dir / "hgb_validation_predictions.csv", val_pred_rows, ["index", "jmkh", "y_true", "p_text", "p_tabular"])

    test_pred_rows = []
    fusion_output_rows = []
    y_pred_fusion = (p_fusion >= 0.5).astype(int)
    y_pred_avg = (p_avg >= 0.5).astype(int)
    for index, jmkh, y_true_value, p_text_value, p_tab_value, p_fusion_value, p_avg_value, pred_fusion_value, pred_avg_value in zip(
        sn1_test["index"].astype(int).to_numpy(),
        jmkh_test,
        y_test,
        p_text_test,
        p_tabular_test,
        p_fusion,
        p_avg,
        y_pred_fusion,
        y_pred_avg,
    ):
        test_pred_rows.append(
            {
                "index": int(index),
                "jmkh": jmkh,
                "y_true": int(y_true_value),
                "p_tabular": f"{float(p_tab_value):.10f}",
            }
        )
        fusion_output_rows.append(
            {
                "index": int(index),
                "jmkh": jmkh,
                "y_true": int(y_true_value),
                "p_text": f"{float(p_text_value):.10f}",
                "p_tabular": f"{float(p_tab_value):.10f}",
                "p_fusion": f"{float(p_fusion_value):.10f}",
                "p_avg": f"{float(p_avg_value):.10f}",
                "y_pred_fusion": int(pred_fusion_value),
                "y_pred_avg": int(pred_avg_value),
            }
        )
    write_csv(output_dir / "hgb_test_predictions.csv", test_pred_rows, ["index", "jmkh", "y_true", "p_tabular"])
    write_csv(
        output_dir / "fusion_predictions.csv",
        fusion_output_rows,
        ["index", "jmkh", "y_true", "p_text", "p_tabular", "p_fusion", "p_avg", "y_pred_fusion", "y_pred_avg"],
    )

    fusion_payload = {
        "sn1_test_metrics": sn1_test_metrics,
        "hgb_validation_metrics": hgb_val_metrics,
        "hgb_test_metrics": hgb_test_metrics,
        "average_ensemble_test_metrics": avg_metrics,
        "logistic_fusion_test_metrics": fusion_metrics,
        "delta_fusion_vs_hgb": {
            "roc_auc": fusion_metrics["roc_auc"] - hgb_test_metrics["roc_auc"],
            "pr_auc": fusion_metrics["pr_auc"] - hgb_test_metrics["pr_auc"],
            "f1": fusion_metrics["f1"] - hgb_test_metrics["f1"],
        },
        "delta_fusion_vs_sn1": {
            "roc_auc": fusion_metrics["roc_auc"] - float(sn1_test_metrics["roc_auc"]),
            "pr_auc": fusion_metrics["pr_auc"] - float(sn1_test_metrics["pr_auc"]),
            "f1": fusion_metrics["f1"] - float(sn1_test_metrics["f1"]),
        },
        "delta_avg_vs_hgb": {
            "roc_auc": avg_metrics["roc_auc"] - hgb_test_metrics["roc_auc"],
            "pr_auc": avg_metrics["pr_auc"] - hgb_test_metrics["pr_auc"],
            "f1": avg_metrics["f1"] - hgb_test_metrics["f1"],
        },
        "delta_avg_vs_sn1": {
            "roc_auc": avg_metrics["roc_auc"] - float(sn1_test_metrics["roc_auc"]),
            "pr_auc": avg_metrics["pr_auc"] - float(sn1_test_metrics["pr_auc"]),
            "f1": avg_metrics["f1"] - float(sn1_test_metrics["f1"]),
        },
        "feature_info": {
            "structured_feature_count": len(structured_columns),
            "drug_code_dimension": drug_dimension,
            "total_feature_count": int(x_train_full.shape[1]),
            "structured_feature_columns": structured_columns,
        },
        "training_setup": {
            "validation_ratio": 0.1,
            "random_state": 42,
            "hgb_params": hgb_params,
            "fusion_params": {"solver": "lbfgs", "max_iter": 1000, "random_state": 42},
        },
        "reference_hgb_metrics": reference_hgb_metrics,
    }
    write_json(output_dir / "fusion_metrics.json", fusion_payload)

    summary_rows = [
        {
            "model_name": "SN1 text-only",
            "roc_auc": sn1_test_metrics["roc_auc"],
            "pr_auc": sn1_test_metrics["pr_auc"],
            "f1": sn1_test_metrics["f1"],
            "accuracy": sn1_test_metrics["accuracy"],
            "precision": sn1_test_metrics["precision"],
            "recall": sn1_test_metrics["recall"],
        },
        {
            "model_name": "HGB structured+drug",
            "roc_auc": hgb_test_metrics["roc_auc"],
            "pr_auc": hgb_test_metrics["pr_auc"],
            "f1": hgb_test_metrics["f1"],
            "accuracy": hgb_test_metrics["accuracy"],
            "precision": hgb_test_metrics["precision"],
            "recall": hgb_test_metrics["recall"],
        },
        {
            "model_name": "Average ensemble",
            "roc_auc": avg_metrics["roc_auc"],
            "pr_auc": avg_metrics["pr_auc"],
            "f1": avg_metrics["f1"],
            "accuracy": avg_metrics["accuracy"],
            "precision": avg_metrics["precision"],
            "recall": avg_metrics["recall"],
        },
        {
            "model_name": "Logistic fusion",
            "roc_auc": fusion_metrics["roc_auc"],
            "pr_auc": fusion_metrics["pr_auc"],
            "f1": fusion_metrics["f1"],
            "accuracy": fusion_metrics["accuracy"],
            "precision": fusion_metrics["precision"],
            "recall": fusion_metrics["recall"],
        },
    ]
    write_csv(output_dir / "fusion_experiment_summary.csv", summary_rows, ["model_name", "roc_auc", "pr_auc", "f1", "accuracy", "precision", "recall"])

    script_copy = output_dir / "run_fusion_strongest_baseline.py"
    shutil.copy2(Path(__file__), script_copy)

    best_models = [
        ("SN1 text-only", float(sn1_test_metrics["roc_auc"]), float(sn1_test_metrics["pr_auc"]), float(sn1_test_metrics["f1"])),
        ("HGB structured+drug", hgb_test_metrics["roc_auc"], hgb_test_metrics["pr_auc"], hgb_test_metrics["f1"]),
        ("Average ensemble", avg_metrics["roc_auc"], avg_metrics["pr_auc"], avg_metrics["f1"]),
        ("Logistic fusion", fusion_metrics["roc_auc"], fusion_metrics["pr_auc"], fusion_metrics["f1"]),
    ]
    best_by_roc = max(best_models, key=lambda item: item[1])
    best_by_pr = max(best_models, key=lambda item: item[2])
    best_by_f1 = max(best_models, key=lambda item: item[3])

    print(f"SN1 test ROC-AUC / PR-AUC / F1: {sn1_test_metrics['roc_auc']} / {sn1_test_metrics['pr_auc']} / {sn1_test_metrics['f1']}")
    print(f"HGB structured+drug test ROC-AUC / PR-AUC / F1: {hgb_test_metrics['roc_auc']} / {hgb_test_metrics['pr_auc']} / {hgb_test_metrics['f1']}")
    print(f"Average ensemble ROC-AUC / PR-AUC / F1: {avg_metrics['roc_auc']} / {avg_metrics['pr_auc']} / {avg_metrics['f1']}")
    print(f"Logistic fusion ROC-AUC / PR-AUC / F1: {fusion_metrics['roc_auc']} / {fusion_metrics['pr_auc']} / {fusion_metrics['f1']}")
    print(f"current best model by ROC-AUC: {best_by_roc[0]}")
    print(f"current best ROC-AUC: {best_by_roc[1]}")
    print(f"current best model by PR-AUC: {best_by_pr[0]}")
    print(f"current best PR-AUC: {best_by_pr[2]}")
    print(f"current best model by F1: {best_by_f1[0]}")
    print(f"current best F1: {best_by_f1[3]}")
    print(f"output directory: {output_dir}")


if __name__ == "__main__":
    main()

