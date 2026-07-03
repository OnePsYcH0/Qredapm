from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input_data"
RESULTS_DIR = ROOT / "results"

STRONG_ROC_AUC = 0.842092753489314
STRONG_PR_AUC = 0.6931922359744075
STRONG_F1 = 0.5808983863933711

EPS = 1e-6


SOURCES = {
    "strong_val": INPUT_DIR / "strong_hgb_validation_predictions.csv",
    "strong_test": INPUT_DIR / "strong_fusion_predictions.csv",
    "qsn2_original_val": INPUT_DIR / "qsn2_original_validation_predictions.csv",
    "qsn2_original_test": INPUT_DIR / "qsn2_original_test_predictions.csv",
    "qsn2_r1_val": INPUT_DIR / "qsn2_r1_validation_predictions.csv",
    "qsn2_r1_test": INPUT_DIR / "qsn2_r1_test_predictions.csv",
    "qsn2_c1_val": INPUT_DIR / "qsn2_c1_validation_predictions.csv",
    "qsn2_c1_test": INPUT_DIR / "qsn2_c1_test_predictions.csv",
    "qsn2_c2_val": INPUT_DIR / "qsn2_c2_validation_predictions.csv",
    "qsn2_c2_test": INPUT_DIR / "qsn2_c2_test_predictions.csv",
    "qsn2_c3_val": INPUT_DIR / "qsn2_c3_validation_predictions.csv",
    "qsn2_c3_test": INPUT_DIR / "qsn2_c3_test_predictions.csv",
}


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def find_prob_col(df: pd.DataFrame, preferred: list[str] | None = None) -> str:
    preferred = preferred or []
    for col in preferred + ["y_prob", "p_fusion", "p_tabular", "p_text", "probability", "score"]:
        if col in df.columns:
            return col
    numeric = [c for c in df.columns if c not in {"index", "jmkh", "y_true", "y_pred", "threshold"}]
    if not numeric:
        raise ValueError(f"No probability column found in columns: {list(df.columns)}")
    return numeric[0]


def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "y_true" not in df.columns:
        raise ValueError(f"{path} lacks y_true")
    if "index" not in df.columns:
        df["index"] = np.arange(len(df))
    return df


def align_by_index(base: pd.DataFrame, other: pd.DataFrame, value_col: str, out_col: str) -> pd.DataFrame:
    cols = ["index", value_col]
    merged = base.merge(other[cols].rename(columns={value_col: out_col}), on="index", how="inner")
    if len(merged) != len(base):
        raise ValueError(f"Alignment changed row count for {out_col}: {len(base)} -> {len(merged)}")
    return merged


def align_validation_by_order(base: pd.DataFrame, other: pd.DataFrame, value_col: str, out_col: str) -> pd.DataFrame:
    """Validation files from different experiments may not preserve original train indices.

    They are generated from the same stratified split family but some REDAPM/QSN2
    files can contain two fewer usable rows. For validation-only meta-model
    selection, keep row-order alignment and record concordance in source_manifest.
    Test files are still index-aligned because final evaluation must be exact.
    """
    n = min(len(base), len(other))
    if n == 0:
        raise ValueError(f"Empty validation alignment for {out_col}")
    out = base.iloc[:n].reset_index(drop=True).copy()
    other2 = other.iloc[:n].reset_index(drop=True)
    out[out_col] = other2[value_col].astype(float)
    label_agreement = float((out["y_true"].astype(int).to_numpy() == other2["y_true"].astype(int).to_numpy()).mean())
    out.attrs[f"{out_col}_label_agreement"] = label_agreement
    out.attrs[f"{out_col}_alignment_rows"] = n
    out.attrs[f"{out_col}_other_rows"] = len(other)
    return out


def threshold_for_best_f1(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.unique(np.r_[np.linspace(0, 1, 201), p])
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        score = f1_score(y, (p >= t).astype(int), zero_division=0)
        if score > best_f1:
            best_t, best_f1 = float(t), float(score)
    return best_t


def sensitivity_at_specificity(y: np.ndarray, p: np.ndarray, target_specificity: float) -> float:
    best = 0.0
    for t in np.unique(np.r_[np.linspace(0, 1, 1001), p]):
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) else math.nan
        sensitivity = tp / (tp + fn) if (tp + fn) else math.nan
        if np.isfinite(specificity) and specificity >= target_specificity and np.isfinite(sensitivity):
            best = max(best, float(sensitivity))
    return best


def ece_score(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            ece += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    x = logit(p).reshape(-1, 1)
    try:
        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        model.fit(x, y)
        return float(model.intercept_[0]), float(model.coef_[0][0])
    except Exception:
        return math.nan, math.nan


def dca_mean_net_benefit(y: np.ndarray, p: np.ndarray, lo: float = 0.10, hi: float = 0.30) -> float:
    thresholds = np.arange(lo, hi + 1e-9, 0.01)
    n = len(y)
    values = []
    for t in thresholds:
        pred = p >= t
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        values.append(tp / n - fp / n * (t / (1 - t)))
    return float(np.mean(values))


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else math.nan
    npv = tn / (tn + fn) if (tn + fn) else math.nan
    lr_pos = recall_score(y, pred, zero_division=0) / (1 - specificity) if specificity < 1 else math.inf
    lr_neg = (1 - recall_score(y, pred, zero_division=0)) / specificity if specificity > 0 else math.inf
    intercept, slope = calibration_slope_intercept(y, p)
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity_recall": recall_score(y, pred, zero_division=0),
        "specificity": specificity,
        "precision_ppv": precision_score(y, pred, zero_division=0),
        "npv": npv,
        "f1": f1_score(y, pred, zero_division=0),
        "brier_score": brier_score_loss(y, p),
        "ece": ece_score(y, p),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "sensitivity_at_specificity_90": sensitivity_at_specificity(y, p, 0.90),
        "sensitivity_at_specificity_95": sensitivity_at_specificity(y, p, 0.95),
        "lr_positive": lr_pos,
        "lr_negative": lr_neg,
        "dca_mean_010_030": dca_mean_net_benefit(y, p),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def make_features(df: pd.DataFrame, features: list[str], transform: str, pairwise: bool) -> np.ndarray:
    raw = df[features].astype(float).to_numpy()
    if transform == "prob":
        x = raw
    elif transform == "logit":
        x = logit(raw)
    elif transform == "prob_logit":
        x = np.column_stack([raw, logit(raw)])
    else:
        raise ValueError(transform)
    if pairwise and len(features) >= 2:
        pairs = []
        for i, j in combinations(range(len(features)), 2):
            pairs.append((raw[:, i] * raw[:, j]).reshape(-1, 1))
            pairs.append(np.abs(raw[:, i] - raw[:, j]).reshape(-1, 1))
        x = np.column_stack([x] + pairs)
    return x


def oof_predict(model, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred = np.zeros(len(y), dtype=float)
    for train_idx, val_idx in splitter.split(x, y):
        m = clone(model)
        m.fit(x[train_idx], y[train_idx])
        pred[val_idx] = m.predict_proba(x[val_idx])[:, 1]
    return pred


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    manifest = []
    for name, path in SOURCES.items():
        manifest.append({"source": name, "path": str(path), "exists": path.exists()})
    missing = [m["path"] for m in manifest if not m["exists"]]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    strong_val = load_frame(SOURCES["strong_val"])
    strong_test = load_frame(SOURCES["strong_test"])
    val = strong_val[["index", "y_true", "p_text", "p_tabular"]].copy()
    test = strong_test[["index", "y_true", "p_text", "p_tabular", "p_fusion", "p_avg"]].copy()

    q_sources = {
        "p_qsn2_original": ("qsn2_original_val", "qsn2_original_test"),
        "p_qsn2_r1": ("qsn2_r1_val", "qsn2_r1_test"),
        "p_qsn2_c1": ("qsn2_c1_val", "qsn2_c1_test"),
        "p_qsn2_c2": ("qsn2_c2_val", "qsn2_c2_test"),
        "p_qsn2_c3": ("qsn2_c3_val", "qsn2_c3_test"),
    }
    for out_col, (v_key, t_key) in q_sources.items():
        v = load_frame(SOURCES[v_key])
        t = load_frame(SOURCES[t_key])
        val = align_validation_by_order(val, v, find_prob_col(v), out_col)
        manifest.append({
            "source": f"{out_col}_validation_alignment",
            "path": str(SOURCES[v_key]),
            "exists": True,
            "alignment": "row_order_min_length",
            "base_rows_after_alignment": len(val),
            "other_rows": len(v),
            "label_agreement": val.attrs.get(f"{out_col}_label_agreement"),
        })
        test = align_by_index(test, t, find_prob_col(t), out_col)

    y_val = val["y_true"].astype(int).to_numpy()
    strong_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=5000, random_state=42),
    )
    x_strong_val = val[["p_text", "p_tabular"]].astype(float).to_numpy()
    x_strong_test = test[["p_text", "p_tabular"]].astype(float).to_numpy()
    val["p_strong_oof"] = oof_predict(strong_model, x_strong_val, y_val)
    strong_model.fit(x_strong_val, y_val)
    test["p_strong_refit"] = strong_model.predict_proba(x_strong_test)[:, 1]

    return val, test, manifest


def candidate_models() -> dict[str, object]:
    cs = [0.03, 0.1, 0.3, 1, 3, 10]
    models = {}
    for c in cs:
        models[f"l2_C{c}"] = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, solver="lbfgs", penalty="l2", max_iter=5000, random_state=42),
        )
        models[f"balanced_l2_C{c}"] = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, solver="lbfgs", penalty="l2", class_weight="balanced", max_iter=5000, random_state=42),
        )
    return models


def experiment9(val: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_val = val["y_true"].astype(int).to_numpy()
    y_test = test["y_true"].astype(int).to_numpy()
    feature_sets = {
        "strong_rebuilt": ["p_text", "p_tabular"],
        "strong_plus_qsn2_r1": ["p_text", "p_tabular", "p_qsn2_r1"],
        "strong_plus_qsn2_original": ["p_text", "p_tabular", "p_qsn2_original"],
        "strong_plus_qsn2_original_r1": ["p_text", "p_tabular", "p_qsn2_original", "p_qsn2_r1"],
        "strong_plus_qsn2_r1_c1": ["p_text", "p_tabular", "p_qsn2_r1", "p_qsn2_c1"],
        "strong_plus_qsn2_r1_c2": ["p_text", "p_tabular", "p_qsn2_r1", "p_qsn2_c2"],
        "pstrong_plus_qsn2_r1": ["p_strong_oof", "p_qsn2_r1"],
        "pstrong_plus_qsn2_original_r1": ["p_strong_oof", "p_qsn2_original", "p_qsn2_r1"],
    }
    test_feature_alias = {"p_strong_oof": "p_strong_refit"}
    rows = []
    best_val_pred = None
    best_test_pred = None
    models = candidate_models()
    for feature_set, features in feature_sets.items():
        test_features = [test_feature_alias.get(f, f) for f in features]
        for transform in ["prob", "logit", "prob_logit"]:
            for pairwise in [False, True]:
                x_val = make_features(val, features, transform, pairwise)
                x_test = make_features(test.rename(columns={"p_strong_refit": "p_strong_oof"}), features, transform, pairwise)
                for model_name, model in models.items():
                    try:
                        p_val = oof_predict(model, x_val, y_val)
                        threshold = threshold_for_best_f1(y_val, p_val)
                        m = clone(model)
                        m.fit(x_val, y_val)
                        p_test = m.predict_proba(x_test)[:, 1]
                        row = {
                            "experiment": "exp9_quantum_enhanced_strong_fusion",
                            "feature_set": feature_set,
                            "features": ",".join(features),
                            "model": model_name,
                            "transform": transform,
                            "pairwise": pairwise,
                            "n_raw_features": len(features),
                            "n_model_features": x_val.shape[1],
                            "threshold": threshold,
                        }
                        row.update({f"val_{k}": v for k, v in binary_metrics(y_val, p_val, threshold).items()})
                        row.update({f"test_{k}": v for k, v in binary_metrics(y_test, p_test, threshold).items()})
                        row["delta_test_roc_vs_strong"] = row["test_roc_auc"] - STRONG_ROC_AUC
                        row["delta_test_pr_vs_strong"] = row["test_pr_auc"] - STRONG_PR_AUC
                        row["delta_test_f1_vs_strong"] = row["test_f1"] - STRONG_F1
                        rows.append(row)
                    except Exception as exc:
                        rows.append({
                            "experiment": "exp9_quantum_enhanced_strong_fusion",
                            "feature_set": feature_set,
                            "features": ",".join(features),
                            "model": model_name,
                            "transform": transform,
                            "pairwise": pairwise,
                            "status": f"failed: {exc}",
                        })
    results = pd.DataFrame(rows)
    ok = results.dropna(subset=["val_roc_auc"]).copy()
    best = ok.sort_values(["val_roc_auc", "val_pr_auc", "val_f1"], ascending=False).head(1)
    if len(best):
        b = best.iloc[0]
        features = str(b["features"]).split(",")
        transform = str(b["transform"])
        pairwise = bool(b["pairwise"])
        model = candidate_models()[str(b["model"])]
        x_val = make_features(val, features, transform, pairwise)
        x_test = make_features(test.rename(columns={"p_strong_refit": "p_strong_oof"}), features, transform, pairwise)
        p_val = oof_predict(model, x_val, y_val)
        model.fit(x_val, y_val)
        p_test = model.predict_proba(x_test)[:, 1]
        threshold = float(b["threshold"])
        best_val_pred = pd.DataFrame({"index": val["index"], "y_true": y_val, "y_prob": p_val, "y_pred": (p_val >= threshold).astype(int)})
        best_test_pred = pd.DataFrame({"index": test["index"], "y_true": y_test, "y_prob": p_test, "y_pred": (p_test >= threshold).astype(int)})
    return results, best, (best_val_pred, best_test_pred)


def experiment10(per_run: pd.DataFrame) -> pd.DataFrame:
    ok = per_run.dropna(subset=["val_roc_auc"]).copy()
    criteria = [
        ("best_val_roc_auc", ["val_roc_auc", "val_pr_auc", "val_f1"]),
        ("best_val_pr_auc", ["val_pr_auc", "val_roc_auc", "val_f1"]),
        ("best_val_f1", ["val_f1", "val_pr_auc", "val_roc_auc"]),
        ("best_val_sens_at_spec90", ["val_sensitivity_at_specificity_90", "val_roc_auc"]),
        ("best_val_sens_at_spec95", ["val_sensitivity_at_specificity_95", "val_roc_auc"]),
        ("best_val_dca_010_030", ["val_dca_mean_010_030", "val_roc_auc"]),
        ("best_val_lr_negative", ["val_lr_negative"]),
    ]
    rows = []
    for label, cols in criteria:
        ascending = [False] * len(cols)
        if label == "best_val_lr_negative":
            ascending = [True]
        selected = ok.sort_values(cols, ascending=ascending).head(1).copy()
        if len(selected):
            row = selected.iloc[0].to_dict()
            row["selection_rule"] = label
            rows.append(row)
    return pd.DataFrame(rows)


def subset_metrics(name: str, y: np.ndarray, p: np.ndarray, threshold: float, mask: np.ndarray) -> dict[str, object]:
    if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
        return {"subset": name, "n": int(mask.sum()), "status": "too_small_or_single_class"}
    row = {"subset": name, "n": int(mask.sum()), "positive_rate": float(y[mask].mean()), "status": "ok"}
    row.update(binary_metrics(y[mask], p[mask], threshold))
    return row


def experiment11(val: pd.DataFrame, test: pd.DataFrame, best_test_pred: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_val = val["y_true"].astype(int).to_numpy()
    y_test = test["y_true"].astype(int).to_numpy()
    p_strong_val = val["p_strong_oof"].astype(float).to_numpy()
    p_strong_test = test["p_strong_refit"].astype(float).to_numpy()
    q_val = val["p_qsn2_r1"].astype(float).to_numpy()
    q_test = test["p_qsn2_r1"].astype(float).to_numpy()
    rows = []
    pred_sets = {
        "strong_rebuilt": (p_strong_val, p_strong_test, threshold_for_best_f1(y_val, p_strong_val)),
        "qsn2_r1": (q_val, q_test, threshold_for_best_f1(y_val, q_val)),
    }
    if best_test_pred is not None:
        # Validation counterpart is not needed for hard-case test subset metrics; use stored threshold from test predictions if possible.
        pred_sets["best_exp9_quantum_fusion"] = (None, best_test_pred["y_prob"].to_numpy(), 0.5)
    hard_masks_test = {
        "strong_abs_margin_lt_0.10": np.abs(p_strong_test - 0.5) < 0.10,
        "strong_abs_margin_lt_0.20": np.abs(p_strong_test - 0.5) < 0.20,
        "strong_lowest_confidence_30pct": np.abs(p_strong_test - 0.5) <= np.quantile(np.abs(p_strong_test - 0.5), 0.30),
        "strong_disagree_qsn2_r1_gt_0.20": np.abs(p_strong_test - q_test) > 0.20,
    }
    for pred_name, (_, p_test, threshold) in pred_sets.items():
        for subset, mask in hard_masks_test.items():
            row = subset_metrics(subset, y_test, p_test, threshold, mask)
            row["prediction"] = pred_name
            rows.append(row)

    gate_rows = []
    gate_feature_sets = {
        "gate_strong_qsn2_r1": ["p_strong_oof", "p_qsn2_r1"],
        "gate_strong_qsn2_r1_diff": ["p_strong_oof", "p_qsn2_r1"],
        "gate_strong_qsn2_original_r1": ["p_strong_oof", "p_qsn2_original", "p_qsn2_r1"],
    }
    for name, features in gate_feature_sets.items():
        raw_val = val[features].astype(float).to_numpy()
        raw_test = test.rename(columns={"p_strong_refit": "p_strong_oof"})[features].astype(float).to_numpy()
        if name.endswith("_diff"):
            raw_val = np.column_stack([raw_val, np.abs(raw_val[:, 0] - raw_val[:, 1]), raw_val[:, 0] * raw_val[:, 1]])
            raw_test = np.column_stack([raw_test, np.abs(raw_test[:, 0] - raw_test[:, 1]), raw_test[:, 0] * raw_test[:, 1]])
        for c in [0.1, 0.3, 1, 3]:
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, solver="lbfgs", max_iter=5000, class_weight="balanced"))
            p_val = oof_predict(model, raw_val, y_val)
            threshold = threshold_for_best_f1(y_val, p_val)
            model.fit(raw_val, y_val)
            p_test = model.predict_proba(raw_test)[:, 1]
            row = {"gate_model": name, "C": c, "threshold": threshold}
            row.update({f"val_{k}": v for k, v in binary_metrics(y_val, p_val, threshold).items()})
            row.update({f"test_{k}": v for k, v in binary_metrics(y_test, p_test, threshold).items()})
            row["delta_test_roc_vs_strong"] = row["test_roc_auc"] - STRONG_ROC_AUC
            row["delta_test_f1_vs_strong"] = row["test_f1"] - STRONG_F1
            gate_rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(gate_rows)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    val, test, manifest = load_dataset()
    pd.DataFrame(manifest).to_csv(RESULTS_DIR / "source_manifest.csv", index=False)

    exp9, exp9_best, (best_val_pred, best_test_pred) = experiment9(val, test)
    exp9.to_csv(RESULTS_DIR / "experiment9_per_run_results.csv", index=False)
    exp9_best.to_csv(RESULTS_DIR / "experiment9_best_by_validation.csv", index=False)
    if best_val_pred is not None:
        best_val_pred.to_csv(RESULTS_DIR / "best_validation_predictions.csv", index=False)
    if best_test_pred is not None:
        best_test_pred.to_csv(RESULTS_DIR / "best_test_predictions.csv", index=False)

    exp10 = experiment10(exp9)
    exp10.to_csv(RESULTS_DIR / "experiment10_operating_point_results.csv", index=False)
    exp11_hard, exp11_gate = experiment11(val, test, best_test_pred)
    exp11_hard.to_csv(RESULTS_DIR / "experiment11_hard_case_results.csv", index=False)
    exp11_gate.to_csv(RESULTS_DIR / "experiment11_gate_results.csv", index=False)

    best_row = exp9_best.iloc[0].to_dict() if len(exp9_best) else {}
    best_gate = exp11_gate.sort_values(["val_roc_auc", "val_f1"], ascending=False).head(1).iloc[0].to_dict() if len(exp11_gate) else {}
    summary = {
        "references": {
            "strong_fusion_roc_auc": STRONG_ROC_AUC,
            "strong_fusion_pr_auc": STRONG_PR_AUC,
            "strong_fusion_f1": STRONG_F1,
        },
        "experiment9_best_by_validation": best_row,
        "experiment11_best_gate_by_validation": best_gate,
        "interpretation_hint": (
            "Use validation-selected rows for primary claims. "
            "If ROC-AUC does not exceed strong fusion, check F1, sensitivity_at_specificity_90/95, LR metrics, and DCA."
        ),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("FINAL SUMMARY")
    if best_row:
        print(
            "Exp9 best:",
            best_row.get("feature_set"),
            best_row.get("model"),
            best_row.get("transform"),
            "pairwise=", best_row.get("pairwise"),
        )
        print(
            "test ROC-AUC / PR-AUC / F1:",
            best_row.get("test_roc_auc"),
            best_row.get("test_pr_auc"),
            best_row.get("test_f1"),
        )
        print(
            "delta vs strong ROC-AUC / PR-AUC / F1:",
            best_row.get("delta_test_roc_vs_strong"),
            best_row.get("delta_test_pr_vs_strong"),
            best_row.get("delta_test_f1_vs_strong"),
        )
    if best_gate:
        print(
            "Exp11 best gate:",
            best_gate.get("gate_model"),
            "test ROC-AUC / PR-AUC / F1:",
            best_gate.get("test_roc_auc"),
            best_gate.get("test_pr_auc"),
            best_gate.get("test_f1"),
        )
    print("results_dir=", RESULTS_DIR)


if __name__ == "__main__":
    main()
