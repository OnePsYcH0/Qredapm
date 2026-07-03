from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from config import (
    DROP_OUT,
    FUSION_HIDDEN_DIM,
    FUSION_TRANSFORMER_HEADS,
    FUSION_TRANSFORMER_LAYERS,
    IMPLEMENTATION_NAME,
    INCLUDE_DISEASE_NAMES,
    LABEL_COLUMN,
    MODEL_VARIANT,
    SEED,
    STRUCT_HIDDEN_DIM,
    STRUCT_NUM_LAYERS,
    TEST_FILE,
    TEXT_PROJECTION_DIM,
    TRAIN_FILE,
    VALIDATION_RATIO,
    VISIT_TRANSFORMER_HEADS,
    VISIT_TRANSFORMER_LAYERS,
)
from dataset import (
    RedapmJsonDataset,
    StandardScaler,
    infer_drug_code_dim,
    infer_feature_columns,
    load_jsonl,
    rows_to_tensors,
)
from model_redapm import REDAPM


def confusion_metrics(labels: List[int], scores: List[float], threshold: float) -> Dict[str, object]:
    tn = fp = fn = tp = 0
    for label, score in zip(labels, scores):
        pred = 1 if score >= threshold else 0
        if label == 1 and pred == 1:
            tp += 1
        elif label == 1 and pred == 0:
            fn += 1
        elif label == 0 and pred == 1:
            fp += 1
        else:
            tn += 1
    total = len(labels)
    positive_count = sum(labels)
    negative_count = total - positive_count
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": float(threshold),
        "sample_count": int(total),
        "positive_count": int(positive_count),
        "negative_count": int(negative_count),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "youden": float(recall + specificity - 1.0),
        "f1_score": float(f1_score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def auc_metrics(labels: List[int], scores: List[float]) -> Dict[str, float]:
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    positive_total = float(labels_tensor.sum().item())
    negative_total = float(labels_tensor.numel() - positive_total)
    if positive_total == 0.0 or negative_total == 0.0:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}

    sorted_scores, order = torch.sort(scores_tensor, descending=False)
    sorted_labels = labels_tensor[order]
    ranks = torch.arange(1, labels_tensor.numel() + 1, dtype=torch.float32)
    unique_scores, inverse, counts = torch.unique_consecutive(sorted_scores, return_inverse=True, return_counts=True)
    rank_sums = torch.zeros(len(unique_scores), dtype=torch.float32).scatter_add_(0, inverse, ranks)
    avg_ranks = rank_sums / counts.float()
    label_sums = torch.zeros(len(unique_scores), dtype=torch.float32).scatter_add_(0, inverse, sorted_labels)
    rank_sum_positive = float((avg_ranks * label_sums).sum().item())
    roc_auc = (rank_sum_positive - positive_total * (positive_total + 1.0) / 2.0) / (positive_total * negative_total)

    descending_order = torch.argsort(scores_tensor, descending=True)
    descending_labels = labels_tensor[descending_order]
    descending_scores = scores_tensor[descending_order]
    _, counts_desc = torch.unique_consecutive(descending_scores, return_counts=True)
    cumulative_counts = torch.cumsum(counts_desc, dim=0)
    group_ends = cumulative_counts - 1
    tp_cumulative = torch.cumsum(descending_labels, dim=0)[group_ends]
    fp_cumulative = cumulative_counts.float() - tp_cumulative
    precisions = tp_cumulative / (tp_cumulative + fp_cumulative).clamp_min(1.0)
    recalls = tp_cumulative / positive_total
    precisions = torch.cat([torch.tensor([1.0]), precisions])
    recalls = torch.cat([torch.tensor([0.0]), recalls])
    pr_auc = float(torch.trapz(precisions, recalls).item())
    return {"roc_auc": float(roc_auc), "pr_auc": pr_auc}


def compute_fast_metrics(labels: List[int], scores: List[float], threshold: float) -> Dict[str, object]:
    metrics = confusion_metrics(labels, scores, threshold)
    metrics.update(auc_metrics(labels, scores))
    return metrics


def find_best_threshold_fast(
    labels: List[int],
    scores: List[float],
    metric: str,
    min_precision: float,
    grid_size: int,
) -> Dict[str, object]:
    best_threshold = 0.5
    best_score = float("-inf")
    best_metrics: Dict[str, object] | None = None
    constraint_satisfied = True
    fallback_metrics = confusion_metrics(labels, scores, 0.5)

    for index in range(grid_size):
        threshold = index / (grid_size - 1)
        metrics = confusion_metrics(labels, scores, threshold)
        if metric == "f1":
            score = float(metrics["f1_score"])
        elif metric == "youden":
            score = float(metrics["youden"])
        elif metric == "balanced_accuracy":
            score = float(metrics["balanced_accuracy"])
        elif metric == "recall_at_precision":
            if float(metrics["precision"]) < min_precision:
                continue
            score = float(metrics["recall"])
        else:
            raise ValueError(f"Unsupported threshold metric: {metric}")

        if (
            score,
            float(metrics["f1_score"]),
            float(metrics["balanced_accuracy"]),
            -abs(threshold - 0.5),
        ) > (
            best_score,
            float(best_metrics["f1_score"]) if best_metrics else float("-inf"),
            float(best_metrics["balanced_accuracy"]) if best_metrics else float("-inf"),
            -abs(best_threshold - 0.5),
        ):
            best_threshold = threshold
            best_score = score
            best_metrics = metrics

    if best_metrics is None:
        constraint_satisfied = False
        best_metrics = fallback_metrics
        best_score = float(best_metrics["precision"]) if metric == "recall_at_precision" else float("-inf")

    return {
        "threshold": float(best_threshold),
        "score": float(best_score),
        "metric": metric,
        "min_precision": float(min_precision),
        "grid_size": int(grid_size),
        "constraint_satisfied": constraint_satisfied,
        "metrics": best_metrics,
    }


def split_indices(total_size: int, val_ratio: float, seed: int) -> tuple[List[int], List[int]]:
    import random

    indices = list(range(total_size))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(total_size * val_ratio))
    return indices[val_size:], indices[:val_size]


def collate_redapm_batch(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch], dim=0),
        "visit_mask": torch.stack([item["visit_mask"] for item in batch], dim=0),
        "features": torch.stack([item["features"] for item in batch], dim=0),
        "drug_code": torch.stack([item["drug_code"] for item in batch], dim=0),
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
    }


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {key: sanitize_for_json(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def build_dataloader(dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_redapm_batch,
        num_workers=0,
        pin_memory=False,
    )


@torch.inference_mode()
def predict_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split_name: str,
) -> tuple[List[int], List[float]]:
    model.eval()
    labels: List[int] = []
    scores: List[float] = []
    for batch in tqdm(loader, desc=f"Predict {split_name}", dynamic_ncols=True):
        batch = {key: value.to(device, non_blocking=False) for key, value in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            visit_mask=batch["visit_mask"],
            features=batch["features"],
            drug_code=batch["drug_code"],
        )
        batch_scores = torch.sigmoid(outputs["logits"]).detach().cpu().tolist()
        batch_labels = batch["labels"].detach().cpu().int().tolist()
        scores.extend(float(score) for score in batch_scores)
        labels.extend(int(label) for label in batch_labels)
    return labels, scores


def save_predictions_csv(path: Path, labels: List[int], scores: List[float], selected_threshold: float) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "y_true", "y_score", "y_pred_0_5", "y_pred_selected_threshold", "y_pred"])
        for index, (label, score) in enumerate(zip(labels, scores)):
            pred_0_5 = 1 if score >= 0.5 else 0
            pred_selected = 1 if score >= selected_threshold else 0
            writer.writerow([index, label, f"{score:.10f}", pred_0_5, pred_selected, pred_selected])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved faithful REDAPM best_model.pt without training.")
    parser.add_argument("--train-file", default=str(TRAIN_FILE))
    parser.add_argument("--test-file", default=str(TEST_FILE))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=400)
    parser.add_argument("--max-visits", type=int, default=12)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--validation-ratio", type=float, default=VALIDATION_RATIO)
    parser.add_argument("--threshold-metric", choices=["f1", "youden", "balanced_accuracy", "recall_at_precision"], default="f1")
    parser.add_argument("--min-precision", type=float, default=0.6)
    parser.add_argument("--threshold-grid-size", type=int, default=101)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    print(f"device={device}", flush=True)
    print(f"checkpoint={args.checkpoint}", flush=True)
    print(f"output_dir={output_dir.resolve()}", flush=True)

    train_rows = load_jsonl(args.train_file)
    test_rows = load_jsonl(args.test_file)
    feature_columns = infer_feature_columns(train_rows)
    drug_code_dim = infer_drug_code_dim(train_rows)
    train_features, _ = rows_to_tensors(train_rows, feature_columns, label_column=LABEL_COLUMN)
    feature_scaler = StandardScaler.fit(train_features)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    full_train_dataset = RedapmJsonDataset(
        train_rows,
        feature_columns=feature_columns,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_visits=args.max_visits,
        drug_code_dim=drug_code_dim,
        scaler=feature_scaler,
        include_disease_names=INCLUDE_DISEASE_NAMES,
    )
    test_dataset = RedapmJsonDataset(
        test_rows,
        feature_columns=feature_columns,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_visits=args.max_visits,
        drug_code_dim=drug_code_dim,
        scaler=feature_scaler,
        include_disease_names=INCLUDE_DISEASE_NAMES,
    )
    _, val_indices = split_indices(len(full_train_dataset), args.validation_ratio, args.seed)
    val_loader = build_dataloader(Subset(full_train_dataset, val_indices), batch_size=args.batch_size)
    test_loader = build_dataloader(test_dataset, batch_size=args.batch_size)

    model = REDAPM(
        model_name=args.model_name,
        structured_input_dim=len(feature_columns),
        drug_code_dim=drug_code_dim,
        struct_hidden_dim=STRUCT_HIDDEN_DIM,
        text_projection_dim=TEXT_PROJECTION_DIM,
        fusion_hidden_dim=FUSION_HIDDEN_DIM,
        struct_num_layers=STRUCT_NUM_LAYERS,
        max_visits=args.max_visits,
        visit_transformer_layers=VISIT_TRANSFORMER_LAYERS,
        visit_transformer_heads=VISIT_TRANSFORMER_HEADS,
        fusion_transformer_layers=FUSION_TRANSFORMER_LAYERS,
        fusion_transformer_heads=FUSION_TRANSFORMER_HEADS,
        dropout=DROP_OUT,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_epoch = int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0)))
    print(f"loaded_best_epoch={best_epoch}", flush=True)

    val_labels, val_scores = predict_scores(model, val_loader, device, "validation")
    print("Finding threshold", flush=True)
    threshold_result = find_best_threshold_fast(
        val_labels,
        val_scores,
        metric=args.threshold_metric,
        min_precision=args.min_precision,
        grid_size=args.threshold_grid_size,
    )
    selected_threshold = float(threshold_result["threshold"])
    print(f"selected_threshold={selected_threshold}", flush=True)

    test_labels, test_scores = predict_scores(model, test_loader, device, "test")
    print("Computing final metrics", flush=True)
    val_metrics_0_5 = compute_fast_metrics(val_labels, val_scores, threshold=0.5)
    val_metrics_selected = compute_fast_metrics(val_labels, val_scores, threshold=selected_threshold)
    test_metrics_0_5 = compute_fast_metrics(test_labels, test_scores, threshold=0.5)
    test_metrics_selected = compute_fast_metrics(test_labels, test_scores, threshold=selected_threshold)

    print("Writing outputs", flush=True)
    save_predictions_csv(output_dir / "validation_predictions.csv", val_labels, val_scores, selected_threshold)
    save_predictions_csv(output_dir / "predictions.csv", test_labels, test_scores, selected_threshold)

    metrics_payload = {
        "implementation_name": IMPLEMENTATION_NAME,
        "model_variant": MODEL_VARIANT,
        "run_timestamp": datetime.now().astimezone().isoformat(),
        "eval_script": str(Path(__file__).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model_name": args.model_name,
        "device": str(device),
        "best_epoch": best_epoch,
        "max_length": args.max_length,
        "max_visits": args.max_visits,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "validation_ratio": args.validation_ratio,
        "drug_code_used": drug_code_dim > 0,
        "drug_code_dim": drug_code_dim,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "selected_threshold": selected_threshold,
        "threshold_tuning": {
            "enabled": True,
            "metric": args.threshold_metric,
            "min_precision": args.min_precision,
            "grid_size": args.threshold_grid_size,
            "selection_source": "validation",
            "constraint_satisfied": threshold_result["constraint_satisfied"],
            "selected_score": threshold_result["score"],
        },
        "best_validation_metrics_at_0_5": val_metrics_0_5,
        "best_validation_metrics_at_selected_threshold": val_metrics_selected,
        "test_metrics_at_0_5": test_metrics_0_5,
        "test_metrics_at_selected_threshold": test_metrics_selected,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(sanitize_for_json(metrics_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_path = output_dir / "experiment_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "best_epoch",
                "selected_threshold",
                "val_roc_auc",
                "val_pr_auc",
                "test_roc_auc",
                "test_pr_auc",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_selected_accuracy",
                "test_selected_precision",
                "test_selected_recall",
                "test_selected_f1",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "best_epoch": best_epoch,
                "selected_threshold": selected_threshold,
                "val_roc_auc": val_metrics_0_5["roc_auc"],
                "val_pr_auc": val_metrics_0_5["pr_auc"],
                "test_roc_auc": test_metrics_0_5["roc_auc"],
                "test_pr_auc": test_metrics_0_5["pr_auc"],
                "test_accuracy": test_metrics_0_5["accuracy"],
                "test_precision": test_metrics_0_5["precision"],
                "test_recall": test_metrics_0_5["recall"],
                "test_f1": test_metrics_0_5["f1_score"],
                "test_selected_accuracy": test_metrics_selected["accuracy"],
                "test_selected_precision": test_metrics_selected["precision"],
                "test_selected_recall": test_metrics_selected["recall"],
                "test_selected_f1": test_metrics_selected["f1_score"],
            }
        )

    print(f"Test ROC-AUC: {test_metrics_0_5['roc_auc']}", flush=True)
    print(f"Test PR-AUC: {test_metrics_0_5['pr_auc']}", flush=True)
    print(f"Test selected F1: {test_metrics_selected['f1_score']}", flush=True)
    print("Evaluation results saved.", flush=True)


if __name__ == "__main__":
    main()
