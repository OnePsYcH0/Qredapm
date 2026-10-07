from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from dataset import (
    LABEL_COLUMN,
    StandardScaler,
    TabularJsonDataset,
    build_dataset_from_rows,
    infer_feature_columns,
    load_jsonl,
    rows_to_tensors,
)
from metrics import compute_binary_metrics
from model_baseline import BaselineMLP


DEFAULT_TRAIN_PATH = "data/train_0.json"
DEFAULT_TEST_PATH = "data/test_0.json"
DEFAULT_OUTPUT_DIR = "outputs/baseline"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_train_validation(dataset: TabularJsonDataset, val_ratio: float, seed: int) -> tuple[Subset, Subset]:
    total_size = len(dataset)
    indices = list(range(total_size))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_size = max(1, int(total_size * val_ratio))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, object]:
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    logits_tensor = torch.cat(all_logits)
    labels_tensor = torch.cat(all_labels)
    scores = torch.sigmoid(logits_tensor).tolist()
    labels = labels_tensor.int().tolist()
    predictions = [1 if score >= 0.5 else 0 for score in scores]
    metrics = compute_binary_metrics(labels, scores)
    return {
        "metrics": metrics,
        "labels": labels,
        "scores": scores,
        "predictions": predictions,
    }


def save_predictions_csv(
    path: Path,
    labels: List[int],
    scores: List[float],
    predictions: List[int],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "y_true", "y_score", "y_pred"])
        for index, (label, score, pred) in enumerate(zip(labels, scores, predictions)):
            writer.writerow([index, label, f"{score:.10f}", pred])


def main() -> None:
    parser = argparse.ArgumentParser(description="训练仅使用结构化特征的 MLP baseline。")
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_PATH, help="训练集 JSON Lines 文件路径。")
    parser.add_argument("--test-file", default=DEFAULT_TEST_PATH, help="测试集 JSON Lines 文件路径。")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="输出目录。")
    parser.add_argument("--batch-size", type=int, default=32, help="训练 batch size。")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数。")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="学习率。")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout 比例。")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="从训练集切出的验证集比例。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(args.train_file)
    test_rows = load_jsonl(args.test_file)
    feature_columns = infer_feature_columns(train_rows)

    train_features, train_labels = rows_to_tensors(train_rows, feature_columns, label_column=LABEL_COLUMN)
    scaler = StandardScaler.fit(train_features)

    full_train_dataset, _, _ = build_dataset_from_rows(train_rows, feature_columns, scaler=scaler)
    test_dataset, _, _ = build_dataset_from_rows(test_rows, feature_columns, scaler=scaler)
    train_subset, val_subset = split_train_validation(full_train_dataset, args.val_ratio, args.seed)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = BaselineMLP(input_dim=len(feature_columns), dropout=args.dropout).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    best_state = None
    best_val_auc = -1.0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.shape[0]
            running_loss += loss.item() * batch_size
            sample_count += batch_size

        train_loss = running_loss / max(1, sample_count)
        val_result = evaluate_model(model, val_loader, device)
        val_metrics = val_result["metrics"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_roc_auc": float(val_metrics["roc_auc"]),
                "val_pr_auc": float(val_metrics["pr_auc"]),
            }
        )
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_roc_auc={val_metrics['roc_auc']:.6f} | "
            f"val_pr_auc={val_metrics['pr_auc']:.6f}"
        )

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = float(val_metrics["roc_auc"])
            best_state = {
                "model_state_dict": model.state_dict(),
                "feature_columns": list(feature_columns),
                "scaler": scaler.state_dict(),
                "best_val_metrics": val_metrics,
                "config": {
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "learning_rate": args.learning_rate,
                    "dropout": args.dropout,
                    "seed": args.seed,
                    "val_ratio": args.val_ratio,
                },
            }

    if best_state is None:
        raise RuntimeError("训练过程中没有得到可保存的最佳模型。")

    model.load_state_dict(best_state["model_state_dict"])
    test_result = evaluate_model(model, test_loader, device)

    torch.save(best_state, output_dir / "best_model.pt")
    save_predictions_csv(
        output_dir / "predictions.csv",
        labels=test_result["labels"],
        scores=test_result["scores"],
        predictions=test_result["predictions"],
    )

    metrics_payload = {
        "device": str(device),
        "label_column": LABEL_COLUMN,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "best_validation_roc_auc": best_val_auc,
        "test_metrics": test_result["metrics"],
        "training_history": history,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metrics_payload["test_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
