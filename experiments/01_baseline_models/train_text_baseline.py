from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import AutoModel, AutoTokenizer

from dataset import LABEL_COLUMN, TextOnlyJsonDataset, build_text_only_input, load_jsonl
from metrics import compute_binary_metrics


DEFAULT_TRAIN_FILE = "data/train_0.json"
DEFAULT_TEST_FILE = "data/test_0.json"
DEFAULT_OUTPUT_DIR = "outputs/baseline"
DEFAULT_MODEL_NAME = "bert-base-chinese"


class BertTextClassifier(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = self.dropout(pooled)
        return self.classifier(pooled).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(total_size: int, val_ratio: float, seed: int) -> tuple[List[int], List[int]]:
    indices = list(range(total_size))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(total_size * val_ratio))
    return indices[val_size:], indices[:val_size]


def collate_text_batch(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch], dim=0),
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
    }


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, object]:
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        all_logits.append(logits.detach().cpu())
        all_labels.append(batch["labels"].detach().cpu())

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


def save_predictions_csv(path: Path, labels, scores, predictions) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "y_true", "y_score", "y_pred"])
        for index, (truth, score, pred) in enumerate(zip(labels, scores, predictions)):
            writer.writerow([index, int(truth), f"{float(score):.10f}", int(pred)])


def append_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def build_comparison_csv(project_root: Path) -> None:
    mapping = {
        "weak_baseline": project_root / "weak_baseline_outputs" / "metrics.json",
        "text_baseline": project_root / "text_baseline_outputs" / "metrics.json",
        "redapm": project_root / "redapm_outputs" / "metrics.json",
    }
    rows = ["model,roc_auc,pr_auc,accuracy,precision,recall,f1"]
    for model_name, path in mapping.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["test_metrics"]
        rows.append(
            ",".join(
                [
                    model_name,
                    str(metrics["roc_auc"]),
                    str(metrics["pr_auc"]),
                    str(metrics["accuracy"]),
                    str(metrics["precision"]),
                    str(metrics["recall"]),
                    str(metrics["f1_score"]),
                ]
            )
        )

    content = "\n".join(rows)
    (project_root / "model_comparison.csv").write_text(content, encoding="utf-8")
    (project_root / "text_baseline_outputs" / "model_comparison.csv").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 BERT text-only baseline。")
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.txt"
    if log_path.exists():
        log_path.unlink()

    project_root = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    append_log(log_path, f"device={device}")
    append_log(log_path, f"model_name={args.model_name}")

    train_rows = load_jsonl(args.train_file)
    test_rows = load_jsonl(args.test_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = TextOnlyJsonDataset(train_rows, tokenizer=tokenizer, max_length=args.max_length)
    test_dataset = TextOnlyJsonDataset(test_rows, tokenizer=tokenizer, max_length=args.max_length)

    train_indices, val_indices = split_indices(len(train_dataset), args.validation_ratio, args.seed)
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_text_batch,
    )
    val_loader = DataLoader(
        Subset(train_dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_text_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_text_batch,
    )

    model = BertTextClassifier(args.model_name).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    smoke_batch = next(iter(train_loader))
    smoke_batch = {key: value.to(device) for key, value in smoke_batch.items()}
    smoke_logits = model(smoke_batch["input_ids"], smoke_batch["attention_mask"])
    smoke_loss = criterion(smoke_logits, smoke_batch["labels"])
    smoke_loss.backward()
    model.zero_grad(set_to_none=True)
    append_log(log_path, "single_batch_smoke_test=passed")

    best_state = None
    best_val_auc = -1.0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()

            current_batch_size = batch["labels"].shape[0]
            running_loss += loss.item() * current_batch_size
            sample_count += current_batch_size

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
        message = (
            f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | "
            f"val_roc_auc={val_metrics['roc_auc']:.6f} | "
            f"val_pr_auc={val_metrics['pr_auc']:.6f}"
        )
        print(message)
        append_log(log_path, message)

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = float(val_metrics["roc_auc"])
            best_state = {
                "model_state_dict": model.state_dict(),
                "best_validation_roc_auc": best_val_auc,
            }

    if best_state is None:
        raise RuntimeError("训练过程中没有得到最佳模型。")

    model.load_state_dict(best_state["model_state_dict"])
    test_result = evaluate_model(model, test_loader, device)

    metrics_payload = {
        "model_name": args.model_name,
        "label_column": LABEL_COLUMN,
        "device": str(device),
        "text_fields": ["disease_names", "drug_names"],
        "drug_names_present_in_data": any("drug_names" in row for row in train_rows),
        "text_template": "disease_names + ' ' + drug_names",
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "best_validation_roc_auc": best_val_auc,
        "test_metrics": test_result["metrics"],
        "notes": "Text-only baseline：仅使用 disease_names 和 drug_names。当前数据中未发现 drug_names 字段时，会自动退化为仅使用 disease_names。",
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_predictions_csv(output_dir / "predictions.csv", test_result["labels"], test_result["scores"], test_result["predictions"])
    append_log(log_path, json.dumps(metrics_payload["test_metrics"], ensure_ascii=False))

    build_comparison_csv(project_root)
    print(json.dumps(metrics_payload["test_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
