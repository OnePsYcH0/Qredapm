from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

import dataset as dataset_module
import model_redapm as model_redapm_module
from config import (
    AMP_DTYPE,
    AMP_ENABLED,
    BATCH_SIZE,
    DROP_OUT,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    FUSION_HIDDEN_DIM,
    FUSION_TRANSFORMER_HEADS,
    FUSION_TRANSFORMER_LAYERS,
    IMPLEMENTATION_NAME,
    INCLUDE_DISEASE_NAMES,
    LABEL_COLUMN,
    LEARNING_RATE,
    MAX_LENGTH,
    MAX_VISITS,
    MODEL_NAME,
    MODEL_VARIANT,
    NUM_WORKERS,
    OUTPUT_DIR,
    PIN_MEMORY,
    PREFETCH_FACTOR,
    PERSISTENT_WORKERS,
    SEED,
    STRUCT_HIDDEN_DIM,
    STRUCT_NUM_LAYERS,
    TEST_FILE,
    TEXT_PROJECTION_DIM,
    TF32_ENABLED,
    TRAIN_FILE,
    VALIDATION_RATIO,
    VISIT_TRANSFORMER_HEADS,
    VISIT_TRANSFORMER_LAYERS,
    WEIGHT_DECAY,
)
from dataset import (
    RedapmJsonDataset,
    StandardScaler,
    infer_drug_code_dim,
    infer_feature_columns,
    load_jsonl,
    rows_to_tensors,
)
from metrics import compute_binary_metrics
from model_redapm import REDAPM


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


def collate_redapm_batch(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch], dim=0),
        "visit_mask": torch.stack([item["visit_mask"] for item in batch], dim=0),
        "features": torch.stack([item["features"] for item in batch], dim=0),
        "drug_code": torch.stack([item["drug_code"] for item in batch], dim=0),
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
    }


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, object]:
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            visit_mask=batch["visit_mask"],
            features=batch["features"],
            drug_code=batch["drug_code"],
        )
        all_logits.append(outputs["logits"].detach().cpu())
        all_labels.append(batch["labels"].detach().cpu())

    logits_tensor = torch.cat(all_logits)
    labels_tensor = torch.cat(all_labels)
    scores = torch.sigmoid(logits_tensor).tolist()
    labels = labels_tensor.int().tolist()
    predictions = [1 if score >= 0.5 else 0 for score in scores]
    try:
        metrics = compute_binary_metrics(labels, scores)
    except ValueError:
        positive_count = sum(labels)
        negative_count = len(labels) - positive_count
        tp = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 1)
        tn = sum(1 for label, pred in zip(labels, predictions) if label == 0 and pred == 0)
        fp = sum(1 for label, pred in zip(labels, predictions) if label == 0 and pred == 1)
        fn = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 0)
        accuracy = (tp + tn) / len(labels)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        metrics = {
            "threshold": 0.5,
            "sample_count": len(labels),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
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


def append_log_line(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def log_and_print(path: Path, message: str) -> None:
    print(message, flush=True)
    append_log_line(path, message)


def resolve_amp_dtype(device: torch.device, amp_enabled: bool, amp_dtype_name: str) -> tuple[bool, torch.dtype | None]:
    if not amp_enabled or device.type != "cuda":
        return False, None
    if amp_dtype_name == "bf16":
        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16
        return True, torch.float16
    if amp_dtype_name == "fp16":
        return True, torch.float16
    raise ValueError(f"不支持的 amp dtype: {amp_dtype_name}")


def get_autocast_context(device: torch.device, amp_enabled: bool, amp_dtype: torch.dtype | None):
    if amp_enabled and device.type == "cuda" and amp_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def create_grad_scaler(device: torch.device, amp_enabled: bool, amp_dtype: torch.dtype | None):
    if amp_enabled and device.type == "cuda" and amp_dtype == torch.float16:
        return torch.amp.GradScaler("cuda")
    return None


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device, non_blocking: bool) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=non_blocking) for key, value in batch.items()}


def build_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    collate_fn,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
):
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**loader_kwargs)


def ranking_metric(metrics: Dict[str, object]) -> float:
    roc_auc = float(metrics["roc_auc"])
    if not math.isnan(roc_auc):
        return roc_auc
    return float("-inf")


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


def run_single_batch_smoke_test(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    scaler,
    non_blocking: bool,
) -> None:
    model.train()
    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device, non_blocking=non_blocking)
    with get_autocast_context(device, amp_enabled=amp_enabled, amp_dtype=amp_dtype):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            visit_mask=batch["visit_mask"],
            features=batch["features"],
            drug_code=batch["drug_code"],
        )
        loss = criterion(outputs["logits"], batch["labels"])
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    model.zero_grad(set_to_none=True)


def build_source_check() -> Dict[str, str]:
    return {
        "model_file": str(Path(model_redapm_module.__file__).resolve()),
        "dataset_file": str(Path(dataset_module.__file__).resolve()),
        "train_script": str(Path(__file__).resolve()),
    }


def save_experiment_summary(path: Path, row: Dict[str, object]) -> None:
    fieldnames = [
        "timestamp",
        "implementation_name",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "max_length",
        "max_visits",
        "dropout",
        "early_stop_patience",
        "best_epoch",
        "best_val_roc_auc",
        "best_val_pr_auc",
        "test_roc_auc",
        "test_pr_auc",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
    ]
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="训练 faithful REDAPM 二分类模型。")
    parser.add_argument("--train-file", default=str(TRAIN_FILE))
    parser.add_argument("--test-file", default=str(TEST_FILE))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--max-visits", type=int, default=MAX_VISITS)
    parser.add_argument("--dropout", type=float, default=DROP_OUT)
    parser.add_argument("--early-stop-patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--prefetch-factor", type=int, default=PREFETCH_FACTOR)
    parser.add_argument("--pin-memory", type=lambda x: str(x).lower() == "true", default=PIN_MEMORY)
    parser.add_argument("--persistent-workers", type=lambda x: str(x).lower() == "true", default=PERSISTENT_WORKERS)
    parser.add_argument("--amp", type=lambda x: str(x).lower() == "true", default=AMP_ENABLED)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default=AMP_DTYPE)
    parser.add_argument("--tf32", type=lambda x: str(x).lower() == "true", default=TF32_ENABLED)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--validation-ratio", type=float, default=VALIDATION_RATIO)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    if output_dir.resolve() == (Path(__file__).resolve().parent / "redapm_outputs").resolve():
        raise ValueError("请不要再写入旧目录 redapm_outputs。请使用 faithful_redapm_outputs 或其他全新目录。")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.txt"
    if not args.resume_checkpoint and log_path.exists():
        log_path.unlink()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.backends.cudnn.benchmark = True
    run_timestamp = datetime.now().astimezone().isoformat()
    source_check = build_source_check()
    amp_enabled, amp_dtype = resolve_amp_dtype(device, amp_enabled=args.amp, amp_dtype_name=args.amp_dtype)
    scaler = create_grad_scaler(device, amp_enabled=amp_enabled, amp_dtype=amp_dtype)
    amp_dtype_name = str(amp_dtype).replace("torch.", "") if amp_dtype is not None else "disabled"
    non_blocking = device.type == "cuda" and args.pin_memory

    log_and_print(log_path, f"implementation_name={IMPLEMENTATION_NAME}")
    log_and_print(log_path, f"model_variant={MODEL_VARIANT}")
    log_and_print(log_path, f"output_dir={output_dir.resolve()}")
    log_and_print(log_path, f"train_script={source_check['train_script']}")
    log_and_print(log_path, f"model_file={source_check['model_file']}")
    log_and_print(log_path, f"dataset_file={source_check['dataset_file']}")
    log_and_print(log_path, f"device={device}")
    log_and_print(log_path, f"model_name={args.model_name}")
    log_and_print(log_path, "per_visit_tokenization=True")
    log_and_print(log_path, f"max_visits={args.max_visits}")
    log_and_print(log_path, f"dropout={args.dropout}")
    log_and_print(log_path, f"early_stop_patience={args.early_stop_patience}")
    log_and_print(log_path, f"amp_enabled={amp_enabled}")
    log_and_print(log_path, f"amp_dtype={amp_dtype_name}")
    log_and_print(log_path, f"num_workers={args.num_workers}")
    log_and_print(log_path, f"pin_memory={args.pin_memory}")
    log_and_print(log_path, f"persistent_workers={args.persistent_workers}")
    log_and_print(log_path, f"prefetch_factor={args.prefetch_factor if args.num_workers > 0 else 'disabled'}")
    log_and_print(log_path, f"tf32={args.tf32}")

    train_rows = load_jsonl(args.train_file)
    test_rows = load_jsonl(args.test_file)
    if args.max_train_samples is not None:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_test_samples is not None:
        test_rows = test_rows[: args.max_test_samples]

    feature_columns = infer_feature_columns(train_rows)
    drug_code_dim = infer_drug_code_dim(train_rows)
    drug_code_used = drug_code_dim > 0
    log_and_print(log_path, f"drug_code_used={drug_code_used}")
    log_and_print(log_path, f"drug_code_dim={drug_code_dim}")

    train_features, _ = rows_to_tensors(train_rows, feature_columns, label_column=LABEL_COLUMN)
    scaler = StandardScaler.fit(train_features)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    full_train_dataset = RedapmJsonDataset(
        train_rows,
        feature_columns=feature_columns,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_visits=args.max_visits,
        drug_code_dim=drug_code_dim,
        scaler=scaler,
        include_disease_names=INCLUDE_DISEASE_NAMES,
    )
    test_dataset = RedapmJsonDataset(
        test_rows,
        feature_columns=feature_columns,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_visits=args.max_visits,
        drug_code_dim=drug_code_dim,
        scaler=scaler,
        include_disease_names=INCLUDE_DISEASE_NAMES,
    )

    train_indices, val_indices = split_indices(len(full_train_dataset), args.validation_ratio, args.seed)
    train_loader = build_dataloader(
        Subset(full_train_dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_redapm_batch,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = build_dataloader(
        Subset(full_train_dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_redapm_batch,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    test_loader = build_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_redapm_batch,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

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
        dropout=args.dropout,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    start_epoch = 1
    best_state = None
    best_selection_score = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_selection_score = float(checkpoint.get("best_val_auc", float("-inf")))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        history = checkpoint.get("training_history", [])
        best_state = checkpoint
        log_and_print(log_path, f"Resuming training from epoch {start_epoch}")

    run_single_batch_smoke_test(
        model,
        train_loader,
        criterion,
        device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        scaler=scaler,
        non_blocking=non_blocking,
    )
    log_and_print(log_path, "single_batch_smoke_test=passed")
    log_and_print(log_path, "Starting training")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        sample_count = 0
        log_and_print(log_path, f"Epoch {epoch} started")
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}",
            dynamic_ncols=True,
            file=sys.stdout,
            mininterval=0.2,
        )

        for step, batch in enumerate(progress_bar, start=1):
            batch = move_batch_to_device(batch, device, non_blocking=non_blocking)
            optimizer.zero_grad()

            with get_autocast_context(device, amp_enabled=amp_enabled, amp_dtype=amp_dtype):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    visit_mask=batch["visit_mask"],
                    features=batch["features"],
                    drug_code=batch["drug_code"],
                )
                loss = criterion(outputs["logits"], batch["labels"])

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            batch_size = batch["labels"].shape[0]
            total_loss += loss.item() * batch_size
            sample_count += batch_size
            current_lr = optimizer.param_groups[0]["lr"]
            progress_bar.set_postfix(
                batch=f"{step}/{len(train_loader)}",
                loss=f"{loss.item():.4f}",
                lr=f"{current_lr:.2e}",
            )

        train_loss = total_loss / max(sample_count, 1)
        log_and_print(log_path, "Validation running")
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

        log_message = (
            f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | "
            f"val_roc_auc={val_metrics['roc_auc']:.6f} | "
            f"val_pr_auc={val_metrics['pr_auc']:.6f}"
        )
        log_and_print(log_path, log_message)

        current_rank = ranking_metric(val_metrics)
        if best_state is None or current_rank > best_selection_score:
            best_selection_score = current_rank
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_val_auc": best_selection_score,
                "best_validation_metrics": val_metrics,
                "scaler": scaler.state_dict(),
                "feature_columns": feature_columns,
                "training_history": history,
                "config": {
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "max_length": args.max_length,
                    "max_visits": args.max_visits,
                    "dropout": args.dropout,
                    "early_stop_patience": args.early_stop_patience,
                    "num_workers": args.num_workers,
                    "pin_memory": args.pin_memory,
                    "persistent_workers": args.persistent_workers,
                    "prefetch_factor": args.prefetch_factor,
                    "amp_enabled": amp_enabled,
                    "amp_dtype": amp_dtype_name,
                    "tf32": args.tf32,
                    "seed": args.seed,
                    "validation_ratio": args.validation_ratio,
                    "include_disease_names": INCLUDE_DISEASE_NAMES,
                    "implementation_name": IMPLEMENTATION_NAME,
                    "model_variant": MODEL_VARIANT,
                    "output_dir": str(output_dir.resolve()),
                    "resume_checkpoint": args.resume_checkpoint,
                },
            }
            torch.save(best_state, output_dir / "best_model.pt")
            log_and_print(log_path, "Best model updated")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.early_stop_patience:
            log_and_print(log_path, "Early stopping triggered")
            break

    if best_state is None:
        raise RuntimeError("训练过程中没有得到可保存的最佳模型。")

    log_and_print(log_path, "Testing best model")
    model.load_state_dict(best_state["model_state_dict"])
    test_result = evaluate_model(model, test_loader, device)

    log_and_print(log_path, "Saving results")
    torch.save(best_state, output_dir / "best_model.pt")
    save_predictions_csv(
        output_dir / "predictions.csv",
        labels=test_result["labels"],
        scores=test_result["scores"],
        predictions=test_result["predictions"],
    )

    best_validation_metrics = best_state["best_validation_metrics"]
    metrics_payload = {
        "implementation_name": IMPLEMENTATION_NAME,
        "model_variant": MODEL_VARIANT,
        "per_visit_tokenization": True,
        "max_visits": args.max_visits,
        "drug_code_used": drug_code_used,
        "visit_transformer_layers": VISIT_TRANSFORMER_LAYERS,
        "fusion_transformer_layers": FUSION_TRANSFORMER_LAYERS,
        "output_dir": str(output_dir.resolve()),
        "run_timestamp": run_timestamp,
        "source_check": source_check,
        "device": str(device),
        "amp_enabled": amp_enabled,
        "amp_dtype": amp_dtype_name,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers,
        "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
        "tf32": args.tf32,
        "label_column": LABEL_COLUMN,
        "model_name": args.model_name,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "drug_code_dim": drug_code_dim,
        "text_builder": {
            "include_disease_names": INCLUDE_DISEASE_NAMES,
            "template": "诊断信息：{disease_names_i}。就诊记录：{visit_sn_i}",
            "max_visits": args.max_visits,
            "per_visit_tokenization": True,
        },
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "best_epoch": best_epoch,
        "best_validation_selection_score": best_selection_score,
        "best_validation_metrics": best_validation_metrics,
        "test_metrics": test_result["metrics"],
        "training_history": history,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(sanitize_for_json(metrics_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    run_identity_payload = {
        "implementation_name": IMPLEMENTATION_NAME,
        "output_dir": str(output_dir.resolve()),
        "model_variant": MODEL_VARIANT,
        "per_visit_tokenization": True,
        "max_visits": args.max_visits,
        "drug_code_used": drug_code_used,
        "run_timestamp": run_timestamp,
        "source_check": source_check,
        "best_validation_roc_auc": best_validation_metrics["roc_auc"],
        "test_roc_auc": test_result["metrics"]["roc_auc"],
        "test_pr_auc": test_result["metrics"]["pr_auc"],
    }
    (output_dir / "run_identity.json").write_text(
        json.dumps(sanitize_for_json(run_identity_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    save_experiment_summary(
        output_dir / "experiment_summary.csv",
        {
            "timestamp": run_timestamp,
            "implementation_name": IMPLEMENTATION_NAME,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_length": args.max_length,
            "max_visits": args.max_visits,
            "dropout": args.dropout,
            "early_stop_patience": args.early_stop_patience,
            "best_epoch": best_epoch,
            "best_val_roc_auc": sanitize_for_json(best_validation_metrics["roc_auc"]),
            "best_val_pr_auc": sanitize_for_json(best_validation_metrics["pr_auc"]),
            "test_roc_auc": test_result["metrics"]["roc_auc"],
            "test_pr_auc": test_result["metrics"]["pr_auc"],
            "test_accuracy": test_result["metrics"]["accuracy"],
            "test_precision": test_result["metrics"]["precision"],
            "test_recall": test_result["metrics"]["recall"],
            "test_f1": test_result["metrics"]["f1_score"],
        },
    )

    append_log_line(log_path, json.dumps(sanitize_for_json(metrics_payload["test_metrics"]), ensure_ascii=False))
    append_log_line(log_path, json.dumps(sanitize_for_json(run_identity_payload), ensure_ascii=False))

    best_val_roc_auc = sanitize_for_json(best_validation_metrics["roc_auc"])
    best_val_pr_auc = sanitize_for_json(best_validation_metrics["pr_auc"])
    test_roc_auc = test_result["metrics"]["roc_auc"]
    test_pr_auc = test_result["metrics"]["pr_auc"]
    surpass_old = test_roc_auc > 0.796
    reached_080 = test_roc_auc >= 0.80

    summary_lines = [
        f"Best validation ROC-AUC: {best_val_roc_auc}",
        f"Best validation PR-AUC: {best_val_pr_auc}",
        f"Best epoch: {best_epoch}",
        f"Test ROC-AUC: {test_roc_auc}",
        f"Test PR-AUC: {test_pr_auc}",
        f"Exceeded old REDAPM 0.796: {surpass_old}",
        f"Reached 0.80 ROC-AUC: {reached_080}",
    ]
    for line in summary_lines:
        log_and_print(log_path, line)

    print(json.dumps(sanitize_for_json(metrics_payload["test_metrics"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
