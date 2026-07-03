from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from config import *
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
from train_redapm import collate_redapm_batch, sanitize_for_json, save_predictions_csv


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("faithful_redapm_outputs")
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)

    train_rows = load_jsonl(TRAIN_FILE)
    test_rows = load_jsonl(TEST_FILE)
    feature_columns = checkpoint.get("feature_columns") or infer_feature_columns(train_rows)
    drug_code_dim = infer_drug_code_dim(train_rows)

    train_features, _ = rows_to_tensors(train_rows, feature_columns, label_column=LABEL_COLUMN)
    feature_scaler = StandardScaler.fit(train_features)

    tokenizer = AutoTokenizer.from_pretrained("./hf_models/bert-base-chinese")
    test_dataset = RedapmJsonDataset(
        test_rows,
        feature_columns=feature_columns,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        max_visits=MAX_VISITS,
        drug_code_dim=drug_code_dim,
        scaler=feature_scaler,
        include_disease_names=INCLUDE_DISEASE_NAMES,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_redapm_batch,
    )

    model = REDAPM(
        model_name="./hf_models/bert-base-chinese",
        structured_input_dim=len(feature_columns),
        drug_code_dim=drug_code_dim,
        struct_hidden_dim=STRUCT_HIDDEN_DIM,
        text_projection_dim=TEXT_PROJECTION_DIM,
        fusion_hidden_dim=FUSION_HIDDEN_DIM,
        struct_num_layers=STRUCT_NUM_LAYERS,
        max_visits=MAX_VISITS,
        visit_transformer_layers=VISIT_TRANSFORMER_LAYERS,
        visit_transformer_heads=VISIT_TRANSFORMER_HEADS,
        fusion_transformer_layers=FUSION_TRANSFORMER_LAYERS,
        fusion_transformer_heads=FUSION_TRANSFORMER_HEADS,
        dropout=DROP_OUT,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    labels = []
    scores = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    visit_mask=batch["visit_mask"],
                    features=batch["features"],
                    drug_code=batch["drug_code"],
                )
            scores.extend(torch.sigmoid(outputs["logits"]).float().cpu().tolist())
            labels.extend(batch["labels"].int().cpu().tolist())

    predictions = [1 if score >= 0.5 else 0 for score in scores]
    test_metrics = compute_binary_metrics(labels, scores)
    save_predictions_csv(output_dir / "predictions.csv", labels, scores, predictions)

    payload = {
        "best_epoch": checkpoint.get("best_epoch"),
        "best_validation_metrics": checkpoint.get("best_validation_metrics"),
        "test_metrics": test_metrics,
        "feature_count": len(feature_columns),
        "drug_code_dim": drug_code_dim,
    }
    text = json.dumps(sanitize_for_json(payload), ensure_ascii=False, indent=2)
    (output_dir / "metrics.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
