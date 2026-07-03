from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset


LABEL_COLUMN = "y2"
EXPLICIT_EXCLUDE_COLUMNS = {
    "jmkh",
    "visit_sn",
    "disease_names",
    "drug_code",
    "sex",
    "y1",
    "y2",
}


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON。") from exc
    return rows


def _is_scalar_numeric(value: Any) -> bool:
    return isinstance(value, (int, float, bool)) and not isinstance(value, bool) or isinstance(value, bool)


def infer_feature_columns(rows: Sequence[Dict[str, Any]]) -> List[str]:
    if not rows:
        raise ValueError("数据为空，无法推断特征列。")

    feature_columns: List[str] = []
    for column in rows[0].keys():
        if column in EXPLICIT_EXCLUDE_COLUMNS:
            continue

        values = [row.get(column) for row in rows]
        if any(isinstance(value, (list, dict, str)) for value in values if value is not None):
            continue
        if any(value is None for value in values):
            continue
        if not all(_is_scalar_numeric(value) for value in values):
            continue

        unique_values = {float(value) for value in values}
        if len(unique_values) <= 1:
            continue

        feature_columns.append(column)

    return feature_columns


@dataclass
class StandardScaler:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, features: torch.Tensor) -> "StandardScaler":
        mean = features.mean(dim=0)
        std = features.std(dim=0, unbiased=False)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        return cls(mean=mean, std=std)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.mean) / self.std

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"mean": self.mean, "std": self.std}


def rows_to_tensors(
    rows: Sequence[Dict[str, Any]],
    feature_columns: Sequence[str],
    label_column: str = LABEL_COLUMN,
) -> tuple[torch.Tensor, torch.Tensor]:
    feature_matrix: List[List[float]] = []
    labels: List[float] = []

    for row in rows:
        feature_matrix.append([float(row[column]) for column in feature_columns])
        labels.append(float(row[label_column]))

    features = torch.tensor(feature_matrix, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    return features, label_tensor


class TabularJsonDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def build_dataset_from_rows(
    rows: Sequence[Dict[str, Any]],
    feature_columns: Sequence[str],
    scaler: Optional[StandardScaler] = None,
) -> tuple[TabularJsonDataset, torch.Tensor, torch.Tensor]:
    features, labels = rows_to_tensors(rows, feature_columns)
    if scaler is not None:
        features = scaler.transform(features)
    return TabularJsonDataset(features, labels), features, labels


def _stringify_list_field(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, list):
        parts = [str(value).strip() for value in values if str(value).strip()]
        return " [SEP] ".join(parts)
    return str(values).strip()


def _safe_visit_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, list):
        return [str(value).strip() if value is not None else "" for value in values]
    text = str(values).strip()
    return [text] if text else []


def align_visit_texts(
    visit_sn: Any,
    disease_names: Any,
    max_visits: int,
) -> tuple[List[str], List[int]]:
    visit_list = _safe_visit_list(visit_sn)
    disease_list = _safe_visit_list(disease_names)
    common_length = max(len(visit_list), len(disease_list))

    aligned_texts: List[str] = []
    visit_mask: List[int] = []
    for index in range(min(common_length, max_visits)):
        disease_text = disease_list[index] if index < len(disease_list) else ""
        visit_text = visit_list[index] if index < len(visit_list) else ""
        aligned_texts.append(f"诊断信息：{disease_text}。就诊记录：{visit_text}")
        visit_mask.append(1)

    while len(aligned_texts) < max_visits:
        aligned_texts.append("诊断信息：。就诊记录：")
        visit_mask.append(0)

    return aligned_texts, visit_mask


def build_text_input(
    row: Dict[str, Any],
    include_disease_names: bool = True,
) -> str:
    visit_text = _stringify_list_field(row.get("visit_sn"))
    disease_text = _stringify_list_field(row.get("disease_names")) if include_disease_names else ""

    if include_disease_names and disease_text and visit_text:
        return f"疾病信息：{disease_text}。就诊记录：{visit_text}"
    if include_disease_names and disease_text:
        return f"疾病信息：{disease_text}"
    if visit_text:
        return f"就诊记录：{visit_text}"
    return "无文本记录"


def build_text_only_input(row: Dict[str, Any]) -> str:
    disease_text = _stringify_list_field(row.get("disease_names"))
    drug_text = _stringify_list_field(row.get("drug_names"))

    pieces = []
    if disease_text:
        pieces.append(disease_text)
    if drug_text:
        pieces.append(drug_text)
    return " ".join(pieces).strip() or "无文本记录"


class RedapmJsonDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        feature_columns: Sequence[str],
        tokenizer,
        max_length: int,
        max_visits: int,
        drug_code_dim: int,
        scaler: Optional[StandardScaler] = None,
        include_disease_names: bool = True,
    ) -> None:
        features, labels = rows_to_tensors(rows, feature_columns)
        if scaler is not None:
            features = scaler.transform(features)

        self.features = features
        self.labels = labels
        self.max_visits = max_visits
        self.max_length = max_length
        self.drug_code_dim = drug_code_dim

        visit_texts: List[List[str]] = []
        visit_masks: List[List[int]] = []
        drug_codes: List[List[float]] = []

        for row in rows:
            texts, mask = align_visit_texts(
                row.get("visit_sn"),
                row.get("disease_names") if include_disease_names else None,
                max_visits=max_visits,
            )
            visit_texts.append(texts)
            visit_masks.append(mask)

            raw_drug_code = row.get("drug_code")
            if isinstance(raw_drug_code, list):
                code = [float(value) for value in raw_drug_code[:drug_code_dim]]
            else:
                code = []
            if len(code) < drug_code_dim:
                code.extend([0.0] * (drug_code_dim - len(code)))
            drug_codes.append(code)

        flat_visit_texts = [text for patient_visits in visit_texts for text in patient_visits]
        tokenized = tokenizer(
            flat_visit_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        num_samples = len(rows)
        self.visit_texts = visit_texts
        self.visit_mask = torch.tensor(visit_masks, dtype=torch.long)
        self.drug_code = torch.tensor(drug_codes, dtype=torch.float32)
        self.input_ids = tokenized["input_ids"].view(num_samples, max_visits, max_length)
        self.attention_mask = tokenized["attention_mask"].view(num_samples, max_visits, max_length)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "visit_mask": self.visit_mask[index],
            "features": self.features[index],
            "drug_code": self.drug_code[index],
            "labels": self.labels[index],
        }


def infer_drug_code_dim(rows: Sequence[Dict[str, Any]]) -> int:
    max_dim = 0
    for row in rows:
        drug_code = row.get("drug_code")
        if isinstance(drug_code, list):
            max_dim = max(max_dim, len(drug_code))
    return max_dim


class TextOnlyJsonDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        tokenizer,
        max_length: int,
        label_column: str = LABEL_COLUMN,
    ) -> None:
        texts = [build_text_only_input(row) for row in rows]
        labels = [float(row[label_column]) for row in rows]
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.input_ids = tokenized["input_ids"]
        self.attention_mask = tokenized["attention_mask"]

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
        }
