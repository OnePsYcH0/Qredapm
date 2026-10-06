from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
from transformers import AutoModel


LATENT_NAMES = [
    "z_demo_visit",
    "z_cardio",
    "z_diabetes_complication",
    "z_acute_metabolic",
    "z_psych_symptom",
    "z_lab_lipid_bp",
    "z_lab_renal_glucose",
    "z_medication",
]

PAIRWISE_CLINICAL_EDGES = [(0, 7), (1, 5), (2, 6), (3, 6), (4, 7), (2, 1), (5, 6), (3, 2)]
RING_EDGES = [(index, (index + 1) % 8) for index in range(8)]


def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(sequence).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class ClinicalGroupCompressor(nn.Module):
    def __init__(self, feature_columns: Sequence[str], group_mapping: Dict[str, List[str]]) -> None:
        super().__init__()
        self.feature_columns = list(feature_columns)
        self.group_mapping = {name: list(columns) for name, columns in group_mapping.items()}
        feature_to_index = {name: index for index, name in enumerate(self.feature_columns)}
        compressors = []
        for idx, latent_name in enumerate(LATENT_NAMES):
            columns = self.group_mapping[latent_name]
            indices = torch.tensor([feature_to_index[column] for column in columns], dtype=torch.long)
            self.register_buffer(f"group_indices_{idx}", indices)
            group_dim = len(indices)
            hidden_dim = max(2, min(8, group_dim * 2))
            compressors.append(nn.Sequential(nn.Linear(group_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)))
        self.compressors = nn.ModuleList(compressors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latents = []
        for idx, compressor in enumerate(self.compressors):
            indices = getattr(self, f"group_indices_{idx}")
            latents.append(compressor(x.index_select(dim=1, index=indices)))
        return torch.cat(latents, dim=1)


class ClinicalGroupLinearCompressor(nn.Module):
    def __init__(self, feature_columns: Sequence[str], group_mapping: Dict[str, List[str]]) -> None:
        super().__init__()
        self.feature_columns = list(feature_columns)
        self.group_mapping = {name: list(columns) for name, columns in group_mapping.items()}
        feature_to_index = {name: index for index, name in enumerate(self.feature_columns)}
        compressors = []
        for idx, latent_name in enumerate(LATENT_NAMES):
            columns = self.group_mapping[latent_name]
            indices = torch.tensor([feature_to_index[column] for column in columns], dtype=torch.long)
            self.register_buffer(f"group_indices_{idx}", indices)
            compressors.append(nn.Linear(len(indices), 1))
        self.compressors = nn.ModuleList(compressors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latents = []
        for idx, compressor in enumerate(self.compressors):
            indices = getattr(self, f"group_indices_{idx}")
            latents.append(compressor(x.index_select(dim=1, index=indices)))
        return torch.cat(latents, dim=1)


class ClinicalGroupMeanCompressor(nn.Module):
    def __init__(self, feature_columns: Sequence[str], group_mapping: Dict[str, List[str]]) -> None:
        super().__init__()
        self.feature_columns = list(feature_columns)
        self.group_mapping = {name: list(columns) for name, columns in group_mapping.items()}
        feature_to_index = {name: index for index, name in enumerate(self.feature_columns)}
        for idx, latent_name in enumerate(LATENT_NAMES):
            columns = self.group_mapping[latent_name]
            indices = torch.tensor([feature_to_index[column] for column in columns], dtype=torch.long)
            self.register_buffer(f"group_indices_{idx}", indices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latents = []
        for idx in range(len(LATENT_NAMES)):
            indices = getattr(self, f"group_indices_{idx}")
            latents.append(x.index_select(dim=1, index=indices).mean(dim=1, keepdim=True))
        return torch.cat(latents, dim=1)


class Exp7Clinical8ClassicalReplacement(nn.Module):
    """SN1 text/visit + clinical8 classical token + drug token + SN3-style fusion."""

    def __init__(
        self,
        sn1_model_dir: str,
        feature_columns: Sequence[str],
        group_mapping: Dict[str, List[str]],
        drug_code_dim: int,
        fusion_hidden_dim: int = 256,
        text_projection_dim: int = 256,
        max_visits: int = 12,
        visit_transformer_layers: int = 2,
        visit_transformer_heads: int = 4,
        fusion_transformer_layers: int = 2,
        fusion_transformer_heads: int = 4,
        dropout: float = 0.3,
        freeze_text_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(sn1_model_dir)
        self.max_visits = max_visits
        self.clinical_compressor = ClinicalGroupCompressor(feature_columns, group_mapping)
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        self.text_projection = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, text_projection_dim),
            nn.LayerNorm(text_projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.visit_positional_embedding = nn.Embedding(max_visits, text_projection_dim)
        self.visit_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=text_projection_dim,
                nhead=visit_transformer_heads,
                dim_feedforward=text_projection_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=visit_transformer_layers,
        )
        self.visit_pooling = AttentionPooling(text_projection_dim)
        self.visit_token_projection = nn.Sequential(
            nn.Linear(text_projection_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.Dropout(dropout),
        )
        self.clinical_token_projection = nn.Sequential(
            nn.Linear(8, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.drug_token_projection = nn.Sequential(
            nn.Linear(drug_code_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.modality_embedding = nn.Embedding(4, fusion_hidden_dim)
        self.fusion_cls = nn.Parameter(torch.zeros(1, 1, fusion_hidden_dim))
        self.fusion_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=fusion_hidden_dim,
                nhead=fusion_transformer_heads,
                dim_feedforward=fusion_hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=fusion_transformer_layers,
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden_dim * 2, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )
        nn.init.normal_(self.fusion_cls, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visit_mask: torch.Tensor,
        features: torch.Tensor,
        drug_code: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, max_visits, seq_length = input_ids.shape
        flat_input_ids = input_ids.view(batch_size * max_visits, seq_length)
        flat_attention_mask = attention_mask.view(batch_size * max_visits, seq_length)

        text_outputs = self.text_encoder(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
        visit_repr = masked_mean_pool(text_outputs.last_hidden_state, flat_attention_mask)
        visit_repr = self.text_projection(visit_repr).view(batch_size, max_visits, -1)

        position_ids = torch.arange(max_visits, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        visit_tokens = visit_repr + self.visit_positional_embedding(position_ids)
        visit_padding_mask = visit_mask == 0
        visit_tokens = self.visit_encoder(visit_tokens, src_key_padding_mask=visit_padding_mask)
        _, visit_attention = self.visit_pooling(visit_tokens, mask=visit_mask.bool())

        visit_fusion_tokens = self.visit_token_projection(visit_tokens)
        clinical8 = self.clinical_compressor(features)
        clinical_token = self.clinical_token_projection(clinical8).unsqueeze(1)
        drug_token = self.drug_token_projection(drug_code).unsqueeze(1)
        cls_token = self.fusion_cls.expand(batch_size, -1, -1)
        fusion_tokens = torch.cat([cls_token, visit_fusion_tokens, clinical_token, drug_token], dim=1)

        cls_modality = torch.zeros((batch_size, 1), dtype=torch.long, device=input_ids.device)
        visit_modality = torch.ones((batch_size, max_visits), dtype=torch.long, device=input_ids.device)
        clinical_modality = torch.full((batch_size, 1), 2, dtype=torch.long, device=input_ids.device)
        drug_modality = torch.full((batch_size, 1), 3, dtype=torch.long, device=input_ids.device)
        modality_ids = torch.cat([cls_modality, visit_modality, clinical_modality, drug_modality], dim=1)
        fusion_tokens = fusion_tokens + self.modality_embedding(modality_ids)

        fusion_padding_mask = torch.cat(
            [
                torch.zeros((batch_size, 1), dtype=torch.bool, device=input_ids.device),
                visit_padding_mask,
                torch.zeros((batch_size, 2), dtype=torch.bool, device=input_ids.device),
            ],
            dim=1,
        )
        fusion_tokens = self.fusion_encoder(fusion_tokens, src_key_padding_mask=fusion_padding_mask)
        fusion_cls = fusion_tokens[:, 0]
        pooled_visit = masked_mean_pool(fusion_tokens[:, 1 : 1 + max_visits], visit_mask)
        logits = self.classifier(torch.cat([fusion_cls, pooled_visit], dim=-1)).squeeze(-1)
        return {"logits": logits, "clinical8": clinical8, "visit_attention": visit_attention, "fusion_cls": fusion_cls}


class Exp7Clinical8LinearReplacement(Exp7Clinical8ClassicalReplacement):
    """Same as 7B, but each clinical group is compressed by Linear(group_dim, 1)."""

    def __init__(
        self,
        sn1_model_dir: str,
        feature_columns: Sequence[str],
        group_mapping: Dict[str, List[str]],
        drug_code_dim: int,
        fusion_hidden_dim: int = 256,
        text_projection_dim: int = 256,
        max_visits: int = 12,
        visit_transformer_layers: int = 2,
        visit_transformer_heads: int = 4,
        fusion_transformer_layers: int = 2,
        fusion_transformer_heads: int = 4,
        dropout: float = 0.3,
        freeze_text_encoder: bool = True,
    ) -> None:
        super().__init__(
            sn1_model_dir=sn1_model_dir,
            feature_columns=feature_columns,
            group_mapping=group_mapping,
            drug_code_dim=drug_code_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            text_projection_dim=text_projection_dim,
            max_visits=max_visits,
            visit_transformer_layers=visit_transformer_layers,
            visit_transformer_heads=visit_transformer_heads,
            fusion_transformer_layers=fusion_transformer_layers,
            fusion_transformer_heads=fusion_transformer_heads,
            dropout=dropout,
            freeze_text_encoder=freeze_text_encoder,
        )
        self.clinical_compressor = ClinicalGroupLinearCompressor(feature_columns, group_mapping)


class Exp7Clinical8MeanReplacement(Exp7Clinical8ClassicalReplacement):
    """Same fusion model, but clinical8 is a fixed standardized group mean."""

    def __init__(
        self,
        sn1_model_dir: str,
        feature_columns: Sequence[str],
        group_mapping: Dict[str, List[str]],
        drug_code_dim: int,
        fusion_hidden_dim: int = 256,
        text_projection_dim: int = 256,
        max_visits: int = 12,
        visit_transformer_layers: int = 2,
        visit_transformer_heads: int = 4,
        fusion_transformer_layers: int = 2,
        fusion_transformer_heads: int = 4,
        dropout: float = 0.3,
        freeze_text_encoder: bool = True,
    ) -> None:
        super().__init__(
            sn1_model_dir=sn1_model_dir,
            feature_columns=feature_columns,
            group_mapping=group_mapping,
            drug_code_dim=drug_code_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            text_projection_dim=text_projection_dim,
            max_visits=max_visits,
            visit_transformer_layers=visit_transformer_layers,
            visit_transformer_heads=visit_transformer_heads,
            fusion_transformer_layers=fusion_transformer_layers,
            fusion_transformer_heads=fusion_transformer_heads,
            dropout=dropout,
            freeze_text_encoder=freeze_text_encoder,
        )
        self.clinical_compressor = ClinicalGroupMeanCompressor(feature_columns, group_mapping)


def edges_for_entanglement(entanglement: str) -> List[tuple[int, int]]:
    if entanglement == "ring":
        return RING_EDGES
    if entanglement == "pairwise_clinical":
        return PAIRWISE_CLINICAL_EDGES
    raise ValueError(f"Unsupported entanglement: {entanglement}")


def build_quantum_layer(
    vqc_layers: int,
    entanglement: str,
    q_device_name: str = "default.qubit",
    diff_method: str = "backprop",
):
    import pennylane as qml

    n_qubits = 8
    dev = qml.device(q_device_name, wires=n_qubits, shots=None)
    edges = edges_for_entanglement(entanglement)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list:
        qml.AngleEmbedding(torch.pi * torch.tanh(inputs), wires=range(n_qubits), rotation="Y")
        for layer in range(vqc_layers):
            for wire in range(n_qubits):
                qml.RY(weights[layer, wire, 0], wires=wire)
                qml.RZ(weights[layer, wire, 1], wires=wire)
            for control, target in edges:
                qml.CNOT(wires=[control, target])
        return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

    layer = qml.qnn.TorchLayer(circuit, {"weights": (vqc_layers, n_qubits, 2)})
    with torch.no_grad():
        layer.weights.uniform_(-0.1, 0.1)
    return layer


def build_data_reupload_quantum_layer(
    reupload_blocks: int,
    entanglement: str,
    angle_scale: float = torch.pi,
    q_device_name: str = "default.qubit",
    diff_method: str = "backprop",
):
    import pennylane as qml

    n_qubits = 8
    dev = qml.device(q_device_name, wires=n_qubits, shots=None)
    edges = edges_for_entanglement(entanglement)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list:
        for block in range(reupload_blocks):
            qml.AngleEmbedding(angle_scale * torch.tanh(inputs), wires=range(n_qubits), rotation="Y")
            for wire in range(n_qubits):
                qml.RY(weights[block, wire, 0], wires=wire)
                qml.RZ(weights[block, wire, 1], wires=wire)
            for control, target in edges:
                qml.CNOT(wires=[control, target])
        return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

    layer = qml.qnn.TorchLayer(circuit, {"weights": (reupload_blocks, n_qubits, 2)})
    with torch.no_grad():
        layer.weights.uniform_(-0.1, 0.1)
    return layer


class Exp7QSN2Replacement(Exp7Clinical8ClassicalReplacement):
    """SN1 text/visit + clinical8->QSN2 token + drug token + SN3-style fusion."""

    def __init__(
        self,
        sn1_model_dir: str,
        feature_columns: Sequence[str],
        group_mapping: Dict[str, List[str]],
        drug_code_dim: int,
        fusion_hidden_dim: int = 256,
        text_projection_dim: int = 256,
        max_visits: int = 12,
        visit_transformer_layers: int = 2,
        visit_transformer_heads: int = 4,
        fusion_transformer_layers: int = 2,
        fusion_transformer_heads: int = 4,
        dropout: float = 0.3,
        freeze_text_encoder: bool = True,
        vqc_layers: int = 2,
        entanglement: str = "pairwise_clinical",
        q_device_name: str = "default.qubit",
        diff_method: str = "backprop",
        residual_qsn2: bool = True,
    ) -> None:
        super().__init__(
            sn1_model_dir=sn1_model_dir,
            feature_columns=feature_columns,
            group_mapping=group_mapping,
            drug_code_dim=drug_code_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            text_projection_dim=text_projection_dim,
            max_visits=max_visits,
            visit_transformer_layers=visit_transformer_layers,
            visit_transformer_heads=visit_transformer_heads,
            fusion_transformer_layers=fusion_transformer_layers,
            fusion_transformer_heads=fusion_transformer_heads,
            dropout=dropout,
            freeze_text_encoder=freeze_text_encoder,
        )
        self.quantum_layer = build_quantum_layer(
            vqc_layers=vqc_layers,
            entanglement=entanglement,
            q_device_name=q_device_name,
            diff_method=diff_method,
        )
        self.residual_qsn2 = residual_qsn2
        qsn2_input_dim = 16 if residual_qsn2 else 8
        self.clinical_token_projection = nn.Sequential(
            nn.Linear(qsn2_input_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visit_mask: torch.Tensor,
        features: torch.Tensor,
        drug_code: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, max_visits, seq_length = input_ids.shape
        flat_input_ids = input_ids.view(batch_size * max_visits, seq_length)
        flat_attention_mask = attention_mask.view(batch_size * max_visits, seq_length)

        text_outputs = self.text_encoder(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
        visit_repr = masked_mean_pool(text_outputs.last_hidden_state, flat_attention_mask)
        visit_repr = self.text_projection(visit_repr).view(batch_size, max_visits, -1)

        position_ids = torch.arange(max_visits, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        visit_tokens = visit_repr + self.visit_positional_embedding(position_ids)
        visit_padding_mask = visit_mask == 0
        visit_tokens = self.visit_encoder(visit_tokens, src_key_padding_mask=visit_padding_mask)
        _, visit_attention = self.visit_pooling(visit_tokens, mask=visit_mask.bool())

        visit_fusion_tokens = self.visit_token_projection(visit_tokens)
        clinical8 = self.clinical_compressor(features)
        clinical8_cpu = clinical8.to("cpu")
        q_features = self.quantum_layer(clinical8_cpu).to(clinical8.device)
        q_features = torch.nan_to_num(q_features, nan=0.0, posinf=1.0, neginf=-1.0)
        clinical8 = torch.nan_to_num(clinical8, nan=0.0, posinf=1.0, neginf=-1.0)
        qsn2_features = torch.cat([clinical8, q_features], dim=1) if self.residual_qsn2 else q_features
        clinical_token = self.clinical_token_projection(qsn2_features).unsqueeze(1)
        drug_token = self.drug_token_projection(drug_code).unsqueeze(1)
        cls_token = self.fusion_cls.expand(batch_size, -1, -1)
        fusion_tokens = torch.cat([cls_token, visit_fusion_tokens, clinical_token, drug_token], dim=1)

        cls_modality = torch.zeros((batch_size, 1), dtype=torch.long, device=input_ids.device)
        visit_modality = torch.ones((batch_size, max_visits), dtype=torch.long, device=input_ids.device)
        clinical_modality = torch.full((batch_size, 1), 2, dtype=torch.long, device=input_ids.device)
        drug_modality = torch.full((batch_size, 1), 3, dtype=torch.long, device=input_ids.device)
        modality_ids = torch.cat([cls_modality, visit_modality, clinical_modality, drug_modality], dim=1)
        fusion_tokens = fusion_tokens + self.modality_embedding(modality_ids)

        fusion_padding_mask = torch.cat(
            [
                torch.zeros((batch_size, 1), dtype=torch.bool, device=input_ids.device),
                visit_padding_mask,
                torch.zeros((batch_size, 2), dtype=torch.bool, device=input_ids.device),
            ],
            dim=1,
        )
        fusion_tokens = self.fusion_encoder(fusion_tokens, src_key_padding_mask=fusion_padding_mask)
        fusion_cls = fusion_tokens[:, 0]
        pooled_visit = masked_mean_pool(fusion_tokens[:, 1 : 1 + max_visits], visit_mask)
        logits = self.classifier(torch.cat([fusion_cls, pooled_visit], dim=-1)).squeeze(-1)
        return {
            "logits": logits,
            "clinical8": clinical8,
            "qsn2_features": qsn2_features,
            "visit_attention": visit_attention,
            "fusion_cls": fusion_cls,
        }


class Exp7DataReuploadQSN2Replacement(Exp7QSN2Replacement):
    """QSN2 replacement with repeated clinical8 angle encoding between VQC blocks."""

    def __init__(
        self,
        sn1_model_dir: str,
        feature_columns: Sequence[str],
        group_mapping: Dict[str, List[str]],
        drug_code_dim: int,
        fusion_hidden_dim: int = 256,
        text_projection_dim: int = 256,
        max_visits: int = 12,
        visit_transformer_layers: int = 2,
        visit_transformer_heads: int = 4,
        fusion_transformer_layers: int = 2,
        fusion_transformer_heads: int = 4,
        dropout: float = 0.3,
        freeze_text_encoder: bool = True,
        reupload_blocks: int = 2,
        entanglement: str = "pairwise_clinical",
        angle_scale: float = torch.pi,
        q_device_name: str = "default.qubit",
        diff_method: str = "backprop",
        residual_qsn2: bool = True,
    ) -> None:
        super().__init__(
            sn1_model_dir=sn1_model_dir,
            feature_columns=feature_columns,
            group_mapping=group_mapping,
            drug_code_dim=drug_code_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            text_projection_dim=text_projection_dim,
            max_visits=max_visits,
            visit_transformer_layers=visit_transformer_layers,
            visit_transformer_heads=visit_transformer_heads,
            fusion_transformer_layers=fusion_transformer_layers,
            fusion_transformer_heads=fusion_transformer_heads,
            dropout=dropout,
            freeze_text_encoder=freeze_text_encoder,
            vqc_layers=1,
            entanglement=entanglement,
            q_device_name=q_device_name,
            diff_method=diff_method,
            residual_qsn2=residual_qsn2,
        )
        self.quantum_layer = build_data_reupload_quantum_layer(
            reupload_blocks=reupload_blocks,
            entanglement=entanglement,
            angle_scale=angle_scale,
            q_device_name=q_device_name,
            diff_method=diff_method,
        )
