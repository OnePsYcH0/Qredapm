from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from layers import AttentionPooling, ResidualMLP


def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class StructuredEncoder(nn.Module):
    def __init__(
        self,
        scalar_input_dim: int,
        drug_code_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.scalar_projection = nn.Sequential(
            nn.Linear(scalar_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.drug_projection = nn.Sequential(
            nn.Linear(drug_code_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fusion_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual_mlp = ResidualMLP(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)

    def forward(self, scalar_features: torch.Tensor, drug_code: torch.Tensor) -> torch.Tensor:
        scalar_repr = self.scalar_projection(scalar_features)
        drug_repr = self.drug_projection(drug_code)
        fused = self.fusion_projection(torch.cat([scalar_repr, drug_repr], dim=-1))
        return self.residual_mlp(fused)


class REDAPM(nn.Module):
    def __init__(
        self,
        model_name: str,
        structured_input_dim: int,
        drug_code_dim: int,
        struct_hidden_dim: int = 128,
        text_projection_dim: int = 128,
        fusion_hidden_dim: int = 128,
        struct_num_layers: int = 3,
        max_visits: int = 12,
        visit_transformer_layers: int = 2,
        visit_transformer_heads: int = 4,
        fusion_transformer_layers: int = 2,
        fusion_transformer_heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.max_visits = max_visits
        self.hidden_dim = text_projection_dim
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
        self.structured_encoder = StructuredEncoder(
            scalar_input_dim=structured_input_dim,
            drug_code_dim=drug_code_dim,
            hidden_dim=struct_hidden_dim,
            num_layers=struct_num_layers,
            dropout=dropout,
        )
        self.structured_token_projection = nn.Sequential(
            nn.Linear(struct_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.visit_token_projection = nn.Sequential(
            nn.Linear(text_projection_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.Dropout(dropout),
        )
        self.modality_embedding = nn.Embedding(3, fusion_hidden_dim)
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
        text_repr, visit_attention = self.visit_pooling(visit_tokens, mask=visit_mask.bool())

        struct_repr = self.structured_encoder(features, drug_code)
        struct_token = self.structured_token_projection(struct_repr).unsqueeze(1)
        visit_fusion_tokens = self.visit_token_projection(visit_tokens)

        cls_token = self.fusion_cls.expand(batch_size, -1, -1)
        fusion_tokens = torch.cat([cls_token, visit_fusion_tokens, struct_token], dim=1)

        cls_modality = torch.zeros((batch_size, 1), dtype=torch.long, device=input_ids.device)
        visit_modality = torch.ones((batch_size, max_visits), dtype=torch.long, device=input_ids.device)
        struct_modality = torch.full((batch_size, 1), 2, dtype=torch.long, device=input_ids.device)
        modality_ids = torch.cat([cls_modality, visit_modality, struct_modality], dim=1)
        fusion_tokens = fusion_tokens + self.modality_embedding(modality_ids)

        fusion_padding_mask = torch.cat(
            [
                torch.zeros((batch_size, 1), dtype=torch.bool, device=input_ids.device),
                visit_padding_mask,
                torch.zeros((batch_size, 1), dtype=torch.bool, device=input_ids.device),
            ],
            dim=1,
        )
        fusion_tokens = self.fusion_encoder(fusion_tokens, src_key_padding_mask=fusion_padding_mask)
        fusion_cls = fusion_tokens[:, 0]
        pooled_visit = masked_mean_pool(
            fusion_tokens[:, 1 : 1 + max_visits],
            visit_mask,
        )
        logits = self.classifier(torch.cat([fusion_cls, pooled_visit], dim=-1)).squeeze(-1)

        return {
            "logits": logits,
            "text_repr": text_repr,
            "struct_repr": struct_repr,
            "visit_tokens": visit_tokens,
            "visit_attention": visit_attention,
            "fusion_cls": fusion_cls,
        }
