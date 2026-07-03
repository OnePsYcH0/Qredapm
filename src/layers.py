from __future__ import annotations

import torch
import torch.nn as nn


class ResidualFeedForwardBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.activation(out + residual)
        out = self.dropout(out)
        return out


class ResidualMLP(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [ResidualFeedForwardBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(sequence).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return pooled, weights
