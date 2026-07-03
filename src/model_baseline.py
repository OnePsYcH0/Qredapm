from __future__ import annotations

import torch.nn as nn


class BaselineMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)
