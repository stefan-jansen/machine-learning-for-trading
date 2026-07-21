"""TSMixer: alternating time/feature mixing MLPs for time series regression.

Input: (batch, seq_len, n_features) -> scalar output.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _TimeMixingMLP(nn.Module):
    """Mix information across the time dimension via transpose + MLP."""

    def __init__(self, seq_len: int, n_features: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, seq_len)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.permute(0, 2, 1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        x = x.permute(0, 2, 1)
        return self.norm(x + residual)


class _FeatureMixingMLP(nn.Module):
    """Mix information across the feature dimension."""

    def __init__(self, n_features: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return self.norm(x + residual)


class _MixerBlock(nn.Module):
    """One TSMixer block: time-mixing followed by feature-mixing."""

    def __init__(self, seq_len: int, n_features: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.time_mix = _TimeMixingMLP(seq_len, n_features, hidden_dim, dropout)
        self.feature_mix = _FeatureMixingMLP(n_features, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.time_mix(x)
        x = self.feature_mix(x)
        return x


class TSMixerRegressor(nn.Module):
    """TSMixer for regression: alternating time/feature mixing + mean pool.

    Input: (batch, seq_len, n_features) -> scalar output.
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        n_blocks: int = 2,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[_MixerBlock(seq_len, n_features, hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = x.mean(dim=1)
        return self.head(x).squeeze(-1)
