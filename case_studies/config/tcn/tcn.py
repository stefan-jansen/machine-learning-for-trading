"""Temporal Convolutional Network for time series regression.

Dilated causal convolutions with exponentially growing receptive field.
Input: (batch, seq_len, n_features) -> scalar output.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _CausalConv1d(nn.Module):
    """1D convolution with causal (left) padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.causal_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.causal_padding,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.causal_padding > 0:
            out = out[:, :, : -self.causal_padding]
        return out


class _TCNBlock(nn.Module):
    """Residual block with two dilated causal convolutions."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = _CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = _CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.dropout(self.relu(self.bn1(self.conv1(x))))
        out = self.dropout(self.relu(self.bn2(self.conv2(out))))
        return self.relu(out + res)


class TCNRegressor(nn.Module):
    """Temporal Convolutional Network for regression.

    Dilated causal convolutions with exponentially growing receptive field.
    Input: (batch, seq_len, n_features) -> scalar output.
    """

    def __init__(
        self,
        n_features: int,
        n_channels: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.1,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()
        blocks: list[nn.Module] = []
        for i, d in enumerate(dilations):
            in_ch = n_features if i == 0 else n_channels
            blocks.append(_TCNBlock(in_ch, n_channels, kernel_size, d, dropout))
        self.tcn = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (batch, n_features, seq_len)
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)
