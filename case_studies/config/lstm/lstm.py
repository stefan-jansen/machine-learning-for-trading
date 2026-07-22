"""LSTM for time series regression.

Architecture: LSTM -> last hidden state -> Linear -> scalar output.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    """LSTM for time series regression.

    Architecture: LSTM -> last hidden state -> Linear -> scalar output.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)
