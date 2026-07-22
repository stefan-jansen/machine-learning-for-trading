"""NLinear: Normalization-Linear baseline for time-series regression.

From Zeng et al. (2023), *Are Transformers Effective for Time Series
Forecasting?*, AAAI 2023. The "N" is literal: subtract the last observation,
apply a linear map over the time axis per channel, then **add the last
observation back**. Without the add-back, the model is not NLinear — just
a dropout-regularized linear regressor.

This file is the authors' reference implementation from
https://github.com/cure-lab/LTSF-Linear (Apache 2.0), adapted minimally:

- The paper's `forward` returns `(batch, pred_len, channels)` for forecasting.
  For cross-sectional scalar regression we set `pred_len=1` and add a single
  `Linear(channels -> 1)` projection head to collapse channels into a scalar.
- The authors' `individual` flag (per-channel weights vs shared) is preserved
  and defaults to False (shared weights — the paper's headline variant).
- Denormalization (`x + seq_last`) is applied BEFORE the regression head so
  the head operates on the same scale as the ground-truth target.

Interface preserved for the factory:
  NLinear(lookback: int, n_features: int, dropout: float = 0.1)
  forward(x: (batch, seq_len, n_features)) -> (batch,)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NLinear(nn.Module):
    """Normalization-Linear baseline (Zeng et al. 2023).

    Channel-wise: subtract last value, Linear(seq_len -> 1) per channel (or
    shared), add last value back, then project to scalar via Linear(n_features
    -> 1).
    """

    def __init__(
        self,
        lookback: int,
        n_features: int,
        dropout: float = 0.1,
        pred_len: int = 1,
        individual: bool = False,
    ):
        super().__init__()
        self.seq_len = lookback
        self.pred_len = pred_len
        self.channels = n_features
        self.individual = individual

        if individual:
            self.Linear = nn.ModuleList()
            for _ in range(n_features):
                self.Linear.append(nn.Linear(lookback, pred_len))
        else:
            self.Linear = nn.Linear(lookback, pred_len)

        self.dropout = nn.Dropout(dropout)
        # Scalar regression head: (pred_len * n_features) -> 1.
        # For pred_len=1 this reduces to a linear aggregation across channels.
        self.head = nn.Linear(pred_len * n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, channels) — matches paper's convention.
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last

        if self.individual:
            output = torch.zeros(
                x.size(0), self.pred_len, x.size(2), dtype=x.dtype, device=x.device
            )
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            # permute so Linear maps over the time axis per channel
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Denormalize — the "N" of NLinear.
        x = x + seq_last  # (batch, pred_len, channels)

        # Scalar regression head.
        x = self.dropout(x.reshape(x.size(0), -1))
        return self.head(x).squeeze(-1)
