"""PatchTST: patching + channel-independent Transformer for time series.

From Nie, Nguyen, Sinthong, Kalagnanam (2023), *A Time Series is Worth 64
Words: Long-term Forecasting with Transformers*, ICLR 2023.

Two structural properties distinguish the paper's PatchTST from naive
"tokenize-the-input-with-a-Transformer" baselines:

1. **Channel-independent patching.** Each feature channel is treated as its
   own univariate sequence and passed through the same shared Transformer
   weights. There is no cross-channel mixing inside the encoder. This file
   delegates to ``PatchTST_backbone`` from the authors' repo to preserve
   this exactly.
2. **RevIN (Reversible Instance Normalization).** Per-sample per-channel
   statistics are removed before the backbone and re-added after, making
   the model robust to distribution shift. The vendored backbone wires
   this up when ``revin=True``.

Additionally, the paper uses **overlapping patches** (stride < patch_len)
and a **flatten + linear** prediction head rather than global mean pooling.

Adapter layer on top of the backbone:
- The backbone returns ``(batch, n_vars, target_window)``. For cross-sectional
  scalar regression we set ``target_window=1`` and then project the per-channel
  outputs to a single scalar via ``Linear(n_vars -> 1)``.

Reference implementation vendored from https://github.com/yuqinie98/PatchTST
(MIT License) into ``_reference/`` with import paths adjusted. RevIN is
vendored from https://github.com/ts-kim/RevIN (MIT License).

Interface preserved for the factory:
  PatchTST(n_features: int, lookback: int, patch_size: int = 16, ...)
  forward(x: (batch, seq_len, n_features)) -> (batch,)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from case_studies.config.patchtst._reference import PatchTST_backbone


class PatchTST(nn.Module):
    """PatchTST channel-independent regressor.

    Wraps the paper authors' ``PatchTST_backbone`` with a scalar regression
    head. RevIN on by default; overlapping patches with stride=patch_size/2.
    """

    def __init__(
        self,
        n_features: int,
        lookback: int,
        patch_size: int = 16,
        stride: int | None = None,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int | None = None,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        padding_patch: str = "end",
    ):
        super().__init__()

        if stride is None:
            stride = max(1, patch_size // 2)
        if d_ff is None:
            d_ff = d_model * 4

        self.backbone = PatchTST_backbone(
            c_in=n_features,
            context_window=lookback,
            target_window=1,
            patch_len=patch_size,
            stride=stride,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            attn_dropout=attn_dropout,
            dropout=dropout,
            revin=revin,
            affine=affine,
            subtract_last=subtract_last,
            padding_patch=padding_patch,
            head_type="flatten",
            individual=False,
        )
        self.head = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features) → backbone wants (batch, n_vars, seq_len)
        z = x.permute(0, 2, 1)
        # backbone out: (batch, n_vars, target_window=1)
        z = self.backbone(z)
        # collapse target_window and project across channels to scalar
        z = z.squeeze(-1)  # (batch, n_vars)
        return self.head(z).squeeze(-1)
