"""Torch datasets for DeePM-style windowed training and inference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import FeaturePanel


@dataclass(frozen=True, slots=True)
class StaticAssetMetadata:
    """Static per-asset metadata used as context."""

    assets: list[str]
    asset_ids: torch.Tensor  # (N,)
    group_ids: torch.Tensor | None  # (N,)
    costs: torch.Tensor | None  # (N, 1)


def build_static_metadata(
    assets: Sequence[str],
    *,
    asset_to_group: dict[str, str] | None = None,
    asset_to_cost_bps: dict[str, float] | None = None,
) -> StaticAssetMetadata:
    """Create tensors for asset id, group id, and costs."""
    assets_list = [str(a) for a in assets]
    n = len(assets_list)

    asset_ids = torch.arange(n, dtype=torch.long)

    group_ids: torch.Tensor | None = None
    if asset_to_group is not None:
        groups = [str(asset_to_group.get(a, "UNKNOWN")) for a in assets_list]
        unique_groups = {g: i for i, g in enumerate(sorted(set(groups)))}
        group_ids = torch.tensor([unique_groups[g] for g in groups], dtype=torch.long)

    costs: torch.Tensor | None = None
    if asset_to_cost_bps is not None:
        cost_vals = [float(asset_to_cost_bps.get(a, 0.0)) / 10000.0 for a in assets_list]
        costs = torch.tensor(cost_vals, dtype=torch.float32).unsqueeze(-1)

    return StaticAssetMetadata(
        assets=assets_list, asset_ids=asset_ids, group_ids=group_ids, costs=costs
    )


class DeepmWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Sliding-window dataset for training.

    Each item returns (x_seq, y_seq, v_seq, m_seq) of shapes
    (L, N, F), (L, N), (L, N), (L, N).
    """

    def __init__(
        self,
        panel: FeaturePanel,
        *,
        seq_len: int,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
        stride: int = 1,
    ) -> None:
        if seq_len <= 1:
            raise ValueError("seq_len must be > 1")

        self._panel = panel
        self.seq_len = int(seq_len)

        dates = panel.dates
        start_idx = 0
        end_idx_exclusive = len(dates)
        if start_date is not None:
            start_idx = int(dates.get_indexer([pd.Timestamp(start_date)], method="bfill")[0])
        if end_date is not None:
            end_idx_exclusive = (
                int(dates.get_indexer([pd.Timestamp(end_date)], method="ffill")[0]) + 1
            )

        t_max_start = (end_idx_exclusive - 1) - self.seq_len
        t_min_start = start_idx
        if t_max_start < t_min_start:
            raise ValueError("Date range too small for given seq_len")

        self.start_indices = np.arange(t_min_start, t_max_start + 1, stride, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.start_indices.shape[0])

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = int(self.start_indices[idx])
        e = s + self.seq_len
        return (
            torch.from_numpy(self._panel.x[s:e]),
            torch.from_numpy(self._panel.y_fwd1[s:e]),
            torch.from_numpy(self._panel.vol_scale[s:e]),
            torch.from_numpy(self._panel.mask[s:e]),
        )


class RollingWindowInferenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    """Dataset for batched rolling-window inference."""

    def __init__(self, panel: FeaturePanel, *, seq_len: int, start_t: int) -> None:
        if seq_len <= 1:
            raise ValueError("seq_len must be > 1")
        if start_t < seq_len - 1:
            raise ValueError("start_t must be >= seq_len - 1")

        self._panel = panel
        self.seq_len = int(seq_len)
        self.times = np.arange(start_t, len(panel.dates) - 1, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.times.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        t = int(self.times[idx])
        s = t - self.seq_len + 1
        e = t + 1
        return (
            torch.from_numpy(self._panel.x[s:e]),
            torch.from_numpy(self._panel.mask[s:e]),
            t,
        )
