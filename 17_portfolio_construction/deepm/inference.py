"""Inference helpers for DeePM-style policies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import RollingWindowInferenceDataset, StaticAssetMetadata
from .features import FeaturePanel


@torch.no_grad()
def infer_risk_weights_rolling(
    model: nn.Module,
    panel: FeaturePanel,
    *,
    static_meta: StaticAssetMetadata,
    seq_len: int,
    batch_size: int = 256,
    device: str = "cpu",
    start_t: int | None = None,
) -> pd.DataFrame:
    """Run rolling-window inference and return a (dates x assets) DataFrame.

    Takes the last step's p_{t} from each window as the portfolio decision.
    """
    dev = torch.device(device)
    model.to(dev)
    model.eval()

    n_dates = len(panel.dates)
    n_assets = len(panel.assets)

    t0 = seq_len - 1 if start_t is None else int(start_t)

    ds = RollingWindowInferenceDataset(panel, seq_len=seq_len, start_t=t0)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    p_out = np.zeros((n_dates, n_assets), dtype=np.float32)

    asset_ids = static_meta.asset_ids.to(dev)
    group_ids = None if static_meta.group_ids is None else static_meta.group_ids.to(dev)
    costs = None if static_meta.costs is None else static_meta.costs.to(dev)

    for x_win, m_win, t_idx in loader:
        x_win = x_win.to(device=dev, dtype=torch.float32)
        m_win = m_win.to(device=dev, dtype=torch.float32)

        p_seq = model(
            x_win,
            mask=m_win,
            asset_ids=asset_ids,
            group_ids=group_ids,
            costs=costs,
        )
        p_last = p_seq[:, -1, :].detach().cpu().numpy()

        t_np = np.asarray(t_idx, dtype=np.int64)
        p_out[t_np, :] = p_last

    return pd.DataFrame(p_out, index=panel.dates, columns=panel.assets)
