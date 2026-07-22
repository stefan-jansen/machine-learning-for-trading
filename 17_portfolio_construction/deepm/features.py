"""Feature engineering for DeePM-style systematic macro models.

Constructs a compact feature set from daily closes:
1. Volatility-normalized returns for horizons h in {1, 21, 63, 252}
2. Multi-scale MACD signals at (8,24), (16,48), (32,96) plus re-normalization
3. Log-price z-scores over windows l in {21, 252}
4. Robust clipping using rolling median and MAD over 252 days
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .configs import FeatureConfig


@dataclass(frozen=True, slots=True)
class FeaturePanel:
    """A fully-aligned feature panel for end-to-end portfolio learning."""

    dates: pd.DatetimeIndex
    assets: list[str]

    x: np.ndarray  # (T, N, F)
    r_fwd1: np.ndarray  # (T, N) forward raw returns
    sigma_hat: np.ndarray  # (T, N) ex-ante volatility
    vol_scale: np.ndarray  # (T, N) = 1 / (sigma_hat + eps)
    y_fwd1: np.ndarray  # (T, N) vol-scaled forward return
    mask: np.ndarray  # (T, N) availability mask

    feature_names: list[str]


def _robust_clip(
    df: pd.DataFrame,
    *,
    window: int,
    k: float,
    mad_scale: float,
) -> pd.DataFrame:
    """Robust, no-lookahead clipping using rolling median and MAD."""
    median = df.rolling(window=window, min_periods=window).median()
    mad = (df - median).abs().rolling(window=window, min_periods=window).median()
    bound = k * mad_scale * mad
    return df.clip(lower=median - bound, upper=median + bound)


def compute_ex_ante_volatility(
    daily_returns: pd.DataFrame,
    *,
    span: int,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """EWMA volatility estimate (sigma_hat) from daily returns."""
    mp = span if min_periods is None else min_periods
    return daily_returns.ewm(span=span, adjust=False, min_periods=mp).std(bias=False)


def build_feature_panel(
    prices: pd.DataFrame,
    cfg: FeatureConfig,
) -> FeaturePanel:
    """Compute DeePM-style features, returns, vol scaling, and masks.

    Parameters
    ----------
    prices:
        Wide DataFrame of close prices (dates x assets). NaNs for missing.

    cfg:
        Feature engineering configuration.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices index must be a pandas.DatetimeIndex")
    prices = prices.sort_index()
    assets = [str(c) for c in prices.columns]

    existence = prices.notna().astype(np.float32)
    prices_ffill = prices.ffill()

    r_daily = prices_ffill.pct_change()
    valid_r_daily = existence.astype(bool) & existence.shift(1).fillna(False).astype(bool)
    r_daily = r_daily.where(valid_r_daily)

    sigma_hat = compute_ex_ante_volatility(r_daily, span=cfg.vol_span)
    vol_scale = 1.0 / (sigma_hat + cfg.vol_eps)

    r_fwd1 = prices_ffill.pct_change().shift(-1)
    valid_trade = existence.astype(bool) & existence.shift(-1).fillna(False).astype(bool)
    r_fwd1 = r_fwd1.where(valid_trade)

    y_fwd1 = vol_scale * r_fwd1

    # Build feature blocks
    feature_dfs: list[tuple[str, pd.DataFrame]] = []

    if cfg.feature_set == "signal":
        horizons: Sequence[int] = cfg.ret_horizons
    elif cfg.feature_set == "raw_momentum":
        horizons = tuple(range(1, int(cfg.raw_momentum_max_horizon) + 1))
    else:
        horizons = cfg.ret_horizons

    for h in horizons:
        ret_h = prices_ffill / prices_ffill.shift(h) - 1.0
        scaled = ret_h / (sigma_hat * np.sqrt(float(h)) + cfg.vol_eps)
        feature_dfs.append((f"ret_h{h}", scaled))

    if cfg.feature_set == "signal":
        for s, l in cfg.macd_pairs:
            ewm_fast = prices_ffill.ewm(span=s, adjust=False, min_periods=s).mean()
            ewm_slow = prices_ffill.ewm(span=l, adjust=False, min_periods=l).mean()
            macd_raw = (ewm_fast - ewm_slow) / (sigma_hat + cfg.vol_eps)
            macd_std = macd_raw.rolling(
                window=cfg.macd_renorm_window, min_periods=cfg.macd_renorm_window
            ).std()
            macd = macd_raw / (macd_std + cfg.vol_eps)
            feature_dfs.append((f"macd_{s}_{l}", macd))

    log_prices = np.log(prices_ffill)
    for w in cfg.zscore_windows:
        mu = log_prices.rolling(window=w, min_periods=w).mean()
        sd = log_prices.rolling(window=w, min_periods=w).std()
        z = (log_prices - mu) / (sd + cfg.vol_eps)
        feature_dfs.append((f"zscore_{w}", z))

    # Robust clipping
    clipped_features: list[np.ndarray] = []
    feature_names: list[str] = []

    for name, df in feature_dfs:
        df_clipped = _robust_clip(
            df, window=cfg.clip_window, k=cfg.clip_k, mad_scale=cfg.clip_mad_scale
        )
        df_filled = df_clipped.fillna(0.0)
        clipped_features.append(df_filled.to_numpy(dtype=np.float32))
        feature_names.append(name)

    x = (
        np.stack(clipped_features, axis=-1)
        if clipped_features
        else np.zeros((len(prices), len(assets), 0))
    )

    if cfg.include_existence:
        x = np.concatenate([x, existence.to_numpy(dtype=np.float32)[..., None]], axis=-1)
        feature_names.append("existence")

    mask = existence.to_numpy(dtype=np.float32)

    return FeaturePanel(
        dates=prices.index,
        assets=assets,
        x=x,
        r_fwd1=r_fwd1.fillna(0.0).to_numpy(dtype=np.float32),
        sigma_hat=sigma_hat.fillna(0.0).to_numpy(dtype=np.float32),
        vol_scale=vol_scale.fillna(0.0).to_numpy(dtype=np.float32),
        y_fwd1=y_fwd1.fillna(0.0).to_numpy(dtype=np.float32),
        mask=mask,
        feature_names=feature_names,
    )
