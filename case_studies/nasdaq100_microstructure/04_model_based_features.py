# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # NASDAQ-100 Microstructure: Temporal Features
#
# **Chapter 9: Time Series Analysis**
#
# This notebook constructs temporal features for the NASDAQ-100 Microstructure
# case study using three complementary approaches: HAR volatility decomposition,
# FFT spectral analysis on intraday volume profiles, and path signatures over
# recent price-volume trajectories.
#
# **Learning Objectives**:
# - Fit the HAR(5,15,60) model on intraday realized volatility (walk-forward)
# - Extract rolling FFT spectral features from volume and volatility series
# - Compute depth-2 path signatures on (price, signed_vol, trades) trajectories
# - Combine temporal features with Ch8 cross-sectional features for modeling
#
# **Temporal Models**:
#
# | Model | Role | Output Features |
# |-------|------|-----------------|
# | HAR(5,15,60) | Primary | Conditional vol forecast, HAR residual |
# | FFT/Spectral | Primary | Spectral energy, dominant period, entropy |
# | Path Signatures | Primary | Depth-2 signature terms (6 features per path) |
#
# **Walk-Forward Discipline**:
# - HAR coefficients fitted on rolling OLS windows within each symbol
# - FFT features use only past data (causal rolling windows)
# - Signatures computed on trailing 30-minute windows (no future leakage)
#
# **Output Contract**:
# - `features/model_based.parquet` -- temporal feature matrix with `fold` column for per-fold CV
#
# **Cross-References**:
# - **Upstream**: Ch8 (`03_financial_features.py` for features), Ch7 ([`02_labels`](02_labels.ipynb) for prices)
# - **Downstream**: Ch11+ (models)
# - **Teaching**: [`09_har_rough_volatility`](../../09_model_based_features/09_har_rough_volatility.ipynb), [`05_spectral_features`](../../09_model_based_features/05_spectral_features.ipynb),
#   [`06_path_signatures`](../../09_model_based_features/06_path_signatures.ipynb)

# %%
"""NASDAQ-100 Microstructure: Temporal Features (Ch9)."""

import warnings
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import polars as pl
import yaml
from ml4t.diagnostic.evaluation.stats import benjamini_hochberg_fdr
from ml4t.diagnostic.metrics import compute_ic_hac_stats

from data import load_nasdaq100_bars
from utils.cv_splits import generate_cv_splits
from utils.paths import get_case_study_dir

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "nasdaq100_microstructure"
START_DATE = "2020-01-01"
END_DATE = "2021-12-31"
MAX_SYMBOLS = 0

# %%
# Configuration
CASE_DIR = get_case_study_dir("nasdaq100_microstructure")
FEATURES_DIR = CASE_DIR / "features"

print(f"Date range: {START_DATE} to {END_DATE}")
if MAX_SYMBOLS:
    print(f"Symbol limit: {MAX_SYMBOLS}")

# %% [markdown]
# ## 1. Load and Prepare Data
#
# Load minute bars with microstructure fields. We need the same raw data as
# `03_financial_features.py` to compute temporal features from time-series models
# applied to volatility, volume, and price paths.

# %%
df = load_nasdaq100_bars(
    start_date=START_DATE,
    end_date=str(END_DATE),
    include_microstructure=True,
)

# Optionally restrict universe
if MAX_SYMBOLS:
    top_syms = (
        df.group_by("symbol")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(MAX_SYMBOLS)["symbol"]
        .to_list()
    )
    df = df.filter(pl.col("symbol").is_in(top_syms))
    print(f"Restricted to {MAX_SYMBOLS} symbols: {top_syms}")

# Filter to regular trading hours
df = df.filter(
    (pl.col("timestamp").dt.hour() >= 10)
    | ((pl.col("timestamp").dt.hour() == 9) & (pl.col("timestamp").dt.minute() >= 30))
)
df = df.filter(pl.col("timestamp").dt.hour() < 16)

# Sort and add session_date
df = df.sort(["symbol", "timestamp"])
df = df.with_columns(pl.col("timestamp").dt.date().alias("session_date"))

print(f"Loaded {len(df):,} minute bars")
print(f"Symbols: {df['symbol'].n_unique()}, Sessions: {df['session_date'].n_unique()}")

# %% [markdown]
# ### Derived Fields
#
# Compute midprice, log returns, signed volume, and bar-of-day from raw minute bars.

# %%

# Compute midprice and basic fields needed for temporal features
mid = (pl.col("close_bid_price") + pl.col("close_ask_price")) / 2
df = df.with_columns(mid_close=mid)
df = df.filter(pl.col("mid_close").is_not_null() & (pl.col("mid_close") > 0))

# 1-minute log mid return (session-bounded)
group_cols = ["symbol", "session_date"]
df = df.with_columns(
    r1m=(pl.col("mid_close").log() - pl.col("mid_close").log().shift(1).over(group_cols))
)

# Signed volume for path signatures
signed_vol = (pl.col("trade_at_ask") + pl.col("trade_at_mid_ask")) - (
    pl.col("trade_at_bid") + pl.col("trade_at_bid_mid")
)
df = df.with_columns(
    signed_vol=signed_vol,
    signed_vol_share=(signed_vol / pl.col("volume").clip(lower_bound=1)),
)

# Bar-of-day for session position
df = df.with_columns(
    bar_of_day=pl.col("timestamp").rank("ordinal").over(group_cols).cast(pl.Int32) - 1,
)

# %% [markdown]
# ## 2. HAR(5,15,60) Volatility Model
#
# The Heterogeneous Autoregressive (HAR) model (Corsi, 2009) decomposes
# realized volatility into components at different horizons, reflecting
# the heterogeneous behavior of market participants:
#
# $$RV_{t+1}^{(5)} = c + \beta_5 \, RV_t^{(5)} + \beta_{15} \, RV_t^{(15)} + \beta_{60} \, RV_t^{(60)} + \varepsilon_{t+1}$$
#
# For intraday microstructure data, we use 5-minute, 15-minute, and 60-minute
# realized volatility components (instead of the standard daily/weekly/monthly
# decomposition). The HAR residual is a useful feature: positive residuals
# indicate realized vol exceeded the model forecast (surprise volatility).
#
# **Walk-forward**: HAR coefficients are fitted on a rolling 120-bar
# fixed window within each symbol using OLS.
#
# **Session boundary note**: HAR regressors are computed across all sessions
# concatenated per symbol. Overnight gaps appear as zero-return bars (via
# `nan_to_num`), which biases windows that span overnight gaps. The fraction
# of contaminated rows varies by window size: ~8% for signatures (30-bar),
# ~15% for FFT (60-bar), and ~31% for HAR (120-bar). A production system
# would use session-bounded windowing (as in `03_financial_features.py`); here we
# accept the approximation for teaching clarity. Note that `r1m` itself is
# session-bounded — the contamination is only in the aggregation windows.


# %%
def build_har_features_intraday(
    r1m: np.ndarray, window_5: int = 5, window_15: int = 15, window_60: int = 60
) -> dict[str, np.ndarray]:
    """Build HAR regressors from 1-minute returns.

    Computes realized volatility at 3 horizons by averaging squared returns
    over trailing windows.

    Returns dict with rv_5m, rv_15m, rv_60m arrays.
    """
    n = len(r1m)
    r2 = r1m**2

    rv_5 = np.full(n, np.nan)
    rv_15 = np.full(n, np.nan)
    rv_60 = np.full(n, np.nan)

    for t in range(window_60, n):
        rv_5[t] = np.mean(r2[t - window_5 : t])
        rv_15[t] = np.mean(r2[t - window_15 : t])
        rv_60[t] = np.mean(r2[t - window_60 : t])

    return {"rv_5m": rv_5, "rv_15m": rv_15, "rv_60m": rv_60}


# %% [markdown]
# ### Rolling HAR Fit
#
# Fit the HAR model on rolling OLS windows and produce walk-forward forecasts.


# %%
def fit_har_rolling(
    rv_5: np.ndarray,
    rv_15: np.ndarray,
    rv_60: np.ndarray,
    fit_window: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit HAR model with rolling OLS and produce walk-forward forecasts.

    Returns:
        har_forecast: 1-step-ahead HAR forecast of rv_5m
        har_residual: Actual rv_5m minus HAR forecast (surprise vol)
    """
    n = len(rv_5)
    har_forecast = np.full(n, np.nan)
    har_residual = np.full(n, np.nan)

    for t in range(fit_window + 1, n):
        # Training window: [t - fit_window, t)
        # Target: rv_5[t-fit_window+1 : t+1] (shifted by 1)
        # Regressors: rv_5, rv_15, rv_60 at [t-fit_window : t]
        start = t - fit_window
        y_train = rv_5[start + 1 : t + 1]
        X_train = np.column_stack(
            [
                np.ones(fit_window),
                rv_5[start:t],
                rv_15[start:t],
                rv_60[start:t],
            ]
        )

        # Check for valid data
        valid_mask = np.isfinite(y_train) & np.all(np.isfinite(X_train), axis=1)
        if valid_mask.sum() < 20:
            continue

        y_fit = y_train[valid_mask]
        X_fit = X_train[valid_mask]

        # OLS: beta = (X'X)^-1 X'y
        try:
            beta = np.linalg.lstsq(X_fit, y_fit, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        # har_forecast[t] = E[rv_5_{t+1} | info_t], so residual at t = rv_5[t] - forecast[t-1]
        x_t = np.array([1.0, rv_5[t], rv_15[t], rv_60[t]])
        if np.all(np.isfinite(x_t)):
            har_forecast[t] = x_t @ beta
            if np.isfinite(rv_5[t]):
                har_residual[t] = (
                    rv_5[t] - har_forecast[t - 1] if np.isfinite(har_forecast[t - 1]) else np.nan
                )

    return har_forecast, har_residual


# %% [markdown]
# ### Per-Symbol HAR Pipeline
#
# Combine regressor construction and rolling OLS fit for a single symbol.


# %%
def compute_har_per_symbol(
    symbol_df: pl.DataFrame,
) -> pl.DataFrame:
    """Compute HAR features for a single symbol's data (all sessions combined).

    Returns DataFrame with timestamp, symbol, har_forecast, har_residual columns.
    """
    r1m = symbol_df["r1m"].to_numpy().copy()
    r1m = np.nan_to_num(r1m, nan=0.0)

    # Build HAR regressors
    har_regs = build_har_features_intraday(r1m)

    # Fit HAR with rolling OLS
    har_forecast, har_residual = fit_har_rolling(
        har_regs["rv_5m"],
        har_regs["rv_15m"],
        har_regs["rv_60m"],
        fit_window=120,
    )

    return pl.DataFrame(
        {
            "timestamp": symbol_df["timestamp"],
            "symbol": symbol_df["symbol"],
            "har_rv5_pred": har_forecast,
            "har_residual": har_residual,
        }
    )


# %%
# Compute HAR features per symbol
symbols = df["symbol"].unique().sort().to_list()
har_results = []

for i, sym in enumerate(symbols):
    sym_df = df.filter(pl.col("symbol") == sym).sort("timestamp")
    result = compute_har_per_symbol(sym_df)
    har_results.append(result)
    if (i + 1) % 20 == 0 or (i + 1) == len(symbols):
        print(f"  HAR: {i + 1}/{len(symbols)} symbols processed")

har_df = pl.concat(har_results)

# Convert NaN to null
for c in ["har_rv5_pred", "har_residual"]:
    har_df = har_df.with_columns(pl.col(c).fill_nan(None))

# Verify
valid_forecasts = har_df["har_rv5_pred"].drop_nulls()
print(f"HAR features computed: {len(valid_forecasts):,} valid forecasts out of {len(har_df):,}")

# %%
# Report average HAR persistence coefficients across symbols
# Fit a single HAR on the last complete symbol for representative coefficients
_last_sym_df = df.filter(pl.col("symbol") == symbols[-1]).sort("timestamp")
_r1m = _last_sym_df["r1m"].to_numpy().copy()
_r1m = np.nan_to_num(_r1m, nan=0.0)
_regs = build_har_features_intraday(_r1m)
_valid = np.isfinite(_regs["rv_5m"]) & np.isfinite(_regs["rv_15m"]) & np.isfinite(_regs["rv_60m"])
_n_valid = _valid.sum()
if _n_valid > 200:
    X = np.column_stack(
        [
            np.ones(_n_valid),
            _regs["rv_5m"][_valid],
            _regs["rv_15m"][_valid],
            _regs["rv_60m"][_valid],
        ]
    )
    y = np.roll(_regs["rv_5m"], -1)[_valid]
    y_valid = np.isfinite(y)
    beta = np.linalg.lstsq(X[y_valid], y[y_valid], rcond=None)[0]
    persistence = beta[1] + beta[2] + beta[3]
    print(f"\nHAR coefficients (representative, {symbols[-1]}):")
    print(f"  beta_5={beta[1]:.3f}, beta_15={beta[2]:.3f}, beta_60={beta[3]:.3f}")
    print(f"  Persistence (sum of betas): {persistence:.3f}")
else:
    persistence = float("nan")
    print("Insufficient data for persistence reporting")

# %% [markdown]
# **HAR Interpretation**: Positive HAR residuals (`rv_5_actual > har_forecast`)
# flag *surprise volatility* — periods where realized vol exceeded the model's
# expectation, often associated with news arrivals or sudden liquidity events.
# Negative residuals indicate unusually calm markets relative to the recent
# volatility regime.

# %% [markdown]
# ## 3. FFT Spectral Features on Intraday Volume
#
# Rolling FFT on volume and volatility series captures periodicity in
# intraday activity patterns. The spectral energy, dominant period, and
# spectral entropy provide regime-sensitive conditioning features.
#
# **Causal design**: We compute FFT on a trailing 60-bar window at each
# bar, using only past data. The features capture whether recent activity
# is concentrated at specific frequencies (structured) or dispersed
# (noisy).


# %%
def rolling_fft_features(
    signal: np.ndarray,
    window: int = 60,
) -> dict[str, np.ndarray]:
    """Compute rolling FFT spectral features on a 1D signal.

    Features:
    - spectral_energy: Total power (excluding DC component)
    - dominant_period: Period of the peak frequency (in bars)
    - spectral_entropy: Normalized entropy of power spectrum
    - low_freq_ratio: Fraction of energy below 1/20 cycles/bar
    """
    n = len(signal)
    spectral_energy = np.full(n, np.nan)
    dominant_period = np.full(n, np.nan)
    spectral_entropy = np.full(n, np.nan)
    low_freq_ratio = np.full(n, np.nan)

    for t in range(window, n):
        segment = signal[t - window : t]

        # Skip if all NaN or constant
        if np.all(np.isnan(segment)) or np.nanstd(segment) < 1e-12:
            continue

        # Replace NaN with 0 and detrend
        seg_clean = np.nan_to_num(segment, nan=0.0)
        seg_clean = seg_clean - seg_clean.mean()

        # FFT
        fft_vals = np.fft.rfft(seg_clean)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(window)

        # Total spectral energy (excluding DC)
        total_power = np.sum(power[1:])
        if total_power <= 0:
            continue

        spectral_energy[t] = total_power

        # Dominant period
        dom_idx = np.argmax(power[1:]) + 1
        if freqs[dom_idx] > 0:
            dominant_period[t] = 1.0 / freqs[dom_idx]

        # Spectral entropy
        p_norm = power[1:] / total_power
        p_norm = p_norm[p_norm > 0]
        spectral_entropy[t] = -np.sum(p_norm * np.log(p_norm))

        # Low frequency ratio (below 1/20 cycles/bar)
        low_mask = freqs[1:] < (1.0 / 20.0)
        if low_mask.any():
            low_freq_ratio[t] = np.sum(power[1:][low_mask]) / total_power

    return {
        "spectral_energy": spectral_energy,
        "dominant_period": dominant_period,
        "spectral_entropy": spectral_entropy,
        "low_freq_ratio": low_freq_ratio,
    }


# %% [markdown]
# ### Per-Symbol FFT Pipeline
#
# Compute rolling FFT spectral features on both volume and squared-return series.


# %%
def compute_fft_per_symbol(
    symbol_df: pl.DataFrame,
    window: int = 60,
) -> pl.DataFrame:
    """Compute FFT spectral features on volume and volatility for one symbol.

    Two signal sources:
    - Volume: captures intraday activity pattern periodicity
    - Squared returns: captures volatility clustering frequency structure
    """
    # Log-transform volume for better scaling (raw volume spans many orders of magnitude)
    vol_raw = symbol_df["volume"].to_numpy().astype(float)
    vol_signal = np.log1p(np.clip(vol_raw, 0, None))

    r1m = symbol_df["r1m"].to_numpy().copy()
    r2_signal = np.nan_to_num(r1m, nan=0.0) ** 2

    # FFT on volume
    vol_fft = rolling_fft_features(vol_signal, window=window)

    # FFT on squared returns (volatility proxy)
    r2_fft = rolling_fft_features(r2_signal, window=window)

    return pl.DataFrame(
        {
            "timestamp": symbol_df["timestamp"],
            "symbol": symbol_df["symbol"],
            # Volume spectral features
            "vol_spectral_energy": vol_fft["spectral_energy"],
            "vol_dominant_period": vol_fft["dominant_period"],
            "vol_spectral_entropy": vol_fft["spectral_entropy"],
            "vol_low_freq_ratio": vol_fft["low_freq_ratio"],
            # Volatility spectral features
            "rv_spectral_energy": r2_fft["spectral_energy"],
            "rv_dominant_period": r2_fft["dominant_period"],
            "rv_spectral_entropy": r2_fft["spectral_entropy"],
            "rv_low_freq_ratio": r2_fft["low_freq_ratio"],
        }
    )


# %%
# Compute FFT features per symbol
fft_results = []

for i, sym in enumerate(symbols):
    sym_df = df.filter(pl.col("symbol") == sym).sort("timestamp")
    result = compute_fft_per_symbol(sym_df, window=60)
    fft_results.append(result)
    if (i + 1) % 20 == 0 or (i + 1) == len(symbols):
        print(f"  FFT: {i + 1}/{len(symbols)} symbols processed")

fft_df = pl.concat(fft_results)

# Convert NaN to null
fft_feature_cols = [c for c in fft_df.columns if c not in ["timestamp", "symbol"]]
for c in fft_feature_cols:
    fft_df = fft_df.with_columns(pl.col(c).fill_nan(None))

valid_fft = fft_df["vol_spectral_energy"].drop_nulls()
print(f"FFT features computed: {len(valid_fft):,} valid out of {len(fft_df):,}")

# %% [markdown]
# **FFT Interpretation**: Volume spectral energy captures how structured the
# intraday volume pattern is — high energy means strong periodicity (typical
# for the open/close U-shape), low energy indicates disrupted patterns (news
# days, half sessions). Spectral entropy measures how concentrated vs dispersed
# the frequency content is; low entropy means a single dominant frequency.

# %% [markdown]
# ## 4. Path Signatures (Depth-2)
#
# Path signatures encode the sequential dynamics of multi-dimensional paths
# into a fixed-size feature vector. For a $d$-dimensional path at depth 2,
# the signature has $d + d^2$ terms:
#
# - **Depth 1** ($d$ terms): Net displacement along each dimension
# - **Depth 2** ($d^2$ terms): Cross-integrals capturing lead-lag structure
#
# We compute signatures on 3D paths: (cumulative mid return, cumulative signed
# volume share, cumulative trade count) over trailing 30-minute windows. No
# external library is needed -- depth-2 signatures have a simple closed-form
# implementation.
#
# **Why these dimensions**:
# - **Price** (mid return): What happened to the price
# - **Signed volume**: Whether buying or selling dominated
# - **Trade count**: How intense the activity was
#
# The cross-terms (e.g., price$\times$signed_vol) capture whether price moved
# before or after order flow -- exactly the lead-lag structure relevant for
# microstructure prediction.


# %%
def compute_depth2_signature(path: np.ndarray) -> np.ndarray:
    """Compute the depth-2 truncated signature of a d-dimensional path.

    For a path of shape (T, d), the depth-2 signature has:
    - d terms at depth 1: S^i = X^i_T - X^i_0 (net displacement)
    - d^2 terms at depth 2: S^{i,j} = integral of dX^i * dX^j (iterated integrals)

    Total: d + d^2 features.

    Args:
        path: Array of shape (T, d) representing the multi-dimensional path

    Returns:
        Array of shape (d + d^2,) with signature terms
    """
    T, d = path.shape
    increments = np.diff(path, axis=0)  # (T-1, d)

    # Depth 1: net displacement
    sig1 = path[-1] - path[0]  # (d,)

    # Depth 2: iterated integrals S^{i,j} = sum_{s<t} dX^i_s * dX^j_t
    sig2 = np.zeros((d, d))
    cumsum = np.zeros(d)
    for t in range(len(increments)):
        # S^{i,j} += cumulative_X^i * dX^j_t
        sig2 += np.outer(cumsum, increments[t])
        cumsum += increments[t]

    return np.concatenate([sig1, sig2.ravel()])


# %%
def _window_normalize(x: np.ndarray) -> np.ndarray:
    """Per-window z-score normalization for signature path dimensions."""
    s = np.std(x)
    if s < 1e-12:
        return x - np.mean(x)
    return (x - np.mean(x)) / s


# %% [markdown]
# ### Per-Symbol Signature Pipeline
#
# Compute rolling depth-2 path signatures on (return, signed volume, trades) paths.


# %%
def compute_signatures_per_symbol(
    symbol_df: pl.DataFrame,
    window: int = 30,
) -> pl.DataFrame:
    """Compute rolling depth-2 path signatures for one symbol.

    3D path: (cumulative mid return, cumulative signed vol share, cumulative trade intensity)
    Window: 30 bars (30 minutes)

    Each window is z-score normalized before computing cumulative sums. This makes
    signature terms scale-invariant across windows — depth-1 terms measure relative
    displacement within the window's own distribution, not absolute price moves.
    """
    r1m = symbol_df["r1m"].to_numpy().copy()
    svs = symbol_df["signed_vol_share"].to_numpy().copy()
    trades = symbol_df["total_trades"].to_numpy().astype(float).copy()

    # Normalize trade count to similar scale as other dimensions
    trades_std = np.nanstd(trades)
    if trades_std > 0:
        trades = trades / trades_std

    # Replace NaN with 0
    r1m = np.nan_to_num(r1m, nan=0.0)
    svs = np.nan_to_num(svs, nan=0.0)
    trades = np.nan_to_num(trades, nan=0.0)

    n = len(r1m)
    d = 3  # dimensions
    n_features = d + d * d  # 3 + 9 = 12

    sig_features = np.full((n, n_features), np.nan)

    for t in range(window, n):
        # Extract window segments
        seg_r_raw = r1m[t - window : t]
        seg_svs_raw = svs[t - window : t]
        seg_trades_raw = trades[t - window : t]

        seg_r_norm = _window_normalize(seg_r_raw)
        seg_svs_norm = _window_normalize(seg_svs_raw)
        seg_trades_norm = _window_normalize(seg_trades_raw)

        # Build path as cumulative sums of normalized increments
        seg_r = np.cumsum(seg_r_norm)
        seg_svs = np.cumsum(seg_svs_norm)
        seg_trades = np.cumsum(seg_trades_norm)

        path = np.column_stack([seg_r, seg_svs, seg_trades])

        # Check for degenerate paths
        if np.all(np.abs(np.diff(path, axis=0)) < 1e-12):
            continue

        sig_features[t] = compute_depth2_signature(path)

    # Build column names
    dims = ["ret", "svs", "trd"]
    col_names = []
    # Depth 1
    for i, name in enumerate(dims):
        col_names.append(f"sig1_{name}")
    # Depth 2
    for i, name_i in enumerate(dims):
        for j, name_j in enumerate(dims):
            col_names.append(f"sig2_{name_i}_{name_j}")

    result = {
        "timestamp": symbol_df["timestamp"],
        "symbol": symbol_df["symbol"],
    }
    for k, col_name in enumerate(col_names):
        result[col_name] = sig_features[:, k]

    return pl.DataFrame(result)


# %%
# Compute signature features per symbol
sig_results = []

for i, sym in enumerate(symbols):
    sym_df = df.filter(pl.col("symbol") == sym).sort("timestamp")
    result = compute_signatures_per_symbol(sym_df, window=30)
    sig_results.append(result)
    if (i + 1) % 20 == 0 or (i + 1) == len(symbols):
        print(f"  Signatures: {i + 1}/{len(symbols)} symbols processed")

sig_df = pl.concat(sig_results)

# Convert NaN to null
sig_feature_cols = [c for c in sig_df.columns if c not in ["timestamp", "symbol"]]
for c in sig_feature_cols:
    sig_df = sig_df.with_columns(pl.col(c).fill_nan(None))

valid_sig = sig_df["sig1_ret"].drop_nulls()
print(f"Signature features computed: {len(valid_sig):,} valid out of {len(sig_df):,}")

# %% [markdown]
# **Signature Interpretation**: The depth-2 cross-terms capture lead-lag structure
# between path dimensions. For example:
#
# - `sig2_ret_svs` > 0 means price moved *before* signed volume increased
#   (price leads flow — informed trading)
# - `sig2_svs_ret` > 0 means signed volume increased *before* price moved
#   (flow leads price — classic Kyle model)
#
# The asymmetry between `sig2_ret_svs` and `sig2_svs_ret` is the key
# microstructure signal — it distinguishes informed from liquidity-driven flow.
#
# **Computational note**: Path signatures with 30-bar windows over millions of
# minute bars per symbol are expensive. With ~100 symbols the loop takes several
# minutes. For faster iteration, use TEST mode or consider 5-minute resampling.

# %% [markdown]
# ## 5. Combine Temporal Features
#
# Join all three temporal feature sets on (timestamp, symbol) and produce
# the final temporal feature matrix.

# %%
# Join HAR, FFT, and signature features
temporal_df = har_df.join(fft_df, on=["timestamp", "symbol"], how="inner")
temporal_df = temporal_df.join(sig_df, on=["timestamp", "symbol"], how="inner")

# List all temporal feature columns
meta_cols = ["timestamp", "symbol"]
temporal_feature_cols = [c for c in temporal_df.columns if c not in meta_cols]

# Convert NaN to null for proper Polars null handling
# (numpy NaN values are not treated as Polars null by default)
for col in temporal_feature_cols:
    temporal_df = temporal_df.with_columns(pl.col(col).fill_nan(None))

print(f"Combined temporal features: {len(temporal_feature_cols)}")
print("  HAR: 2 features (har_rv5_pred, har_residual)")
print("  FFT: 8 features (4 volume + 4 volatility spectral)")
print("  Signatures: 12 features (3 depth-1 + 9 depth-2)")
print(f"Total temporal matrix: {temporal_df.shape}")

# %%
# Drop rows where key temporal features are null (warm-up period)
warmup_cols = ["har_rv5_pred", "vol_spectral_energy", "sig1_ret"]
temporal_clean = temporal_df.drop_nulls(subset=warmup_cols)

dropped = len(temporal_df) - len(temporal_clean)
print(f"Rows dropped (warm-up): {dropped:,}")
print(f"Clean rows: {len(temporal_clean):,}")

# Replace infinities with null
for col in temporal_feature_cols:
    if col in temporal_clean.columns:
        temporal_clean = temporal_clean.with_columns(
            pl.when(pl.col(col).is_infinite()).then(None).otherwise(pl.col(col)).alias(col)
        )

# %% [markdown]
# ## 6. Temporal Feature Summary

# %%
# Summary statistics
print("=== Temporal Feature Summary ===\n")
for col in temporal_feature_cols:
    vals = temporal_clean[col].drop_nulls()
    if len(vals) > 0:
        print(
            f"  {col:<25s}: n={len(vals):>10,}, mean={vals.mean():>12.6f}, std={vals.std():>12.6f}"
        )
    else:
        print(f"  {col:<25s}: all null")

# %% [markdown]
# ## 7. Join Validation
#
# Verify that temporal features join correctly with Ch8 features and Ch7 labels.
# This confirms compatible keys (`timestamp`, `symbol`) and overlapping row counts.

# %%
labels_path = CASE_DIR / "labels" / "fwd_ret_15m.parquet"
features_path = CASE_DIR / "features" / "financial.parquet"

if labels_path.exists() and features_path.exists():
    labels_keys = pl.scan_parquet(labels_path).select("timestamp", "symbol").collect()
    features_keys = pl.scan_parquet(features_path).select("timestamp", "symbol").collect()
    temporal_keys = temporal_clean.select("timestamp", "symbol")

    label_feat = features_keys.join(labels_keys, on=["timestamp", "symbol"], how="inner")
    full_join = label_feat.join(temporal_keys, on=["timestamp", "symbol"], how="inner")

    print(f"Labels rows:   {len(labels_keys):>12,}")
    print(f"Features rows: {len(features_keys):>12,}")
    print(f"Temporal rows: {len(temporal_keys):>12,}")
    print(f"All-three join:{len(full_join):>12,}")
    print(f"Join coverage:  {len(full_join) / len(temporal_keys) * 100:.1f}% of temporal rows")
else:
    print(
        "Upstream artifacts not yet available — run 02_labels.py and 03_financial_features.py first"
    )

# %% [markdown]
# ## 8. Incremental IC Evaluation
#
# Evaluate whether temporal features add predictive content beyond the Ch8
# cross-sectional features. We compute the Information Coefficient (IC) —
# cross-sectional Spearman rank correlation between each temporal feature and
# `fwd_ret_15m` — then apply HAC standard errors and FDR correction.
#
# **Comparison**: Load Ch8 feature evaluation results to compare temporal
# feature IC magnitudes against cross-sectional feature ICs.

# %%
# Load labels and join with temporal features
labels_path = CASE_DIR / "labels" / "fwd_ret_15m.parquet"

eval_results = {}  # Store for evaluation metrics

if not labels_path.exists():
    raise FileNotFoundError("Labels not available — run 02_labels.py first.")

labels_15m = pl.read_parquet(labels_path)
eval_df = temporal_clean.join(labels_15m, on=["timestamp", "symbol"], how="inner")
print(f"Evaluation DataFrame: {len(eval_df):,} rows")

# Sample every 15th timestamp for approximate independence
all_timestamps = eval_df["timestamp"].unique().sort()
sample_ts = all_timestamps.gather_every(15)
eval_sample = eval_df.filter(pl.col("timestamp").is_in(sample_ts))
print(f"Sampled {len(sample_ts):,} timestamps ({len(eval_sample):,} rows)")

# Compute IC series for each temporal feature
n_symbols = eval_sample["symbol"].n_unique()
min_cs_size = min(10, n_symbols)

ic_data = {}
for feat in temporal_feature_cols:
    ic_by_ts = (
        eval_sample.filter(pl.col(feat).is_not_null() & pl.col("fwd_ret_15m").is_not_null())
        .group_by("timestamp")
        .agg(
            pl.corr(feat, "fwd_ret_15m", method="spearman").alias("ic"),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") >= min_cs_size)
    )
    if len(ic_by_ts) >= 20:
        ic_data[feat] = ic_by_ts

print(f"IC series computed for {len(ic_data)}/{len(temporal_feature_cols)} features")

# %% [markdown]
# ### HAC-Adjusted Significance + FDR
#
# Apply Newey-West HAC standard errors and Benjamini-Hochberg FDR correction
# to the temporal feature IC series.

# %%
hac_rows = []
for feat, ic_df in ic_data.items():
    stats = compute_ic_hac_stats(ic_df, ic_col="ic", maxlags=26)
    stats["feature"] = feat
    hac_rows.append(stats)

if hac_rows:
    hac_df = pl.DataFrame(hac_rows)
    p_values = hac_df["p_value"].to_list()
    fdr_result = benjamini_hochberg_fdr(p_values, alpha=0.05, return_details=True)
    hac_df = hac_df.with_columns(fdr_significant=pl.Series(fdr_result["rejected"].tolist()))
    hac_df = hac_df.sort(pl.col("mean_ic").abs(), descending=True)
else:
    hac_df = pl.DataFrame(
        schema={
            "feature": pl.Utf8,
            "mean_ic": pl.Float64,
            "hac_se": pl.Float64,
            "t_stat": pl.Float64,
            "p_value": pl.Float64,
            "naive_t_stat": pl.Float64,
            "fdr_significant": pl.Boolean,
        }
    )

# %%
n_tested = len(hac_df)
n_naive_sig = (
    int(hac_df.filter(pl.col("naive_t_stat").abs() > 1.96).shape[0]) if n_tested > 0 else 0
)
n_fdr_sig = int(hac_df.filter(pl.col("fdr_significant")).shape[0]) if n_tested > 0 else 0
if n_tested > 0:
    _hac_mean = hac_df["t_stat"].abs().mean()
    inflation = (
        round(float(hac_df["naive_t_stat"].abs().mean() / _hac_mean), 2)
        if _hac_mean and _hac_mean > 0
        else 1.0
    )
else:
    inflation = 1.0

print(f"\nTemporal features tested: {n_tested}")
print(f"Naive significant (|t|>1.96): {n_naive_sig}")
print(f"FDR significant (alpha=0.05): {n_fdr_sig}")
print(f"Inflation factor (naive/HAC): {inflation:.1f}x")

# %%
if n_tested > 0:
    print("Top temporal features by |IC|:")
    for row in hac_df.head(10).iter_rows(named=True):
        sig = "**" if row["fdr_significant"] else ("*" if abs(row["naive_t_stat"]) > 1.96 else " ")
        print(
            f"  {sig} {row['feature']:<30s} IC={row['mean_ic']:+.5f}  "
            f"HAC t={row['t_stat']:+.2f}  naive t={row['naive_t_stat']:+.2f}"
        )
    print("  ** = FDR-significant  * = naive-significant only")
else:
    print("Insufficient cross-sectional data for IC testing")

# Store evaluation metrics
top_feats = []
for row in hac_df.head(10).iter_rows(named=True):
    top_feats.append(
        {
            "name": row["feature"],
            "ic_mean": round(row["mean_ic"], 5),
            "hac_tstat": round(row["t_stat"], 2),
            "hac_pval": round(row["p_value"], 4),
            "fdr_significant": bool(row["fdr_significant"]),
        }
    )

eval_results = {
    "primary_label": "fwd_ret_15m",
    "n_features_tested": n_tested,
    "n_significant_naive05": n_naive_sig,
    "n_significant_fdr05": n_fdr_sig,
    "inflation_factor": inflation,
    "top_features": top_feats,
}

# %% [markdown]
# ### Incremental Value Assessment
#
# Temporal features capture time-series dynamics (volatility persistence, spectral
# structure, path geometry) that are orthogonal to cross-sectional features (order
# flow, liquidity, microstructure). The IC comparison shows whether these temporal
# patterns carry additional predictive content for 15-minute forward returns.
#
# In a cost-dominant regime like NASDAQ-100 microstructure, even statistically
# significant features may not translate to economic edge. The evaluation here
# establishes *statistical* significance; economic viability is assessed in Ch11+
# when transaction costs are applied to model-generated signals.

# %% [markdown]
# ## 9. Tag with CV Folds and Save
#
# The temporal features (HAR, FFT, signatures) are inherently causal — each
# value depends only on a trailing window of past data — so the computed
# feature values are identical regardless of fold. We still tag rows with a
# `fold` column so that downstream `load_modeling_dataset` can provide
# per-fold temporal features to training functions.
#
# For each fold we include **train + test** rows (not test-only) because
# downstream models need temporal features for the full fold period.

# %%
# Generate CV splits from setup.yaml
setup_cfg = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())
eval_cfg = setup_cfg["evaluation"]

# generate_cv_splits needs a DataFrame with the timestamp column to derive
# fold boundaries. We use temporal_clean since it has the relevant date range.
splits = generate_cv_splits(
    temporal_clean,
    case_study_id=CASE_STUDY_ID,
    label_buffer=setup_cfg.get("labels", {}).get("buffer", "0D"),
    date_col="timestamp",
)

holdout_start = pd.Timestamp(eval_cfg["holdout_start"])
holdout_end = pd.Timestamp(eval_cfg["holdout_end"])

print(f"CV folds: {len(splits)}")
for s in splits:
    print(
        f"  Fold {s['fold']}: train [{s['train_start']} .. {s['train_end']}] "
        f"test [{s['val_start']} .. {s['val_end']}]"
    )
print(f"  Holdout: [{holdout_start.date()} .. {holdout_end.date()}]")

# %%
# Replicate temporal features per fold (train+test period each) + holdout
fold_frames = []

for s in splits:
    train_start = pd.Timestamp(s["train_start"])
    val_end_key = "val_end" if "val_end" in s else "test_end"
    test_end = pd.Timestamp(s[val_end_key])
    fold_df = temporal_clean.filter(
        (pl.col("timestamp") >= train_start) & (pl.col("timestamp") <= test_end)
    ).with_columns(pl.lit(s["fold"]).alias("fold"))
    fold_frames.append(fold_df)
    print(f"  Fold {s['fold']}: {len(fold_df):,} rows (train+test)")

# Holdout fold: train up to holdout_start, test through holdout_end
# Use fold index = n_splits (one past last CV fold)
holdout_fold_idx = len(splits)
last_train_start = pd.Timestamp(splits[-1]["train_start"]) if splits else holdout_start
holdout_df = temporal_clean.filter(
    (pl.col("timestamp") >= last_train_start) & (pl.col("timestamp") <= holdout_end)
).with_columns(pl.lit(holdout_fold_idx).alias("fold"))
fold_frames.append(holdout_df)
print(f"  Holdout fold {holdout_fold_idx}: {len(holdout_df):,} rows")

temporal_with_folds = pl.concat(fold_frames)
print(
    f"\nTotal with folds: {len(temporal_with_folds):,} rows "
    f"({temporal_with_folds['fold'].n_unique()} folds)"
)

# %%
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

output_path = FEATURES_DIR / "model_based.parquet"
temporal_with_folds.write_parquet(output_path)
print(f"Saved: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

# %%
# Results JSON
results = {
    "case_study_id": "nasdaq100_microstructure",
    "chapter": 9,
    "stage": "temporal",
    "timestamp": datetime.now(UTC).isoformat(),
    "git_commit": "unknown",
    "notebook": "case_studies/nasdaq100_microstructure/code/04_temporal.py",
    "summary": {
        "n_rows": len(temporal_with_folds),
        "n_rows_unique": len(temporal_clean),
        "n_features": len(temporal_feature_cols),
        "n_symbols": temporal_clean["symbol"].n_unique(),
        "n_folds": int(temporal_with_folds["fold"].n_unique()),
        "feature_groups": {
            "har": 2,
            "fft_volume": 4,
            "fft_volatility": 4,
            "path_signatures": 12,
        },
    },
    "techniques": [
        "HAR(5,15,60) intraday volatility (Corsi 2009)",
        "Rolling FFT on volume profiles (60-bar window)",
        "Rolling FFT on squared returns (60-bar window)",
        "Depth-2 path signatures on (ret, signed_vol, trades) paths",
        "Walk-forward HAR fitting (120-bar rolling OLS)",
    ],
    "diagnostics": {
        "har_valid_forecasts": int(har_df["har_rv5_pred"].drop_nulls().len()),
        "fft_valid": int(fft_df["vol_spectral_energy"].drop_nulls().len()),
        "sig_valid": int(sig_df["sig1_ret"].drop_nulls().len()),
        "rows_after_warmup": len(temporal_clean),
    },
    "key_findings": [
        f"Total {len(temporal_feature_cols)} temporal features across 3 model families",
        f"HAR(5,15,60): {int(har_df['har_rv5_pred'].drop_nulls().len()):,} valid walk-forward forecasts",
        "FFT on volume reveals intraday periodicity structure",
        "Depth-2 signatures capture price-flow lead-lag in 30-min windows",
        f"Temporal feature matrix: {len(temporal_clean):,} unique rows x {len(temporal_feature_cols)} features",
        f"Saved with fold column: {int(temporal_with_folds['fold'].n_unique())} folds, {len(temporal_with_folds):,} total rows",
    ],
}

# %%
if eval_results:
    results["incremental_evaluation"] = eval_results
    n_fdr_sig_r = eval_results.get("n_significant_fdr05", 0)
    n_tested_r = eval_results.get("n_features_tested", 0)
    infl_r = eval_results.get("inflation_factor", 0)
    results["key_findings"].append(
        f"IC evaluation: {n_fdr_sig_r}/{n_tested_r} features FDR-significant "
        f"(inflation {infl_r:.1f}x)"
    )

# %% [markdown]
# ## Key Takeaways
#
# ### Three Temporal Models for Intraday Microstructure
#
# | Model | Features | Signal Type | Window |
# |-------|----------|-------------|--------|
# | HAR(5,15,60) | Conditional vol, residual | Multi-horizon vol dynamics | 120-bar OLS |
# | FFT | Energy, period, entropy | Frequency structure | 60-bar rolling |
# | Path Signatures | 12 depth-2 terms | Path geometry (lead-lag) | 30-bar trailing |
#
# ### Implementation Highlights
#
# 1. **HAR adapted for intraday**: Uses 5/15/60-minute components instead of
#    daily/weekly/monthly. Walk-forward OLS avoids look-ahead bias
# 2. **Dual FFT targets**: Volume and volatility have different spectral
#    signatures -- volume has strong intraday periodicity (U-shape), while
#    volatility clustering shows in lower frequencies
# 3. **Manual signatures**: Depth-2 closed-form implementation avoids library
#    dependencies. Cross-terms (e.g., sig2_ret_svs) capture whether price
#    moved before or after order flow -- the key microstructure question
# 4. **Walk-forward discipline**: HAR is fitted on rolling windows. FFT and
#    signatures are inherently causal (use only trailing data)
#
# ### Feature Evaluation
#
# 5. **HAC adjustment is essential**: Overlapping 15-minute returns create strong
#    IC autocorrelation; HAC standard errors correct for this
# 6. **Incremental value**: Comparing temporal feature ICs against Ch8
#    cross-sectional feature ICs shows whether time-series dynamics add
#    predictive content beyond liquidity and order flow measures
#
# **Next**: Ch11+ combines Ch8 cross-sectional features with Ch9 temporal
# features for model training.
