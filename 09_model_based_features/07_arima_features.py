# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ARIMA Feature Extraction
#
# **Docker image**: `ml4t`
#
# This notebook demonstrates ARIMA as a **feature extractor** rather than a
# standalone forecaster. The output is the ARIMA point forecast, evaluated as a
# predictive feature (IC/RMSE) across orders and symbols.
#
# **Learning Objectives**:
# - Select ARIMA order using ACF/PACF and information criteria (AIC/BIC)
# - Build ARIMA point forecasts and evaluate them as features (IC, RMSE)
# - Compare AR, AIC-selected, and naive baselines across the ETF universe
# - Diagnose why simple ARIMA mean models extract limited signal from daily returns
#
# **Book Reference**: Chapter 9, Section 9.3 (Volatility Features)
#
# **Prerequisites**: `01_visual_diagnostics` for stationarity testing,
# `03_fractional_differencing` for memory-preserving transforms.

# %% [markdown]
# ## 1. Setup and Imports

# %%
"""ARIMA Feature Extraction — build ARIMA point forecasts and evaluate them as features (IC/RMSE)."""

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from IPython.display import display
from ml4t.diagnostic.evaluation.autocorrelation import analyze_autocorrelation
from ml4t.diagnostic.evaluation.stationarity import analyze_stationarity
from ml4t.diagnostic.metrics import pooled_ic
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, adfuller, pacf

warnings.filterwarnings("ignore", category=FutureWarning)

from data import load_etfs
from utils.paths import get_case_study_dir
from utils.reproducibility import set_global_seeds

# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI
MAX_SYMBOLS = 0  # 0 = all symbols
START_DATE = "2015-01-01"
END_DATE = "2024-12-01"
TEST_START = "2024-01-01"
SEED = 42

# %%
set_global_seeds(SEED)

# Configuration

# ETF symbols
etf_data = load_etfs()
ALL_SYMBOLS = etf_data["symbol"].unique().sort().to_list()
SYMBOLS = ALL_SYMBOLS[:MAX_SYMBOLS] if MAX_SYMBOLS > 0 else ALL_SYMBOLS

# For single-symbol demonstration (used in educational sections)
SYMBOL = "SPY"

print("ARIMA Baseline Configuration:")
print(f"  Symbols: {len(SYMBOLS)} ({', '.join(SYMBOLS)})")
print(f"  Train: {START_DATE} to {TEST_START}")
print(f"  Test: {TEST_START} to {END_DATE}")

# %% [markdown]
# ## 2. Load ETF Price Data
#
# Load from ETF universe for cross-chapter consistency.


# %%
# Filter ETF data to date range (once, for all symbols)
start_dt = pl.col("timestamp") >= pl.lit(START_DATE).str.to_date()
end_dt = pl.col("timestamp") <= pl.lit(END_DATE).str.to_date()
etf_data = etf_data.filter(start_dt & end_dt)

print(f"  ETF data: {len(etf_data):,} observations across {etf_data['symbol'].n_unique()} assets")


def get_symbol_data(symbol: str) -> pd.DataFrame:
    """Extract single symbol from pre-loaded ETF data.

    Uses the already-loaded etf_data for efficiency.
    """
    data = (
        etf_data.filter(pl.col("symbol") == symbol)
        .sort("timestamp")
        .with_columns(returns=pl.col("close").pct_change())
        .drop_nulls()
    )

    # Convert to pandas with date index (statsmodels expects pandas)
    # NOTE: statsmodels ARIMA requires pandas DataFrame with DatetimeIndex
    df = data.select(["timestamp", "close", "returns"]).to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    return df


print(f"Loading {SYMBOL} data from ETF universe...")
df = get_symbol_data(SYMBOL)
print(f"  Observations: {len(df)}")
print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")

# %% [markdown]
# ## 3. Stationarity Testing
#
# ARIMA requires stationary series. We test prices (non-stationary) and returns (stationary).

# %%
print("Augmented Dickey-Fuller Test for Stationarity:")
print("-" * 50)

# Test prices
adf_price = adfuller(df["close"].dropna())
print(f"Prices:   ADF={adf_price[0]:.4f}, p={adf_price[1]:.4f}")
print(f"          {'STATIONARY' if adf_price[1] < 0.05 else 'NON-STATIONARY'}")

# Test returns
adf_ret = adfuller(df["returns"].dropna())
print(f"Returns:  ADF={adf_ret[0]:.4f}, p={adf_ret[1]:.4f}")
print(f"          {'STATIONARY' if adf_ret[1] < 0.05 else 'NON-STATIONARY'}")

print("\n→ We model RETURNS (stationary), not prices.")

# %% [markdown]
# ## 4. ACF/PACF Analysis
#
# ACF helps identify MA order (q), PACF helps identify AR order (p).


# %%
def plot_acf_pacf(series: pd.Series, title: str):
    """Plot ACF and PACF for a series."""
    acf_vals = acf(series.dropna(), nlags=20)
    pacf_vals = pacf(series.dropna(), nlags=20)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF", "PACF"))

    # ACF
    fig.add_trace(
        go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name="ACF"),
        row=1,
        col=1,
    )

    # PACF
    fig.add_trace(
        go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, name="PACF"),
        row=1,
        col=2,
    )

    # Significance bands
    n = len(series)
    sig = 1.96 / np.sqrt(n)
    for col in [1, 2]:
        fig.add_hline(y=sig, line_dash="dash", line_color="red", row=1, col=col)
        fig.add_hline(y=-sig, line_dash="dash", line_color="red", row=1, col=col)

    fig.update_layout(height=300, title_text=title, showlegend=False)
    return fig


fig = plot_acf_pacf(df["returns"], f"{SYMBOL} Returns ACF/PACF")
fig.show()

print("\nACF/PACF Interpretation:")
print("  - Returns show little autocorrelation (efficient market)")
print("  - Low ACF/PACF suggests AR(0) or AR(1) may suffice")
print("  - This is typical: returns are hard to predict from past returns")

# %% [markdown]
# ### ml4t-diagnostic: Automated Order Suggestion
#
# The manual ACF/PACF interpretation above requires visual inspection.
# `analyze_autocorrelation()` examines significant lags programmatically
# and suggests an ARIMA order — a useful sanity check before grid search.

# %%
stat_check = analyze_stationarity(df["returns"].dropna().values)
acf_analysis = analyze_autocorrelation(df["returns"].dropna().values)

print("=== ml4t-diagnostic: Pre-Modeling Diagnostics ===")
print(f"Stationarity: {stat_check.consensus} (agreement: {stat_check.agreement_score:.2f})")
print(f"Suggested ARIMA order: {acf_analysis.suggested_arima_order}")

# %% [markdown]
# The diagnostic confirms returns are stationary (no differencing needed, d=0)
# and suggests a low-order ARIMA — consistent with the efficient market
# expectation of minimal autocorrelation in returns.

# %% [markdown]
# ## 5. Train/Test Split

# %%
# Split at TEST_START
test_start_dt = pd.Timestamp(TEST_START)
train = df[df.index < test_start_dt]
test = df[df.index >= test_start_dt]

print(f"Train: {len(train)} obs ({train.index.min().date()} to {train.index.max().date()})")
print(f"Test:  {len(test)} obs ({test.index.min().date()} to {test.index.max().date()})")

# %% [markdown]
# ## 6. Baseline Models

# %%
results = {}


# Model 1: Naive (predict last value = 0 for returns)
def naive_forecast(train_ret, n_forecast):
    """Naive forecast: predict 0 (random walk for prices = 0 for returns)."""
    return np.zeros(n_forecast)


naive_pred = naive_forecast(train["returns"], len(test))
naive_ic = pooled_ic(test["returns"].values, naive_pred)
naive_rmse = np.sqrt(np.mean((test["returns"].values - naive_pred) ** 2))

results["Naive (0)"] = {"ic": naive_ic, "rmse": naive_rmse, "predictions": naive_pred}
print("Model 1: Naive (predict 0)")
print(f"  IC: {naive_ic:.4f}, RMSE: {naive_rmse:.6f}")


# %%
# Model 2: Naive (predict mean)
def mean_forecast(train_ret, n_forecast):
    """Predict historical mean."""
    return np.full(n_forecast, train_ret.mean())


mean_pred = mean_forecast(train["returns"], len(test))
mean_ic = pooled_ic(test["returns"].values, mean_pred)
mean_rmse = np.sqrt(np.mean((test["returns"].values - mean_pred) ** 2))

results["Mean"] = {"ic": mean_ic, "rmse": mean_rmse, "predictions": mean_pred}
print("\nModel 2: Historical Mean")
print(f"  IC: {mean_ic:.4f}, RMSE: {mean_rmse:.6f}")

# %% [markdown]
# ## 7. AR Models

# %%
# Model 3: AR(1)
print("\nModel 3: AR(1)")
ar1 = ARIMA(train["returns"], order=(1, 0, 0))
ar1_fit = ar1.fit()
ar1_pred = ar1_fit.forecast(steps=len(test))

ar1_ic = pooled_ic(test["returns"].values, ar1_pred.values)
ar1_rmse = np.sqrt(np.mean((test["returns"].values - ar1_pred.values) ** 2))

results["AR(1)"] = {"ic": ar1_ic, "rmse": ar1_rmse, "predictions": ar1_pred.values}
print(f"  AR(1) coef: {ar1_fit.params.get('ar.L1', ar1_fit.params.iloc[1]):.4f}")
print(f"  IC: {ar1_ic:.4f}, RMSE: {ar1_rmse:.6f}")

# Model 4: AR(5)
print("\nModel 4: AR(5)")
ar5 = ARIMA(train["returns"], order=(5, 0, 0))
ar5_fit = ar5.fit()
ar5_pred = ar5_fit.forecast(steps=len(test))

ar5_ic = pooled_ic(test["returns"].values, ar5_pred.values)
ar5_rmse = np.sqrt(np.mean((test["returns"].values - ar5_pred.values) ** 2))

results["AR(5)"] = {"ic": ar5_ic, "rmse": ar5_rmse, "predictions": ar5_pred.values}
print(f"  IC: {ar5_ic:.4f}, RMSE: {ar5_rmse:.6f}")

# %% [markdown]
# ## 8. ARIMA Model Selection
#
# We search over a grid of (p, d, q) and select based on AIC/BIC.

# %%
print("\nModel Selection: Grid Search")
print("-" * 50)

best_aic = np.inf
best_order = None
best_model = None

# Grid search (limited for speed)
p_range = range(0, 4)
q_range = range(0, 3)
d = 0  # Returns are already stationary

model_results = []

for p in p_range:
    for q in q_range:
        try:
            model = ARIMA(train["returns"], order=(p, d, q))
            fit = model.fit()
            model_results.append({"p": p, "q": q, "aic": fit.aic, "bic": fit.bic})

            if fit.aic < best_aic:
                best_aic = fit.aic
                best_order = (p, d, q)
                best_model = fit
        except (ValueError, np.linalg.LinAlgError):
            continue

if model_results:
    model_df = pd.DataFrame(model_results).sort_values("aic")
    print("Top 5 models by AIC:")
    display(model_df.head())

    print(f"Best model: ARIMA{best_order} (AIC: {best_aic:.2f})")

# %%
# Forecast with best model
if best_model:
    best_pred = best_model.forecast(steps=len(test))
    best_ic = pooled_ic(test["returns"].values, best_pred.values)
    best_rmse = np.sqrt(np.mean((test["returns"].values - best_pred.values) ** 2))

    results[f"ARIMA{best_order}"] = {
        "ic": best_ic,
        "rmse": best_rmse,
        "predictions": best_pred.values,
    }
    print(f"\nBest ARIMA{best_order}:")
    print(f"  IC: {best_ic:.4f}, RMSE: {best_rmse:.6f}")

# %% [markdown]
# ## 9. Rolling Forecast (Walk-Forward)
#
# For a more realistic evaluation, we use rolling 1-step-ahead forecasts.

# %%
print("\nRolling 1-Step-Ahead Forecast (Walk-Forward):")
print("-" * 50)

# Rolling forecast with AR(1)
rolling_preds = []
rolling_actuals = []

# Initial training window
window_size = len(train)
full_returns = df["returns"].values

# Rolling forecast iterations
n_rolling = len(test)

for i in range(n_rolling):
    # Fit on expanding window
    train_window = full_returns[: window_size + i]

    # Quick AR(1) fit; statsmodels can fail on rank-deficient or non-stationary
    # windows — fall back to zero forecast in that rare case.
    try:
        model = ARIMA(train_window, order=(1, 0, 0))
        fit = model.fit()
        pred = fit.forecast(steps=1)[0]
    except (ValueError, np.linalg.LinAlgError):
        pred = 0.0

    rolling_preds.append(pred)
    rolling_actuals.append(full_returns[window_size + i])

rolling_preds = np.array(rolling_preds)
rolling_actuals = np.array(rolling_actuals)

rolling_ic = pooled_ic(rolling_actuals, rolling_preds)
rolling_rmse = np.sqrt(np.mean((rolling_actuals - rolling_preds) ** 2))

results["AR(1) Rolling"] = {"ic": rolling_ic, "rmse": rolling_rmse, "predictions": rolling_preds}
print(f"AR(1) Rolling IC: {rolling_ic:.4f}, RMSE: {rolling_rmse:.6f}")

# %% [markdown]
# ## 10. Results Comparison

# %%
summary_rows = [{"model": name, "ic": r["ic"], "rmse": r["rmse"]} for name, r in results.items()]
summary_df = pd.DataFrame(summary_rows)
display(summary_df)

# Best model by IC (skip NaN entries — constant predictions yield undefined IC)
finite = summary_df[summary_df["ic"].notna()]
if not finite.empty:
    best_row = finite.loc[finite["ic"].idxmax()]
    print(
        f"Best by IC (excluding constant predictions): {best_row['model']} ({best_row['ic']:.4f})"
    )

# %% [markdown]
# ## 11. Visualization

# %%
# Plot comparison
models_to_plot = ["Naive (0)", "AR(1)", f"ARIMA{best_order}" if best_order else "AR(1)"]
colors = {"Naive (0)": "gray", "AR(1)": "steelblue", f"ARIMA{best_order}": "coral"}

fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=("Actual vs Predicted Returns (Sample)", "Prediction Distributions"),
)

# Sample of actual vs predicted
n_plot = min(100, len(test))
dates = test.index[:n_plot]

fig.add_trace(
    go.Scatter(
        x=list(range(n_plot)),
        y=test["returns"].values[:n_plot],
        name="Actual",
        line=dict(color="black"),
    ),
    row=1,
    col=1,
)

for model_name in models_to_plot:
    if model_name in results:
        fig.add_trace(
            go.Scatter(
                x=list(range(n_plot)),
                y=results[model_name]["predictions"][:n_plot],
                name=model_name,
                line=dict(dash="dot"),
            ),
            row=1,
            col=1,
        )

# %%
# Histograms
fig.add_trace(
    go.Histogram(x=test["returns"].values, name="Actual", opacity=0.5, nbinsx=50),
    row=2,
    col=1,
)
fig.add_trace(
    go.Histogram(x=results["AR(1)"]["predictions"], name="AR(1) Pred", opacity=0.5, nbinsx=50),
    row=2,
    col=1,
)

fig.update_layout(height=500, title_text="ARIMA Baseline: Predictions vs Actual", barmode="overlay")
fig.show()

# %% [markdown]
# **Reading the table.** AR(1) on this train/test split lands a small positive
# IC (≈ +0.07), which is the most that any single-asset ARIMA model produces
# here. The AIC-best ARIMA — `ARIMA(3, 0, 2)` — flips sign and lands a small
# *negative* IC; AR(5) is also negative. This is a useful illustration of two
# things at once: longer AR/MA orders overfit in-sample at the cost of OOS
# rank correlation, and the AIC-best model is not the IC-best model. The
# Naive (0) and Mean baselines have constant predictions, so their IC is
# undefined (NaN) — they are RMSE benchmarks, not rank-correlation
# benchmarks.
#
# **Why ARIMA struggles with returns on this dataset**:
# 1. **Low autocorrelation**: the ACF of daily SPY returns is statistically
#    indistinguishable from zero at most lags, leaving little linear structure
#    for an ARIMA mean model to exploit
# 2. **Heteroskedasticity**: variance changes over time, so a constant-variance
#    mean model is misspecified (GARCH addresses this in `08_garch_volatility`)
#
# **When ARIMA works**:
# - Trending series (moving averages, cumulative metrics)
# - Seasonal patterns (explicit or multiplicative)
# - Non-financial time series (weather, sales, etc.)

# %% [markdown]
# ## 13. Multi-Symbol ARIMA Forecasting
#
# Process all symbols in parallel and collect predictions for downstream chapters.

# %%
print("\n" + "=" * 60)
print("MULTI-SYMBOL ARIMA FORECASTING")
print("=" * 60)


def run_arima_for_symbol(symbol: str) -> tuple[pl.DataFrame | None, dict]:
    """Run ARIMA forecasting for a single symbol.

    Returns (predictions DataFrame or None, summary dict) — summary is always
    populated with the symbol and an `ic` (NaN if skipped).
    """
    summary = {"symbol": symbol, "n_predictions": 0, "ic": float("nan"), "status": "ok"}

    symbol_df = get_symbol_data(symbol)
    if len(symbol_df) < 252:
        summary["status"] = "insufficient_data"
        return None, summary

    test_start_dt = pd.Timestamp(TEST_START)
    train_data = symbol_df[symbol_df.index < test_start_dt]
    test_data = symbol_df[symbol_df.index >= test_start_dt]

    if len(train_data) < 100 or len(test_data) < 10:
        summary["status"] = "insufficient_split"
        return None, summary

    try:
        model = ARIMA(train_data["returns"], order=(1, 0, 0))
        fit = model.fit()
        preds = fit.forecast(steps=len(test_data))
    except (ValueError, np.linalg.LinAlgError) as exc:
        summary["status"] = f"fit_failed: {type(exc).__name__}"
        return None, summary

    ic = pooled_ic(test_data["returns"].values, preds.values)
    summary["n_predictions"] = len(test_data)
    summary["ic"] = ic

    pred_df = pl.DataFrame(
        {
            "timestamp": test_data.index.values,
            "symbol": symbol,
            "y_true": test_data["returns"].values,
            "y_pred": preds.values,
            "model_id": "arima_ar1",
            "fold_id": 0,
            "horizon": "1d",
            "dataset": "etf",
        }
    )
    return pred_df, summary


# Process all symbols
all_predictions = []
symbol_summaries = []
print(f"Processing {len(SYMBOLS)} symbols...")

for symbol in SYMBOLS:
    pred_df, summary = run_arima_for_symbol(symbol)
    symbol_summaries.append(summary)
    if pred_df is not None:
        all_predictions.append(pred_df)

ic_summary_df = pd.DataFrame(symbol_summaries)
display(ic_summary_df.sort_values("ic", ascending=False, na_position="last"))

# Combine predictions
if all_predictions:
    multi_symbol_df = pl.concat(all_predictions)
    print(
        f"Total predictions: {len(multi_symbol_df):,} rows across {len(all_predictions)} symbols. "
        f"Mean IC: {ic_summary_df['ic'].mean():+.4f}, "
        f"share positive: {(ic_summary_df['ic'] > 0).mean():.1%}"
    )
else:
    multi_symbol_df = pl.DataFrame()
    print("No predictions generated")

# %% [markdown]
# ## 14. Save Predictions for Downstream Chapters
#
# Output standardized predictions file for cross-model comparison.
# Schema: date, asset, y_true, y_pred, model_id, fold_id, horizon, dataset

# %%
# Save ARIMA predictions
MODEL_DIR = get_case_study_dir("etfs") / "models" / "time_series"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

if len(multi_symbol_df) > 0:
    output_path = MODEL_DIR / "arima_predictions.parquet"
    multi_symbol_df.write_parquet(output_path)
    print(f"Saved multi-symbol predictions to {output_path}")
    print(f"  Shape: {multi_symbol_df.shape}")
    print(f"  Assets: {multi_symbol_df['symbol'].n_unique()}")
    print(
        f"  Date range: {multi_symbol_df['timestamp'].min()} to {multi_symbol_df['timestamp'].max()}"
    )
else:
    print("WARNING: No predictions to save")

print("ARIMA baseline demonstration complete")
