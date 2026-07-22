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

# %% [markdown] tags=[]
# # From Signals to Returns: The Reality Check
#
# **Docker image**: `ml4t`
#
# **Purpose**: pedagogical end-to-end backtest comparing ML-generated signals
# against momentum and equal-weight baselines on the ETF panel. Shows that
# positive IC does not guarantee portfolio profitability — turnover and
# transaction costs eat the predictive edge.
#
# **Learning objectives**
#
# - Train Ridge and Logistic models on the canonical 8-fold walk-forward CV
# - Convert signals to long-only top-10 portfolios with equal weights
# - Compute gross / net Sharpe, annualized return, volatility, drawdown, and
#   turnover
# - Quantify how transaction-cost drag (10 bps per side) discriminates
#   high-turnover ML strategies from low-turnover baselines
#
# **Book reference**: Section 11.6 — Linear Models Across Nine Case Studies
# (the chapter synthesis paragraph on IC vs net Sharpe).
#
# **Prerequisites**
#
# - Ch7 21-day forward return labels at `case_studies/etfs/labels/fwd_ret_21d.parquet`
# - Ch8 ETF features at `case_studies/etfs/features/financial.parquet`
# - ETF prices via `data.load_etfs()`
# - `setup.yaml` evaluation section for canonical walk-forward splits
#
# **Caveat**: this is a deliberately simplified backtest for pedagogy.
# Production backtesting with proper execution modeling, slippage, and risk
# management is covered in *Chapter 16*. Portfolio construction with turnover
# constraints is *Chapter 17*; transaction-cost modeling is *Chapter 18*.

# %% tags=[]
"""From Signals to Returns: The Reality Check — pedagogical backtest showing IC does not guarantee profitability."""

import warnings
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from ml4t.diagnostic.metrics import cross_sectional_ic_series
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from data import load_etfs
from utils.cv_splits import generate_cv_splits
from utils.paths import get_case_study_dir
from utils.reproducibility import set_global_seeds
from utils.style import COLORS

# %% tags=["parameters"]
SEED = 42
TOP_N = 10
COST_BPS = 10
MAX_SYMBOLS = 0
MAX_FOLDS = 0

# %% tags=[]
set_global_seeds(SEED)

# %% tags=[]
CASE_DIR = get_case_study_dir("etfs")

TRADING_DAYS_PER_YEAR = 252

# %% [markdown] tags=[]
# ## Load Data
#
# Features from Ch8, labels from Ch7, ETF prices from the canonical loader, and the canonical
# walk-forward CV splits from the `setup.yaml` evaluation section.

# %% tags=[]
features = pl.read_parquet(CASE_DIR / "features" / "financial.parquet")
labels = pl.read_parquet(CASE_DIR / "labels" / "fwd_ret_21d.parquet")

prices = (
    load_etfs()
    .sort("symbol", "timestamp")
    .with_columns(daily_ret=pl.col("close").pct_change().over("symbol"))
)

print(f"Features: {features.shape[0]:,} rows, {features['symbol'].n_unique()} assets")
print(f"Labels:   {labels.shape[0]:,} rows")

# %% tags=[]
# Join features + labels
data = features.join(labels, on=["timestamp", "symbol"], how="inner").drop_nulls(
    subset=["fwd_ret_21d"]
)

if MAX_SYMBOLS > 0:
    keep_assets = data["symbol"].unique().sort().head(MAX_SYMBOLS).to_list()
    data = data.filter(pl.col("symbol").is_in(keep_assets))
    prices = prices.filter(pl.col("symbol").is_in(keep_assets))

EXCLUDE = {"timestamp", "symbol", "regime", "fwd_ret_21d"}
feature_cols = [c for c in data.columns if c not in EXCLUDE and data[c].dtype.is_numeric()]
print(f"Combined: {data.shape[0]:,} rows, {len(feature_cols)} features")

# %% tags=[]
# Generate walk-forward CV splits from setup.yaml evaluation section
splits = generate_cv_splits(data, case_study_id="etfs", label_buffer="21D")
if MAX_FOLDS > 0:
    splits = splits[:MAX_FOLDS]
print(f"CV folds: {len(splits)}")

# %% [markdown] tags=[]
# ## Walk-Forward Prediction (8 Folds)
#
# We use the canonical CV splits from the `setup.yaml` evaluation section: 8 folds, each with
# 10Y training and 1Y validation, stepping forward annually with a 1-month purge gap.
#
# For each fold we train Ridge and Logistic once, then predict across the entire validation window.
# Momentum uses `ret_126d` directly — no training needed.


# %% tags=[]
def rank_top_n(assets, scores, top_n):
    """Select top-N symbols by score, return equal-weight dict."""
    valid = ~np.isnan(scores)
    effective_n = min(top_n, int(valid.sum()))
    if effective_n == 0:
        return {}
    order = np.argsort(-np.where(valid, scores, -np.inf))
    selected = [assets[i] for i in order[:effective_n]]
    w = 1.0 / effective_n
    return {s: w for s in selected}


# %% tags=[]
# Train Ridge and Logistic models on each fold's training set.
# These are simple default hyperparameters (alpha=1.0, C=1.0) for pedagogy —
# production runs would use tuned hyperparameters from the walk-forward pipeline.
fold_models = []

for fold in splits:
    fold_num = fold["fold"]
    train_start = date.fromisoformat(str(fold["train_start"])[:10])
    train_end = date.fromisoformat(str(fold["train_end"])[:10])
    val_start = date.fromisoformat(str(fold["val_start"])[:10])
    val_end = date.fromisoformat(str(fold["val_end"])[:10])

    train = data.filter((pl.col("timestamp") >= train_start) & (pl.col("timestamp") <= train_end))
    val = data.filter((pl.col("timestamp") >= val_start) & (pl.col("timestamp") <= val_end))

    if len(train) == 0 or len(val) == 0:
        print(f"  Fold {fold_num}: skipped (no data)")
        continue

    X_train = np.nan_to_num(train.select(feature_cols).to_numpy(), nan=0.0)
    y_train = train["fwd_ret_21d"].to_numpy()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_s, y_train)

    y_dir = (y_train > 0).astype(int)
    logit = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
    logit.fit(X_train_s, y_dir)

    fold_models.append((fold_num, scaler, ridge, logit, val, len(train)))

train_summary = pl.DataFrame(
    {"Fold": [fm[0] for fm in fold_models], "Train rows": [fm[5] for fm in fold_models]}
)
train_summary

# %% [markdown] tags=[]
# ### Signal-to-Portfolio Conversion
#
# For each month-end rebalance date within the validation window, we rank
# symbols by each signal and select the top-N for equal-weight long portfolios.
# Momentum uses `ret_126d` directly — no model needed.

# %% tags=[]
all_predictions = []
all_weights = []

for fold_num, scaler, ridge, logit, val, _ in fold_models:
    val_dates = val.select("timestamp").unique().sort("timestamp")["timestamp"].to_list()
    reb_dates = []
    for i, d in enumerate(val_dates):
        if i + 1 < len(val_dates):
            if val_dates[i + 1].month != d.month:
                reb_dates.append(d)
    if val_dates:
        reb_dates.append(val_dates[-1])

    for reb_date in reb_dates:
        cs = val.filter(pl.col("timestamp") == reb_date)
        if len(cs) < TOP_N:
            continue

        assets = cs["symbol"].to_list()
        n_assets = len(assets)
        y_actual = cs["fwd_ret_21d"].to_numpy()
        mom_scores = cs["ret_126d"].to_numpy()

        X_val = np.nan_to_num(cs.select(feature_cols).to_numpy(), nan=0.0)
        X_val_s = scaler.transform(X_val)
        ridge_preds = ridge.predict(X_val_s)
        logit_probs = logit.predict_proba(X_val_s)[:, 1]

        all_weights.append((reb_date, "equal", {a: 1.0 / n_assets for a in assets}))
        all_weights.append((reb_date, "momentum", rank_top_n(assets, mom_scores, TOP_N)))
        all_weights.append((reb_date, "ridge", rank_top_n(assets, ridge_preds, TOP_N)))
        all_weights.append((reb_date, "logistic", rank_top_n(assets, logit_probs, TOP_N)))
        all_predictions.append((reb_date, assets, y_actual, mom_scores, ridge_preds, logit_probs))

print(f"Total: {len(all_weights)} weight snapshots across {len(fold_models)} folds")

# %% [markdown] tags=[]
# ## Compute Portfolio Returns
#
# Forward-fill weights to daily frequency and compute daily portfolio returns as
# $r_{p,t} = \sum_i w_{i,t} \cdot r_{i,t}$. Turnover is measured at each rebalance date.

# %% tags=[]
strategies = ["equal", "momentum", "ridge", "logistic"]

# Daily return matrix (date x asset)
daily_rets = (
    prices.select(["timestamp", "symbol", "daily_ret"])
    .pivot(on="symbol", index="timestamp", values="daily_ret")
    .sort("timestamp")
)

all_assets = [c for c in daily_rets.columns if c != "timestamp"]
dates_array = daily_rets["timestamp"].to_list()
ret_matrix = daily_rets.select(all_assets).to_numpy()
sym_to_idx = {s: i for i, s in enumerate(all_assets)}

# %% tags=[]
# Get validation period boundaries
first_val = min(date.fromisoformat(str(s["val_start"])[:10]) for s in splits)
last_val = max(date.fromisoformat(str(s["val_end"])[:10]) for s in splits)

results = {}

for strat in strategies:
    strat_weights = [(d, w) for d, s, w in all_weights if s == strat]
    if not strat_weights:
        continue
    # CV splits arrive newest-first, so sort rebalance snapshots chronologically
    # before the daily simulation walk. Without this, only the most recent fold's
    # weights ever fire because the chronological dates_array loop never sees the
    # earlier-dated snapshots come back in time.
    strat_weights.sort(key=lambda dw: dw[0])

    T = len(dates_array)
    N = len(all_assets)
    port_ret = np.full(T, np.nan)
    turnover_series = np.zeros(T)

    weight_snapshots = []
    for d, w_dict in strat_weights:
        w_arr = np.zeros(N)
        for sym, wt in w_dict.items():
            if sym in sym_to_idx:
                w_arr[sym_to_idx[sym]] = wt
        weight_snapshots.append((d, w_arr))

    current_w = np.zeros(N)
    snap_idx = 0

    for t, d in enumerate(dates_array):
        # Earn today's return on the weights held into today's close FIRST, then
        # rebalance at the close so the new weights apply from the next bar. The
        # signal at a rebalance date is only known at that day's close, so the
        # positions it implies cannot capture the same day's return — applying
        # them before computing the return would be a one-day look-ahead.
        day_rets_row = ret_matrix[t]
        valid = ~np.isnan(day_rets_row)
        if current_w.sum() > 0 and valid.any():
            safe_rets = np.where(valid, day_rets_row, 0.0)
            port_ret[t] = np.dot(current_w, safe_rets)

        if snap_idx < len(weight_snapshots) and d >= weight_snapshots[snap_idx][0]:
            new_w = weight_snapshots[snap_idx][1]
            turnover_series[t] = np.sum(np.abs(new_w - current_w)) / 2.0
            current_w = new_w.copy()
            snap_idx += 1

    results[strat] = {"daily_ret": port_ret, "turnover": turnover_series}

# %% [markdown] tags=[]
# ## Performance Summary
#
# Annualized Sharpe (gross and net of costs), return, volatility, max drawdown,
# and average annual turnover across the 2016–2023 test period.


# %% tags=[]
def max_drawdown(cum_returns):
    """Maximum drawdown from cumulative return series."""
    peak = np.maximum.accumulate(cum_returns)
    dd = (cum_returns - peak) / peak
    return float(np.nanmin(dd))


# %% tags=[]
# Compute annualized metrics for each strategy over the test period
mask = np.array([first_val <= d <= last_val for d in dates_array])
dates_bt = [d for d, m in zip(dates_array, mask, strict=False) if m]

summary_rows = []
for strat in strategies:
    r = results[strat]["daily_ret"]
    to = results[strat]["turnover"]

    r_bt = r[mask]
    to_bt = to[mask]
    valid = ~np.isnan(r_bt)
    r_clean = r_bt[valid]

    if len(r_clean) == 0:
        continue

    ann_ret = float(np.mean(r_clean) * TRADING_DAYS_PER_YEAR)
    ann_vol = float(np.std(r_clean, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe_gross = ann_ret / ann_vol if ann_vol > 0 else 0.0

    # Turnover is one-sided; multiply by 2 so COST_BPS is charged on each leg
    # (buy + sell), i.e. COST_BPS bps per side.
    cost_per_day = to_bt * 2 * COST_BPS / 10_000
    r_net = r_bt - cost_per_day
    r_net_clean = r_net[valid]
    ann_ret_net = float(np.mean(r_net_clean) * TRADING_DAYS_PER_YEAR)
    sharpe_net = ann_ret_net / ann_vol if ann_vol > 0 else 0.0

    cum = np.cumprod(1 + r_clean)
    mdd = max_drawdown(cum)

    n_years = len(r_clean) / TRADING_DAYS_PER_YEAR
    ann_turnover = float(to_bt.sum() / n_years) if n_years > 0 else 0.0

    summary_rows.append(
        {
            "strategy": strat,
            "sharpe_gross": round(sharpe_gross, 2),
            "sharpe_net": round(sharpe_net, 2),
            "ann_return_pct": round(ann_ret * 100, 1),
            "ann_vol_pct": round(ann_vol * 100, 1),
            "max_dd_pct": round(mdd * 100, 1),
            "ann_turnover_pct": round(ann_turnover * 100, 0),
        }
    )

summary = pl.DataFrame(summary_rows)
summary

# %% [markdown] tags=[]
# ## Equity Curves and Turnover
#
# Two-panel comparison: growth of \$1 (top; solid = gross, dashed = net of cost)
# and monthly turnover (bottom) across all four strategies over the 2016–2023
# test period.

# %% tags=[]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[3, 1], sharex=True)

colors = {
    "equal": COLORS["neutral"],
    "momentum": COLORS["blue"],
    "ridge": COLORS["amber"],
    "logistic": COLORS["copper"],
}
labels_map = {
    "equal": "Equal-Weight (1/N)",
    "momentum": "Momentum (ret_126d)",
    "ridge": "Ridge Regression",
    "logistic": "Logistic Regression",
}

for strat in strategies:
    r = results[strat]["daily_ret"][mask]
    to = results[strat]["turnover"][mask]
    valid = ~np.isnan(r)
    # Gross returns (solid lines)
    cum = np.cumprod(1 + np.where(valid, r, 0.0))
    lw = 1.0 if strat == "equal" else 1.5
    alpha = 0.5 if strat == "equal" else 1.0
    ax1.plot(dates_bt, cum, label=labels_map[strat], color=colors[strat], linewidth=lw, alpha=alpha)
    # Net-of-cost returns (dashed lines, skip equal-weight)
    if strat != "equal":
        cost_daily = to * 2 * COST_BPS / 10_000
        r_net = np.where(valid, r - cost_daily, 0.0)
        cum_net = np.cumprod(1 + r_net)
        ax1.plot(dates_bt, cum_net, color=colors[strat], linewidth=1.0, alpha=0.5, linestyle="--")

ax1.set_ylabel(r"Growth of \$1")
ax1.legend(loc="upper left", frameon=False, fontsize=8)
ax1.set_title(
    "ETF Strategies: ML vs Momentum vs Equal-Weight (2016-2023)\n"
    "solid = gross, dashed = net of cost",
    fontsize=11,
)

# Panel (b): monthly turnover bars
for strat in ["momentum", "ridge", "logistic"]:
    to = results[strat]["turnover"][mask]
    reb_mask = to > 0
    reb_dates_plot = [d for d, m in zip(dates_bt, reb_mask, strict=False) if m]
    reb_to_plot = to[reb_mask] * 100
    ax2.bar(
        reb_dates_plot,
        reb_to_plot,
        width=15,
        alpha=0.5,
        label=labels_map[strat],
        color=colors[strat],
    )

ax2.set_ylabel("Turnover (%)")
ax2.set_xlabel("Date")
ax2.legend(loc="upper right", frameon=False, fontsize=8)

fig.tight_layout()
plt.show()

# %% [markdown] tags=[]
# ## IC Comparison
#
# Rolling cross-sectional IC from cached predictions. Positive IC for ML models does not
# guarantee portfolio profitability — turnover costs close the gap.

# %% tags=[]
# Long-format DataFrame across all rebalance dates and assets, with one
# prediction column per signal. Cross-sectional IC per date is then a single
# vectorized call per signal.
panel_rows = []
for reb_date, assets, y_actual, mom_scores, ridge_preds, logit_probs in all_predictions:
    for j, sym in enumerate(assets):
        panel_rows.append(
            {
                "timestamp": reb_date,
                "symbol": sym,
                "fwd_ret": float(y_actual[j]),
                "momentum": float(mom_scores[j]),
                "ridge": float(ridge_preds[j]),
                "logistic": float(logit_probs[j]),
            }
        )

panel_df = pl.DataFrame(panel_rows)
ret_df = panel_df.select(["timestamp", "symbol", "fwd_ret"]).rename({"fwd_ret": "forward_return"})


def _ic_series(signal_col: str) -> pl.DataFrame:
    pred_df = panel_df.select(["timestamp", "symbol", signal_col]).rename(
        {signal_col: "prediction"}
    )
    return cross_sectional_ic_series(
        pred_df,
        ret_df,
        pred_col="prediction",
        ret_col="forward_return",
        date_col="timestamp",
        entity_col="symbol",
        min_obs=5,
    ).select(["timestamp", pl.col("ic").alias(signal_col)])


ic_df = (
    _ic_series("momentum")
    .join(_ic_series("ridge"), on="timestamp", how="full", coalesce=True)
    .join(_ic_series("logistic"), on="timestamp", how="full", coalesce=True)
    .sort("timestamp")
)

for col in ["momentum", "ridge", "logistic"]:
    ic_df = ic_df.with_columns(pl.col(col).rolling_mean(12).alias(f"{col}_12m"))

# %% tags=[]
fig, ax = plt.subplots(figsize=(10, 4))

for col, color, label in [
    ("momentum_12m", COLORS["blue"], "Momentum"),
    ("ridge_12m", COLORS["amber"], "Ridge"),
    ("logistic_12m", COLORS["copper"], "Logistic"),
]:
    vals = ic_df[col].to_numpy()
    dates_ic = ic_df["timestamp"].to_list()
    ax.plot(dates_ic, vals, label=label, color=color)

ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_ylabel("Rolling 12-Month IC (Spearman)")
ax.set_xlabel("Date")
ax.set_title("Cross-Sectional IC: Positive IC Does Not Guarantee Profitability")
ax.legend(frameon=False)
fig.tight_layout()
plt.show()

print("\n=== Mean Cross-Sectional IC ===")
for col in ["momentum", "ridge", "logistic"]:
    mean_ic = ic_df[col].drop_nulls().mean()
    print(f"  {col:12s}: {mean_ic:.4f}")

# %% [markdown] tags=[]
# **Interpretation**: Ridge achieves the highest IC ($+0.022$) yet
# equal-weight — with no informational signal — delivers the best net Sharpe
# (0.64) thanks to its low turnover (~$9\%$/year, the drift correction
# back to $1/N$ at each monthly rebalance). Active strategies turn over far
# more aggressively — momentum at $\approx 491\%$/year, Ridge at
# $\approx 665\%$/year, Logistic at $\approx 678\%$/year, roughly 50–75× the
# equal-weight baseline. The turnover penalty is cleanest between momentum and
# Ridge: they earn the *same* gross Sharpe (0.48), but Ridge nets only 0.41
# versus momentum's 0.43 — the difference is entirely the cost of its higher
# turnover. Logistic's negative cross-sectional IC ($-0.005$) shows that
# probability-based ranking does not translate to a return-ordered cross-section
# on this horizon (its high net Sharpe rides its low realized volatility, not IC).
# The cost gap is clearest in the gross-to-net spread: equal-weight loses ~0.00
# Sharpe, momentum 0.05, Ridge 0.07, Logistic 0.10 — exactly the ordering
# predicted by their turnover. This motivates turnover-penalized objectives and
# portfolio constraints in *Chapters 17–18*.

# %% [markdown] tags=[]
# ## Key Takeaways
#
# 1. **Positive IC does not guarantee portfolio profitability.** Ridge ranks
#    future returns with the highest IC ($+0.022$) yet finishes last on net
#    Sharpe among the four strategies — the rank-correlation signal is real
#    but the implementation costs it.
# 2. **Turnover is the discriminator.** Equal-weight rebalancing back to
#    $1/N$ generates only $\approx 9\%$ annual turnover (drift correction),
#    while momentum runs at $\approx 491\%$ and the ML models at
#    $\approx 665$–$678\%$. The gross-to-net Sharpe spread tracks turnover
#    exactly: equal $\approx 0.00$, momentum $0.05$, Ridge $0.07$, Logistic $0.10$
#    at 10 bps per side.
# 3. **Transaction costs belong in the objective.** Regularization controls
#    coefficient magnitude but not position changes. Turnover-penalized
#    objectives or trading constraints (*Chapters 17–18*) are needed to
#    make ML signals cost-effective on this kind of cross-section.
#
# **Next**: *Chapter 16* develops production backtesting with proper execution
# modeling. *Chapter 17* adds portfolio construction with turnover constraints,
# and *Chapter 18* layers in transaction-cost modeling.
