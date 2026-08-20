# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     formats: py:percent,ipynb
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
# # OLS and the Inferential Toolkit
#
# **Chapter 11 | Section 11.1**
#
# **Docker image**: `ml4t`
#
# This notebook shows what classical inference looks like before we leave it behind.
# Using the same ETF features and labels as the rest of Chapter 11, we fit a
# statsmodels OLS model and walk through the full inferential toolkit: coefficient
# significance, Gauss-Markov diagnostics, and robust standard errors.
#
# **Learning objectives**
#
# - Interpret a statsmodels OLS summary: coefficients, standard errors, t-statistics, p-values
# - Test the spherical-errors condition on a panel: heteroscedasticity and residual autocorrelation
# - Identify multicollinearity via Variance Inflation Factors (VIF)
# - Separate what robust standard errors repair from what they cannot
# - Understand why inference diagnostics do not answer the prediction question
#
# **Book reference**
#
# Section 11.1 ("From Inference to Prediction"). The next notebook,
# `02_regularization_paths`, transitions to the prediction-oriented framework.
#
# **Prerequisites**
#
# - Familiarity with linear regression and the Gauss-Markov assumptions.
# - Conceptual exposure to heteroscedasticity, autocorrelation, and VIF.
# - ETF case-study features at `case_studies/etfs/features/financial.parquet`
#   (Ch8 output) and labels at `case_studies/etfs/labels/fwd_ret_21d.parquet`
#   (Ch7 output).

# %% [markdown]
# ## Setup

# %%
"""OLS and the Inferential Toolkit - classical inference diagnostics before the prediction pivot."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from ml4t.diagnostic.metrics import compute_ic_hac_stats, cross_sectional_ic_series
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera

from utils.cv_splits import generate_cv_splits
from utils.paths import get_case_study_dir
from utils.reproducibility import set_global_seeds
from utils.style import show_with_alt

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
SEED = 42
MAX_SYMBOLS = 0
MAX_TRAIN_ROWS = 0
VIF_MAX_ROWS = 0

# %%
set_global_seeds(SEED)

# %% [markdown]
# ## Load Features and Labels
#
# We use the same ETF case study data as the rest of Chapter 11: financial
# features from Ch8 and 21-day forward return labels from Ch7.

# %%
CASE_DIR = get_case_study_dir("etfs")

FEATURES_PATH = CASE_DIR / "features" / "financial.parquet"
LABELS_PATH = CASE_DIR / "labels" / "fwd_ret_21d.parquet"

assert FEATURES_PATH.exists(), f"Features not found: {FEATURES_PATH}\nRun Ch8 ETF features first."
assert LABELS_PATH.exists(), f"Labels not found: {LABELS_PATH}\nRun Ch7 ETF labels first."

features_df = pl.read_parquet(FEATURES_PATH).with_columns(pl.col("timestamp").cast(pl.Date))
labels_df = pl.read_parquet(LABELS_PATH).with_columns(pl.col("timestamp").cast(pl.Date))

# %%
TARGET_COL = "fwd_ret_21d"
ASSET_COL = "symbol"
LABEL_HORIZON_DAYS = 21  # the forward window in TARGET_COL; sets every overlap correction below

df = features_df.join(labels_df, on=["timestamp", ASSET_COL], how="inner")

META_COLS = {"timestamp", ASSET_COL, TARGET_COL}
FEATURE_COLS = sorted(c for c in df.columns if c not in META_COLS)

# Drop features that are entirely null (can happen with reduced test universes)
all_null = [c for c in FEATURE_COLS if df[c].null_count() == df.height]
if all_null:
    print(f"Dropping {len(all_null)} all-null features: {all_null}")
    df = df.drop(all_null)
    FEATURE_COLS = [c for c in FEATURE_COLS if c not in all_null]

# Replace inf/NaN with null, then drop all nulls
df = df.with_columns(
    [
        pl.when(pl.col(c).is_nan() | pl.col(c).is_infinite())
        .then(None)
        .otherwise(pl.col(c))
        .alias(c)
        for c in FEATURE_COLS
    ]
)
df = df.drop_nulls(subset=FEATURE_COLS + [TARGET_COL]).sort(["timestamp", ASSET_COL])

# %% [markdown]
# Ch8 builds the `skip_recent` and `mom_accel` features as differences of return
# horizons, so `skip_recent_12_1` is `ret_252d - ret_21d`. Keeping both the raw
# returns and the derived differences makes the design matrix singular, which is
# a violation of the no-perfect-multicollinearity assumption severe enough that
# the robust covariance estimators below cannot be computed at all. We drop the
# five composites.

# %%
REDUNDANT = {
    "mom_accel_short",
    "mom_accel_medium",
    "mom_accel_long",
    "skip_recent_6_1",
    "skip_recent_12_1",
}
n_redundant = len(REDUNDANT & set(FEATURE_COLS))
FEATURE_COLS = [c for c in FEATURE_COLS if c not in REDUNDANT]
if n_redundant:
    print(f"Dropped {n_redundant} linearly dependent features (return-difference composites)")

if MAX_SYMBOLS > 0:
    assets = df[ASSET_COL].unique().sort().head(MAX_SYMBOLS).to_list()
    df = df.filter(pl.col(ASSET_COL).is_in(assets))

print(f"Shape: {df.height:,} rows x {len(FEATURE_COLS)} features")
print(f"Assets: {df[ASSET_COL].n_unique()}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# %% [markdown]
# ### What is in the panel
#
# Two properties of this dataset decide how the diagnostics below have to be
# computed, so we look at them before computing anything.
#
# The first is what the features are. They are not unrelated measurements:
# they come in families that describe the same price history over different
# windows - returns over several horizons, volatility over several windows,
# moving-average ratios, oscillators. Members of a family move together, which
# is what the collinearity diagnostic will find.
#
# The second is how much history each ETF has. The panel is unbalanced. An ETF
# that listed in 2015 has no rows before it, and a row is dropped wherever a
# feature could not be computed, so a symbol's rows can skip trading sessions
# that other symbols have. Any statistic that reads "the previous observation"
# off row order will silently cross those gaps.

# %%
FEATURE_FAMILY_PREFIXES = (
    "ret_",
    "vol_",
    "sharpe_",
    "sma_",
    "ema_",
    "rsi_",
    "bb_",
    "atr_",
    "natr_",
    "max_dd_",
    "volume_",
    "dollar_vol_",
)


def feature_family(name: str) -> str:
    """Group a feature under the price-history family it is computed from."""
    for prefix in FEATURE_FAMILY_PREFIXES:
        if name.startswith(prefix):
            return prefix.rstrip("_")
    return "other"


family_counts = (
    pl.DataFrame({"feature": FEATURE_COLS})
    .with_columns(family=pl.col("feature").map_elements(feature_family, return_dtype=pl.String))
    .group_by("family")
    .agg(pl.len().alias("n_features"), pl.col("feature").sort().str.join(", ").alias("members"))
    .sort("n_features", descending=True)
)
family_counts

# %% [markdown]
# The history each symbol brings. One row per ETF, drawn only over the sessions
# it actually appears on, so a break in a row is a stretch of sessions the panel
# has for other symbols and not for this one.

# %%
sessions_per_symbol = (
    df.group_by(ASSET_COL)
    .agg(
        pl.len().alias("sessions"),
        pl.col("timestamp").min().alias("first"),
        pl.col("timestamp").max().alias("last"),
    )
    .sort(["first", "sessions"])
)

fig, ax = plt.subplots(figsize=(9, 6))
for row, symbol in enumerate(sessions_per_symbol[ASSET_COL].to_list()):
    days = df.filter(pl.col(ASSET_COL) == symbol)["timestamp"].unique().sort().to_numpy()
    breaks = np.flatnonzero(np.diff(days).astype("timedelta64[D]").astype(int) > 5)
    for run in np.split(days, breaks + 1):
        ax.hlines(row, run[0], run[-1], linewidth=1.2)
ax.set_ylim(-1, sessions_per_symbol.height)
ax.set_yticks([])
ax.set_ylabel(f"{sessions_per_symbol.height} ETFs, ordered by first session")
ax.set_xlabel("Session")
ax.set_title("Symbols enter over time and some stop quoting for years")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
show_with_alt(
    fig,
    "One horizontal line per ETF over the sessions it appears on, ordered by first "
    "session. Lines start at different dates and a few break in the middle.",
)

print(f"Sessions in the panel: {df['timestamp'].n_unique():,}")
print(
    "Sessions per symbol: "
    f"shortest {sessions_per_symbol['sessions'].min():,}, "
    f"median {int(sessions_per_symbol['sessions'].median()):,}, "
    f"longest {sessions_per_symbol['sessions'].max():,}"
)

# %% [markdown]
# ## Select a Single Training Fold
#
# Inference is an in-sample exercise - we fit one model on one training window
# and examine its properties. We use the first walk-forward fold and hold back
# the validation set for a prediction comparison at the end.

# %%
splits = generate_cv_splits(df, case_study_id="etfs", label_buffer="21D", date_col="timestamp")

s = splits[0]
tr_start = np.datetime64(s["train_start"])
tr_end = np.datetime64(s["train_end"])
val_start = np.datetime64(s["val_start"])
val_end = np.datetime64(s["val_end"])

dates_np = df["timestamp"].to_numpy()
train_mask = (dates_np >= tr_start) & (dates_np <= tr_end)
val_mask = (dates_np >= val_start) & (dates_np <= val_end)

train_df = df.filter(train_mask)
X_train_raw = train_df.select(FEATURE_COLS).to_numpy()
y_train = train_df[TARGET_COL].to_numpy()
train_meta = train_df.select(["timestamp", ASSET_COL])
X_val_raw = df.filter(val_mask).select(FEATURE_COLS).to_numpy()
y_val = df.filter(val_mask)[TARGET_COL].to_numpy()

if MAX_TRAIN_ROWS > 0 and len(y_train) > MAX_TRAIN_ROWS:
    # Whole symbols, never individual rows: the lag-1 and lag-21 diagnostics below
    # read off each symbol's own series, and a random draw of rows would leave gaps.
    symbols_arr = train_meta[ASSET_COL].to_numpy()
    unique_symbols = np.unique(symbols_arr)
    rows_per_symbol = max(1, len(y_train) // len(unique_symbols))
    n_keep = max(1, min(len(unique_symbols), MAX_TRAIN_ROWS // rows_per_symbol))
    keep = np.random.default_rng(SEED).choice(unique_symbols, size=n_keep, replace=False)
    idx = np.flatnonzero(np.isin(symbols_arr, keep))
    X_train_raw = X_train_raw[idx]
    y_train = y_train[idx]
    train_meta = train_meta[idx]
    print(f"Subsampled to {n_keep} whole symbols ({len(idx):,} rows) for faster diagnostics")

# Standardize features (fitted on training data only)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)

# Keep feature names for statsmodels output (x1/x2/... is hard to interpret).
X_train_df = pd.DataFrame(X_train, columns=FEATURE_COLS)
X_val_df = pd.DataFrame(X_val, columns=FEATURE_COLS)
y_train_s = pd.Series(y_train, name=TARGET_COL)

print(f"Training: {len(y_train):,} observations ({s['train_start']} to {s['train_end']})")
print(f"Validation: {len(y_val):,} observations ({s['val_start']} to {s['val_end']})")

# %% [markdown]
# ## Section 1: Statsmodels OLS Summary
#
# Statsmodels provides the full inferential output that scikit-learn omits:
# coefficient standard errors, t-statistics, p-values, and model-level diagnostics.

# %%
X_train_c = sm.add_constant(X_train_df, has_constant="add")
ols_model = sm.OLS(y_train_s, X_train_c).fit()

# %%
print(ols_model.summary())

# %% [markdown]
# **Reading the summary**: The coefficients table shows many features with small
# p-values in the `P>|t|` column, which might seem to indicate "significant" predictors.
# But the overall $R^2$ is very low - typical for cross-sectional return prediction.
# A large sample (hundreds of thousands of observations) makes even tiny effects
# "significant" in the statistical sense, while the economic magnitude may be
# negligible. Note that even after removing the five exact linear combinations,
# the remaining features are still pervasively correlated (as VIF will confirm),
# so coefficient estimates are unstable across samples.
#
# Crucially, these standard errors assume **spherical errors**: constant variance,
# uncorrelated across observations. We now test that condition, and the
# no-perfect-multicollinearity condition alongside it.

# %% [markdown]
# ## Section 2: Gauss-Markov Diagnostic Battery
#
# Section 11.1 states the four Gauss-Markov assumptions: linearity in the
# parameters, strict exogeneity, no perfect multicollinearity, and spherical
# errors. When all four hold, OLS is the Best Linear Unbiased Estimator (BLUE).
#
# Only two of the four are testable from residuals. Spherical errors decompose
# into constant variance (Breusch-Pagan, below) and no correlation across
# observations (residual autocorrelation, below); multicollinearity is read off
# the design matrix via VIF. Linearity and exogeneity are assumptions about the
# data-generating process that residuals cannot confirm - a misspecified model
# can produce well-behaved residuals.
#
# We also test normality, which is **not** a Gauss-Markov assumption. It is
# required for exact finite-sample t and F distributions, not for BLUE.

# %% [markdown]
# ### Heteroscedasticity: Breusch-Pagan Test
#
# Tests whether the error variance depends on the regressors. Financial returns
# exhibit time-varying volatility (volatility clustering), so we expect rejection.

# %%
residuals = ols_model.resid
bp_stat, bp_pvalue, bp_fstat, bp_fpvalue = het_breuschpagan(residuals, X_train_c)

print(f"Breusch-Pagan LM statistic: {bp_stat:.1f}")
print(f"p-value: {bp_pvalue:.2e}")
print(f"F-statistic: {bp_fstat:.1f} (p = {bp_fpvalue:.2e})")

if bp_pvalue < 0.05:
    print("\nResult: REJECT homoscedasticity.")
    print("Consequence: OLS standard errors are biased - t-stats and p-values are unreliable.")
else:
    print("\nResult: Cannot reject homoscedasticity at 5% level.")

# %% [markdown]
# ### Correlated Errors: Residual Autocorrelation Within Each Asset
#
# This is a panel: many symbols observed on the same dates. The residual vector
# is stored in date order, so its neighbouring entries are *different assets on
# the same day*, not consecutive observations of one asset. A Durbin-Watson or
# Breusch-Godfrey statistic computed on that vector measures cross-sectional
# dependence and reports it under the name of serial correlation.
#
# To test serial correlation we regroup the residuals into one time-ordered
# series per symbol and measure autocorrelation within each. Three lags are
# informative here. Lag 1 asks whether yesterday's error predicts today's. Lag 5
# is one trading week. Lag 21 sits at the label horizon: `fwd_ret_21d` is a
# 21-day forward return, so consecutive daily observations of one asset share 20
# of 21 days of outcome window. That overlap induces autocorrelation by
# construction out to lag 20 and is the dominant source of the correlation we
# find. At lag 21 the outcome windows no longer overlap, so what is left is
# whatever serial correlation the data has of its own.
#
# The lag has to be counted on the session grid the panel is dated on, not on
# row position. As the unbalanced histories above showed, a symbol's rows skip
# sessions wherever a feature could not be computed or the ETF was not trading,
# and three symbols in this training window resume after a break of a year or
# more. Pairing rows by position would file those gap-crossing pairs under "lag
# 1". So each residual is placed at the number of the session it belongs to, and
# a pair counts towards a lag only when its two sessions are exactly that far
# apart. Durbin-Watson is computed the same way, over consecutive-session pairs
# only.

# %%
session_number = {
    day: number for number, day in enumerate(df["timestamp"].unique().sort().to_list())
}
resid_panel = train_meta.with_columns(
    residual=pl.Series(np.asarray(residuals)),
    session=pl.col("timestamp").replace_strict(session_number, return_dtype=pl.Int32),
).sort([ASSET_COL, "session"])


# %%
LAGS = (1, 5, 21)
MIN_PAIRS = 60  # a correlation on fewer pairs than a quarter of sessions says little


# %%
def lagged_correlation(values: np.ndarray, sessions: np.ndarray, lag: int) -> float:
    """Correlate residual pairs exactly `lag` sessions apart, ignoring gaps."""
    on_grid = np.full(sessions[-1] - sessions[0] + 1, np.nan)
    on_grid[sessions - sessions[0]] = values
    earlier, later = on_grid[:-lag], on_grid[lag:]
    both = np.isfinite(earlier) & np.isfinite(later)
    if both.sum() < MIN_PAIRS:
        return float("nan")
    return float(np.corrcoef(earlier[both], later[both])[0, 1])


def gap_aware_durbin_watson(values: np.ndarray, sessions: np.ndarray) -> float:
    """Durbin-Watson counting only pairs on consecutive sessions."""
    consecutive = np.diff(sessions) == 1
    return float((np.diff(values)[consecutive] ** 2).sum() / (values**2).sum())


# %%
per_symbol = {lag: [] for lag in LAGS}
dw_per_symbol = []
for (_symbol,), group in resid_panel.group_by([ASSET_COL], maintain_order=True):
    r = group["residual"].to_numpy()
    sessions = group["session"].to_numpy()
    if len(r) < max(LAGS) + 2:
        continue
    dw_per_symbol.append(gap_aware_durbin_watson(r, sessions))
    for lag in LAGS:
        correlation = lagged_correlation(r, sessions, lag)
        if np.isfinite(correlation):
            per_symbol[lag].append(correlation)

autocorr_df = pl.DataFrame(
    {
        "lag": list(LAGS),
        "symbols_measured": [len(per_symbol[lag]) for lag in LAGS],
        "median_autocorr": [float(np.median(per_symbol[lag])) for lag in LAGS],
        "share_positive": [float(np.mean(np.array(per_symbol[lag]) > 0)) for lag in LAGS],
    }
)

print(f"Symbols with enough history for the test: {len(dw_per_symbol)}")
print(f"Median within-symbol Durbin-Watson: {float(np.median(dw_per_symbol)):.4f}")
print("  (2.0 = no autocorrelation; <2 = positive; >2 = negative)")
autocorr_df

# %% [markdown]
# ### Normality: Jarque-Bera Test
#
# Normality of the errors is not one of the four Gauss-Markov assumptions, and
# OLS remains BLUE without it. It is what makes the t and F statistics follow
# exactly those distributions in a finite sample. At this sample size the
# central limit theorem delivers approximate normality of the coefficient
# estimates regardless, so a rejection here is the least consequential of the
# diagnostics in this section.
#
# Financial returns have fat tails and excess kurtosis, so this will almost
# certainly reject.

# %%
jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals)

print(f"Jarque-Bera statistic: {jb_stat:.1f}")
print(f"p-value: {jb_pvalue:.2e}")
print(f"Skewness: {skew:.3f} (normal = 0)")
print(f"Excess kurtosis: {kurtosis:.3f} (normal = 0)")

if jb_pvalue < 0.05:
    print("\nResult: REJECT normality.")
    print("Consequence: exact finite-sample t and F distributions do not hold; at this")
    print("sample size the central limit theorem makes the normal approximation adequate.")
else:
    print("\nResult: Cannot reject normality at 5% level.")

# %% [markdown]
# ### Multicollinearity: Variance Inflation Factors
#
# VIF measures how much each coefficient's variance is inflated by correlation
# with other features. VIF > 5 is a common concern threshold; VIF > 10 indicates
# severe multicollinearity. Financial features (momentum at different horizons,
# volatility measures) are pervasively correlated.

# %%
X_vif = X_train_c
if VIF_MAX_ROWS > 0 and len(X_vif) > VIF_MAX_ROWS:
    idx = np.random.default_rng(SEED).choice(len(X_vif), size=VIF_MAX_ROWS, replace=False)
    X_vif = X_vif.iloc[idx]
    print(f"VIF computed on {VIF_MAX_ROWS:,} sampled rows")

x_vif_values = X_vif.to_numpy()
vif_values = [variance_inflation_factor(x_vif_values, i) for i in range(1, x_vif_values.shape[1])]
vif_df = pl.DataFrame({"feature": FEATURE_COLS, "vif": vif_values}).sort("vif", descending=True)

n_high = vif_df.filter(pl.col("vif") > 10).height
n_moderate = vif_df.filter((pl.col("vif") > 5) & (pl.col("vif") <= 10)).height

print(f"VIF > 10 (severe): {n_high} features")
print(f"VIF 5-10 (moderate): {n_moderate} features")
vif_df.head(15)

# %% [markdown]
# High VIF confirms what Section 11.1 discusses: financial features are pervasively
# correlated. Momentum at 5-day and 21-day horizons, different volatility measures,
# and volume-price indicators share overlapping information. OLS distributes weight
# arbitrarily among correlated features, producing unstable coefficient estimates
# that flip sign across samples.

# %% [markdown]
# ## Section 3: When Standard Errors Fail - Robust Alternatives
#
# Robust standard errors address exactly one failure: non-spherical errors. The
# point estimates are unchanged and, if linearity and exogeneity hold, they stay
# consistent; only the variance estimate around them is repaired.
#
# They do nothing for the other two assumptions. If the model omits a relevant
# variable or gets the functional form wrong, exogeneity fails, the coefficients
# themselves are not consistent, and a robust standard error is a more careful
# statement about the wrong quantity. Section 11.1 makes this point directly:
# when the model is misspecified, the unbiasedness guarantee fails and the
# coefficients stop being interpretable as structural parameters. No covariance
# estimator recovers it.

# %% [markdown]
# ### Comparing Standard Error Estimates
#
# Each estimator below assumes some pairs of residuals are uncorrelated, and the
# question is whether the diagnostics above have already shown that assumption to
# be false.
#
# - **OLS** (default): every pair uncorrelated, and equal variance throughout
# - **HC3**: still every pair uncorrelated, but variance may vary with the regressors
# - **Clustered by date**: any two observations sharing a date may correlate;
#   different dates may not
# - **Clustered by date and symbol**: adds any two observations of one symbol;
#   still assumes independence when the symbol *and* the date both differ
# - **Driscoll-Kraay**: no independence assumption within a window of nearby
#   dates, across the whole cross-section
#
# That last assumption is the one this panel breaks in a way the others do not.
# `fwd_ret_21d` is a 21-day forward return, so one symbol on date *t* and a
# *different* symbol on date *t+5* share sixteen days of outcome window, and
# whatever moved the market in those sixteen days is in both residuals. Two-way
# clustering treats that pair as independent. Driscoll-Kraay does not: it sums
# the moment conditions across the cross-section within each date and then applies
# a Newey-West correction over dates, so arbitrary cross-sectional correlation and
# serial correlation out to the lag length are both admitted. We set that lag to
# the label horizon, which is where the mechanical overlap ends.
#
# Plain HAC (Newey-West) is the usual answer to autocorrelation, but it reads its
# lags off the row order of a single time series. On a pooled panel stored in date
# order those lags run across the cross-section, so it does not estimate what its
# name promises here. It becomes the right tool once the panel is collapsed to
# one observation per date, which is what the IC calculation below does.

# %%
ols_hc3 = sm.OLS(y_train_s, X_train_c).fit(cov_type="HC3")
date_groups = train_meta["timestamp"].to_physical().to_numpy()
symbol_groups = train_meta[ASSET_COL].cast(pl.Categorical).to_physical().to_numpy()
# Driscoll-Kraay sums within each date, so it needs consecutive period codes and
# rows already ordered by date - which is how `df` was sorted on load.
time_codes = np.unique(date_groups, return_inverse=True)[1]

ols_cluster_date = sm.OLS(y_train_s, X_train_c).fit(
    cov_type="cluster", cov_kwds={"groups": date_groups}
)
# A second cluster dimension needs at least two of it, which a reduced run
# (MAX_SYMBOLS=1, or a small MAX_TRAIN_ROWS) may not leave.
n_symbols_train = int(len(np.unique(symbol_groups)))
if n_symbols_train >= 2:
    ols_cluster_2way = sm.OLS(y_train_s, X_train_c).fit(
        cov_type="cluster", cov_kwds={"groups": np.column_stack([date_groups, symbol_groups])}
    )
    se_2way = ols_cluster_2way.bse[1:]
    n_sig_2way = int((np.abs(ols_cluster_2way.tvalues[1:]) > 1.96).sum())
else:
    se_2way = np.full(len(FEATURE_COLS), np.nan)
    n_sig_2way = None

ols_cluster = sm.OLS(y_train_s, X_train_c).fit(
    cov_type="nw-groupsum", cov_kwds={"time": time_codes, "maxlags": LABEL_HORIZON_DAYS}
)
print(f"Dates:  {len(np.unique(date_groups)):,}")
print(f"Symbols: {n_symbols_train:,}")
print(f"Driscoll-Kraay lags: {LABEL_HORIZON_DAYS} (the label horizon)")

# Compare SEs and t-stats for features (skip constant at index 0)
se_comparison = pl.DataFrame(
    {
        "feature": FEATURE_COLS,
        "coef": ols_model.params[1:],
        "se_ols": ols_model.bse[1:],
        "se_hc3": ols_hc3.bse[1:],
        "se_date": ols_cluster_date.bse[1:],
        "se_2way": se_2way,
        "se_cluster": ols_cluster.bse[1:],
        "t_ols": ols_model.tvalues[1:],
        "t_cluster": ols_cluster.tvalues[1:],
    }
).with_columns(
    se_ratio_cluster=pl.col("se_cluster") / pl.col("se_ols"),
)

# %% [markdown]
# Each estimator admits a different set of correlated pairs, and admitting a pair
# costs precision: the standard error grows. The chart shows, for each estimator,
# how much larger its standard error is than the OLS one, over all
# features. A value of 1 means the estimator finds no dependence to correct for;
# a value of 3 means OLS was overstating that coefficient's precision threefold.

# %%
SE_ESTIMATORS = {
    "HC3": "se_hc3",
    "Clustered by date": "se_date",
    "Clustered by date + symbol": "se_2way",
    "Driscoll-Kraay": "se_cluster",
}

ratios, labels = [], []
for label, column in SE_ESTIMATORS.items():
    ratio = (se_comparison[column] / se_comparison["se_ols"]).to_numpy()
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size:
        ratios.append(ratio)
        labels.append(label)

fig, ax = plt.subplots(figsize=(9, 4))
ax.boxplot(ratios, vert=False, tick_labels=labels, widths=0.6)
ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
ax.set_xlabel("Standard error relative to the OLS standard error")
ax.set_title("Each dependence the estimator admits widens the standard errors")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
show_with_alt(
    fig,
    "Box plots of each estimator's standard error divided by the OLS standard error, "
    "one box per estimator, against a dashed reference line at one.",
)

# %%
# Count how many features change significance at 5% level
n_sig_ols = int((np.abs(ols_model.tvalues[1:]) > 1.96).sum())
n_sig_date = int((np.abs(ols_cluster_date.tvalues[1:]) > 1.96).sum())
n_sig_cluster = int((np.abs(ols_cluster.tvalues[1:]) > 1.96).sum())

print("\nFeatures significant at 5% level:")
print(f"  OLS standard errors:         {n_sig_ols} / {len(FEATURE_COLS)}")
print(f"  Clustered by date:           {n_sig_date} / {len(FEATURE_COLS)}")
if n_sig_2way is not None:
    print(f"  Clustered by date + symbol:  {n_sig_2way} / {len(FEATURE_COLS)}")
print(f"  Driscoll-Kraay:              {n_sig_cluster} / {len(FEATURE_COLS)}")
print(f"  Difference: {n_sig_ols - n_sig_cluster} features lose significance")

# %% [markdown]
# ### Coefficients with Panel-Robust Confidence Intervals
#
# Rank features by absolute coefficient magnitude and show the Driscoll-Kraay
# confidence interval around each estimate. The interval width is set by the
# `CONF_Z` multiplier below; a bar that crosses the dashed line at zero is a
# coefficient whose sign the data does not pin down.

# %%
TOP_COEFS = 20
CONF_Z = 1.96  # two-sided 5% normal critical value

coef_plot = (
    se_comparison.with_columns(abs_coef=pl.col("coef").abs())
    .sort("abs_coef", descending=True)
    .head(TOP_COEFS)
    .sort("coef")
)

y_pos = np.arange(coef_plot.height)
coef_values = coef_plot["coef"].to_numpy()
cluster_se = coef_plot["se_cluster"].to_numpy()

fig, ax = plt.subplots(figsize=(8, 7))
ax.errorbar(
    x=coef_values,
    y=y_pos,
    xerr=CONF_Z * cluster_se,
    fmt="o",
    capsize=3,
    linewidth=1.5,
    markersize=4,
)
ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(coef_plot["feature"].to_list())
ax.set_xlabel("Coefficient estimate (standardized feature, 21-day forward return)")
ax.set_title("Most large coefficients cannot be signed once dependence is admitted")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
show_with_alt(
    fig,
    "The largest coefficient estimates with Driscoll-Kraay confidence intervals, "
    "sorted by value, against a dashed vertical line at zero.",
)

n_crosses_zero = int((np.abs(coef_values) <= CONF_Z * cluster_se).sum())
print(
    f"Of the {TOP_COEFS} largest coefficients, {n_crosses_zero} have a "
    f"Driscoll-Kraay interval that includes zero."
)

# %% [markdown]
# Robust standard errors are typically *larger* than OLS standard errors, making
# fewer features appear significant. This is the correct adjustment: OLS was
# overstating precision by ignoring the correlation structure in the errors.
#
# The econometric toolkit offers further specialized corrections:
#
# - **WLS** (Weighted Least Squares): explicitly models heteroscedasticity when
#   the variance function is known or estimable
# - **GLS/FGLS** (Generalized / Feasible GLS): corrects for both heteroscedasticity
#   and autocorrelation by transforming the model
# - **HAC (Newey-West)**: the standard correction for autocorrelation in a single
#   time series, and the right tool once the panel is collapsed to one series per
#   date - which is exactly what the IC calculation in the next section does
# - **Fama-MacBeth**: estimates the cross-section separately on each date and
#   draws inference from the time series of those estimates, which sidesteps the
#   within-date dependence rather than modelling it
#
# These are valuable tools for the data modeling culture - but they fix *inference*
# quality, not *prediction* quality. A coefficient with correct standard errors
# still tells you about in-sample relationships, not out-of-sample forecasting power.

# %% [markdown]
# ## Section 4: Why This Doesn't Help Prediction
#
# We now compute the one metric that matters for trading: the out-of-sample
# Information Coefficient (IC). Because trading decisions rank assets at each
# decision date, IC is computed cross-sectionally - Spearman rank correlation
# between predicted and realized returns *within* each date - and then summarized
# across the validation period via mean, IR ($\bar{IC} / \sigma_{IC}$), and a
# t-statistic on the IC time series.
#
# That t-statistic needs the same care Section 2 demanded. Each date's IC is
# measured against a 21-day forward return, so consecutive daily ICs share 20 of
# 21 days of outcome window and the series is autocorrelated by construction.
# Dividing by $\sigma_{IC} / \sqrt{n}$ treats the dates as independent and
# overstates significance by roughly the square root of the overlap. We report
# the HAC-corrected statistic with the lag set to the label horizon, and the
# naive one beside it to show the size of the error.
#
# Two dates are dropped before that average is taken. A date where every symbol
# gets the same prediction has no ranking to correlate, so its Spearman
# coefficient is undefined and comes back as NaN; a date with too few symbols
# quoting comes back as null. Both have to go, and in polars they are separate
# values: dropping nulls leaves a NaN in place, and one NaN makes the mean of
# the whole series NaN.

# %%
X_val_c = sm.add_constant(X_val_df, has_constant="add")
y_pred_val = np.asarray(ols_model.predict(X_val_c))

val_meta = df.filter(val_mask).select(["timestamp", ASSET_COL])
pred_df = val_meta.with_columns(prediction=pl.Series(y_pred_val))
ret_df = val_meta.with_columns(forward_return=pl.Series(y_val))

ic_per_date = cross_sectional_ic_series(
    pred_df,
    ret_df,
    pred_col="prediction",
    ret_col="forward_return",
    date_col="timestamp",
    entity_col=ASSET_COL,
)
# Sorted by date because the HAC autocovariance is read off the row order.
ic_clean = ic_per_date.drop_nans("ic").drop_nulls("ic").sort("timestamp")
ic_mean = float(ic_clean["ic"].mean())
ic_std = float(ic_clean["ic"].std())
n_periods = ic_clean.height
ic_ir = ic_mean / ic_std if ic_std > 0 else float("nan")

ic_stats = compute_ic_hac_stats(ic_clean, ic_col="ic", label_horizon=LABEL_HORIZON_DAYS)

r2_train = ols_model.rsquared
r2_val = 1 - np.sum((y_val - y_pred_val) ** 2) / np.sum((y_val - y_val.mean()) ** 2)

print(f"In-sample R²:            {r2_train:.4f}")
print(f"Out-of-sample R²:        {r2_val:.4f}")
print(f"Out-of-sample IC mean:   {ic_mean:.4f}")
print(f"Out-of-sample IC IR:     {ic_ir:.2f}")
print(f"IC t-stat, HAC:          {ic_stats['t_stat']:.2f}  (p = {ic_stats['p_value']:.3f},")
print(f"                          {ic_stats['effective_lags']} lags, n={n_periods} dates)")
print(f"IC t-stat, naive:        {ic_stats['naive_t_stat']:.2f}  (treats dates as independent)")
print(f"Significant features:    {n_sig_cluster} (Driscoll-Kraay)")

# %% [markdown]
# ### Validation Predictions and Residuals
#
# The left panel compares predicted vs realized returns; the right panel shows
# residuals against predictions. We sample for readability on large validation sets.

# %%
MAX_PLOT_POINTS = 20000
if len(y_val) > MAX_PLOT_POINTS:
    idx = np.random.default_rng(SEED).choice(len(y_val), size=MAX_PLOT_POINTS, replace=False)
    y_val_plot = y_val[idx]
    y_pred_plot = y_pred_val[idx]
else:
    y_val_plot = y_val
    y_pred_plot = y_pred_val

residuals_plot = y_val_plot - y_pred_plot

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].scatter(y_pred_plot, y_val_plot, s=8, alpha=0.25, edgecolor="none")
diag_min = min(y_pred_plot.min(), y_val_plot.min())
diag_max = max(y_pred_plot.max(), y_val_plot.max())
axes[0].plot([diag_min, diag_max], [diag_min, diag_max], linestyle="--", linewidth=1.2)
axes[0].set_xlabel("Predicted Return")
axes[0].set_ylabel("Realized Return")
axes[0].set_title("Predictions span a fraction of the realized return range")
axes[0].grid(alpha=0.3)

axes[1].scatter(y_pred_plot, residuals_plot, s=8, alpha=0.25, edgecolor="none")
axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
axes[1].set_xlabel("Predicted Return")
axes[1].set_ylabel("Residual (Realized - Predicted)")
axes[1].set_title("The largest predictions carry the largest negative residuals")
axes[1].grid(alpha=0.3)

plt.tight_layout()
show_with_alt(
    fig,
    "Two scatter plots over the validation set: realized against predicted return "
    "with a 45-degree reference line, and residual against predicted return.",
)

# %% [markdown]
# ### Why out-of-sample $R^2$ can go negative
#
# Both in-sample and out-of-sample $R^2$ use the same formula:
#
# $$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}
#       = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$
#
# The denominator $SS_{\text{tot}}$ measures total variance around the mean; the
# numerator $SS_{\text{res}}$ measures residual variance after the model's
# predictions. When $R^2 = 0$, the model is no better than predicting $\bar{y}$
# for every observation.
#
# **In-sample**, OLS with an intercept guarantees $R^2 \geq 0$. The intercept
# ensures that the predicted mean equals the observed mean
# ($\bar{\hat{y}} = \bar{y}$), so the model can always do at least as well as
# the constant-mean baseline. The fitted coefficients minimize $SS_{\text{res}}$
# by construction.
#
# **Out-of-sample**, both guarantees break. First, the coefficients were optimized
# for the training set, not the validation set - they may amplify noise rather
# than capture signal, making $SS_{\text{res}} > SS_{\text{tot}}$ and driving
# $R^2$ below zero. Second, the intercept was calibrated to the training-period
# mean return. If the validation period has a different mean (as it usually does
# with non-stationary financial data), predictions are systematically shifted to
# the wrong level. This level error inflates $SS_{\text{res}}$ even if the model
# ranks returns correctly.
#
# Both effects are visible in the out-of-sample $R^2$ printed above. Read it
# against zero: below zero, the fitted coefficients and the training-period
# intercept together predict worse than having used the validation period's own
# mean return for every observation.
#
# The IC answers a different question, and the two can disagree. IC (Spearman
# rank correlation) is invariant to level shifts and scaling, so a model whose
# magnitudes are badly calibrated can still rank correctly - which is why it,
# not $R^2$, is the standard metric for cross-sectional return prediction in
# quantitative finance. Read the IC against its HAC p-value rather than its
# sign: this is one validation fold of one case study, and the naive statistic
# printed beside it overstates the evidence, because it counts overlapping
# windows as independent observations.
#
# None of the diagnostic tests - Breusch-Pagan, residual autocorrelation,
# Jarque-Bera, VIF - measure out-of-sample ranking accuracy. A coefficient can be
# statistically significant yet contribute nothing to prediction (large sample,
# tiny effect), and vice versa. The diagnostics tell us whether our *inference*
# about parameter values is reliable; they say nothing about whether those
# parameters produce useful *forecasts*.
#
# This is the core insight of Section 11.1: the inferential toolkit and the
# predictive toolkit answer different questions. For algorithmic trading, the
# relevant question is "does this model rank future returns accurately?" - and
# the relevant tool is out-of-sample evaluation, not hypothesis testing.
#
# Section 11.2 introduces regularization - the tool that directly targets
# prediction quality by trading bias for variance.

# %% [markdown]
# ## Key Takeaways
#
# 1. **Statsmodels provides the full inferential toolkit**: standard errors,
#    t-statistics, p-values, F-tests, and a battery of diagnostic tests that
#    scikit-learn intentionally omits.
#
# 2. **The ETF panel violates the spherical-errors assumption in both of its
#    parts**: the error variance depends on the regressors (Breusch-Pagan
#    rejects) and residuals are autocorrelated within each asset, largely
#    because a 21-day forward label makes consecutive observations overlap.
#    Features are also pervasively collinear (many with VIF > 10).
#
# 3. **A statistic computed on a panel measures what its row order says it
#    measures.** Durbin-Watson on a date-ordered panel reports cross-sectional
#    dependence under the name of serial correlation; the same ordering makes
#    Newey-West lag across assets rather than across time. Regroup by asset, pick
#    a covariance estimator that assumes independence only where the diagnostics
#    did not find dependence, and say which one the number came from. Here the
#    overlapping label correlates residuals across symbols *and* across nearby
#    dates, which rules out clustering on either dimension and leaves
#    Driscoll-Kraay.
#
# 4. **Robust standard errors repair one failure only.** They fix the variance
#    estimate when errors are non-spherical. They cannot rescue a model whose
#    exogeneity or functional form is wrong, where the point estimates
#    themselves are not consistent - and they do not improve predictions.
#
# 5. **For prediction, we need a different approach**: regularization +
#    out-of-sample evaluation. The next notebook introduces Ridge, LASSO,
#    and Elastic Net.
#
# **Next**: `02_regularization_paths` introduces the prediction-oriented framework
# with regularized regression and walk-forward cross-validation.
