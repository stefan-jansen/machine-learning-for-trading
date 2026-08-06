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
# # S&P 500 Equity Option Analytics: Temporal Features (GARCH)
#
# **Chapter 9: Model-Based Feature Extraction**
# **Section Reference**: 9.3 - Volatility Features
#
# This notebook fits GJR-GARCH(1,1) per stock to produce forward-looking
# conditional volatility estimates. The primary contribution is an alternative
# VRP feature: `garch_ivrv_spread = iv_30_atm - garch_cond_vol`.
#
# ## Walk-Forward Protocol
# - Fit GJR-GARCH on training window returns (per CV fold)
# - Run variance recursion on full train+test window with frozen parameters
# - No re-estimation within fold (use fitted parameters)
#
# ## Output Features (3)
# - `garch_cond_vol`: 1-step-ahead conditional volatility (annualized)
# - `garch_ivrv_spread`: IV minus GARCH forecast (forward-looking VRP)
# - `garch_vol_surprise`: |return| / garch_cond_vol (news impact)
#
# ## Cross-References
# - **Teaching**: [`08_garch_volatility`](../../09_model_based_features/08_garch_volatility.ipynb)
# - **Upstream**: [`02_labels`](02_labels.ipynb) (label parquet files and canonical splits),
#   `03_financial_features.py`
# - **Downstream**: Ch11+ (ML models use combined features + temporal)

# %%
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from arch import arch_model

from case_studies.utils.cv_window import modeling_fold_boundaries
from data import load_sp500_daily_bars
from utils.artifact_specs import load_setup_config, resolve_label_horizon
from utils.cv_splits import load_evaluation_config
from utils.paths import get_case_study_dir
from utils.style import ml4t_palette


def _normalize_symbol_column(df: pl.DataFrame) -> pl.DataFrame:
    if "symbol" in df.columns:
        return df
    msg = f"Expected symbol-like column in frame, found columns={df.columns}"
    raise KeyError(msg)


# %% tags=["parameters"]
CASE_STUDY_ID = "sp500_equity_option_analytics"
MIN_OBS = 252  # Minimum observations for fitting (relaxed in TEST)
MAX_SYMBOLS = None  # Limit symbols in TEST mode
start_date = "2017-01-01"
end_date = "2021-12-31"

# %%
CASE_DIR = get_case_study_dir("sp500_equity_option_analytics")
LABELS_DIR = CASE_DIR / "labels"
FEATURES_DIR = CASE_DIR / "features"

# %% [markdown]
# ## Configuration
#
# Prices, the canonical fold timeline and the Chapter 8 feature matrix. `MIN_OBS` and
# `MAX_SYMBOLS` are declared in the parameters cell above and bound here, and never
# re-assigned below it, so the value papermill injects is the value the notebook runs on.
#
# Load prices, CV config, and Ch8 features. We need:
# - Prices for GARCH fitting (daily returns)
# - Canonical label timeline for walk-forward fold boundaries
# - Ch8 features for IV data (to compute garch_ivrv_spread)

# %%
# Load prices from canonical loader
prices = load_sp500_daily_bars(start_date=start_date, end_date=end_date)
print(f"Loaded prices: {prices.shape[0]:,} rows, {prices['symbol'].n_unique()} symbols")

prices = prices.sort(["symbol", "timestamp"])

# %% [markdown]
# The producer folds resolve from the same label timeline and split generator every
# downstream model consumer reads, so a fold id means the same thing here as it does
# in `05_evaluation` and in the Ch11 models that join on it.

# %%
SETUP = load_setup_config(CASE_STUDY_ID)
PRIMARY_LABEL = SETUP["labels"]["primary"]
# The outcome horizon, not the CV buffer: `labels.buffer` is 10D here because the
# purge has to clear the longest variant, while this label realizes in 5 sessions.
# Section F's HAC bandwidth is the one that has to cover the overlap.
_horizon = resolve_label_horizon(CASE_STUDY_ID, PRIMARY_LABEL, SETUP)
if not _horizon or not str(_horizon).endswith("D"):
    raise ValueError(f"Expected a daily label horizon for {PRIMARY_LABEL}, found {_horizon!r}")
LABEL_HORIZON = int(str(_horizon)[:-1])

canonical_splits = modeling_fold_boundaries(CASE_STUDY_ID, PRIMARY_LABEL)
if not canonical_splits:
    raise RuntimeError(f"Canonical {PRIMARY_LABEL} modeling folds are unavailable")
evaluation_config = load_evaluation_config(CASE_STUDY_ID)
print(f"Canonical modeling folds: {len(canonical_splits)} validation splits")
print(f"Primary label {PRIMARY_LABEL}, outcome horizon {LABEL_HORIZON} sessions")

# %%
# Load Ch8 features for IV data (needed for garch_ivrv_spread)
features_path = FEATURES_DIR / "financial.parquet"
if features_path.exists():
    ch8_features = _normalize_symbol_column(pl.read_parquet(features_path)).select(
        ["timestamp", "symbol", "iv_30_atm"]
    )
    print(f"Loaded Ch8 features: {ch8_features.shape[0]:,} rows")
    has_iv = True
else:
    print("Ch8 features not found - will compute garch_cond_vol only (no garch_ivrv_spread)")
    ch8_features = None
    has_iv = False

# %% [markdown]
# ## A. Why a fitted feature is different
#
# Every feature in `03_financial_features` is a rule written in advance. These three are not:
# their parameters are estimated from the data, which is what makes the fold contract in B
# load-bearing rather than a formality. A rule cannot leak by being fitted; a GARCH
# parameter can, and only the window it was fitted on decides whether it did.
#
# GARCH is a **secondary** model here. The primary value is in the options-derived features,
# not temporal dynamics, and its narrow role is to replace the VRP denominator with a
# forward-looking volatility estimate rather than a backward-looking one.

# %% [markdown]
# ## B. The fold contract
#
# GARCH is fitted on each fold's training window. For the test window,
# we use the fitted model to generate 1-step-ahead conditional variance
# forecasts without re-estimation.
#
# **Note on holdout fold**: We include a third fold that covers the 2021 holdout
# period. This is a deliberate deviation from the strict "sealed holdout" protocol:
# GARCH fitting uses only training-window prices (no label leakage), so generating
# conditional volatility features for the holdout window is methodologically safe.
# Downstream Ch11+ notebooks expect temporal features to cover the full date range
# including holdout. The holdout fold is flagged with `is_holdout=True` and reported
# separately in results.

# %%
# Build fold boundaries from the canonical backward walk-forward protocol.
folds = [
    {
        "fold": split["fold"],
        "train_start": str(split["train_start"]),
        "train_end": str(split["train_end"]),
        "test_start": str(split["val_start"]),
        "test_end": str(split["val_end"]),
    }
    for split in canonical_splits
]

# Add holdout fold (train on the last development years, test on holdout). The
# training span is the one setup.yaml declares for every other fold rather than a
# second number kept here.
_train_size = str(evaluation_config["train_size"])
if not _train_size.endswith("Y"):
    raise ValueError(f"Expected an annual evaluation.train_size, found {_train_size!r}")
train_years = int(_train_size[:-1])
holdout_start = str(evaluation_config["holdout_start"])[:10]
holdout_end = str(evaluation_config["holdout_end"])[:10]
holdout_train_start = f"{int(holdout_start[:4]) - train_years}-01-01"
holdout_train_end = f"{int(holdout_start[:4]) - 1}-12-31"
folds.append(
    {
        "fold": len(folds),
        "train_start": holdout_train_start,
        "train_end": holdout_train_end,
        "test_start": holdout_start,
        "test_end": holdout_end,
        "is_holdout": True,
    }
)

print(f"Walk-forward folds: {len(folds)}")
for fold in folds:
    print(
        f"  Fold {fold['fold']}: train {fold['train_start']} to {fold['train_end']}, "
        f"test {fold['test_start']} to {fold['test_end']}"
    )

# %% [markdown]
# ### F1. The fold contract
#
# The figure the fold table cannot replace: every fitted parameter comes from the left-hand bar of
# its own row, and the inference bar it feeds sits entirely to the right of it. The holdout rule
# marks where the sealed period begins. The last row is the holdout fold, and it is the one to
# read carefully - it exists because a conditional-volatility feature has to be defined over the
# holdout for a later stage to score it, and it is fitted on development prices only.

# %%
fig = go.Figure()
palette = ml4t_palette(2)
for row, fold in enumerate(folds):
    label = f"fold {fold['fold']}"
    for span, colour, name in (
        (("train_start", "train_end"), palette[0], "fitted on"),
        (("test_start", "test_end"), palette[1], "inferred over"),
    ):
        fig.add_trace(
            go.Scatter(
                x=[fold[span[0]], fold[span[1]]],
                y=[label, label],
                mode="lines",
                line=dict(color=colour, width=14),
                name=name,
                showlegend=row == 0,
            )
        )
fig.add_vline(x=holdout_start, line_dash="dash", line_color="crimson")
fig.update_layout(
    title="No parameter comes from the right of its own training bar",
    xaxis_title=f"Fitted and inference spans per fold; the rule marks {holdout_start}",
    height=320,
)
fig.update_yaxes(autorange="reversed")
fig.show()

# %% [markdown]
# ## C. One section per model
#
# The GJR-GARCH(1,1) conditional variance model:
#
# $$\sigma^2_t = \omega + (\alpha + \gamma \mathbb{1}_{r_{t-1}<0}) r^2_{t-1} + \beta \sigma^2_{t-1}$$
#
# where $\gamma > 0$ captures the asymmetric leverage effect: negative returns
# increase volatility more than positive returns of equal magnitude. This is
# well-documented for equities (Glosten, Jagannathan, and Runkle 1993).


# %%
def fit_gjr_garch_for_symbol(
    returns_pd: pd.Series,
) -> dict | None:
    """Fit GJR-GARCH(1,1) on a pandas Series of returns (scaled by 100).

    Args:
        returns_pd: Daily log returns * 100 (percentage scale for numerical stability)

    Returns:
        Dict with fitted parameters, or None if fitting fails
    """
    if len(returns_pd) < MIN_OBS:
        return None

    try:
        model = arch_model(
            returns_pd,
            mean="Constant",
            vol="GARCH",
            p=1,
            o=1,
            q=1,  # GJR-GARCH: p=1, o=1 (asymmetric), q=1
            dist="Normal",
        )
        result = model.fit(disp="off", show_warning=False)

        return {
            "result": result,
            "alpha": result.params.get("alpha[1]", 0),
            "gamma": result.params.get("gamma[1]", 0),  # Asymmetry parameter
            "beta": result.params.get("beta[1]", 0),
            "persistence": (
                result.params.get("alpha[1]", 0)
                + result.params.get("gamma[1]", 0) / 2
                + result.params.get("beta[1]", 0)
            ),
        }
    except Exception:
        return None


# %%
def generate_garch_forecasts(
    symbol_prices: pl.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> pl.DataFrame | None:
    """Fit GARCH on train window, produce conditional vol for train+test.

    Fits GJR-GARCH on training data, then uses model.fix() to run the variance
    recursion over the full train+test window with frozen parameters. This
    produces conditional volatility for both periods so downstream models have
    features available during training (not just test).

    Args:
        symbol_prices: DataFrame with date, close for a single symbol
        train_start/end: Training period boundaries
        test_start/end: Test period boundaries

    Returns:
        DataFrame with date, garch_cond_vol (annualized) for train+test period,
        or None if fitting fails
    """
    # Compute log returns scaled by 100 for numerical stability
    df = (
        symbol_prices.sort("timestamp")
        .with_columns((pl.col("close").log().diff() * 100).alias("ret_pct"))
        .drop_nulls()
    )

    # Need full data for the model; last_obs controls where fitting stops
    full = df.select(["timestamp", "ret_pct"]).to_pandas().set_index("timestamp")["ret_pct"]
    train_start_dt = pd.Timestamp(train_start)
    train_end_dt = pd.Timestamp(train_end)
    test_end_dt = pd.Timestamp(test_end)

    train_mask = (full.index >= train_start_dt) & (full.index <= train_end_dt)
    if train_mask.sum() < MIN_OBS:
        return None

    train_returns = full[train_mask]

    # Fit GJR-GARCH on training data only
    try:
        train_model = arch_model(
            train_returns,
            mean="Constant",
            vol="GARCH",
            p=1,
            o=1,
            q=1,
            dist="Normal",
        )
        train_result = train_model.fit(disp="off", show_warning=False)

        # Run variance recursion on full train+test window with frozen params
        full_mask = (full.index >= train_start_dt) & (full.index <= test_end_dt)
        full_returns = full[full_mask]

        if len(full_returns) == 0:
            return None

        full_model = arch_model(
            full_returns,
            mean="Constant",
            vol="GARCH",
            p=1,
            o=1,
            q=1,
            dist="Normal",
        )
        fixed_result = full_model.fix(train_result.params)
        full_vol = fixed_result.conditional_volatility

        if len(full_vol) == 0:
            return None

        # Convert to annualized vol: cond_vol is in percentage scale, so /100 * sqrt(252)
        annualized_vol = full_vol.values * np.sqrt(252) / 100

        return pl.DataFrame(
            {
                "timestamp": full_vol.index.values,
                "garch_cond_vol": annualized_vol,
            }
        ).with_columns(pl.col("timestamp").cast(pl.Date))

    except Exception:
        return None


# %% [markdown]
# ### C.1 Walk-forward fitting
#
# For each fold, fit GJR-GARCH per symbol on the training window,
# then run the variance recursion on the full train+test window
# with frozen parameters to produce features for both periods.

# %%
symbols = prices["symbol"].unique().sort().to_list()
if MAX_SYMBOLS is not None:
    symbols = symbols[:MAX_SYMBOLS]
    print(f"TEST mode: limiting to {MAX_SYMBOLS} symbols")

print(f"\nProcessing {len(symbols)} symbols across {len(folds)} folds...")

all_garch_results = []
fit_stats = {"success": 0, "fail": 0, "skip": 0}

for fold in folds:
    fold_id = fold["fold"]
    print(
        f"\n=== Fold {fold_id}: train {fold['train_start']}-{fold['train_end']}, "
        f"test {fold['test_start']}-{fold['test_end']} ==="
    )

    fold_results = []

    for i, symbol in enumerate(symbols):
        symbol_prices = prices.filter(pl.col("symbol") == symbol).select(["timestamp", "close"])

        result = generate_garch_forecasts(
            symbol_prices,
            fold["train_start"],
            fold["train_end"],
            fold["test_start"],
            fold["test_end"],
        )

        if result is not None and len(result) > 0:
            result = result.with_columns(
                pl.lit(symbol).alias("symbol"),
                pl.lit(fold_id).alias("fold"),
            )
            fold_results.append(result)
            fit_stats["success"] += 1
        else:
            fit_stats["fail"] += 1

        # Progress update every 100 symbols
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(symbols)} symbols...")

    if fold_results:
        fold_df = pl.concat(fold_results)
        all_garch_results.append(fold_df)
        print(
            f"  Fold {fold_id}: {fold_df['symbol'].n_unique()} symbols, {len(fold_df):,} forecasts"
        )

success_rate = fit_stats["success"] / max(fit_stats["success"] + fit_stats["fail"], 1)
print(
    f"\nFitting stats: {fit_stats['success']} success, {fit_stats['fail']} fail ({success_rate:.1%})"
)

# %% [markdown]
# ## D. Fit stability across folds
#
# Sample symbol-level GARCH parameters from the first fold to characterize
# the fitted models. The persistence parameter $\alpha + \gamma/2 + \beta$
# should be close to but below one for well-behaved volatility dynamics.

# %%
# Fit a sample of symbols on fold 0 to extract parameter estimates
if len(folds) > 0:
    from datetime import date as dt_date

    fold0 = folds[0]
    _t_start = dt_date.fromisoformat(fold0["train_start"])
    _t_end = dt_date.fromisoformat(fold0["train_end"])
    sample_symbols = symbols[: min(len(symbols), 200)]
    persistence_vals = []

    for symbol in sample_symbols:
        sym_prices = prices.filter(pl.col("symbol") == symbol).select(["timestamp", "close"])
        train_rets = (
            sym_prices.sort("timestamp")
            .with_columns((pl.col("close").log().diff() * 100).alias("ret_pct"))
            .filter((pl.col("timestamp") >= _t_start) & (pl.col("timestamp") <= _t_end))
            .drop_nulls()
        )
        if len(train_rets) >= MIN_OBS:
            fit_result = fit_gjr_garch_for_symbol(train_rets["ret_pct"].to_pandas())
            if fit_result is not None:
                persistence_vals.append(
                    {
                        "symbol": symbol,
                        "alpha": fit_result["alpha"],
                        "gamma": fit_result["gamma"],
                        "beta": fit_result["beta"],
                        "persistence": fit_result["persistence"],
                    }
                )

    if persistence_vals:
        persist_df = pl.DataFrame(persistence_vals)
        median_persist = float(persist_df["persistence"].median())
        median_alpha = float(persist_df["alpha"].median())
        median_gamma = float(persist_df["gamma"].median())
        median_beta = float(persist_df["beta"].median())
        print(f"\nGARCH Persistence (n={len(persist_df)} symbols, fold 0):")
        print(f"  Median persistence (α + γ/2 + β): {median_persist:.4f}")
        print(f"  Median α: {median_alpha:.4f}, γ: {median_gamma:.4f}, β: {median_beta:.4f}")
    else:
        median_persist = median_alpha = median_gamma = median_beta = None
        print("No persistence diagnostics (no successful fits)")
else:
    persist_df = None
    median_persist = median_alpha = median_gamma = median_beta = None

# %% [markdown]
# The histogram below shows the fitted persistence ($\alpha + \gamma/2 + \beta$)
# across the sampled stocks. The mass sits just below one, the signature of
# highly persistent equity volatility: shocks decay slowly, so a conditional-vol
# forecast carries real information about tomorrow's variance.

# %%
if persist_df is not None and median_persist is not None:
    fig = go.Figure()
    fig.add_histogram(
        x=persist_df["persistence"].to_list(),
        marker_color=ml4t_palette(1)[0],
        nbinsx=30,
    )
    fig.add_vline(
        x=median_persist,
        line_width=2,
        line_dash="dash",
        line_color="black",
        annotation_text=f"median {median_persist:.2f}",
        annotation_position="top left",
    )
    fig.update_layout(
        title="Fitted GJR-GARCH persistence concentrates just below one",
        xaxis_title="GJR-GARCH persistence (α + γ/2 + β); dashed rule marks the median",
        yaxis_title="Number of stocks",
        height=420,
    )
    fig.show()

# %% [markdown]
# The success rate indicates how many symbol-fold combinations produced valid
# GARCH estimates. Failures typically occur for stocks with insufficient history
# (new listings) or extreme return patterns that prevent convergence. Downstream
# models should treat missing GARCH features as informative (potentially a quality signal).

# %%
# Combine all fold results
if all_garch_results:
    garch_df = pl.concat(all_garch_results).sort(["symbol", "timestamp"])
    print(f"Total GARCH forecasts: {len(garch_df):,} rows, {garch_df['symbol'].n_unique()} symbols")
else:
    print("No GARCH results - creating empty DataFrame")
    garch_df = pl.DataFrame(
        {
            "timestamp": pl.Series([], dtype=pl.Date),
            "symbol": pl.Series([], dtype=pl.Utf8),
            "garch_cond_vol": pl.Series([], dtype=pl.Float64),
            "fold": pl.Series([], dtype=pl.Int32),
        }
    )

# %% [markdown]
# ## E. Combine and emit
#
# Using the GARCH conditional volatility, compute:
# - `garch_ivrv_spread`: IV - GARCH forecast (forward-looking VRP)
# - `garch_vol_surprise`: |return| / GARCH forecast (news impact)

# %%
# Add daily returns for vol surprise
returns = (
    prices.sort(["symbol", "timestamp"])
    .with_columns(pl.col("close").pct_change().over("symbol").alias("_daily_ret"))
    .select(["timestamp", "symbol", "_daily_ret"])
)

temporal = garch_df.join(returns, on=["timestamp", "symbol"], how="left")

# Vol surprise: |actual return| / predicted vol
temporal = temporal.with_columns(
    (
        pl.col("_daily_ret").abs()
        / pl.col("garch_cond_vol").clip(lower_bound=0.001)
        * (252**0.5)  # Annualize daily return for comparison
    ).alias("garch_vol_surprise")
)

# Forward-looking VRP: IV - GARCH conditional vol
if has_iv and ch8_features is not None:
    temporal = temporal.join(ch8_features, on=["timestamp", "symbol"], how="left")
    temporal = temporal.with_columns(
        (pl.col("iv_30_atm") - pl.col("garch_cond_vol")).alias("garch_ivrv_spread")
    ).drop("iv_30_atm")
    print("Computed garch_ivrv_spread (forward-looking VRP)")
else:
    print("Skipping garch_ivrv_spread (no Ch8 IV data available)")

# Drop intermediate columns
temporal = temporal.drop("_daily_ret")

print(f"\nTemporal features: {temporal.shape}")
print(f"Columns: {temporal.columns}")

# %% [markdown]
# ### E.1 Feature summary

# %%
feat_cols = [c for c in temporal.columns if c not in ("timestamp", "symbol", "fold")]
summary_rows = []
for col in feat_cols:
    stats = temporal.select(
        pl.col(col).is_not_null().sum().alias("n_valid"),
        pl.col(col).mean().alias("mean"),
        pl.col(col).std().alias("std"),
        pl.col(col).median().alias("median"),
    ).row(0, named=True)
    stats["feature"] = col
    stats["coverage_pct"] = round(stats["n_valid"] / max(len(temporal), 1) * 100, 1)
    summary_rows.append(stats)

summary_df = pl.DataFrame(summary_rows).select(
    "feature", "n_valid", "coverage_pct", "mean", "std", "median"
)
print(summary_df)

# %% [markdown]
# ### Downstream Merge Strategy
#
# Temporal features cover **both train and test periods** for each fold. GARCH
# parameters are fitted on training data only, then `model.fix()` runs the
# variance recursion over the full train+test window with frozen parameters.
# This means conditional vol estimates for the training period use only the
# model's own parameters (no re-estimation), providing valid features for
# downstream model training.
#
# The `fold` column is preserved in `model_based.parquet` so downstream
# consumers can join per-fold features correctly.
#
# Downstream consumers (Ch11+ models) should:
# - **Left-join** `model_based.parquet` onto `financial.parquet` by
#   `[timestamp, symbol, fold]`
# - The `fold` column ensures each CV fold gets its own GARCH features
#   (fitted on that fold's training window)
# - **Ablation test**: compare model performance with and without temporal features
#   to isolate their incremental contribution

# %% [markdown]
# ### E.2 Save the temporal features

# %%
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

output_path = FEATURES_DIR / "model_based.parquet"
temporal.write_parquet(output_path)
print(f"Saved model_based.parquet ({output_path.stat().st_size / 1024:.1f} KB)")
print(f"  Shape: {temporal.shape}")
# %% [markdown]
# ## F. Incremental evaluation
#
# Compute IC for temporal features against the primary label, the five-day forward return.
# Compare static VRP (`ivrv_spread` from Ch8) vs dynamic VRP (`garch_ivrv_spread`)
# to test whether GARCH improves the VRP signal.
#
# **Context**: Ch8 feature evaluation found weak individual signal strength
# (no feature clears the false-discovery threshold). The temporal features attempt to improve
# the VRP signal specifically.
#
# **Scope of this evaluation**: the IC below uses only each canonical validation
# fold - the holdout fold added in section B is excluded, so nothing here reads a
# holdout row. It selects or drops **no** feature: all three temporal features are
# written to `model_based.parquet` in section E.2 before this diagnostic runs. The
# authoritative feature evaluation remains `05_evaluation.py`; this section only
# characterizes the dynamic-vs-static VRP comparison.
#
# The two VRP variants differ only in the denominator, so both are scored on the
# rows where both are defined. The GARCH spread is missing wherever a fit failed or
# the surface is thin, and scoring the Ch8 spread over its own wider panel would put
# a difference in sample into a comparison that is supposed to isolate the
# denominator.

# %%
from ml4t.diagnostic.metrics import compute_ic_hac_stats, cross_sectional_ic_series

# Load primary label
label_path = CASE_DIR / "labels" / f"{PRIMARY_LABEL}.parquet"
if label_path.exists() and len(temporal) > 0:
    label_df = pl.read_parquet(label_path)
    label_col = PRIMARY_LABEL

    validation_parts = [
        temporal.filter(
            (pl.col("fold") == split["fold"])
            & (pl.col("timestamp") >= split["val_start"])
            & (pl.col("timestamp") <= split["val_end"])
        )
        for split in canonical_splits
    ]
    validation_temporal = pl.concat(validation_parts).sort(["timestamp", "symbol"])
    if validation_temporal.n_unique(["timestamp", "symbol"]) != len(validation_temporal):
        raise RuntimeError("Canonical validation GARCH keys are not unique")

    temporal_eval = validation_temporal.join(
        label_df.rename({label_col: "forward_return"}),
        on=["timestamp", "symbol"],
        how="inner",
    )

    temporal_ic_results = []
    temp_feat_cols = [c for c in temporal.columns if c not in ("timestamp", "symbol", "fold")]
    for feat in temp_feat_cols:
        feat_eval = temporal_eval.select(
            ["timestamp", "symbol", feat, "forward_return"]
        ).drop_nulls()
        if len(feat_eval) < 100:
            continue
        feat_eval = feat_eval.rename({feat: "prediction"})
        ic_series = cross_sectional_ic_series(
            feat_eval,
            feat_eval,
            date_col="timestamp",
            entity_col="symbol",
            method="spearman",
            min_obs=10,
        )
        if len(ic_series) > 10:
            hac_stats = compute_ic_hac_stats(ic_series, label_horizon=LABEL_HORIZON)
            temporal_ic_results.append(
                {
                    "name": feat,
                    "ic_mean": round(hac_stats["mean_ic"], 4),
                    "hac_tstat": round(hac_stats["t_stat"], 2),
                    "hac_pval": round(hac_stats["p_value"], 4),
                }
            )

    print("Temporal feature IC (vs fwd_ret_5d):")
    for r in temporal_ic_results:
        print(f"  {r['name']:25s}: IC={r['ic_mean']:.4f}, t={r['hac_tstat']:.2f}")

    # Compare static vs dynamic VRP on the rows where both are defined.
    base_features_path = FEATURES_DIR / "financial.parquet"
    if base_features_path.exists() and "garch_ivrv_spread" in temporal_eval.columns:
        base = _normalize_symbol_column(pl.read_parquet(base_features_path)).select(
            ["timestamp", "symbol", "ivrv_spread"]
        )
        vrp_eval = (
            temporal_eval.select(["timestamp", "symbol", "garch_ivrv_spread", "forward_return"])
            .join(base, on=["timestamp", "symbol"], how="inner")
            .drop_nulls()
        )
        print(
            f"\nShared VRP rows: {len(vrp_eval):,} over {vrp_eval['timestamp'].n_unique():,} dates"
        )

        def _vrp_ic_series(frame: pl.DataFrame, column: str) -> pl.DataFrame:
            scored = frame.select(["timestamp", "symbol", column, "forward_return"]).rename(
                {column: "prediction"}
            )
            return cross_sectional_ic_series(
                scored,
                scored,
                date_col="timestamp",
                entity_col="symbol",
                method="spearman",
                min_obs=10,
            )

        static_series = _vrp_ic_series(vrp_eval, "ivrv_spread")
        dynamic_series = _vrp_ic_series(vrp_eval, "garch_ivrv_spread")

        # The two variants are scored on the same rows, so they produce an IC on the
        # same dates: pair them date by date rather than comparing two independent
        # summaries of them. The sort is load-bearing - `compute_ic_hac_stats` reads
        # the row order as the time order, and a join does not promise one.
        paired_ic = (
            static_series.select(["timestamp", pl.col("ic").alias("ic_static")])
            .join(
                dynamic_series.select(["timestamp", pl.col("ic").alias("ic_dynamic")]),
                on="timestamp",
                how="inner",
            )
            .drop_nulls()
            .sort("timestamp")
            .with_columns((pl.col("ic_dynamic") - pl.col("ic_static")).alias("ic"))
        )
        if not paired_ic["timestamp"].is_sorted():
            raise RuntimeError("Paired IC series is not in date order")

        static_stats = compute_ic_hac_stats(static_series, label_horizon=LABEL_HORIZON)
        dynamic_stats = compute_ic_hac_stats(dynamic_series, label_horizon=LABEL_HORIZON)
        static_ic_mean = round(static_stats["mean_ic"], 4)
        static_tstat = round(static_stats["t_stat"], 2)
        dynamic_ic_mean = round(dynamic_stats["mean_ic"], 4)
        dynamic_tstat = round(dynamic_stats["t_stat"], 2)

        # Neither t-statistic above tests the question the section asks. Each tests its
        # own mean IC against zero, and two dependent series can differ from each other
        # by more than either differs from zero - or by less, depending on the sign of
        # the dependence. The difference needs its own test, on the paired daily series.
        paired_corr = paired_ic.select(pl.corr("ic_static", "ic_dynamic")).item()
        diff_stats = compute_ic_hac_stats(paired_ic, label_horizon=LABEL_HORIZON)
        diff_mean = round(diff_stats["mean_ic"], 4)
        diff_tstat = round(diff_stats["t_stat"], 2)
        diff_pval = round(diff_stats["p_value"], 4)

        print("Static vs Dynamic VRP, on the same rows:")
        print(f"  Static (ivrv_spread):     IC={static_ic_mean:.4f}, t={static_tstat:.2f}")
        print(f"  Dynamic (garch_ivrv):     IC={dynamic_ic_mean:.4f}, t={dynamic_tstat:.2f}")
        print(f"  Daily IC correlation:     {paired_corr:.3f} over {len(paired_ic):,} dates")
        print(
            f"  Paired difference:        {diff_mean:+.4f}, "
            f"HAC t={diff_tstat:.2f}, p={diff_pval:.4f}"
        )
    else:
        static_ic_mean, static_tstat = 0, 0
        dynamic_ic_mean, dynamic_tstat = 0, 0
        diff_mean = diff_tstat = diff_pval = paired_corr = 0
else:
    temporal_ic_results = []
    static_ic_mean = static_tstat = dynamic_ic_mean = dynamic_tstat = 0
    diff_mean = diff_tstat = diff_pval = paired_corr = 0
    print("Skipping incremental evaluation (label or temporal data not available)")

# %% [markdown]
# The chart ranks the three temporal features by their standalone information
# coefficient against the 5-day forward return, and prints each one's HAC
# t-statistic beside its bar. The three disagree about the sign of the
# relationship and none of them reaches a t-statistic the sample can separate
# from zero, which is the same result Ch8 reached for the options-derived
# features. These are candidate inputs to the multivariate models downstream;
# what they contribute there is a question this notebook does not answer.

# %%
if temporal_ic_results:
    ic_sorted = sorted(temporal_ic_results, key=lambda r: abs(r["ic_mean"]), reverse=True)
    fig = go.Figure()
    fig.add_bar(
        x=[r["ic_mean"] for r in ic_sorted],
        y=[r["name"] for r in ic_sorted],
        orientation="h",
        marker_color=ml4t_palette(1)[0],
        text=[f"HAC t {r['hac_tstat']:+.2f}" for r in ic_sorted],
        textposition="outside",
        cliponaxis=False,
    )
    fig.add_vline(x=0.0, line_width=1, line_color="black")
    # Room for the t-statistic printed past each bar end, on whichever side the bar runs.
    _span = max(abs(r["ic_mean"]) for r in ic_sorted) * 1.6
    fig.update_layout(
        title="The three temporal features split in sign and all sit near zero",
        xaxis_title=(
            f"Mean cross-sectional IC vs {LABEL_HORIZON}-day forward return; "
            "a |t| near two would be the conventional threshold"
        ),
        xaxis=dict(range=[-_span, _span]),
        yaxis_title=None,
        yaxis=dict(autorange="reversed"),
        margin=dict(l=160, r=60),
        height=360,
    )
    fig.show()

# %% [markdown]
# The question this notebook was written to ask: does the GARCH conditional-vol
# denominator change what the variance risk premium tells you, against the
# backward-looking realized-vol denominator from Ch8? Both variants are scored on
# the same rows, so the only thing that differs between the two bars is the
# denominator.
#
# The first two bars carry their own HAC t-statistics, and neither of those answers
# the question. Each tests one mean IC against zero; the question is whether the two
# differ from *each other*. Scored on the same rows, the two variants produce an IC
# on the same dates, and those two daily series are dependent - so the difference
# has a variance of its own that neither level's standard error contains. Dependent
# series can separate from each other while neither separates from zero, and can
# fail to separate from each other while both do; which way it goes is set by the
# sign of the dependence, printed on the axis below. So the difference gets its own
# test, on the paired daily series - one observation per date,
# `ic_dynamic - ic_static` - and that is the third bar. Signed IC is plotted rather
# than |IC| so the quantity on the axis is the quantity the test is about.

# %%
if temporal_ic_results and (static_ic_mean or dynamic_ic_mean):
    # The two levels are the same kind of quantity; the difference is derived from
    # them and is the one the section's claim rests on, so it gets the accent.
    level_color, diff_color = ml4t_palette(2, categorical=True)
    fig = go.Figure()
    fig.add_bar(
        x=[
            "Static VRP<br>(realized-vol denom, Ch8)",
            "Dynamic VRP<br>(GARCH denom)",
            "Paired difference<br>(dynamic - static)",
        ],
        y=[static_ic_mean, dynamic_ic_mean, diff_mean],
        marker_color=[level_color, level_color, diff_color],
        text=[
            f"IC {static_ic_mean:+.4f}<br>HAC t {static_tstat:+.2f}",
            f"IC {dynamic_ic_mean:+.4f}<br>HAC t {dynamic_tstat:+.2f}",
            f"Δ {diff_mean:+.4f}<br>HAC t {diff_tstat:+.2f}",
        ],
        textposition="outside",
        cliponaxis=False,
    )
    fig.add_hline(y=0.0, line_width=1, line_color="black")
    fig.update_layout(
        title="Neither level nor the paired difference separates from zero",
        yaxis_title=f"Mean cross-sectional IC vs {LABEL_HORIZON}-day forward return",
        xaxis_title=(
            "Both variants scored on the rows where both are defined; "
            f"their daily ICs correlate at {paired_corr:.2f}"
        ),
        margin=dict(t=90),
        height=440,
    )
    fig.show()

# %% [markdown]
# ## Key Takeaways
#
# 1. **The GARCH denominator does not settle the VRP question either way**: on the
#    rows where both variants are defined, neither the dynamic spread nor the static
#    `ivrv_spread` it was meant to improve on reaches an HAC t-statistic anywhere
#    near two, and neither does the paired difference between them, which is tested
#    separately because two levels that each fail to separate from zero can still
#    separate from each other. Reading either denominator as the better one is
#    reading noise, and that now rests on a test of the difference rather than on an
#    inference from two tests that were not about it. The GARCH features enter the
#    downstream multivariate models as candidate inputs, and the ablation there is
#    what decides whether they earned the place.
#
# 2. **Walk-forward discipline**: GARCH is fitted only on training data for
#    each CV fold. The variance recursion runs on the full train+test window
#    with frozen parameters, providing features for both periods without
#    look-ahead bias.
#
# 3. **Minimal scope by design**: Only 3 features (cond_vol, ivrv_spread,
#    vol_surprise) -- this case study's value is in options-derived features
#    (Ch8), not temporal dynamics.
#
# 4. **GJR asymmetry matters for equities**: The leverage effect (negative
#    returns increase vol more than positive) is well-documented and directly
#    relevant because the VRP is larger after negative shocks. The fitted
#    persistence sits just below one, confirming slow-decaying, forecastable
#    volatility.
#
# **Next**: Ch11+ models combine Ch8 features with these temporal features and
# run the ablation that isolates their incremental contribution.
