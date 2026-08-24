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
# # ML-Based Exit Signals: Two-Model Architecture
# **Docker image**: `ml4t-gpu`
#
# ## Purpose
# Demonstrate a two-model exit architecture in which entry-model confidence becomes an
# explicit feature of the exit model, and judge the resulting decision rule against a
# basic exit baseline using both AUC and realized trade-path metrics.
#
# ## Learning Objectives
# After completing this notebook, you will be able to:
# - Train separate entry and exit prediction models on a shared feature set.
# - Inject entry-probability into the exit model and read its feature importance.
# - Evaluate exit logic by AUC *and* by realized trade-path metrics, not AUC alone.
# - Recognize when an architectural change does not pay off on AUC but still matters operationally.
#
# ## Book reference
# §19.7 Adaptive Risk Controls, Figure 19.5 (signal-strength-conditioned barrier outcomes).
#
# ## Prerequisites
# Complete [`02_exit_strategies`](02_exit_strategies.ipynb) first for the rule-based exit
# baseline (fixed stops, trailing stops, volatility-adjusted exits, whipsaw analysis) that
# this notebook extends with an ML-driven exit policy.

# %% [markdown]
# ## Setup

# %%
"""Train a causal two-model architecture where entry confidence drives exit timing."""

from datetime import UTC, datetime, timedelta

import lightgbm as lgb
import numpy as np
import plotly.graph_objects as go
import polars as pl
from IPython.display import Markdown, display
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score

from data import load_crypto_perps
from utils.paths import get_output_dir
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, show_plotly_with_alt, show_with_alt

# %% tags=["parameters"]
SEED = 42
LGB_DEVICE = "cuda"
FORWARD_HOURS = 24
N_OOF_FOLDS = 5
N_IMPORTANCE_REPEATS = 5
TEST_FRACTION = 0.20
ENTRY_QUANTILE = 0.95
ENTRY_PROBABILITY_THRESHOLD = 0.5
PROBABILITY_BIN_WIDTH = 0.02
ENTRY_CONFIDENCE_DROP = 0.30

# %%
set_global_seeds(SEED)
OUTPUT_DIR = get_output_dir(19, "ml_exit_signals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"LightGBM device: {LGB_DEVICE} (GPU production path required)")

# %% [markdown]
# ## 1. Data Loading
#
# Three crypto perpetuals (BTC, ETH, SOL) at hourly frequency from 2023-01-01. High
# volatility and round-the-clock trading make exit timing the dominant risk control
# rather than a corner-case concern.

# %%
ohlcv = load_crypto_perps(frequency="1h")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
START_DATE = "2023-01-01"

# Filter data
timestamp_dtype = ohlcv.schema["timestamp"]
start_ts = datetime.fromisoformat(f"{START_DATE}T00:00:00")
if getattr(timestamp_dtype, "time_zone", None):
    start_ts = start_ts.replace(tzinfo=UTC)

df = ohlcv.filter(
    (pl.col("symbol").is_in(SYMBOLS))
    & (pl.col("timestamp") >= pl.lit(start_ts, dtype=timestamp_dtype))
)

print(f"Loaded {len(df):,} rows for {df['symbol'].n_unique()} symbols")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# %% [markdown]
# **Interpretation**: The sample is intentionally short-horizon and volatile, so the
# exit model has enough adverse moves to learn from. That makes crypto perpetuals a
# good stress test for conviction-driven exits.

# %% [markdown]
# ## 2. Feature Engineering
#
# Create technical features for both entry and exit models.

# %% [markdown]
# The feature builder combines momentum, volatility, mean-reversion, and volume
# signals into a single tabular dataset for both the entry and exit models.


# %%
def create_features(df: pl.DataFrame, forward_hours: int = 24) -> pl.DataFrame:
    """
    Create features for entry/exit models.

    Features focus on momentum, volatility, and mean reversion signals.
    """
    close = pl.col("close")
    returns = close.pct_change()
    feature_exprs = [
        returns.over("symbol").alias("ret_1h"),
        close.pct_change(4).over("symbol").alias("ret_4h"),
        close.pct_change(8).over("symbol").alias("ret_8h"),
        close.pct_change(24).over("symbol").alias("ret_24h"),
        close.pct_change(72).over("symbol").alias("ret_72h"),
        returns.rolling_std(24).over("symbol").alias("vol_24h"),
        returns.rolling_std(72).over("symbol").alias("vol_72h"),
        (
            returns.rolling_mean(14).over("symbol") / returns.abs().rolling_mean(14).over("symbol")
        ).alias("rsi_proxy"),
        (close / close.rolling_mean(24).over("symbol") - 1).alias("dist_ma24"),
        (close / close.rolling_mean(72).over("symbol") - 1).alias("dist_ma72"),
        (pl.col("volume") / pl.col("volume").rolling_mean(24).over("symbol")).alias("vol_ratio"),
        ((pl.col("high") - pl.col("low")) / close).alias("hl_range"),
        (((pl.col("high") - pl.col("low")) / close).rolling_mean(24).over("symbol")).alias(
            "hl_range_ma"
        ),
        close.pct_change(forward_hours).shift(-forward_hours).over("symbol").alias("fwd_return"),
        # True one-step realized return: close[t] -> close[t+1], per symbol.
        # Used for the trade-path backtest so realized PnL is summed from
        # actual next-bar returns rather than a forward label divided by h.
        close.pct_change().shift(-1).over("symbol").alias("realized_ret_1step"),
    ]
    return df.sort(["symbol", "timestamp"]).with_columns(feature_exprs).drop_nulls()


# %%
# Create features
df_feat = create_features(df, forward_hours=FORWARD_HOURS)

# Define feature columns
FEATURE_COLS = [
    "ret_1h",
    "ret_4h",
    "ret_8h",
    "ret_24h",
    "ret_72h",
    "vol_24h",
    "vol_72h",
    "rsi_proxy",
    "dist_ma24",
    "dist_ma72",
    "vol_ratio",
    "hl_range",
    "hl_range_ma",
]

print(f"Feature dataset: {len(df_feat):,} samples")
df_feat.select(
    pl.len().alias("Samples"),
    pl.col("fwd_return").mean().alias("Mean forward return"),
    pl.col("fwd_return").std().alias("Forward-return volatility"),
    pl.col("fwd_return").quantile(0.05).alias("5th percentile"),
    pl.col("fwd_return").quantile(ENTRY_QUANTILE).alias("entry quantile"),
)

# %% [markdown]
# **Interpretation**: The forward-return distribution is heavy enough in both tails to
# justify separate entry and exit labels. A symmetric Gaussian world would not make
# the exit model especially useful.

# %% [markdown]
# ## 3. Chronological Split and Train-Only Labels
#
# The split is chronological and the last part of the sample is the test interval. Between the two
# sits a purge of `FORWARD_HOURS`: each label reads that far ahead, so without it the last training
# rows would carry outcomes from inside the test window.
#
# The entry threshold is estimated on the training interval alone and then applied unchanged. That
# ordering is what makes the test interval a test - a threshold picked to look good on the test
# rows would report the fit of a choice made after seeing them.
#
# This is a single fixed interval used once for demonstration. Nothing here is selected on it, and
# it is not large enough to distinguish two models whose scores are close.

# %%
timestamps = df_feat["timestamp"].unique().sort()
test_start = timestamps[int((1 - TEST_FRACTION) * len(timestamps))]
purge_start = test_start - timedelta(hours=FORWARD_HOURS)

train_df = df_feat.filter(pl.col("timestamp") < purge_start).sort(["timestamp", "symbol"])
test_df = df_feat.filter(pl.col("timestamp") >= test_start).sort(["timestamp", "symbol"])
embargo_rows = df_feat.filter(
    (pl.col("timestamp") >= purge_start) & (pl.col("timestamp") < test_start)
).height
entry_return_threshold = train_df["fwd_return"].quantile(ENTRY_QUANTILE)


# %% [markdown]
# The same train-only threshold creates both label frames. Exit labels need no fitted
# threshold: they indicate whether the next 24-hour return is negative.


# %%
def add_labels(frame: pl.DataFrame, entry_threshold: float) -> pl.DataFrame:
    """Attach entry and exit labels using a threshold learned outside the frame."""

    return frame.with_columns(
        (pl.col("fwd_return") >= entry_threshold).cast(pl.Int8).alias("y_entry"),
        (pl.col("fwd_return") < 0).cast(pl.Int8).alias("y_exit"),
    )


# %%
train_df = add_labels(train_df, entry_return_threshold)
test_df = add_labels(test_df, entry_return_threshold)

X_train = train_df.select(FEATURE_COLS).to_numpy()
X_test = test_df.select(FEATURE_COLS).to_numpy()
y_entry_train = train_df["y_entry"].to_numpy()
y_entry_test = test_df["y_entry"].to_numpy()
y_exit_train = train_df["y_exit"].to_numpy()
y_exit_test = test_df["y_exit"].to_numpy()

split_summary = pl.DataFrame(
    {
        "Interval": ["Purged train", "Embargo", "Demonstration test"],
        "Start": [train_df["timestamp"].min(), purge_start, test_df["timestamp"].min()],
        "End": [train_df["timestamp"].max(), test_start, test_df["timestamp"].max()],
        "Rows": [train_df.height, embargo_rows, test_df.height],
    }
)
split_summary

# %%
label_summary = pl.DataFrame(
    {
        "Sample": ["Train", "Test"],
        "Entry positive share": [train_df["y_entry"].mean(), test_df["y_entry"].mean()],
        "Exit positive share": [train_df["y_exit"].mean(), test_df["y_exit"].mean()],
        "Train-only entry threshold": [entry_return_threshold, entry_return_threshold],
    }
)
label_summary

# %% [markdown]
# The split is chronological by timestamp across all three symbols. The 24-hour gap
# prevents a training label from reaching into the test interval; the threshold and
# every fitted model state are learned without test rows.

# %% [markdown]
# ## 4. Train Entry Model
#
# The entry model predicts an exceptional positive return - the top slice of the forward-return
# distribution set by `ENTRY_QUANTILE`. Because the class is rare by construction, the model spends
# most of its capacity learning what an ordinary bar looks like, and its probabilities are
# correspondingly low almost everywhere.
#
# The exit model that follows will take the entry probability as one of its inputs. That creates a
# trap: a probability produced by a model that was fitted on the same row is not the probability
# that model would have produced in production, and an exit model trained on it learns a
# relationship that will not exist live. The expanding-window out-of-fold pass below produces each
# training row's entry probability from a model that never saw that row, with a purge at every
# fold boundary for the same reason as the main split.


# %%
def make_classifier(seed: int) -> lgb.LGBMClassifier:
    """Create the pinned CUDA LightGBM classifier used throughout the notebook."""

    return lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=5,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        max_bin=63,
        importance_type="gain",
        device_type=LGB_DEVICE,
        n_jobs=1,
        verbosity=-1,
        random_state=seed,
        data_random_seed=seed,
        feature_fraction_seed=seed,
        bagging_seed=seed,
        extra_seed=seed,
    )


# %% [markdown]
# Probability extraction uses the fitted LightGBM booster directly so inference follows
# the same pinned one-thread path without sklearn feature-name conversion warnings.


# %%
def predict_positive_probability(model: lgb.LGBMClassifier, features: np.ndarray) -> np.ndarray:
    """Predict binary positive-class probabilities without sklearn name warnings."""

    probability = np.asarray(model.booster_.predict(features, num_threads=1))
    if probability.shape != (len(features),) or not np.isfinite(probability).all():
        raise ValueError("binary LightGBM prediction must return one finite value per row")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("binary LightGBM probabilities must lie in [0, 1]")
    return probability


# %% [markdown]
# Expanding validation blocks preserve order. Each fold model stops at least one
# label horizon before its validation block, so the entry-probability feature is
# point-in-time for every row used to train the enhanced exit model.


# %% [markdown]
# Each validation block is contiguous in timestamp space. This helper types its boundaries
# to the frame schema and returns purged fit rows plus the bounded validation interval.


# %%
def chronological_fold_rows(
    ordered: pl.DataFrame,
    validation_times: list[datetime],
    label_horizon_hours: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return purged fit rows and dtype-safe validation rows for one fold."""

    timestamp_dtype = ordered.schema["timestamp"]
    validation_values = pl.Series("validation_timestamp", validation_times, dtype=timestamp_dtype)
    validation_start, validation_end = validation_values[0], validation_values[-1]
    fit_end = validation_start - timedelta(hours=label_horizon_hours)
    fit_rows = ordered.filter(pl.col("timestamp") < pl.lit(fit_end, dtype=timestamp_dtype))[
        "row_id"
    ].to_numpy()
    validation_rows = ordered.filter(
        (pl.col("timestamp") >= pl.lit(validation_start, dtype=timestamp_dtype))
        & (pl.col("timestamp") <= pl.lit(validation_end, dtype=timestamp_dtype))
    )["row_id"].to_numpy()
    if not len(fit_rows) or not len(validation_rows):
        raise ValueError("each OOF fold requires nonempty fit and validation rows")
    return fit_rows, validation_rows


# %% [markdown]
# Each OOF model defines its rare-event label from that fold's purged fit rows. Later
# outer-training returns therefore cannot move an earlier fold's cutoff or fit labels.


# %%
def fold_entry_labels(ordered: pl.DataFrame, fit_rows: np.ndarray) -> tuple[np.ndarray, float]:
    """Build binary entry labels from one fold's purged fit-return distribution."""

    fit_returns = ordered[fit_rows, "fwd_return"]
    threshold = float(fit_returns.quantile(ENTRY_QUANTILE))
    labels = (fit_returns >= threshold).cast(pl.Int8).to_numpy()
    if len(np.unique(labels)) != 2:
        raise ValueError("each OOF fit requires both entry-label classes")
    return labels, threshold


# %% [markdown]
# The OOF driver stays focused on expanding folds, local label fits, model fits, and
# prediction placement.


# %%
def chronological_oof_entry_probabilities(
    frame: pl.DataFrame,
    feature_cols: list[str],
    n_folds: int,
    label_horizon_hours: int,
    seed: int,
) -> tuple[np.ndarray, list[lgb.LGBMClassifier], list[float]]:
    """Generate purged expanding-window entry probabilities for stacking."""

    ordered = frame.sort(["timestamp", "symbol"]).with_row_index("row_id")
    unique_times = ordered["timestamp"].unique().sort().to_list()
    initial = max(int(0.4 * len(unique_times)), label_horizon_hours + 1)
    validation_blocks = [
        [unique_times[index] for index in block]
        for block in np.array_split(np.arange(initial, len(unique_times)), n_folds)
    ]
    features = ordered.select(feature_cols).to_numpy()
    oof = np.full(ordered.height, np.nan)
    models: list[lgb.LGBMClassifier] = []
    thresholds: list[float] = []

    for fold, validation_times in enumerate(validation_blocks):
        if not validation_times:
            continue
        fit_rows, validation_rows = chronological_fold_rows(
            ordered, validation_times, label_horizon_hours
        )
        fit_labels, fold_threshold = fold_entry_labels(ordered, fit_rows)
        model = make_classifier(seed + fold)
        model.fit(features[fit_rows], fit_labels)
        oof[validation_rows] = predict_positive_probability(model, features[validation_rows])
        models.append(model)
        thresholds.append(fold_threshold)

    if not models or np.isnan(oof).all():
        raise ValueError("chronological OOF construction produced no predictions")
    return oof, models, thresholds


# %%
entry_proba_oof, entry_fold_models, entry_fold_thresholds = chronological_oof_entry_probabilities(
    train_df,
    FEATURE_COLS,
    n_folds=N_OOF_FOLDS,
    label_horizon_hours=FORWARD_HOURS,
    seed=SEED,
)
meta_train_mask = np.isfinite(entry_proba_oof)
print(
    f"Purged chronological OOF coverage: {meta_train_mask.sum():,}/{len(meta_train_mask):,} "
    f"rows across {len(entry_fold_models)} folds"
)
print(
    f"Fold-local entry cutoffs: {min(entry_fold_thresholds):.6f} to "
    f"{max(entry_fold_thresholds):.6f}"
)

# %%
entry_model = make_classifier(SEED)
entry_model.fit(X_train, y_entry_train)
entry_model_device = str(entry_model.booster_.params.get("device_type", "")).lower()
if entry_model_device != LGB_DEVICE:
    raise RuntimeError(
        f"LightGBM device mismatch: required {LGB_DEVICE}, observed {entry_model_device or 'unset'}"
    )
print(
    f"GPU completeness: LightGBM {lgb.__version__}; "
    f"trained booster device_type={entry_model_device}; CPU helper threads=1"
)
entry_proba_test = predict_positive_probability(entry_model, X_test)

entry_auc = roc_auc_score(y_entry_test, entry_proba_test)
print(f"Entry Model AUC-ROC: {entry_auc:.4f}")

# %% [markdown]
# **Interpretation**: The entry model sets the benchmark for conviction. Its AUC and
# feature ranking tell us whether the signal has enough structure to be useful as an
# input to the exit model.

# %% [markdown]
# ## 5. Train Exit Model (Basic)
#
# Predicts negative returns from technical features. For a fair comparison, it uses
# exactly the OOF-covered training rows available to the enhanced model.

# %%
exit_model_basic = make_classifier(SEED + 100)
exit_model_basic.fit(X_train[meta_train_mask], y_exit_train[meta_train_mask])

exit_proba_basic = predict_positive_probability(exit_model_basic, X_test)

exit_auc_basic = roc_auc_score(y_exit_test, exit_proba_basic)
print(f"Basic Exit Model AUC-ROC: {exit_auc_basic:.4f}")

# %% [markdown]
# **Interpretation**: The basic exit model shows how much adverse-move prediction is
# already present in the raw feature set before we feed in entry conviction.

# %% [markdown]
# ## 6. Train Enhanced Exit Model
#
# Add the entry-model probability as an extra feature. The hypothesis is that fading
# conviction is informative about adverse moves beyond the raw technicals. Training
# uses only OOF entry probabilities; the final entry model supplies test probabilities.

# %%
X_train_enhanced = np.column_stack([X_train[meta_train_mask], entry_proba_oof[meta_train_mask]])
X_test_enhanced = np.column_stack([X_test, entry_proba_test])

FEATURE_COLS_ENHANCED = FEATURE_COLS + ["entry_prediction"]

exit_model_enhanced = make_classifier(SEED + 200)
exit_model_enhanced.fit(X_train_enhanced, y_exit_train[meta_train_mask])

# Predictions
exit_proba_enhanced = predict_positive_probability(exit_model_enhanced, X_test_enhanced)

# Evaluate
exit_auc_enhanced = roc_auc_score(y_exit_test, exit_proba_enhanced)
improvement = (exit_auc_enhanced - exit_auc_basic) / exit_auc_basic * 100

print(f"Enhanced Exit Model AUC-ROC: {exit_auc_enhanced:.4f}")
print(f"AUC delta vs basic:          {improvement:+.2f}%")

# %% tags=["results"]
display(
    Markdown(
        f"""**Interpretation**: The enhanced exit AUC is {exit_auc_enhanced:.3f} versus
{exit_auc_basic:.3f} for the basic model, a relative change of {improvement:+.2f}%. These values
describe one fixed test interval. Neither model was selected on it, and an interval this size
cannot separate two AUCs this close.
The realized trade-path comparison below is a separate operational diagnostic."""
    )
)

# %% [markdown]
# ## 7. Model Comparison
#
# The summary table puts the three models on one page so we can judge whether the
# extra architecture earns its complexity.

# %%
delta_label = f"AUC delta: {improvement:+.2f}%"
comparison_summary = pl.DataFrame(
    {
        "Model": ["Entry (Top 5%)", "Exit (Basic)", "Exit (Enhanced)"],
        "AUC-ROC": [entry_auc, exit_auc_basic, exit_auc_enhanced],
        "Features": [
            "Technical indicators",
            "Technical indicators only",
            "Technical + Entry Prediction",
        ],
        "Notes": [
            "Identify exceptional opportunities",
            "Baseline exit model",
            delta_label,
        ],
    }
)
comparison_summary

# %% tags=["results"]
display(
    Markdown(
        f"""**Interpretation**: Entry AUC is {entry_auc:.3f}; basic and enhanced exit AUC are
{exit_auc_basic:.3f} and {exit_auc_enhanced:.3f}. AUC measures ranking on the fixed test interval,
while Section 9 asks a different question: how the predeclared exit rules change realized holding
periods and compounded trade returns."""
    )
)

# %% [markdown]
# ## 8. Signal Analysis
#
# How do entry and exit signals interact?

# %% [markdown]
# The entry label marks a rare event, so demanding a probability above one half before entering is
# a severe filter: it admits only the bars where the model thinks a rare move is more likely than
# not. Almost no bars clear it, which is the intent - the policy buys precision with coverage, and
# spends most of the sample flat rather than trying to hold a position most of the time.

# %% [markdown]
# ### Three Ways to Say "Get Out"
#
# The exit rules compared below differ in what evidence they act on. One exits when the exit model
# calls for it. One exits when the *entry* model's confidence has fallen far enough that the reason
# for holding has gone, even if the exit model has said nothing. The third fires on either.
#
# `ENTRY_CONFIDENCE_DROP` is the level the entry probability must fall below for the second rule.
# It sits well under the threshold required to open a position, so the rule asks whether the case
# for the trade has collapsed rather than merely weakened.

# %%
entry_threshold = ENTRY_PROBABILITY_THRESHOLD
exit_threshold = ENTRY_PROBABILITY_THRESHOLD
entry_confidence_drop = ENTRY_CONFIDENCE_DROP

# Classify signals
entry_signal = entry_proba_test > entry_threshold
exit_signal_model = exit_proba_enhanced > exit_threshold
exit_signal_confidence = entry_proba_test < entry_confidence_drop

# Combined exit: model says exit OR entry confidence too low
exit_signal_combined = exit_signal_model | exit_signal_confidence

# %%
signal_counts = pl.DataFrame(
    {
        "Rule": [
            f"Entry signal (p_entry > {entry_threshold})",
            f"Exit model (p_exit > {exit_threshold})",
            f"Confidence drop (p_entry < {entry_confidence_drop})",
            "Combined exit (OR)",
        ],
        "Count": [
            int(entry_signal.sum()),
            int(exit_signal_model.sum()),
            int(exit_signal_confidence.sum()),
            int(exit_signal_combined.sum()),
        ],
        "Share of bars": [
            f"{entry_signal.mean():.1%}",
            f"{exit_signal_model.mean():.1%}",
            f"{exit_signal_confidence.mean():.1%}",
            f"{exit_signal_combined.mean():.1%}",
        ],
    }
)
signal_counts

# %% tags=["results"]
display(
    Markdown(
        f"""**Interpretation**: The confidence-drop clause fires on
{exit_signal_confidence.mean():.1%} of test bars and the combined rule fires on
{exit_signal_combined.mean():.1%}. Because the rare-event entry model usually assigns low
probability, most bars sit below the confidence-drop level already, so that clause fires on nearly
everything and the combined rule inherits its behaviour rather than the exit model's. The next
section reads the holding periods and trade returns instead of assuming more exits are better."""
    )
)

# %% [markdown]
# ## 9. Backtest Simulation
#
# Compare exit strategies on actual returns.


# %% [markdown]
# Signals are observed at an hourly close. An entry at close t earns the return from
# close t to close t+1. An exit at close u does not earn the return from u to u+1.
# This explicit ordering makes a k-bar holding period contain exactly k one-step returns.


# %%
def build_trade(
    symbol: str,
    entry_idx: int,
    exit_idx: int,
    timestamps: np.ndarray,
    returns: np.ndarray,
    exit_reason: str,
) -> dict:
    """Create one close-to-close trade with compounded simple returns."""

    path = returns[entry_idx:exit_idx]
    if not len(path) or np.any(path <= -1) or not np.isfinite(path).all():
        raise ValueError("a trade requires finite simple returns above -100%")
    return {
        "symbol": symbol,
        "entry_timestamp": timestamps[entry_idx],
        "exit_timestamp": timestamps[exit_idx],
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
        "return": np.prod(1 + path) - 1,
        "holding_bars": exit_idx - entry_idx,
        "exit_reason": exit_reason,
    }


# %% [markdown]
# The simulator processes each symbol independently, so a position can never cross an
# asset boundary. Open positions close at the last observed close without inventing a
# return beyond the evaluation sample.


# %% [markdown]
# A separate summary step keeps variable-duration trade accounting explicit and avoids
# burying the reported statistics inside the event loop.


# %%
def summarize_trades(trades: list[dict]) -> dict:
    """Summarize compounded trade paths without annualizing unequal horizons."""

    if not trades:
        return {
            "n_trades": 0,
            "mean_return": 0.0,
            "median_return": 0.0,
            "win_rate": 0.0,
            "avg_holding": 0.0,
            "signal_exits": 0,
            "timeout_exits": 0,
            "sample_end_exits": 0,
            "trades": [],
        }
    trade_returns = np.array([trade["return"] for trade in trades])
    return {
        "n_trades": len(trades),
        "mean_return": float(trade_returns.mean()),
        "median_return": float(np.median(trade_returns)),
        "win_rate": float((trade_returns > 0).mean()),
        "avg_holding": float(np.mean([trade["holding_bars"] for trade in trades])),
        "signal_exits": sum(trade["exit_reason"] == "signal" for trade in trades),
        "timeout_exits": sum(trade["exit_reason"] == "timeout" for trade in trades),
        "sample_end_exits": sum(trade["exit_reason"] == "sample_end" for trade in trades),
        "trades": trades,
    }


# %% [markdown]
# The event loop handles only entry, signal/timeout exits, and sample-end closure by symbol.


# %%
def simulate_strategy(
    symbols: np.ndarray,
    timestamps: np.ndarray,
    returns: np.ndarray,
    entry_signals: np.ndarray,
    exit_signals: np.ndarray,
    max_hold: int = 24,
) -> dict:
    """Simulate independent symbol paths with explicit close-time event ordering."""

    lengths = {len(symbols), len(timestamps), len(returns), len(entry_signals), len(exit_signals)}
    if lengths != {len(returns)}:
        raise ValueError("strategy arrays must have equal length")
    trades: list[dict] = []

    for symbol in dict.fromkeys(symbols):
        symbol_rows = np.flatnonzero(symbols == symbol)
        position = False
        entry_idx = -1
        for i in symbol_rows:
            if not position and entry_signals[i]:
                position = True
                entry_idx = i
                continue
            if not position:
                continue
            holding_time = i - entry_idx
            if not (exit_signals[i] or holding_time >= max_hold):
                continue
            exit_reason = "signal" if exit_signals[i] else "timeout"
            trades.append(build_trade(symbol, entry_idx, i, timestamps, returns, exit_reason))
            position = False

        final_idx = symbol_rows[-1]
        if position and final_idx > entry_idx:
            trades.append(
                build_trade(symbol, entry_idx, final_idx, timestamps, returns, "sample_end")
            )

    return summarize_trades(trades)


# %% [markdown]
# The model's predictions are in the order `test_df` was built in, and the simulator needs the rows
# regrouped so each symbol forms an independent path. Carrying an explicit row index through the
# regrouping is what keeps every prediction attached to the bar it was made for; re-sorting without
# it would silently pair predictions with other symbols' bars.

# %%
bt_frame = test_df.with_row_index("model_row").sort(["symbol", "timestamp"])
bt_pos = bt_frame["model_row"].to_numpy()
bt_symbols = bt_frame["symbol"].to_numpy()
bt_timestamps = bt_frame["timestamp"].to_numpy()
actual_returns = bt_frame["realized_ret_1step"].to_numpy()
entry_signal_bt = entry_signal[bt_pos]
exit_basic_bt = (exit_proba_basic > ENTRY_PROBABILITY_THRESHOLD)[bt_pos]
exit_model_bt = exit_signal_model[bt_pos]
exit_confidence_bt = exit_signal_confidence[bt_pos]
exit_combined_bt = exit_signal_combined[bt_pos]

# Strategy variants
strategies = {
    "No Exit Signal": simulate_strategy(
        bt_symbols,
        bt_timestamps,
        actual_returns,
        entry_signal_bt,
        np.zeros_like(entry_signal_bt, dtype=bool),
    ),
    "Basic Exit Model": simulate_strategy(
        bt_symbols, bt_timestamps, actual_returns, entry_signal_bt, exit_basic_bt
    ),
    "Enhanced Exit Model": simulate_strategy(
        bt_symbols, bt_timestamps, actual_returns, entry_signal_bt, exit_model_bt
    ),
    "Entry Confidence Drop": simulate_strategy(
        bt_symbols, bt_timestamps, actual_returns, entry_signal_bt, exit_confidence_bt
    ),
    "Combined Exit": simulate_strategy(
        bt_symbols, bt_timestamps, actual_returns, entry_signal_bt, exit_combined_bt
    ),
}

# %%
results_df = pl.DataFrame(
    [
        {
            "Strategy": name,
            **{key: value for key, value in result.items() if key != "trades"},
        }
        for name, result in strategies.items()
    ]
)
results_df

# %% tags=["results"]
best_mean_row = results_df.sort("mean_return", descending=True).row(0, named=True)
shortest_row = results_df.sort("avg_holding").row(0, named=True)
display(
    Markdown(
        f"""**Interpretation**: {best_mean_row["Strategy"]} has the highest mean compounded trade
return ({best_mean_row["mean_return"]:.2%}) on this fixed test interval. {shortest_row["Strategy"]}
has the shortest average holding period ({shortest_row["avg_holding"]:.1f} hourly bars). These are
descriptive trade-level metrics over a single test interval. They are not annualized, carry no
cost, and are not an estimate of what the strategy would earn."""
    )
)

# %% [markdown]
# ## 10. Visualization
#
# The figures test whether the architecture's behavior is explainable. GPU LightGBM
# is best-effort reproducible rather than bitwise deterministic, so importance is
# summarized across seeded repeats with random row and feature subsampling.


# %%
def repeated_gain_importance(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    seed: int,
    repeats: int,
) -> pl.DataFrame:
    """Estimate normalized gain importance dispersion across seeded GPU repeats."""

    records = []
    for repeat in range(repeats):
        model = make_classifier(seed + repeat)
        model.fit(features, labels)
        gain = model.feature_importances_.astype(float)
        if not np.isfinite(gain).all() or gain.sum() <= 0:
            raise ValueError("gain importance must be finite with positive total")
        for feature, importance in zip(feature_names, 100 * gain / gain.sum(), strict=True):
            records.append({"feature": feature, "repeat": repeat, "importance": importance})

    return (
        pl.DataFrame(records)
        .group_by("feature")
        .agg(
            pl.col("importance").mean().alias("mean_importance"),
            pl.col("importance").std().fill_null(0).alias("std_importance"),
        )
        .sort("mean_importance")
    )


# %%
entry_importance = repeated_gain_importance(
    X_train,
    y_entry_train,
    FEATURE_COLS,
    seed=SEED + 300,
    repeats=N_IMPORTANCE_REPEATS,
)
enhanced_importance = repeated_gain_importance(
    X_train_enhanced,
    y_exit_train[meta_train_mask],
    FEATURE_COLS_ENHANCED,
    seed=SEED + 400,
    repeats=N_IMPORTANCE_REPEATS,
)
entry_prediction_rank = (
    enhanced_importance.sort("mean_importance", descending=True)
    .with_row_index("rank", offset=1)
    .filter(pl.col("feature") == "entry_prediction")["rank"]
    .item()
)

# %%
fig = make_subplots(
    rows=1, cols=2, subplot_titles=["Entry Model Features", "Enhanced Exit Model Features"]
)

entry_imp_plot = entry_importance.to_pandas()
_ = fig.add_trace(
    go.Bar(
        x=entry_imp_plot["mean_importance"],
        y=entry_imp_plot["feature"],
        error_x={"type": "data", "array": entry_imp_plot["std_importance"], "visible": True},
        orientation="h",
        marker_color=COLORS["blue"],
        name="Entry",
    ),
    row=1,
    col=1,
)

# %% [markdown]
# The second panel uses the same normalized-gain scale and highlights only the stacked
# entry-prediction feature before both panels receive a common range and title.

# %%
exit_imp_plot = enhanced_importance.to_pandas()
colors = [
    COLORS["amber"] if feature == "entry_prediction" else COLORS["blue"]
    for feature in exit_imp_plot["feature"]
]
fig.add_trace(
    go.Bar(
        x=exit_imp_plot["mean_importance"],
        y=exit_imp_plot["feature"],
        error_x={"type": "data", "array": exit_imp_plot["std_importance"], "visible": True},
        orientation="h",
        marker_color=colors,
        name="Exit",
    ),
    row=1,
    col=2,
)

importance_xmax = 1.1 * max(
    (entry_importance["mean_importance"] + entry_importance["std_importance"]).max(),
    (enhanced_importance["mean_importance"] + enhanced_importance["std_importance"]).max(),
)
fig.update_layout(
    title=(
        f"Entry confidence ranks #{entry_prediction_rank} in the enhanced exit model"
        f"<br><sup>Mean normalized gain across {N_IMPORTANCE_REPEATS} seeded GPU repeats; "
        "error bars show ±1 SD</sup>"
    ),
    width=950,
    height=500,
    margin={"l": 120, "r": 40, "t": 110, "b": 70},
    showlegend=False,
)
fig.update_xaxes(title_text="Mean normalized gain importance (%)", range=[0, importance_xmax])
show_plotly_with_alt(
    fig,
    "Two histograms of predicted probability on a shared scale and bin width, entry above and exit below, each with its decision threshold marked. Nearly all the mass sits well below the threshold in both.",
)

# %% tags=["results"]
display(
    Markdown(
        f"""**Interpretation**: Across {N_IMPORTANCE_REPEATS} seeded GPU fits,
`entry_prediction` ranks #{entry_prediction_rank} by mean normalized gain in the enhanced exit
model. Gain importance is an impurity-based allocation measure, not a signed or causal effect; the
error bars show its repeat-to-repeat dispersion."""
    )
)

# %%
# Signal distribution
fig = make_subplots(
    rows=1, cols=2, subplot_titles=["Entry Signal Distribution", "Exit Signal Distribution"]
)

fig.add_trace(
    go.Histogram(
        x=entry_proba_test,
        xbins={"start": 0, "end": 1, "size": PROBABILITY_BIN_WIDTH},
        name="Entry probability",
        marker_color=COLORS["blue"],
    ),
    row=1,
    col=1,
)
fig.add_vline(x=entry_threshold, line_dash="dash", line_color=COLORS["amber"], row=1, col=1)

fig.add_trace(
    go.Histogram(
        x=exit_proba_enhanced,
        xbins={"start": 0, "end": 1, "size": PROBABILITY_BIN_WIDTH},
        name="Exit probability",
        marker_color=COLORS["slate"],
    ),
    row=1,
    col=2,
)
fig.add_vline(x=exit_threshold, line_dash="dash", line_color=COLORS["amber"], row=1, col=2)

fig.update_layout(
    title="The fixed rules select only the upper tail of each distribution",
    height=400,
    showlegend=False,
)
fig.update_xaxes(title_text="Predicted probability", range=[0, 1])
fig.update_yaxes(title_text="Test bars (count)")
show_plotly_with_alt(
    fig,
    "Grouped bars comparing the exit rules on mean trade return and average holding period, showing that the rules differ far more in how long they hold than in what they earn.",
)

# %% [markdown]
# **Interpretation**: Both panels share a bin width and a common probability scale. Letting each
# panel choose its own bins would make two distributions look different because their histograms
# were built differently, which is the wrong reason for a reader to see a difference.

# %%
signal_quintile_breaks = np.quantile(entry_proba_test, [0.2, 0.4, 0.6, 0.8])
signal_quintile_labels = ["Q1 (Weak)", "Q2", "Q3", "Q4", "Q5 (Strong)"]
publication_df = pl.DataFrame(
    {
        "signal": entry_proba_test,
        "fwd_return": test_df["fwd_return"],
        "signal_quintile": [
            signal_quintile_labels[index]
            for index in np.digitize(entry_proba_test, signal_quintile_breaks)
        ],
    }
).with_columns(
    pl.when(pl.col("fwd_return") < 0)
    .then(pl.lit("Adverse move"))
    .when(pl.col("fwd_return") >= entry_return_threshold)
    .then(pl.lit("Strong upside"))
    .otherwise(pl.lit("Neutral"))
    .alias("outcome")
)

# %% [markdown]
# Aggregate each ordered quintile into exhaustive outcome shares and persist the exact
# table used by the publication figure.

# %%
plot_data = (
    publication_df.group_by("signal_quintile", "outcome")
    .len()
    .pivot(on="outcome", index="signal_quintile", values="len")
    .join(
        pl.DataFrame({"signal_quintile": signal_quintile_labels, "quintile_order": range(5)}),
        on="signal_quintile",
        how="right",
    )
    .fill_null(0)
    .sort("quintile_order")
    .with_columns(pl.sum_horizontal("Adverse move", "Neutral", "Strong upside").alias("total"))
    .with_columns(
        (100 * pl.col(outcome) / pl.col("total")).alias(outcome)
        for outcome in ["Adverse move", "Neutral", "Strong upside"]
    )
)

# Persist quintile-conditioned outcome shares for the figure-19.5 publication script.
plot_data.drop("quintile_order", "total").write_parquet(
    OUTPUT_DIR / "signal_quintile_outcomes.parquet"
)

# %%
strong_upside = plot_data["Strong upside"].to_numpy()
adverse_move = plot_data["Adverse move"].to_numpy()
strong_direction = (
    "rises monotonically" if np.all(np.diff(strong_upside) >= 0) else "is non-monotonic"
)
adverse_direction = "declines" if adverse_move[-1] < adverse_move[0] else "increases"
outcome_title = (
    "Stronger entry signals concentrate strong-upside outcomes"
    if np.all(np.diff(strong_upside) >= 0)
    else "Signal quintiles reveal non-monotonic outcome conditioning"
)

fig_pub, ax_pub = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
bottom = np.zeros(plot_data.height)
for label, color in zip(
    ["Adverse move", "Neutral", "Strong upside"],
    [COLORS["negative"], COLORS["silver_muted"], COLORS["positive"]],
    strict=False,
):
    values = plot_data[label].to_numpy()
    ax_pub.bar(
        plot_data["signal_quintile"].to_list(),
        values,
        bottom=bottom,
        label=label,
        color=color,
        edgecolor="white",
        linewidth=0.6,
    )
    bottom = bottom + values
ax_pub.set_ylabel("Outcome share (%)")
ax_pub.set_xlabel("Signal quintile")
ax_pub.set_title(outcome_title)
ax_pub.set_ylim(0, 112)
ax_pub.set_yticks(range(0, 101, 20))
ax_pub.legend(loc="upper center", ncols=3, frameon=False)
show_with_alt(
    fig_pub,
    "Stacked bars of outcome share by signal quintile: adverse move, neutral, and strong upside. "
    "The strong-upside share grows across the quintiles from near zero, while the adverse-move "
    "share stays broadly flat.",
)

# %% tags=["results"]
display(
    Markdown(
        f"""**Interpretation**: Across test-sample probability quintiles, the strong-upside share
{strong_direction} from {strong_upside[0]:.1f}% to {strong_upside[-1]:.1f}%. The adverse-move share
{adverse_direction} from {adverse_move[0]:.1f}% to {adverse_move[-1]:.1f}%. These are descriptive
test-sample bins whose cut points are estimated from the displayed test probabilities; they are not
an independently validated trading rule."""
    )
)

# %% [markdown]
# ## 11. Key Takeaways

# %% tags=["results"]
takeaways = pl.DataFrame(
    {
        "Metric": [
            "Entry AUC",
            "Exit AUC (basic)",
            "Exit AUC (enhanced)",
            "AUC delta (enhanced − basic)",
            "Confidence-drop rule firing rate",
        ],
        "Value": [
            f"{entry_auc:.3f}",
            f"{exit_auc_basic:.3f}",
            f"{exit_auc_enhanced:.3f}",
            f"{(exit_auc_enhanced - exit_auc_basic):+.3f} ({improvement:+.2f}%)",
            f"{exit_signal_confidence.mean():.1%} of bars",
        ],
    }
)
takeaways

# %% tags=["results"]
display(
    Markdown(
        f"The purged out-of-fold pass supplies an entry probability for "
        f"**{meta_train_mask.sum():,} training rows** without any of them coming from a model that "
        f"saw its own row. Basic and enhanced exit AUC are **{exit_auc_basic:.3f}** and "
        f"**{exit_auc_enhanced:.3f}**. On the test interval, **{best_mean_row['Strategy']}** has "
        f"the highest mean compounded trade return at **{best_mean_row['mean_return']:.2%}**, and "
        f"**{shortest_row['Strategy']}** the shortest average hold at "
        f"**{shortest_row['avg_holding']:.1f} bars**."
    )
)

# %% [markdown]
# 1. **A feature built from another model has to be produced out of sample.** The exit model takes
#    the entry model's probability as an input. If that probability came from a model fitted on the
#    same row, it is more accurate than anything production will ever supply, and the exit model
#    learns to rely on an accuracy that will not be there. The expanding-window pass, with a purge
#    at every fold boundary, is what makes the training feature resemble the live one.
#
# 2. **Purge wherever a forward-looking label meets a boundary.** Not only at the train-test split:
#    every fold boundary inside the out-of-fold pass has the same problem, because the label on the
#    last row of a fold reads into the next one.
#
# 3. **A better AUC is not a better strategy.** AUC scores the ranking of every bar. A trading rule
#    acts on a threshold, holds a position for a while, and is judged on what the position earned.
#    The two can move in opposite directions, which is why the comparison here reports both and
#    reads the trade-level result rather than inferring it from the score.
#
# 4. **Check what a combined rule actually fires on.** Exiting when either of two conditions holds
#    sounds like a compromise between them. When one condition is satisfied on almost every bar -
#    as the confidence-drop clause is, because a rare-event model assigns low probability nearly
#    everywhere - the combination is that condition, and the other contributes nothing.
#
# 5. **A threshold estimated on the training rows and applied unchanged is a test; one refitted on
#    the test rows is not.** The entry cutoff here is a quantile of the training returns. Recomputing
#    it on the test period would produce a better-looking result and measure nothing.
#
# ### Known limitations
#
# - One fixed test interval, used once. It is not large enough to separate two models whose AUCs
#   differ in the third decimal, and nothing here reports how much of that difference is sampling.
# - No cost is charged. The rules differ mainly in how often they exit, which is exactly the
#   dimension trading costs price, so a cost-aware comparison could reorder them.
# - The quintile cut points in the signal analysis are estimated from the test probabilities being
#   displayed, so those bins describe this sample rather than defining a rule that could be applied
#   forward.
# - The entry policy is deliberately sparse and spends most of the sample flat. Trade-level
#   averages therefore rest on relatively few positions.
#
# **Next**: `09_deep_hedging` replaces a rule that decides when to exit with a network that learns
# how much to hold.
#
# **Book reference**: Chapter 19, Section 19.7.
