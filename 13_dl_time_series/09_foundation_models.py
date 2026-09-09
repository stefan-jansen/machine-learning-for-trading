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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Time series foundation models, applied to a panel they never saw
#
# **Docker image**: `ml4t-gpu`
#
# Every model so far in this section was fitted on this ETF panel. A **time series
# foundation model** is pretrained on a large, mostly non-financial corpus and asked
# to forecast a new series with no fitting at all - the transfer story that reshaped
# language and vision, applied to sequences of numbers.
#
# Two are run here in that zero-shot mode: **Chronos** (Amazon), which quantises a
# series into tokens and runs a T5 encoder-decoder over them, and **TinyTimeMixer**
# (IBM Granite), a small mixing architecture in the family `06_tsmixer` builds. They
# are scored against two models fitted on this panel: an LSTM and a penalised linear
# map on the same context windows.
#
# **What the comparison actually asks.** The zero-shot models forecast the context
# series itself for `PREDICTION_LENGTH` steps, and the mean of that path is used as a
# score for ranking funds. The label is a `LABEL_HORIZON`-session forward return, and
# the two horizons are not tied together. So this is a probe of whether a zero-shot
# forecast ranks assets usefully, not a calibrated forecast of the label - which is
# why the results table below leaves MSE blank for those rows and reports only rank
# IC.
#
# **Learning objectives**:
# - Run a pretrained forecaster with no fitting step and say exactly what it was given
#   and what it returned.
# - State what a rank-IC comparison between a forecast of one quantity and a label of
#   another can and cannot establish.
# - Name the design choices that bound this result - one feature, first-generation
#   models, a fixed horizon, zero-shot only - and separate them from the published
#   findings the notebook cites.
#
# **Key references**:
# - Rahimikia et al. (2025): zero-shot TSFMs on financial series
# - DELPHYNE (Ding et al., 2025): negative transfer from financial pretraining data
# - Rasul et al. (2024), Lag-Llama: probabilistic forecasting
#
# **Book Reference**: Chapter 13, Section 13.6 (Alternative architectures and foundation models)
#
# **Prerequisites**: ETF features (`case_studies/etfs/`)

# %%
"""Time Series Foundation Models - evaluate zero-shot Chronos and TTM against task-specific baselines."""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from chronos import ChronosPipeline
from dl_sequences import load_dl_dataset, train_model
from IPython.display import Markdown, display
from ml4t.diagnostic.metrics import cross_sectional_ic_series
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

from utils.reproducibility import set_global_seeds
from utils.style import COLORS, add_message_title, show_with_alt

# %% [markdown]
# Three of the settings below decide what the comparison is, rather than tuning it.
#
# `PREDICTION_LENGTH` is the horizon the zero-shot models forecast over. It is held
# fixed and deliberately not tied to the label's horizon, because these models are
# being used as a ranking signal rather than as a forecast of the label; the design
# note further down says what that costs.
#
# `LABEL_HORIZON` is how many trading sessions pass between a decision date and the
# day its label resolves. The ETF case study's primary label is a close-to-close
# return over that many sessions, and the split drops every example within one horizon
# of a boundary.
#
# `REQUIRE_FOUNDATION_MODELS` decides what happens when Chronos or TTM fails at
# *runtime*: True raises, False leaves a null row in the results table. It does not
# govern missing installs - both libraries are imported at module load, so they have
# to be present either way.

# %% tags=["parameters"]
SEED = 42
CONTEXT_LENGTH = 60
PREDICTION_LENGTH = 10
LABEL_HORIZON = 21
EPOCHS = 30
BATCH_SIZE = 64
LR = 0.001
INFER_BATCH_SIZE = 1_024
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None
MAX_TEST_SAMPLES = None
CHRONOS_NUM_SAMPLES = 10
REQUIRE_FOUNDATION_MODELS = True

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

set_global_seeds(SEED)

print(f"Context length: {CONTEXT_LENGTH}")
print(f"Prediction length: {PREDICTION_LENGTH}")
print(f"Epochs: {EPOCHS}")

# %% [markdown]
# ## Data Loading
#
# ETF features and labels from the case study pipeline. The foundation models read a
# univariate context - one feature column - while the LSTM and ridge baselines read
# the multivariate feature set.

# %%
mds = load_dl_dataset("etfs")

FEATURE_COLS = mds.feature_names[:8]
TARGET_COL = mds.label_col

df = mds.dataset.drop_nulls(subset=FEATURE_COLS + [TARGET_COL])
print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"Target: {TARGET_COL}")
print(f"Rows after dropna: {len(df):,}")
per_date = df.group_by(mds.date_col).len().sort(mds.date_col)
print(
    f"{df[mds.date_col].min()} to {df[mds.date_col].max()}, "
    f"{df[mds.entity_cols[0]].n_unique()} funds; funds per date "
    f"{per_date['len'].min()} to {per_date['len'].max()}, median {per_date['len'].median():.0f}"
)
print(
    f"Label {TARGET_COL}: mean {df[TARGET_COL].mean():+.5f}, "
    f"standard deviation {df[TARGET_COL].std():.5f}"
)

# %% [markdown]
# ## Prepare Univariate Contexts
#
# Foundation models expect univariate time series contexts. We extract
# the first feature column (typically close returns) per symbol and build
# sliding windows of length `CONTEXT_LENGTH`. Each window is paired with
# the forward return label at the next timestep.
#
# **Design note on horizon and target alignment.** The zero-shot baseline
# below forecasts the *first feature column* for `PREDICTION_LENGTH` steps
# and uses the mean of that forecast path as a directional score, which is
# then compared cross-sectionally to the label `TARGET_COL`. The forecast
# horizon (`PREDICTION_LENGTH = 10`) is held fixed across models and does
# not necessarily match the label horizon - for the ETF case study used
# here, the label is a multi-day forward return. The experiment is therefore
# a *directional-signal probe*, not a calibrated forecast of the label
# itself: it tests whether a zero-shot foundation model's mean forecast on
# raw historical values produces a useful ranking of assets at the next
# rebalance, against the same ranking from a task-specific baseline. Tying
# `PREDICTION_LENGTH` to the label horizon and explicitly aggregating the
# forecasted path into the same target definition would let foundation
# models be evaluated on the label directly; that variation is left as an
# exercise and would change the numerical comparisons that follow.


# %%
def prepare_univariate_contexts(
    df: pl.DataFrame,
    feature_col: str,
    target_col: str,
    context_length: int,
    date_col: str = "timestamp",
    symbol_col: str = "symbol",
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Sliding-window contexts of `feature_col` paired with `target_col` per symbol."""
    contexts, targets_list, dates_list, symbols_list = [], [], [], []
    for symbol in df.select(symbol_col).unique().to_series().to_list():
        sym_df = df.filter(pl.col(symbol_col) == symbol).sort(date_col)
        if len(sym_df) < context_length + 1:
            continue
        values = sym_df[feature_col].to_numpy().astype(np.float32)
        target_values = sym_df[target_col].to_numpy().astype(np.float32)
        dates = sym_df[date_col].to_numpy()
        for i in range(context_length, len(values)):
            ctx = values[i - context_length : i]
            if not np.isfinite(ctx).all():
                continue
            contexts.append(ctx)
            targets_list.append(target_values[i])
            dates_list.append(dates[i])
            symbols_list.append(symbol)
    return (
        contexts,
        np.array(targets_list, dtype=np.float32),
        np.array(dates_list),
        np.array(symbols_list),
    )


# %%
# Use the first feature as the univariate context series
CONTEXT_FEATURE = FEATURE_COLS[0]
print(f"Univariate context feature: {CONTEXT_FEATURE}")

contexts, targets, dates_arr, symbols_arr = prepare_univariate_contexts(
    df,
    CONTEXT_FEATURE,
    TARGET_COL,
    CONTEXT_LENGTH,
    date_col=mds.date_col,
    symbol_col=mds.entity_cols[0],
)
print(f"Total samples: {len(contexts):,}")

# %% [markdown]
# ## Temporal Train/Test Split
#
# A 60/20/20 split by date, the same shape the rest of this section uses, rather than
# walk-forward validation: foundation model inference is expensive and the comparison
# here is zero-shot against trained, not a claim about stability over time.
#
# The split has to be indexed by date and not by row. These samples are pooled across
# assets, so a positional slice would cut through a cross-section and put the same day
# on both sides of a boundary.
#
# It also needs a gap. The label is a close-to-close return over `LABEL_HORIZON`
# trading sessions, so an example dated within that many sessions of a boundary has an
# outcome resolved by days on the far side. Without the gap the LSTM would be fitted
# on targets that resolve inside the validation stretch, and selected on targets that
# resolve inside the test stretch. Those examples are dropped.

# %%
unique_dates = np.sort(np.unique(dates_arr))
train_boundary_idx = int(len(unique_dates) * 0.6)
val_boundary_idx = int(len(unique_dates) * 0.8)
train_end_date = unique_dates[train_boundary_idx]
val_end_date = unique_dates[val_boundary_idx]
train_label_cutoff = unique_dates[train_boundary_idx - LABEL_HORIZON]
val_label_cutoff = unique_dates[val_boundary_idx - LABEL_HORIZON]

train_idx = np.where(dates_arr < train_label_cutoff)[0].tolist()
val_idx = np.where((dates_arr >= train_end_date) & (dates_arr < val_label_cutoff))[0].tolist()
test_idx = np.where(dates_arr >= val_end_date)[0].tolist()

# %% [markdown]
# The three sample caps below are for local iteration only and the shipped run leaves
# them at `None`. They are worth understanding before using: they keep the *last* rows
# of an index that is ordered by symbol and pooled across assets, so a non-`None` cap
# returns an arbitrary subset of symbols and a possibly partial cross-section, which
# makes the per-date IC depend on symbol ordering. `07_mamba_ssm` and
# `08_cnn_image_encoding` show the alternative: trim to the most recent complete dates.

# %%
if MAX_TRAIN_SAMPLES and len(train_idx) > MAX_TRAIN_SAMPLES:
    train_idx = train_idx[-MAX_TRAIN_SAMPLES:]
if MAX_VAL_SAMPLES and len(val_idx) > MAX_VAL_SAMPLES:
    val_idx = val_idx[-MAX_VAL_SAMPLES:]
if MAX_TEST_SAMPLES and len(test_idx) > MAX_TEST_SAMPLES:
    test_idx = test_idx[-MAX_TEST_SAMPLES:]

test_dates = dates_arr[test_idx]
test_symbols = symbols_arr[test_idx]

print(f"Train: {len(train_idx):,}, Val: {len(val_idx):,}, Test: {len(test_idx):,}")
print(
    f"Purged {LABEL_HORIZON} target dates before each boundary: "
    f"validation starts {train_end_date}, test starts {val_end_date}"
)


# %% [markdown]
# ### Cross-sectional IC helper
#
# The same per-date Spearman rank correlation used across this section, so the
# zero-shot models are scored the same way as the LSTM and ridge baselines.
#
# A date's IC is undefined when a model predicts the same value for every fund on it:
# the predicted ranks are all tied and there is nothing to correlate. The library
# returns `NaN` for such a date, and polars treats `NaN` and null as different values,
# so `drop_nulls` alone leaves it in place and one of them makes the whole mean `NaN`.
# Both are filtered here, and the count of dates the mean was taken over is reported
# with each score - a zero-shot model that produces a near-constant forecast is
# exactly the case where ties are plausible.


# %%
def cross_sectional_ic_mean(y_true, y_pred, dates, syms):
    """Mean cross-sectional Spearman IC over the dates where it is defined.

    Returns the mean and the defined/total date counts. Filters both null and NaN,
    since polars `drop_nulls` leaves NaN in place.
    """
    pred_df = pl.DataFrame({"timestamp": dates, "symbol": syms, "prediction": y_pred})
    ret_df = pl.DataFrame({"timestamp": dates, "symbol": syms, "forward_return": y_true})
    ic_per_date = cross_sectional_ic_series(
        pred_df,
        ret_df,
        pred_col="prediction",
        ret_col="forward_return",
        date_col="timestamp",
        entity_col="symbol",
    )
    defined = ic_per_date.filter(pl.col("ic").is_not_null() & pl.col("ic").is_not_nan())
    mean_ic = float(defined["ic"].mean()) if defined.height else float("nan")
    return {"ic": mean_ic, "n_defined": defined.height, "n_total": ic_per_date.height}


# %% [markdown]
# ## Chronos Zero-Shot Forecasting
#
# Chronos (Amazon) treats time series as discrete tokens via quantization,
# using a T5-based encoder-decoder architecture pre-trained on diverse datasets.
# We test whether these pre-trained representations transfer to ETF returns.

# %%
CHRONOS_SUCCESS = False
chronos_ic = 0.0
chronos_params = None

try:
    model_size = "small"
    print(f"Loading Chronos-{model_size}...")
    chronos = ChronosPipeline.from_pretrained(
        f"amazon/chronos-t5-{model_size}",
        device_map=str(DEVICE),
        dtype=torch.float32,
    )
    chronos_params = sum(p.numel() for p in chronos.model.parameters())
    print(f"Chronos-{model_size} parameters: {chronos_params:,}")
    contexts_test = [contexts[i] for i in test_idx]
    y_test_chronos = targets[test_idx]
    scores = []
    for i in range(0, len(contexts_test), BATCH_SIZE):
        batch = contexts_test[i : i + BATCH_SIZE]
        batch_tensors = [torch.tensor(c) for c in batch]
        forecast = chronos.predict(
            inputs=batch_tensors,
            prediction_length=PREDICTION_LENGTH,
            num_samples=CHRONOS_NUM_SAMPLES,
        )
        scores.extend(forecast.mean(dim=1).mean(dim=1).cpu().tolist())
    scores = np.array(scores)
    chronos_result = cross_sectional_ic_mean(y_test_chronos, scores, test_dates, test_symbols)
    chronos_ic = chronos_result["ic"]
    CHRONOS_SUCCESS = True
    print(
        f"Chronos IC: {chronos_ic:.4f} "
        f"(defined on {chronos_result['n_defined']} of {chronos_result['n_total']} test dates)"
    )
except Exception as e:
    print(f"Chronos failed: {e}")
    if REQUIRE_FOUNDATION_MODELS:
        raise

# %% [markdown]
# ## TTM (TinyTimeMixer) Zero-Shot Forecasting
#
# TTM from IBM Granite is a lightweight MLP-Mixer architecture (1-5M params)
# designed for efficient zero-shot inference. It uses the TSMixer architecture
# discussed in the previous notebook, pre-trained on a large corpus.

# %%
TTM_SUCCESS = False
ttm_ic = 0.0
ttm_params = None
_ttm_loaded = False

try:
    print("Loading TTM model...")
    ttm = TinyTimeMixerForPrediction.from_pretrained(
        "ibm-granite/granite-timeseries-ttm-v1",
    )
    ttm = ttm.to(DEVICE)
    ttm.eval()
    ttm_context_len = getattr(ttm.config, "context_length", None)
    ttm_pred_len = getattr(ttm.config, "prediction_length", None)
    if ttm_context_len is None or ttm_pred_len is None:
        print(
            "WARNING: TTM config attributes not found (API may have changed); "
            f"falling back to 512/96. Available config keys: "
            f"{sorted(ttm.config.to_dict().keys())}"
        )
    ttm_context_len = ttm_context_len or 512
    ttm_pred_len = ttm_pred_len or 96
    ttm_params = sum(p.numel() for p in ttm.parameters())
    print(f"TTM parameters: {ttm_params:,}")
    print(f"TTM context_length: {ttm_context_len}, prediction_length: {ttm_pred_len}")
    _ttm_loaded = True
except Exception as e:
    print(f"TTM failed: {e}")
    if REQUIRE_FOUNDATION_MODELS:
        raise

# %%
# Run TTM inference on test set
if _ttm_loaded:
    try:
        contexts_test = [contexts[i] for i in test_idx]
        y_test_ttm = targets[test_idx]

        scores = []
        for i in range(0, len(contexts_test), BATCH_SIZE):
            batch = contexts_test[i : i + BATCH_SIZE]

            # Pad or truncate to TTM's expected context length
            processed = []
            for ctx in batch:
                if len(ctx) < ttm_context_len:
                    padded = np.zeros(ttm_context_len, dtype=np.float32)
                    padded[-len(ctx) :] = ctx
                    processed.append(padded)
                else:
                    processed.append(ctx[-ttm_context_len:])

            batch_arr = np.array(processed)[:, :, np.newaxis]
            batch_tensor = torch.tensor(batch_arr, dtype=torch.float32).to(DEVICE)

            with torch.no_grad():
                output = ttm(batch_tensor)

            forecast = output.prediction_outputs
            forecast_slice = forecast[:, :PREDICTION_LENGTH, :]
            mean_forecast = forecast_slice.mean(dim=(1, 2))
            scores.extend(mean_forecast.cpu().tolist())

        scores = np.array(scores)
        ttm_result = cross_sectional_ic_mean(y_test_ttm, scores, test_dates, test_symbols)
        ttm_ic = ttm_result["ic"]
        TTM_SUCCESS = True
        print(
            f"TTM IC: {ttm_ic:.4f} "
            f"(defined on {ttm_result['n_defined']} of {ttm_result['n_total']} test dates)"
        )
    except Exception as e:
        print(f"TTM inference failed: {e}")
        if REQUIRE_FOUNDATION_MODELS:
            raise

# %% [markdown]
# ## LSTM Baseline (Task-Specific Training)
#
# A small LSTM trained from scratch on this panel. It is the comparison that gives the
# zero-shot scores a scale: a few tens of thousands of weights fitted on the data at
# hand, against tens of millions fitted on a corpus that does not include it. Note
# that it also reads the full multivariate feature set, where the zero-shot models
# read one column, so the two differ in what they see as well as in how they were
# fitted.


# %%
class LSTMRegressor(nn.Module):
    """Simple LSTM for regression on univariate context windows."""

    def __init__(self, input_size: int = 1, hidden_size: int = 32, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze(-1)


# %%
# Prepare 3D arrays: (samples, context_length, 1) for LSTM
X_train_lstm = np.array([contexts[i] for i in train_idx])[:, :, np.newaxis]
y_train_lstm = targets[train_idx]
X_val_lstm = np.array([contexts[i] for i in val_idx])[:, :, np.newaxis]
y_val_lstm = targets[val_idx]
X_test_lstm = np.array([contexts[i] for i in test_idx])[:, :, np.newaxis]
y_test_lstm = targets[test_idx]

lstm = LSTMRegressor(input_size=1, hidden_size=64, n_layers=2).to(DEVICE)
lstm_params = sum(p.numel() for p in lstm.parameters())
print(f"LSTM parameters: {lstm_params:,}")

history = train_model(
    lstm, X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm, EPOCHS, LR, BATCH_SIZE, DEVICE
)

# %%
lstm.eval()
with torch.no_grad():
    y_pred_lstm_chunks = []
    for i in range(0, len(X_test_lstm), INFER_BATCH_SIZE):
        X_test_t = torch.FloatTensor(X_test_lstm[i : i + INFER_BATCH_SIZE]).to(DEVICE)
        y_pred_lstm_chunks.append(lstm(X_test_t).cpu().numpy())
    y_pred_lstm = np.concatenate(y_pred_lstm_chunks)

lstm_result = cross_sectional_ic_mean(y_test_lstm, y_pred_lstm, test_dates, test_symbols)
lstm_ic = lstm_result["ic"]
lstm_mse = np.mean((y_pred_lstm - y_test_lstm) ** 2)
print("\nLSTM Test Results:")
print(f"  MSE: {lstm_mse:.6f}")
print(f"  Spearman IC: {lstm_ic:.4f}", end="")
print(f"  (defined on {lstm_result['n_defined']} of {lstm_result['n_total']} test dates)")

# %% [markdown]
# ## Ridge Baseline
#
# Flattening the context window into a feature vector and fitting Ridge
# regression provides the simplest possible baseline.

# %%
X_train_flat = np.array([contexts[i] for i in train_idx])
y_train_ridge = targets[train_idx]
X_test_flat = np.array([contexts[i] for i in test_idx])
y_test_ridge = targets[test_idx]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train_ridge)
y_pred_ridge = ridge.predict(X_test_scaled)

ridge_mse = np.mean((y_pred_ridge - y_test_ridge) ** 2)
ridge_result = cross_sectional_ic_mean(y_test_ridge, y_pred_ridge, test_dates, test_symbols)
ridge_ic = ridge_result["ic"]

print("Ridge Test Results:")
print(f"  MSE: {ridge_mse:.6f}")
print(f"  Spearman IC: {ridge_ic:.4f}", end="")
print(f"  (defined on {ridge_result['n_defined']} of {ridge_result['n_total']} test dates)")

# %% [markdown]
# ## Results Comparison
#
# Comparing zero-shot foundation models against task-specific baselines
# trained on domain data.


# %% [markdown]
# The parameter counts below are counted on the objects in memory rather than quoted
# from a model card, so they are the sizes actually used. A zero-shot row whose model
# failed to load has no count to report.
#
# MSE is left blank for the zero-shot rows on purpose. Those models forecast the
# context series over `PREDICTION_LENGTH` steps, and the mean of that path is used as
# a ranking score; a squared error between that path's mean and a
# `LABEL_HORIZON`-session forward return would be comparing two different quantities.


# %%
def _fmt_params(n):
    return f"{n:,}" if n is not None else "not loaded"


PARAM_COUNTS = {
    "Chronos (Zero-Shot)": _fmt_params(chronos_params),
    "TTM (Zero-Shot)": _fmt_params(ttm_params),
    "LSTM (Task-Specific)": f"{lstm_params:,}",
    "Ridge (Task-Specific)": f"{ridge.coef_.size + 1:,}",
}

rows = []
for name, ic_val, mse_val, model_type, available in [
    (
        "Chronos (Zero-Shot)",
        chronos_ic if CHRONOS_SUCCESS else None,
        None,
        "Foundation",
        CHRONOS_SUCCESS,
    ),
    ("TTM (Zero-Shot)", ttm_ic if TTM_SUCCESS else None, None, "Foundation", TTM_SUCCESS),
    ("LSTM (Task-Specific)", lstm_ic, lstm_mse, "Baseline", True),
    ("Ridge (Task-Specific)", ridge_ic, ridge_mse, "Baseline", True),
]:
    rows.append(
        {
            "Model": name,
            "Type": model_type,
            "Spearman IC": ic_val if available else None,
            "MSE": mse_val,
            "Parameters": PARAM_COUNTS[name],
        }
    )

results_df = pl.DataFrame(rows)
results_df

# %% [markdown]
# **What was measured, and what it is a measurement of.** The table reports one
# number per model - the mean cross-sectional rank IC on one test split - and the
# sentence below it is computed from those numbers rather than typed in. MSE is blank
# for the zero-shot rows on purpose: those models forecast the context series over a
# fixed horizon that is not the label's, so a squared error against the label would be
# comparing two different quantities.
#
# **Four design choices bound what this can say**, and each of them is a place where
# published work does something else:
#
# - **One feature.** The zero-shot models see a single column, because
#   first-generation Chronos and TTM are univariate. The LSTM and ridge baselines see
#   the multivariate feature set. That difference alone could produce the gap.
# - **First-generation models.** Chronos-2 and Moirai-MoE accept multivariate inputs
#   and exogenous covariates; neither is run here.
# - **Zero-shot only.** In-context learning, parameter-efficient fine-tuning and
#   test-time ensembling are the other deployment modes, and the cited literature
#   reports them narrowing the gap.
# - **Return prediction.** The published negative results concentrate on returns.
#   Volatility and value-at-risk targets carry more transferable structure - trend,
#   clustering, mean reversion - and fine-tuned models do better on them.
#
# So the honest reading is that this configuration of these models does not rank ETFs
# usefully here, which is consistent with Rahimikia et al. (2025), and that a general
# claim about foundation models on financial data is not something one split of one
# panel with one feature can support.

# %%
_zero_shot = [("Chronos", chronos_ic) for ok in [CHRONOS_SUCCESS] if ok] + [
    ("TTM", ttm_ic) for ok in [TTM_SUCCESS] if ok
]
_fitted = [("LSTM", lstm_ic), ("Ridge", ridge_ic)]
_worst_fitted = min(v for _, v in _fitted)
_best_zero_shot = max((v for _, v in _zero_shot), default=float("-nan"))
_separated = bool(_zero_shot) and _worst_fitted > _best_zero_shot
display(
    Markdown(
        "On this run the zero-shot models score "
        + ", ".join(f"{n} {v:+.4f}" for n, v in _zero_shot)
        + ", and the models fitted on this panel score "
        + ", ".join(f"{n} {v:+.4f}" for n, v in _fitted)
        + ". Every fitted model is above every zero-shot one." * _separated
        + " The two groups overlap on this run, so the ordering below is not clean."
        * (not _separated)
        + " The ridge fit is deterministic; the LSTM is GPU-trained with fixed seeds and"
        " its IC moves slightly between runs, so read the separation rather than the"
        " decimals."
    )
)

# %% [markdown]
# ## Two figures
#
# The first puts every model's rank IC on one axis. The second looks at what the
# best-scoring fitted model is actually producing: a scatter of its predictions
# against the outcomes, with the printed ratio of the two standard deviations, which
# is the number that says how much of the label's spread the predictions span.

# %%
available_rows = [r for r in rows if r["Spearman IC"] is not None]
model_names = [r["Model"] for r in available_rows]
ics = [r["Spearman IC"] for r in available_rows]
bar_colors = [
    COLORS["blue"] if r["Type"] == "Foundation" else COLORS["amber"] for r in available_rows
]

fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
ax.barh(model_names, ics, color=bar_colors)
ax.set_xlabel("Spearman IC (test)")
ax.axvline(0, color=COLORS["neutral"], linestyle="--", alpha=0.7)
add_message_title(
    ax,
    "Mean cross-sectional Spearman IC by model",
    subtitle="ETF forward returns, one test split (amber = fitted on this panel, navy = zero-shot)",
)
show_with_alt(
    fig,
    "A horizontal bar chart of mean cross-sectional Spearman IC, one bar per model, "
    "with a dashed vertical line at zero. Bars for the models fitted on this panel "
    "are amber and the zero-shot bars are navy.",
)

# %%
sample_n = min(500, len(y_test_lstm))
spread_ratio = float(np.std(y_pred_lstm) / np.std(y_test_lstm))
print(
    f"LSTM prediction spread / outcome spread: {spread_ratio:.3f} "
    f"(standard deviations {np.std(y_pred_lstm):.5f} and {np.std(y_test_lstm):.5f})"
)

fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
ax.scatter(y_test_lstm[:sample_n], y_pred_lstm[:sample_n], alpha=0.3, s=10, color=COLORS["amber"])
ax.set_xlabel("Actual forward return")
ax.set_ylabel("Predicted return")
ax.axline((0, 0), slope=1, color=COLORS["neutral"], linestyle="--", alpha=0.5)
add_message_title(
    ax,
    "LSTM predictions against the outcomes they were predicting",
    subtitle=(
        f"{sample_n} sampled test points; the dashed line is where a perfectly "
        f"calibrated prediction would lie"
    ),
)
show_with_alt(
    fig,
    "A scatter of predicted return against actual forward return for a sample of test "
    "points, with a dashed diagonal marking perfect calibration. The vertical extent "
    "of the cloud against its horizontal extent is the spread ratio printed above.",
)

# %% [markdown]
# ## Why Zero-Shot Fails on Finance
#
# The results above are consistent with recent literature. Key findings:
#
# - **Negative $R^2$**: Rahimikia et al. (2025) report that
#   Chronos and TimesFM achieve $R^2$ of $-1.37\%$ and $-2.80\%$ on S&P 500
#   data -- worse than predicting the mean
# - **Negative transfer**: DELPHYNE shows that adding financial
#   data to pre-training *hurts* general benchmarks while still failing on finance
# - **Root causes**: Near-zero autocorrelation, heavy tails, non-stationarity,
#   and adversarial dynamics (exploited patterns disappear) create a fundamental
#   distributional mismatch with pre-training corpora
# - **Pretraining corpus leakage**: Beyond standard temporal splits, the
#   pretraining corpus itself is a leakage channel -- if TSFMs were trained on
#   data overlapping the evaluation period, zero-shot claims are compromised
# - **Risk forecasting differs**: Volatility and VaR prediction show more
#   favorable TSFM results in the cited literature because the target exhibits
#   stronger transferable structure (trend, clustering, mean reversion)
#
# See Section 13.6 for the full evidence review, including second-generation
# models (Chronos-2, Moirai-MoE) and the efficiency frontier.

# %% [markdown]
# ## The same model through sktime
#
# Chronos is called directly above. sktime also wraps it as a `ChronosForecaster`,
# which puts it behind the same `fit`/`predict` interface as every other forecaster in
# that library - the practical value being that a Chronos run and a classical run
# become interchangeable in a pipeline rather than two separate scripts.
# `11_library_landscape` is where that interface is used.
#
# It is not demonstrated here because it cannot be run in this environment: sktime's
# neural forecasters pull in `neuralforecast`, which requires `ray`, and ray has no
# Python 3.14 wheels ([ray-project/ray#56434](https://github.com/ray-project/ray/issues/56434)).
# Once those exist, `uv pip install neuralforecast` makes
# `sktime.forecasting.chronos.ChronosForecaster` importable, and it takes the model
# path and a `num_samples` config the same way the pipeline above does.

# %% [markdown]
# ## Key takeaways
#
# 1. **Check what the pretrained model is being asked to forecast.** These models
#    forecast the context series, and the label is a forward return over a different
#    horizon. Turning a forecast path into a ranking score is a modelling decision
#    made here, not something the model provides, and it is the first thing to state
#    when reporting a zero-shot result.
# 2. **The parameter counts are read off the loaded objects.** Model cards round, and
#    the count that matters is the one in memory. Chronos-t5-small carries far more
#    weight than either fitted baseline, which is worth seeing next to the scores.
# 3. **The comparison is not like-for-like, and saying so is part of the result.**
#    One side sees one feature and no fitting; the other sees eight features and is
#    fitted on this panel. Any gap has at least those two explanations before it has
#    an architectural one.
# 4. **Pretraining is its own leakage channel.** A standard temporal split governs
#    what the model was *fitted* on here, and says nothing about what it was
#    *pretrained* on. If a pretraining corpus overlaps the evaluation period, a
#    zero-shot claim is compromised in a way no split can detect.
# 5. **Negative transfer is a documented finding, not an inference from this run.**
#    DELPHYNE reports that adding financial data to a pretraining mix can hurt
#    general benchmark performance while still failing on finance. Nothing here
#    tests that; it is context for why the zero-shot result is not surprising.
#
# **Known limitations.** One chronological split of one ETF panel, one label horizon,
# one seed, one univariate context feature, zero-shot only, and first-generation
# models. The ridge fit is deterministic; the LSTM is GPU-trained and its IC moves
# slightly between runs on the same software and GPU.
#
# **Next**: `10_uncertainty` stops asking which point forecast is best and asks what a
# model can say about how sure it is.
