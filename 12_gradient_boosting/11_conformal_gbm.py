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
# # Conformal Prediction for Gradient Boosting Models
#
# **Docker image**: `ml4t`
#
# **Chapter 12, Section 12.5**: From Explanation to Uncertainty and Robustness
#
# > **Execution profile**: `ml4t` on CPU. The LightGBM models use deterministic CPU
# > settings with a fixed thread count. GPU training can be faster, but parallel histogram
# > accumulation is not bitwise reproducible even with fixed seeds.
#
#
# ## Purpose
# This notebook demonstrates **conformal prediction** for uncertainty
# quantification with GBMs. It also tests where the finite-sample coverage
# guarantee weakens when financial observations are temporally dependent.
#
# ## Learning Objectives
# - Implement split conformal prediction from scratch for GBM regression
# - Evaluate coverage properties across ETF, crypto, and futures asset classes
# - Implement CQR and ACI extensions for non-stationary settings
# - Compare split conformal, QR, CQR, and ACI interval behavior
# - Apply conformal interval width as a position sizing signal
#
# **Prerequisites**: Requires feature datasets from `case_studies/{etfs,crypto_perps_funding,cme_futures}/features/`.
#
# ## Cross-References
# - **Section 12.5**: Conformal prediction for UQ, SHAP drift + conformal feedback
# - **Chapter 8**: Feature engineering (input data)
# - **Chapter 19**: Position sizing with uncertainty
# - **Related**: `08_shap_analysis` (drift detection), `09_xai_limitations` (explanation limits)

# %% [markdown] tags=[]
# ## 1. Setup

# %% tags=[]
"""Conformal Prediction for GBMs - construct prediction intervals for financial forecasts."""

import warnings

# lightgbm must be imported before anything that loads scikit-learn, and
# ml4t.diagnostic loads it transitively. Both ship their own OpenMP runtime and
# the first one loaded wins for the whole process; on macOS ARM64, getting
# scikit-learn's libomp first makes LightGBM's next multithreaded fit segfault
# in __kmp_suspend_initialize_thread, killing the kernel with no traceback.
# Plain `import` statements sort ahead of `from ... import` ones, so one
# canonical block keeps this order and isort will not undo it.
import lightgbm as lgb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from ml4t.diagnostic.metrics import cross_sectional_ic_series

warnings.filterwarnings("ignore")

from utils.modeling import fold_temporal_frame, load_modeling_dataset, temporal_fold_index
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, FIGSIZE, add_message_title, format_pct_axis

# %% tags=["parameters"]
MAX_SYMBOLS = 0  # 0 = all symbols
CONFORMAL_ALPHA = 0.10  # 90% prediction intervals
SEED = 42
NUM_THREADS = 4
REFERENCE_INTERVAL_WIDTH = 0.20  # fixed ex-ante uncertainty budget


# %% tags=[]
set_global_seeds(SEED)
CAL_FRACTION = 0.2  # 20% calibration set

# %% [markdown] tags=[]
# ## 2. Conformal Prediction Framework
#
# **Split Conformal Prediction** (Vovk et al., 2005; Lei et al., 2018):
#
# Given training data $(X_1, Y_1), \ldots, (X_n, Y_n)$:
#
# 1. **Split** into proper training $D_{\text{train}}$ and calibration $D_{\text{cal}}$
# 2. **Train** model $\hat{f}$ on $D_{\text{train}}$
# 3. **Compute** residuals $R_i = |Y_i - \hat{f}(X_i)|$ for $(X_i, Y_i) \in D_{\text{cal}}$
# 4. **Order statistic**: $k = \min\{\lceil (n_{\text{cal}} + 1)(1 - \alpha) \rceil,
#    n_{\text{cal}}\}$ and $\hat{q} = R_{(k)}$
# 5. **Interval**: $C(X_{\text{new}}) = [\hat{f}(X_{\text{new}}) - \hat{q}, \hat{f}(X_{\text{new}}) + \hat{q}]$
#
# **Guarantee**: For exchangeable data, $P(Y_{\text{new}} \in C(X_{\text{new}})) \geq 1 - \alpha$


# %% tags=[]
def embargo_steps_from_buffer(label_buffer: str, dates: np.ndarray) -> int:
    """Convert a label horizon to decision-time steps on the observed calendar."""
    unit = label_buffer[-1].upper()
    value = int(label_buffer[:-1])
    unit_ns = {"D": 86_400_000_000_000, "H": 3_600_000_000_000}
    if unit not in unit_ns:
        raise ValueError(f"Unsupported label buffer: {label_buffer}")

    unique_dates = np.unique(dates.astype("datetime64[ns]"))
    median_step_ns = int(
        np.median(np.diff(unique_dates).astype("timedelta64[ns]").astype(np.int64))
    )
    return max(1, int(np.ceil(value * unit_ns[unit] / median_step_ns)))


# %% [markdown] tags=[]
# ### Purged Calibration Split
#
# Calibration is the final 20% of unique training timestamps. The observations immediately
# before it are removed for at least one full label horizon, so no forward label can cross
# from proper training into calibration. Grouping on timestamps also keeps a panel date wholly
# on one side of every boundary.


# %% tags=[]
def chronological_calibration_masks(
    dates: np.ndarray, cal_frac: float, embargo_steps: int
) -> tuple[np.ndarray, np.ndarray, np.datetime64, np.datetime64]:
    """Return proper-training and calibration masks separated by an embargo."""
    normalized_dates = dates.astype("datetime64[ns]")
    unique_dates = np.unique(normalized_dates)
    n_cal_dates = max(1, int(np.ceil(len(unique_dates) * cal_frac)))
    cal_start_idx = len(unique_dates) - n_cal_dates
    proper_end_idx = cal_start_idx - embargo_steps - 1
    if proper_end_idx < 0:
        raise ValueError("Not enough timestamps for calibration after applying the embargo")

    proper_end = unique_dates[proper_end_idx]
    cal_start = unique_dates[cal_start_idx]
    return normalized_dates <= proper_end, normalized_dates >= cal_start, proper_end, cal_start


# %% [markdown] tags=[]
# ### Exact Finite-Sample Order Statistic
#
# The conformal correction selects the $k$th sorted score directly. Passing the
# finite-sample fraction to an interpolating quantile API can select the next rank.


# %% tags=[]
def conformal_order_statistic(scores: np.ndarray, alpha: float) -> float:
    """Return the exact finite-sample conformal score at one-based rank k."""
    values = np.asarray(scores)
    if values.size == 0:
        raise ValueError("Conformal calibration requires at least one score")
    rank = min(int(np.ceil((values.size + 1) * (1 - alpha))), values.size)
    return float(np.partition(values, rank - 1)[rank - 1])


# %% [markdown] tags=[]
# ### Split Conformal Estimator


# %% tags=[]
class ConformalRegressor:
    """Split conformal prediction wrapper for any regression model."""

    def __init__(self, model_factory, alpha: float = 0.10, cal_frac: float = 0.2):
        self.model_factory = model_factory
        self.alpha = alpha
        self.cal_frac = cal_frac
        self.model = None
        self.conformal_width = None
        self.calibration_residuals = None

    def fit(self, X, y, dates, label_buffer: str) -> "ConformalRegressor":
        self.embargo_steps = embargo_steps_from_buffer(label_buffer, dates)
        proper_mask, cal_mask, proper_end, cal_start = chronological_calibration_masks(
            dates, self.cal_frac, self.embargo_steps
        )
        X_proper, X_cal = X[proper_mask], X[cal_mask]
        y_proper, y_cal = y[proper_mask], y[cal_mask]
        self.model = self.model_factory()
        self.model.fit(X_proper, y_proper)
        y_cal_pred = self.model.predict(X_cal)
        residuals = np.abs(y_cal - y_cal_pred)
        self.calibration_residuals = residuals
        self.conformal_width = conformal_order_statistic(residuals, self.alpha)
        self.calibration_bounds = (proper_end, cal_start)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        y_pred = self.model.predict(X)
        lower = y_pred - self.conformal_width
        upper = y_pred + self.conformal_width
        return y_pred, lower, upper


# %% [markdown] tags=[]
# ### Deterministic LightGBM Parameters

# %% tags=[]
from sklearn.base import BaseEstimator, RegressorMixin


def deterministic_lgb_parameters(objective: str, alpha: float | None = None) -> dict:
    """Build the fixed CPU configuration used by every model in this notebook."""
    params = {
        "objective": objective,
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "verbose": -1,
        "seed": SEED,
        "data_random_seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": NUM_THREADS,
        "device_type": "cpu",
    }
    if alpha is not None:
        params["alpha"] = alpha
    return params


# %% [markdown] tags=[]
# ### LightGBM Estimator Wrapper


# %% tags=[]
class LGBWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, n_rounds=200):
        self.n_rounds = n_rounds
        self.model_ = None

    def fit(self, X, y):
        params = deterministic_lgb_parameters("regression")
        train_data = lgb.Dataset(X, label=y)
        self.model_ = lgb.train(params, train_data, num_boost_round=self.n_rounds)
        return self

    def predict(self, X):
        return self.model_.predict(X)


# %% [markdown] tags=[]
# ### Wrapper Factory


# %% tags=[]
def lgb_model_factory():
    return LGBWrapper(n_rounds=200)


# %% [markdown] tags=[]
# ## 3. Load Multi-Asset Data
#
# We compare conformal coverage across three asset classes with
# different return distributions and signal-to-noise ratios.

# %% tags=[]
ASSET_CONFIGS = [
    ("etfs", "fwd_ret_21d", "ETF"),
    ("crypto_perps_funding", "fwd_ret_8h", "Crypto"),
    ("cme_futures", "fwd_ret_5d", "Futures"),
]


# %% [markdown] tags=[]
# ### Resolve Canonical Modeling Inputs
#
# The loader joins each case study's financial, model-based, and label artifacts. The
# composite entity key preserves all panel dimensions, including CME product and position,
# when cross-sectional IC is computed later. Because the loader's combined frame carries
# fold 0 temporal columns only as a schema placeholder, each fold below drops those columns
# and joins the temporal state fitted specifically for that fold.


# %% tags=[]
def replace_temporal_state(mds, temporal, fold_id: int) -> pl.DataFrame:
    """Replace the loader's schema placeholder with one fold's temporal state."""
    frame = mds.dataset
    if temporal is None:
        return frame
    fold_temporal = fold_temporal_frame(
        temporal, fold_id, temporal_keys=mds.temporal_keys, schema=frame.schema
    )
    if fold_temporal.is_empty():
        raise ValueError(f"Temporal artifact has no rows for fold {fold_id}")
    return frame.drop(mds.temporal_feature_names).join(
        fold_temporal, on=mds.temporal_keys, how="left"
    )


# %% [markdown] tags=[]
# ### Fold-Specific Arrays
#
# The joined frame is sorted before conversion so row order, feature order, and complete
# timestamp groups remain stable across deterministic CPU fits.


# %% tags=[]
def build_fold_arrays(mds, split: dict, temporal: pl.DataFrame | None) -> dict:
    """Build one canonical fold with its own temporal learned state."""
    frame = replace_temporal_state(mds, temporal, int(split["fold"]))

    entity = pl.concat_str(
        [pl.col(col).cast(pl.String) for col in mds.entity_cols], separator="|"
    ).alias("__entity")
    frame = frame.filter(pl.col(mds.label_col).is_finite()).with_columns(entity)
    frame = frame.sort([mds.date_col, "__entity"])
    dates = frame[mds.date_col].to_numpy()
    train_start, train_end, val_start, val_end = (
        np.datetime64(split[key].to_datetime64())
        for key in ("train_start", "train_end", "val_start", "val_end")
    )
    train_mask = (dates >= train_start) & (dates <= train_end)
    val_mask = (dates >= val_start) & (dates <= val_end)
    X = frame.select(mds.feature_names).to_numpy()
    return {
        "fold": int(split["fold"]),
        "X_train": X[train_mask],
        "y_train": frame[mds.label_col].to_numpy()[train_mask],
        "dates_train": dates[train_mask],
        "X_val": X[val_mask],
        "y_val": frame[mds.label_col].to_numpy()[val_mask],
        "dates_val": dates[val_mask],
        "symbols_val": frame["__entity"].to_numpy()[val_mask],
    }


# %% [markdown] tags=[]
# Each temporal artifact must contain every requested canonical fold ID. This explicit
# mapping is the provenance link between the split record and its learned feature state.


# %% tags=[]
processed_datasets = {}

for cs_id, label, display_name in ASSET_CONFIGS:
    try:
        mds = load_modeling_dataset(cs_id, label, max_symbols=MAX_SYMBOLS)
        temporal = mds.temporal_by_fold
        requested_splits = mds.splits[:5]
        if temporal is not None:
            available_folds = set(
                temporal_fold_index(temporal, mds.date_col)["fold"].unique().to_list()
            )
            required_folds = {int(split["fold"]) for split in requested_splits}
            if not required_folds.issubset(available_folds):
                raise ValueError(
                    f"Missing temporal folds: {sorted(required_folds - available_folds)}"
                )

        fold_data = [build_fold_arrays(mds, split, temporal) for split in requested_splits]
        processed_datasets[display_name] = {
            "fold_data": fold_data,
            "feature_cols": mds.feature_names,
            "label_buffer": mds.label_buffer,
        }
        print(
            f"  {display_name}: {len(fold_data)} fold-aware matrices "
            f"({len(mds.feature_names)} features; embargo={mds.label_buffer})"
        )
    except FileNotFoundError as missing:
        # The only tolerable absence: a reader who has not built this case study's inputs.
        print(f"  {display_name}: skipped, inputs not built ({missing})")

if not processed_datasets:
    raise RuntimeError(
        "no asset class loaded - every one was skipped. A conformal interval needs a fitted "
        "model, so there is nothing below this cell to render."
    )
print(f"\nLoaded {len(processed_datasets)} asset classes")

# %% [markdown] tags=[]
# ## 4. Conformal Prediction Evaluation
#
# We use each case study's canonical pre-holdout walk-forward folds. Every outer
# train-validation boundary carries the label horizon configured in `setup.yaml`, and
# the calibration split inside each training fold applies the same embargo. The sealed
# 2024-2025 holdouts play no role in model fitting, calibration, method comparison, or
# interpretation.


# %% [markdown] tags=[]
# Each fold fits on the canonical training window, calibrates inside that window after a
# second purge, and evaluates once on the corresponding validation dates.


# %% tags=[]
def run_conformal_splits(
    fold_data: list[dict],
    label_buffer: str,
    alpha: float,
    n_splits: int,
) -> list[dict]:
    fold_results = []

    for fold in fold_data[:n_splits]:
        if len(fold["X_train"]) < 200 or len(fold["X_val"]) < 50:
            continue

        conformal = ConformalRegressor(lgb_model_factory, alpha=alpha)
        conformal.fit(fold["X_train"], fold["y_train"], fold["dates_train"], label_buffer)
        y_pred, lower, upper = conformal.predict(fold["X_val"])
        fold_results.append(
            {
                "y_true": fold["y_val"],
                "y_pred": y_pred,
                "dates": fold["dates_val"],
                "symbols": fold["symbols_val"],
                "lower": lower,
                "upper": upper,
                "half_width": conformal.conformal_width,
                "fold": fold["fold"],
                "embargo_steps": conformal.embargo_steps,
            }
        )
    return fold_results


# %% [markdown] tags=[]
# ### Cross-Sectional Information Coefficient


# %% tags=[]
def mean_cross_sectional_ic(dates, symbols, predictions, returns) -> float:
    """Compute mean per-date Spearman IC on unique timestamp-entity keys."""
    pred_df = pl.DataFrame({"timestamp": dates, "symbol": symbols, "prediction": predictions})
    ret_df = pl.DataFrame({"timestamp": dates, "symbol": symbols, "forward_return": returns})
    ic_per_date = cross_sectional_ic_series(
        pred_df,
        ret_df,
        pred_col="prediction",
        ret_col="forward_return",
        date_col="timestamp",
        entity_col="symbol",
    )
    ic_clean = ic_per_date.drop_nans("ic").drop_nulls("ic")
    return float(ic_clean["ic"].mean()) if ic_clean.height else float("nan")


# %% [markdown] tags=[]
# ### Metric Aggregation


# %% tags=[]
def summarize_conformal_results(results: list[dict], alpha: float) -> dict | None:
    if not results:
        return None

    y_true_all = np.concatenate([r["y_true"] for r in results])
    y_pred_all = np.concatenate([r["y_pred"] for r in results])
    dates_all = np.concatenate([r["dates"] for r in results])
    symbols_all = np.concatenate([r["symbols"] for r in results])
    lower_all = np.concatenate([r["lower"] for r in results])
    upper_all = np.concatenate([r["upper"] for r in results])

    coverage = ((y_true_all >= lower_all) & (y_true_all <= upper_all)).mean()
    ic = mean_cross_sectional_ic(dates_all, symbols_all, y_pred_all, y_true_all)
    avg_width = np.mean(upper_all - lower_all)

    return {
        "coverage": coverage,
        "target_coverage": 1 - alpha,
        "coverage_error": abs(coverage - (1 - alpha)),
        "ic": ic,
        "avg_width": avg_width,
        "n_samples": len(y_true_all),
        "results": results,
    }


# %% [markdown] tags=[]
# ### Conformal Evaluation Wrapper


# %% tags=[]
def evaluate_conformal(
    fold_data: list[dict],
    label_buffer: str,
    alpha: float = 0.10,
    n_splits: int = 5,
) -> dict | None:
    results = run_conformal_splits(
        fold_data,
        label_buffer,
        alpha=alpha,
        n_splits=n_splits,
    )
    return summarize_conformal_results(results, alpha)


# %% tags=[]
all_metrics = {}

for asset_name, data in processed_datasets.items():
    metrics = evaluate_conformal(
        data["fold_data"],
        data["label_buffer"],
        alpha=CONFORMAL_ALPHA,
        n_splits=5,
    )
    if metrics:
        all_metrics[asset_name] = metrics

# %% tags=[]
eval_df = pl.DataFrame(
    [
        {
            "symbol": name,
            "coverage": round(m["coverage"], 3),
            "target": round(m["target_coverage"], 2),
            "coverage_error": round(m["coverage_error"], 3),
            "IC": round(m["ic"], 3),
            "interval_width": round(m["avg_width"], 5),
            "n_samples": m["n_samples"],
        }
        for name, m in all_metrics.items()
    ]
)
eval_df

# %% [markdown] tags=[]
# **Interpretation**: With complete timestamp groups and horizon-sized embargoes,
# the coverage column in the table above does not land on its 0.90 target: one
# asset class overshoots it and the other two fall short, ETFs by the widest
# margin. The finite-sample guarantee assumes exchangeability between calibration
# and validation residuals; these walk-forward results show why empirical coverage
# still matters when financial residuals shift over time. Predictive ordering is
# modest throughout, and the ranking by coverage is not the ranking by IC - the
# best-covered asset class here is not the best-ranked one, so read both columns.

# %% [markdown] tags=[]
# ## 5. Coverage Visualization


# %% tags=[]
def label_coverage_bars(ax, bars, values) -> None:
    """Place percentage labels immediately above coverage bars."""
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{value:.1%}",
            ha="center",
        )


assets = list(all_metrics.keys())
coverages = [all_metrics[a]["coverage"] for a in assets]
ics = [all_metrics[a]["ic"] for a in assets]


# %% [markdown] tags=[]
# The paired view separates calibration from predictive ordering: empirical coverage is
# judged against its nominal target, while IC is judged against a zero-information baseline.


# %% tags=[]
if all_metrics:
    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE["dual_v"])

    ax1 = axes[0]
    bars = ax1.bar(assets, coverages, color=COLORS["blue"])
    ax1.axhline(
        1 - CONFORMAL_ALPHA,
        linestyle="--",
        color=COLORS["amber"],
        linewidth=1.5,
    )
    ax1.set_ylabel("Empirical coverage (%)")
    ax1.set_ylim(0, 1)
    format_pct_axis(ax1)
    add_message_title(
        ax1,
        "Coverage varies under temporal dependence",
        subtitle="Nominal target = 90%; purged walk-forward validation",
    )
    label_coverage_bars(ax1, bars, coverages)

    ax2 = axes[1]
    ax2.bar(assets, ics, color=COLORS["slate"])
    ax2.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=1)
    ax2.set_ylabel("Mean daily Spearman IC")
    add_message_title(
        ax2,
        "Predictive rank correlation differs across markets",
        subtitle="Cross-sectional IC on the same validation folds",
    )

    plt.tight_layout()
    plt.show()

# %% [markdown] tags=[]
# ## 6. Conformal Variants: Split, CQR, and Adaptive
#
# | Method | Core Idea | Strength | Limitation |
# |--------|-----------|----------|------------|
# | **Split Conformal** | Calibrate absolute residuals | Finite-sample coverage under exchangeability | Symmetric intervals |
# | **Quantile Regression (QR)** | Learn lower/upper quantiles directly | Asymmetric intervals | No finite-sample guarantee |
# | **Conformalized QR (CQR)** | Quantile models + conformal calibration | Asymmetric + calibrated | Needs 2 models + calibration split |
# | **Adaptive Conformal (ACI)** | Update miscoverage target online | Tracks regime shifts | Online update hyperparameter sensitivity |

# %% tags=[]
# The section below is written about ETFs: the figure title, the interpretation under the
# coverage table and takeaway 2 all read the 2023 ETF validation fold specifically. Taking
# whichever asset class happened to load first would caption crypto or futures intervals as
# ETF ones for any reader who built only those, so the choice is named rather than positional.
COMPARISON_ASSET = "ETF"

if processed_datasets:
    if COMPARISON_ASSET not in processed_datasets:
        raise RuntimeError(
            f"the method comparison is written about {COMPARISON_ASSET} results, and only "
            f"{sorted(processed_datasets)} loaded. Build case_studies/etfs/ features, or change "
            f"COMPARISON_ASSET and the interpretation below together - the prose reads one "
            f"specific fold and does not transfer."
        )
    test_asset = COMPARISON_ASSET
    test_data = processed_datasets[test_asset]
    comparison_fold = test_data["fold_data"][0]

    X_tr = comparison_fold["X_train"]
    y_tr = comparison_fold["y_train"]
    dates_tr = comparison_fold["dates_train"]
    X_te = comparison_fold["X_val"]
    y_te = comparison_fold["y_val"]
    dates_te = comparison_fold["dates_val"]
    symbols_te = comparison_fold["symbols_val"]

    conformal = ConformalRegressor(lgb_model_factory, alpha=CONFORMAL_ALPHA)
    conformal.fit(X_tr, y_tr, dates_tr, test_data["label_buffer"])
    y_pred_conf, lower_conf, upper_conf = conformal.predict(X_te)

    coverage_conf = ((y_te >= lower_conf) & (y_te <= upper_conf)).mean()
    width_conf = np.mean(upper_conf - lower_conf)

# %% [markdown] tags=[]
# ### Train Quantile Regression Models


# %% tags=[]
def train_quantile_model(X: np.ndarray, y: np.ndarray, alpha: float, n_rounds: int = 200):
    """Fit a deterministic CPU LightGBM quantile model."""
    train_data = lgb.Dataset(X, label=y)
    return lgb.train(
        deterministic_lgb_parameters("quantile", alpha),
        train_data,
        num_boost_round=n_rounds,
    )


# %% [markdown] tags=[]
# The uncalibrated baseline fits both quantiles on the complete outer training window. It
# never sees the validation fold, which remains a method-comparison set rather than a final
# holdout estimate.


# %% tags=[]
if processed_datasets:
    lower_q = CONFORMAL_ALPHA / 2
    upper_q = 1 - CONFORMAL_ALPHA / 2
    n_rounds = 200

    model_lower = train_quantile_model(X_tr, y_tr, lower_q, n_rounds)
    model_upper = train_quantile_model(X_tr, y_tr, upper_q, n_rounds)

    lower_qr = model_lower.predict(X_te)
    upper_qr = model_upper.predict(X_te)

    coverage_qr = ((y_te >= lower_qr) & (y_te <= upper_qr)).mean()
    width_qr = np.mean(upper_qr - lower_qr)

# %% [markdown] tags=[]
# ### Conformalized Quantile Regression (CQR)
#
# $$s_i = \max\left(\hat{q}_{\ell}(x_i) - y_i,\ y_i - \hat{q}_{u}(x_i)\right), \quad
# \hat{q}_{\mathrm{cqr}} = Q_{1-\alpha}(s)$$
#
# $$C_{\mathrm{CQR}}(x) = [\hat{q}_{\ell}(x) - \hat{q}_{\mathrm{cqr}},\ \hat{q}_{u}(x) + \hat{q}_{\mathrm{cqr}}]$$

# %% tags=[]
if processed_datasets:
    cqr_embargo_steps = embargo_steps_from_buffer(test_data["label_buffer"], dates_tr)
    proper_mask, cal_mask, cqr_train_end, cqr_cal_start = chronological_calibration_masks(
        dates_tr, CAL_FRACTION, cqr_embargo_steps
    )
    X_tr_q, y_tr_q = X_tr[proper_mask], y_tr[proper_mask]
    X_cal_q, y_cal_q = X_tr[cal_mask], y_tr[cal_mask]

    model_lower_cqr = train_quantile_model(X_tr_q, y_tr_q, lower_q, n_rounds)
    model_upper_cqr = train_quantile_model(X_tr_q, y_tr_q, upper_q, n_rounds)

    lower_cal = model_lower_cqr.predict(X_cal_q)
    upper_cal = model_upper_cqr.predict(X_cal_q)
    cqr_scores = np.maximum(lower_cal - y_cal_q, y_cal_q - upper_cal)

    cqr_qhat = conformal_order_statistic(cqr_scores, CONFORMAL_ALPHA)

# %% tags=[]
if processed_datasets:
    lower_cqr = model_lower_cqr.predict(X_te) - cqr_qhat
    upper_cqr = model_upper_cqr.predict(X_te) + cqr_qhat

    coverage_cqr = ((y_te >= lower_cqr) & (y_te <= upper_cqr)).mean()
    width_cqr = np.mean(upper_cqr - lower_cqr)


# %% [markdown] tags=[]
# ### Adaptive Conformal Inference (ACI)
#
# ACI updates the effective miscoverage target online and recomputes interval
# width from a rolling residual buffer. A forward label enters that buffer only
# after its full horizon has elapsed.


# %% tags=[]
def compute_adaptive_intervals(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    dates: np.ndarray,
    cal_scores: np.ndarray,
    alpha: float = 0.10,
    gamma: float = 0.01,
    window: int = 250,
    label_horizon_steps: int = 21,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = list(cal_scores[-window:])
    alpha_t = alpha
    alpha_path = np.empty_like(y_pred)
    lower = np.empty_like(y_pred)
    upper = np.empty_like(y_pred)
    normalized_dates = dates.astype("datetime64[ns]")

    unique_dates = np.unique(normalized_dates)
    for date_idx, timestamp in enumerate(unique_dates):
        if date_idx >= label_horizon_steps:
            matured_timestamp = unique_dates[date_idx - label_horizon_steps]
            matured_mask = normalized_dates == matured_timestamp
            misses = (y_true[matured_mask] < lower[matured_mask]) | (
                y_true[matured_mask] > upper[matured_mask]
            )
            alpha_t = float(np.clip(alpha_t + gamma * (alpha - misses.mean()), 0.01, 0.30))
            scores.extend(np.abs(y_true[matured_mask] - y_pred[matured_mask]))
            if len(scores) > window:
                scores = scores[-window:]

        date_mask = normalized_dates == timestamp
        q_t = conformal_order_statistic(np.asarray(scores), alpha_t)
        lower[date_mask] = y_pred[date_mask] - q_t
        upper[date_mask] = y_pred[date_mask] + q_t
        alpha_path[date_mask] = alpha_t

    return lower, upper, alpha_path


# %% tags=[]
if processed_datasets:
    lower_aci, upper_aci, alpha_path = compute_adaptive_intervals(
        y_pred_conf,
        y_te,
        dates_te,
        conformal.calibration_residuals,
        alpha=CONFORMAL_ALPHA,
        gamma=0.01,
        window=250,
        label_horizon_steps=conformal.embargo_steps,
    )
    coverage_aci = ((y_te >= lower_aci) & (y_te <= upper_aci)).mean()
    width_aci = np.mean(upper_aci - lower_aci)

# %% tags=[]
comparison_df = (
    pl.DataFrame(
        {
            "method": [
                "Split Conformal",
                "Quantile Regression",
                "Conformalized Quantile (CQR)",
                "Adaptive Conformal (ACI)",
                "Target",
            ],
            "coverage": [
                round(coverage_conf, 3),
                round(coverage_qr, 3),
                round(coverage_cqr, 3),
                round(coverage_aci, 3),
                round(1 - CONFORMAL_ALPHA, 2),
            ],
            "interval_width": [
                round(width_conf, 5),
                round(width_qr, 5),
                round(width_cqr, 5),
                round(width_aci, 5),
                None,
            ],
        }
    )
    if processed_datasets
    else pl.DataFrame()
)
comparison_df

# %% tags=[]
# Visualization indices on a shared subsample
if processed_datasets:
    unique_symbols, symbol_counts = np.unique(symbols_te, return_counts=True)
    display_symbol = unique_symbols[np.argmax(symbol_counts)]
    symbol_idx = np.flatnonzero(symbols_te == display_symbol)
    n_show = min(200, len(symbol_idx))
    idx = symbol_idx[-n_show:]
    x_dates = dates_te[idx]

# %% tags=[]
if processed_datasets:
    methods = [
        ("Split conformal", lower_conf, upper_conf, COLORS["slate"]),
        ("Quantile regression", lower_qr, upper_qr, COLORS["amber"]),
        ("Conformalized quantile", lower_cqr, upper_cqr, COLORS["neutral"]),
        ("Adaptive conformal", lower_aci, upper_aci, COLORS["blue"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE["dashboard_2x2"], sharex=True, sharey=True)
    for ax, (method, lower, upper, color) in zip(axes.flat, methods, strict=True):
        ax.fill_between(x_dates, 100 * lower[idx], 100 * upper[idx], alpha=0.18, color=color)
        ax.scatter(x_dates, 100 * y_te[idx], s=5, color=COLORS["blue"], zorder=3)
        add_message_title(ax, method, subtitle=f"{display_symbol}; final {n_show} validation dates")
        ax.set_ylabel("Forward return (%)")
    for ax in axes[-1]:
        ax.set_xlabel("Validation date")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    fig.suptitle(
        f"Label-mature adaptation reshapes {test_asset} prediction intervals through time",
        x=0.01,
        ha="left",
    )
    plt.tight_layout()
    plt.show()

# %% tags=[]
if processed_datasets:
    fig, ax = plt.subplots(figsize=FIGSIZE["single_wide"])
    ax.plot(x_dates, alpha_path[idx], color=COLORS["slate"], linewidth=1.8)
    ax.axhline(CONFORMAL_ALPHA, color=COLORS["amber"], linestyle="--", linewidth=1.5)
    ax.set_xlabel("Validation date")
    ax.set_ylabel(r"Miscoverage target $\alpha_t$")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    add_message_title(
        ax,
        "ACI updates only after forward outcomes mature",
        subtitle=f"{display_symbol}; nominal miscoverage = {CONFORMAL_ALPHA:.0%}",
    )
    plt.tight_layout()
    plt.show()

# %% [markdown] tags=[]
# **Interpretation** (purged 2023 ETF validation fold), reading the table above:
# - Split conformal undercovers the most, showing that a fixed calibration
#   quantile can fail when the next residual regime changes sharply.
# - Quantile regression buys the narrowest intervals and still undercovers, and
#   it carries no conformal coverage guarantee at all.
# - CQR is the only one of the four to reach the 0.90 target, by calibrating the
#   asymmetric quantile models rather than a single symmetric width.
# - ACI uses the widest intervals of the four and still misses, because each
#   update waits for its forward outcome to mature.
# Width and coverage do not move together here: the widest method is not the best
# covered, which is the point of the comparison.

# %% [markdown] tags=[]
# ## 7. Position Sizing Application
#
# Wider intervals signal higher uncertainty, naturally reducing position
# size. A fixed ex-ante interval-width budget $B=0.20$ sets the scale, so no
# fold depends on widths observed in another period:
#
# $$\text{Relative Exposure} = \frac{B}{\text{Interval Width}}$$


# %% [markdown] tags=[]
# The scaling function depends only on each fold's calibration width and the fixed budget.
# Changing another fold therefore cannot alter an earlier displayed exposure.


# %% tags=[]
def fixed_budget_exposure(widths: list[float], budget: float) -> np.ndarray:
    """Scale interval widths by an ex-ante fixed uncertainty budget."""
    width_array = np.asarray(widths)
    if np.any(width_array <= 0):
        raise ValueError("Interval widths must be positive")
    return budget / width_array


# %% tags=[]
if all_metrics:
    example_metrics = list(all_metrics.values())[0]
    widths = [2 * r["half_width"] for r in example_metrics["results"]]
    position_sizes = fixed_budget_exposure(widths, REFERENCE_INTERVAL_WIDTH)

    sizing_df = pl.DataFrame(
        {
            "fold": list(range(len(widths))),
            "interval_width": [round(w, 5) for w in widths],
            "relative_exposure": [round(p, 2) for p in position_sizes],
        }
    )
else:
    sizing_df = pl.DataFrame()
sizing_df

# %% [markdown] tags=[]
# Narrow intervals (high confidence) scale up exposure; wide intervals
# (high uncertainty) scale it down. The fixed 0.20 budget is illustrative,
# not an estimated optimum. Chapter 19 integrates uncertainty into a complete
# position-sizing rule with return forecasts and risk constraints.
#
# Static split conformal widths remain fixed until recalibration. Adaptive conformal
# methods can instead update their residual buffer after forward outcomes mature,
# widening or narrowing later intervals without using unavailable labels.
# Section 12.5 connects that update cycle to SHAP-based drift monitoring.

# %% [markdown] tags=[]
# ## 8. Key Takeaways
#
# 1. **Always verify empirical coverage**: across purged walk-forward folds no
#    asset class lands on its 0.90 target - one overshoots, the rest fall short,
#    ETFs furthest. The exchangeability condition is not exact for financial
#    panels, so the theoretical guarantee does not replace a time-ordered
#    empirical check.
#
# 2. **Adaptivity can matter more than width**: on the 2023 ETF validation fold,
#    CQR is the only method to reach the target while ACI uses the widest
#    intervals of the four and still misses, because the residual regime changes
#    underneath it. Wider is not better covered.
#
# 3. **Match the wrapper to the estimator**: Split conformal and ACI can wrap
#    a point-prediction model without an error-distribution assumption. QR and
#    CQR instead require an estimator that learns conditional quantiles.
#
# 4. **Risk management**: Interval width is a natural uncertainty signal
#    for position sizing. Wider intervals automatically scale positions
#    down. Chapter 19 integrates conformal widths into Kelly sizing.
#
# **Next**: See `08_shap_analysis` for SHAP-based drift detection that
# complements conformal uncertainty, or Chapter 19 for position sizing.
