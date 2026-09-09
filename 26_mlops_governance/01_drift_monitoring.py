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
# # Drift Monitoring on Real Case-Study Artifacts
#
# **Chapter 26: MLOps and Governance**
# **Docker image**: `ml4t`
# **Book Reference**: Chapter 26, Sections 26.2-26.3
# **Prerequisites**: Chapter 25 deployment verification and basic performance metrics.
#
# **Learning Objectives**:
# - Measure how far a feature's distribution has moved since a model went live, using the
#   population stability index and the two-sample Kolmogorov-Smirnov test.
# - Tell a broken data feed apart from a genuine change in the market, by validating the
#   incoming feature panel before any drift statistic is computed.
# - Track whether a live model's predictions are still informative, by recomputing its rank
#   correlation with realized returns over a rolling window.
# - Assemble those measurements into one dashboard a desk can read before deciding whether
#   to investigate, reduce exposure, or retrain.

# %%
"""Drift Monitoring on Real Case-Study Artifacts: distribution and performance drift on a real holdout window."""

# %% [markdown]
# ## Settings
#
# `CASE_STUDY_ID` names the case study whose deployed model is being monitored.
#
# `LABEL` is the return the model predicts. Left unset it is read from the registry: whichever
# label has a holdout prediction set on disk, preferring the case study's configured primary
# label when more than one qualifies. It is read rather than typed because only a holdout
# prediction set will do here, and which labels have one changes as a case study is rebuilt. A
# validation prediction set covers the sessions the model was selected on, so monitoring one
# would measure the window the model was chosen to fit and call the result drift.
#
# Set it to a label and that label is required: the notebook stops rather than monitoring a
# different return horizon than the one asked for.
#
# `REFERENCE_START` is the beginning of the distribution that everything is compared against.
# Left unset, it is derived from the case study's own fold geometry rather than typed. A date
# typed here would be a claim about where a fold boundary falls, and fold boundaries move when a
# case study is rebuilt: the same literal then lands wherever the new geometry puts it, silently,
# with the notebook still reporting a reference window. Set it to a date string to pin one by
# hand.
#
# `LOOKBACK_DAYS` is the width of both the baseline and the current window, in sessions. Sixty
# three is a quarter, long enough for a rolling information coefficient to mean anything and
# short enough to notice a change within a monitoring cycle.
#
# `PSI_WATCH` and `PSI_ALERT` are the population-stability thresholds. They are the conventional
# pair and they are conventions, not estimates: the population stability index has no
# distribution under a null hypothesis, so no threshold on it is a significance level. They rank
# features by how far a distribution moved and say where to start looking.
#
# `KS_WATCH_PVALUE` is a significance level, and the Kolmogorov-Smirnov test does have a null.
# Against tens of thousands of daily observations it will reject on differences too small to act
# on, which is why it is read beside the stability index rather than instead of it.
# `PSI_KS_FLOOR` is how the two are read together: a K-S rejection only raises a feature to watch
# when the stability index has also moved off zero. Without a floor the test's sensitivity at this
# sample size would put every feature on watch on almost every cycle, and a monitor that always
# alerts is a monitor nobody reads.
#
# `IC_WATCH_DROP` and friends are absolute drops from the launch baseline for the information
# coefficient and the hit rate, and relative increases for mean squared error. Absolute for the
# first two because a coefficient near zero makes a relative change meaningless, relative for the
# third because squared error has no natural scale.

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"
LABEL = None
REFERENCE_START = None
LOOKBACK_DAYS = 63
PSI_WATCH = 0.10
PSI_ALERT = 0.25
KS_WATCH_PVALUE = 0.05
PSI_KS_FLOOR = 0.02
IC_WATCH_DROP = 0.005
IC_ALERT_DROP = 0.01
MSE_WATCH_INCREASE = 0.05
MSE_ALERT_INCREASE = 0.10
SEED = 42

# %%
import json
import sqlite3
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import yaml
from ml4t.diagnostic.validation import DataFrameValidator
from scipy import stats

from utils.cv_splits import generate_cv_splits
from utils.paths import get_case_study_dir, get_output_dir
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, FIGSIZE, add_message_title, format_pct_axis, show_with_alt

# Named, not blanket: a bare ignore would also hide the convergence and numerical
# warnings a reader needs to see.
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

set_global_seeds(SEED)

CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
SETUP_PATH = CASE_DIR / "config" / "setup.yaml"
REGISTRY_PATH = CASE_DIR / "run_log" / "registry.db"

FEATURE_COLUMNS = [
    "past_ret_21d",
    "vol_21d",
    "rsi_14",
    "volume_ratio",
    "garch_cond_vol",
]

print("Drift Monitoring on Real Holdout Data")
print("=" * 60)


# %% [markdown]
# ## 1. Load the real monitoring boundary
#
# The *holdout* is the stretch of history a case study reserves and never reads while models
# are being chosen. It is the closest thing an offline notebook has to live production data: the
# model saw none of it, so its behaviour there is what its behaviour after deployment would look
# like. Both ends of that window and the case study's primary label come from its own
# `setup.yaml`, so the monitoring boundary is the one the case study declares rather than one
# this notebook asserts.


# %%
def load_holdout_window(setup_path: Path) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    setup = yaml.safe_load(setup_path.read_text())
    evaluation = setup["evaluation"]
    return (
        pd.Timestamp(evaluation["holdout_start"]),
        pd.Timestamp(evaluation["holdout_end"]),
        str(setup["labels"]["primary"]),
    )


# %% [markdown]
# ### Load the holdout prediction artifact
# Find the content-addressed prediction hash for the holdout model run.


# %%
def load_holdout_prediction_hash(
    registry_path: Path, preferred: str, required: bool
) -> tuple[str, str, str]:
    """The label, prediction hash and model family of a holdout prediction set on disk.

    Only a prediction set the registry records as ``split='holdout'`` qualifies, and only
    one whose parquet has been written.

    When *required* is true, *preferred* was named explicitly and nothing else will do:
    monitoring a different return horizon than the one asked for would produce a dashboard
    that is correct about a question nobody asked. When it is false, *preferred* came from
    the case study's configuration and is a preference, so any label with a holdout
    prediction set is acceptable and the configured one wins if it has one.
    """
    query = """
        SELECT tr.label, ps.prediction_hash, tr.family
        FROM training_runs tr
        JOIN prediction_sets ps ON tr.training_hash = ps.training_hash
        WHERE ps.split = 'holdout' {label_filter}
        ORDER BY tr.label = ? DESC, tr.created_at DESC
    """
    pred_dir = registry_path.parent / "predictions"
    with sqlite3.connect(registry_path) as conn:
        rows = conn.execute(
            query.format(label_filter="AND tr.label = ?" if required else ""),
            (preferred, preferred) if required else (preferred,),
        ).fetchall()
    for label, pred_hash, family in rows:
        if (pred_dir / str(pred_hash) / "predictions.parquet").exists():
            return str(label), str(pred_hash), str(family)
    scope = f"for {preferred}" if required else "for any label"
    raise RuntimeError(
        f"{CASE_STUDY_ID} has no materialized holdout prediction set {scope}. Drift can "
        "only be measured against sessions the model was scored on out of sample, and "
        "this case study's holdout stage has not written one."
    )


# %%
holdout_start, holdout_end, configured_primary_label = load_holdout_window(SETUP_PATH)
monitored_label, holdout_run_hash, holdout_model_family = load_holdout_prediction_hash(
    REGISTRY_PATH, LABEL or configured_primary_label, required=LABEL is not None
)

_pred_check_path = CASE_DIR / "run_log" / "predictions" / holdout_run_hash / "predictions.parquet"
_pred_bounds = (
    pl.scan_parquet(_pred_check_path)
    .select(pl.min("timestamp").alias("start"), pl.max("timestamp").alias("end"))
    .collect()
    .row(0)
)
_pred_start, _pred_end = (pd.Timestamp(value) for value in _pred_bounds)
_label_bounds = (
    pl.scan_parquet(CASE_DIR / "labels" / f"{monitored_label}.parquet")
    .filter(pl.col("timestamp").is_between(holdout_start.date(), holdout_end.date()))
    .select(pl.min("timestamp").alias("start"), pl.max("timestamp").alias("end"))
    .collect()
    .row(0)
)
_label_start, _label_end = (pd.Timestamp(value) for value in _label_bounds)
assert (_pred_start, _pred_end) == (_label_start, _label_end), (
    f"Holdout predictions cover {_pred_start.date()} to {_pred_end.date()}, "
    f"not all label-eligible sessions {_label_start.date()} to {_label_end.date()}"
)

print(f"Holdout window: {holdout_start.date()} to {holdout_end.date()}")
print(f"Holdout prediction run: {holdout_run_hash} ({holdout_model_family}, {monitored_label})")


# %% [markdown]
# ## 2. Validate incoming features before measuring drift
#
# A broken join or null-filled feature column can look like drift. Validation
# happens first so monitoring is not diagnosing a pipeline failure as a market
# change.


# %%
def validate_feature_data(features_df: pl.DataFrame, required_columns: list[str]) -> None:
    validator = DataFrameValidator(features_df)
    validator.check_empty().check_min_rows(min_rows=100).require_columns(required_columns)
    validator.check_nulls(columns=required_columns, allow_nulls=False)
    print(f"Validated {len(features_df):,} rows across {len(required_columns)} monitored features.")


# %% [markdown]
# ### Restrict the panel to the sessions the model was evaluated on
#
# A walk-forward split divides the history into folds: each fold fits on the sessions before
# a cut-off and is scored on the sessions just after it, its *validation window*. Only those
# windows and the holdout are sessions on which this model ever produced an out-of-sample
# number, so they are the only sessions a drift measurement can be taken over. Everything
# else in the file is a session the model trained on.
#
# `model_based.parquet` carries one row per stock-date. Where it also carries a `fold`
# column, the same stock-date appears once per fold with the value that fold's transform
# produced, and the window and the fold have to be applied together; where it does not, the
# date range alone selects the row.


# %%
def evaluation_spans() -> tuple[list[tuple[object, object]], tuple[object, object]]:
    """The validation windows and the holdout, as date ranges.

    Derived from `generate_cv_splits`, which is the same function the modelling
    notebooks split on, rather than typed here. A boundary written as a literal is a
    claim about the case study's fold geometry that nothing re-checks, and rebuilding
    the case study with a different number of folds moves every one of them.
    """
    timeline = (
        pl.scan_parquet(CASE_DIR / "features" / "financial.parquet")
        .select("timestamp")
        .unique()
        .collect()
    )
    splits = generate_cv_splits(timeline, case_study_id=CASE_STUDY_ID, label_buffer="1D")
    validation = [
        (pd.Timestamp(split["val_start"]).date(), pd.Timestamp(split["val_end"]).date())
        for split in splits
    ]
    return validation, (holdout_start.date(), holdout_end.date())


MODEL_BASED_PATH = CASE_DIR / "features" / "model_based.parquet"
FOLD_KEYED_ARTIFACT = "fold" in pl.scan_parquet(MODEL_BASED_PATH).collect_schema().names()


# %%
def folds_by_coverage(path, windows: list[tuple]) -> list[object]:
    """Pair each evaluation window with the artifact fold that covers it, by date.

    A fold id is a label the writer chose, not a property of the data, and two tools can
    number the same folds in opposite directions. Joining a stored id to a freshly
    generated one then pairs each date with the wrong fitted state and still succeeds,
    because both ids exist. Ordering both sides by the dates they cover is the pairing
    that holds whatever numbering wrote the file.
    """
    coverage = (
        pl.scan_parquet(path)
        .group_by("fold")
        .agg(pl.max("timestamp").alias("last_covered"))
        .collect()
        .sort("last_covered")
    )
    stored = coverage["fold"].to_list()
    if len(stored) != len(windows):
        raise ValueError(
            f"model_based.parquet carries {len(stored)} folds and this notebook derived "
            f"{len(windows)} evaluation windows. They cannot be paired by date, so the "
            "artifact does not describe the geometry this case study now declares; "
            "regenerate it against the current stage 04."
        )
    order = sorted(range(len(windows)), key=lambda i: windows[i])
    paired: list[object] = [None] * len(windows)
    for position, window_index in enumerate(order):
        paired[window_index] = stored[position]
    return paired


VALIDATION_SPANS, HOLDOUT_SPAN = evaluation_spans()
EVALUATION_SPANS = [*VALIDATION_SPANS, HOLDOUT_SPAN]
# Fold ids are resolved only where the artifact carries them.
EVALUATION_FOLDS = (
    folds_by_coverage(MODEL_BASED_PATH, EVALUATION_SPANS)
    if FOLD_KEYED_ARTIFACT
    else [None] * len(EVALUATION_SPANS)
)
# By date, not by list position: which end of the list holds the latest window depends on
# the numbering convention, and the dates do not.
LAST_VALIDATION_SPAN = max(VALIDATION_SPANS)
print(
    f"{len(VALIDATION_SPANS)} validation windows, "
    f"{min(VALIDATION_SPANS)[0]} to {LAST_VALIDATION_SPAN[1]}; holdout {HOLDOUT_SPAN[0]}"
)

reference_start = REFERENCE_START or str(LAST_VALIDATION_SPAN[0])
print(f"Reference window starts {reference_start} (latest validation window)")


# %%
def window_terms(clipped: list[tuple]) -> list[pl.Expr]:
    """One predicate per evaluation window, to be OR-ed into a single filter.

    Where the artifact carries no fold column a window is a date range and nothing else.
    Where it does, the same window also names the fold whose fitted state that range is
    evaluated under, and the two are applied together: each fold holds its own value for
    the same stock-date, so a date range alone returns one row per fold with no rule for
    choosing between them.
    """
    if FOLD_KEYED_ARTIFACT:
        return [
            (pl.col("fold") == fold) & pl.col("timestamp").is_between(start, end)
            for fold, start, end in clipped
        ]
    return [pl.col("timestamp").is_between(start, end) for _, start, end in clipped]


# %%
def load_temporal_panel(start_date: object, end_date: object, columns: list[str]) -> pl.DataFrame:
    """Model-based features over the requested range, restricted to evaluated sessions.

    One filter over the union of the windows, not one frame per window concatenated. A
    concat double-counts any date two windows cover and would do it silently, so the
    duplicate-key assertion below could only report it after the fact; under a single
    filter it states what it is for, that one stock-date resolves to one row.
    """
    clipped = [
        (fold, max(start_date, span_start), min(end_date, span_end))
        for fold, (span_start, span_end) in zip(EVALUATION_FOLDS, EVALUATION_SPANS, strict=True)
        if span_start <= end_date and span_end >= start_date
    ]
    if not clipped:
        raise ValueError(
            f"No validation window and not the holdout covers {start_date}..{end_date}; "
            "the requested range lies outside every window this model was evaluated on"
        )
    result = (
        pl.scan_parquet(MODEL_BASED_PATH)
        .filter(pl.any_horizontal(*window_terms(clipped)))
        .select(["symbol", "timestamp", *columns])
        .collect()
        .sort(["timestamp", "symbol"])
    )
    duplicate_keys = result.select(pl.struct("symbol", "timestamp").is_duplicated().any()).item()
    assert duplicate_keys is False, (
        "a stock-date resolved to more than one model-based row; the evaluation windows "
        "this notebook derived are not disjoint over the requested range"
    )
    return result


# %% [markdown]
# ### Load feature panel
# Join financial and model-based feature parquets over the requested date range.


# %%
def load_feature_panel(start: str, end: str, feature_columns: list[str]) -> pl.DataFrame:
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date()
    financial_cols = [c for c in feature_columns if c != "garch_cond_vol"]
    model_cols = [c for c in feature_columns if c == "garch_cond_vol"]

    financial = (
        pl.scan_parquet(CASE_DIR / "features" / "financial.parquet")
        .filter(
            (pl.col("timestamp") >= pl.lit(start_date)) & (pl.col("timestamp") <= pl.lit(end_date))
        )
        .select(["symbol", "timestamp", *financial_cols])
    )
    model_based = load_temporal_panel(start_date, end_date, model_cols).lazy()

    return (
        financial.join(model_based, on=["symbol", "timestamp"], how="inner")
        .drop_nulls(feature_columns)
        .collect()
    )


# %% [markdown]
# ### Load holdout predictions
# Read the content-addressed prediction parquet for the holdout run.


# %%
def load_holdout_predictions(run_hash: str) -> pl.DataFrame:
    pred_path = CASE_DIR / "run_log" / "predictions" / run_hash / "predictions.parquet"
    lf = pl.scan_parquet(pred_path)
    # Prediction sets are written under either naming; accept both.
    cols = lf.collect_schema().names()
    renames = {}
    if "actual" in cols and "y_true" not in cols:
        renames["actual"] = "y_true"
    if "prediction" in cols and "y_score" not in cols:
        renames["prediction"] = "y_score"
    if renames:
        lf = lf.rename(renames)
    return (
        lf.select(
            pl.col("timestamp").cast(pl.Date).alias("timestamp"),
            pl.col("symbol"),
            pl.col("y_score").alias("score"),
            pl.col("y_true").alias("actual_return"),
        )
        .collect()
        .sort(["timestamp", "symbol"])
    )


# %%
reference_features = load_feature_panel(
    reference_start, str(holdout_start.date() - pd.Timedelta(days=1)), FEATURE_COLUMNS
)
# Restrict the artifact to the configured holdout window.
holdout_predictions = load_holdout_predictions(holdout_run_hash).filter(
    (pl.col("timestamp") >= holdout_start.date()) & (pl.col("timestamp") <= holdout_end.date())
)
holdout_dates = holdout_predictions.get_column("timestamp").unique().sort().to_list()
assert len(holdout_dates) >= 2 * LOOKBACK_DAYS, (
    "Holdout is too short for distinct monitoring windows"
)
launch_dates = holdout_dates[:LOOKBACK_DAYS]
current_window_dates = holdout_dates[-LOOKBACK_DAYS:]
current_window_start = pd.Timestamp(current_window_dates[0])
current_window_end = pd.Timestamp(current_window_dates[-1])
current_features = load_feature_panel(
    str(current_window_start.date()), str(current_window_end.date()), FEATURE_COLUMNS
)

validate_feature_data(current_features, FEATURE_COLUMNS)
validate_feature_data(reference_features, FEATURE_COLUMNS)

print(f"Reference feature rows: {reference_features.height:,}")
print(f"Current feature rows:   {current_features.height:,}")


# %% [markdown]
# ## 3. Feature and prediction drift diagnostics
#
# The reference set is the model's last validation window, which is the most recent stretch
# of history it was scored on before going live. The current slice is the most recent
# `LOOKBACK_DAYS` sessions inside the holdout.
#
# The two statistics answer different questions. The population stability index bins the
# reference sample, counts how the current sample falls into the same bins, and sums a
# weighted log ratio over them: it measures how far the distribution moved, in a unit with
# no null distribution behind it. The two-sample Kolmogorov-Smirnov test takes the largest
# gap between the two empirical cumulative distributions and returns the probability of
# seeing a gap that large if both samples came from one distribution. So the index says how
# much, and the test says whether the difference is larger than sampling noise.


# %%
def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> tuple[float, np.ndarray]:
    inner = np.linspace(reference.min(), reference.max(), n_bins + 1)[1:-1]
    bin_edges = np.concatenate([[-np.inf], inner, [np.inf]])
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)
    ref_pct = ref_counts / len(reference) + epsilon
    cur_pct = cur_counts / len(current) + epsilon
    bin_psi = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    return float(np.sum(bin_psi)), bin_psi


# %% [markdown]
# ### Drift metric container
# A typed container for per-feature drift statistics and alert status.


# %%
@dataclass
class DriftMetric:
    name: str
    psi: float
    ks_stat: float
    ks_pvalue: float
    reference_mean: float
    current_mean: float
    status: Literal["OK", "WATCH", "ALERT"]


# %% [markdown]
# ### Polars-to-numpy helper
# Extract a single column as a float numpy array, dropping nulls.


# %%
def to_numpy(frame: pl.DataFrame, column: str) -> np.ndarray:
    values = frame.get_column(column).drop_nulls().to_numpy()
    return np.asarray(values, dtype=float)


# %% [markdown]
# ### Summarize feature drift
# Compute PSI and K-S for each monitored feature and assign alert levels.


# %%
def summarize_feature_drift(
    reference_frame: pl.DataFrame,
    current_frame: pl.DataFrame,
    feature_columns: list[str],
) -> list[DriftMetric]:
    metrics: list[DriftMetric] = []
    for feature in feature_columns:
        reference = to_numpy(reference_frame, feature)
        current = to_numpy(current_frame, feature)
        psi, _ = compute_psi(reference, current)
        ks_stat, ks_pvalue = stats.ks_2samp(reference, current)
        if psi >= PSI_ALERT:
            status = "ALERT"
        elif psi >= PSI_WATCH or (psi >= PSI_KS_FLOOR and ks_pvalue < KS_WATCH_PVALUE):
            status = "WATCH"
        else:
            status = "OK"
        metrics.append(
            DriftMetric(
                name=feature,
                psi=psi,
                ks_stat=float(ks_stat),
                ks_pvalue=float(ks_pvalue),
                reference_mean=float(reference.mean()),
                current_mean=float(current.mean()),
                status=status,
            )
        )
    return metrics


# %%
feature_drift = summarize_feature_drift(reference_features, current_features, FEATURE_COLUMNS)
feature_drift_df = pd.DataFrame([metric.__dict__ for metric in feature_drift]).sort_values(
    "psi", ascending=False
)
feature_drift_df


# %%
baseline_predictions = (
    holdout_predictions.filter(pl.col("timestamp").is_in(launch_dates))
    .select("score")
    .to_series()
    .to_numpy()
)
recent_predictions = (
    holdout_predictions.filter(pl.col("timestamp").is_in(current_window_dates))
    .select("score")
    .to_series()
    .to_numpy()
)

prediction_psi, _ = compute_psi(baseline_predictions, recent_predictions)
prediction_ks = stats.ks_2samp(baseline_predictions, recent_predictions)

print(f"Prediction PSI: {prediction_psi:.4f}")
print(f"Prediction K-S p-value: {prediction_ks.pvalue:.4f}")


# %% [markdown]
# Read the two together. The feature with the largest stability index says which input moved
# most; the prediction-side index says whether that movement reached the model's output.
#
# The interesting case is when they disagree. A feature that crosses the watch threshold while
# the prediction distribution sits still means the model was not leaning on that feature much,
# which is worth knowing and is not an emergency. Predictions moving while every feature looks
# stable is the alarming direction: the inputs the monitor watches are not the ones that changed,
# and the cause is upstream of them.


# %% [markdown]
# ## 4. Rolling performance diagnostics on the holdout stream
#
# The first 63 holdout sessions act as the launch baseline. Monitoring then
# tracks whether later 63-day windows retain similar IC and hit-rate behavior.


# %%
daily_metrics = (
    holdout_predictions.group_by("timestamp")
    .agg(
        pl.corr("score", "actual_return", method="spearman").alias("ic"),
        ((pl.col("score") * pl.col("actual_return")) > 0).mean().alias("hit_rate"),
        ((pl.col("actual_return") - pl.col("score")) ** 2).mean().alias("mse"),
        pl.col("score").mean().alias("score_mean"),
        pl.col("score").std().alias("score_std"),
        pl.len().alias("n_assets"),
    )
    .sort("timestamp")
    .to_pandas()
)

ROLLING_IC = f"rolling_ic_{LOOKBACK_DAYS}"
ROLLING_HIT_RATE = f"rolling_hit_rate_{LOOKBACK_DAYS}"
ROLLING_MSE = f"rolling_mse_{LOOKBACK_DAYS}"
MIN_ROLLING_SESSIONS = LOOKBACK_DAYS // 3

daily_metrics["timestamp"] = pd.to_datetime(daily_metrics["timestamp"])
for source, rolled in (("ic", ROLLING_IC), ("hit_rate", ROLLING_HIT_RATE), ("mse", ROLLING_MSE)):
    daily_metrics[rolled] = (
        daily_metrics[source].rolling(LOOKBACK_DAYS, min_periods=MIN_ROLLING_SESSIONS).mean()
    )

baseline_slice = daily_metrics.iloc[:LOOKBACK_DAYS]
current_slice = daily_metrics.iloc[-LOOKBACK_DAYS:]

baseline_ic = baseline_slice["ic"].mean()
baseline_hit_rate = baseline_slice["hit_rate"].mean()
baseline_mse = baseline_slice["mse"].mean()
current_ic = current_slice["ic"].mean()
current_hit_rate = current_slice["hit_rate"].mean()
current_mse = current_slice["mse"].mean()

# %%
alert_rows = [
    {
        "metric": "prediction_distribution",
        "baseline": 0.0,
        "current": prediction_psi,
        "threshold": PSI_WATCH,
        "status": "ALERT"
        if prediction_psi >= PSI_ALERT
        else ("WATCH" if prediction_psi >= PSI_WATCH else "OK"),
    },
    {
        "metric": ROLLING_IC,
        "baseline": baseline_ic,
        "current": current_ic,
        "threshold": baseline_ic - IC_ALERT_DROP,
        "status": "ALERT"
        if current_ic < baseline_ic - IC_ALERT_DROP
        else ("WATCH" if current_ic < baseline_ic - IC_WATCH_DROP else "OK"),
    },
    {
        "metric": ROLLING_HIT_RATE,
        "baseline": baseline_hit_rate,
        "current": current_hit_rate,
        "threshold": baseline_hit_rate - IC_ALERT_DROP,
        "status": "ALERT"
        if current_hit_rate < baseline_hit_rate - IC_ALERT_DROP
        else ("WATCH" if current_hit_rate < baseline_hit_rate - IC_WATCH_DROP else "OK"),
    },
    {
        "metric": ROLLING_MSE,
        "baseline": baseline_mse,
        "current": current_mse,
        "threshold": baseline_mse * (1 + MSE_ALERT_INCREASE),
        "status": "ALERT"
        if current_mse > baseline_mse * (1 + MSE_ALERT_INCREASE)
        else ("WATCH" if current_mse > baseline_mse * (1 + MSE_WATCH_INCREASE) else "OK"),
    },
]

alert_table = pd.DataFrame(alert_rows)
alert_table


# %% [markdown]
# ## 5. Monitoring dashboard
#
# The dashboard combines feature drift, output drift, and rolling signal quality
# into one review surface. A desk can inspect this view before deciding whether
# to investigate data quality, reduce exposure, or start a model update.


# %%
fig, axes = plt.subplots(2, 2, figsize=FIGSIZE["dashboard_2x2"], constrained_layout=True)
ax1 = axes[0, 0]
colors = feature_drift_df["status"].map(
    {"OK": COLORS["blue"], "WATCH": COLORS["amber"], "ALERT": COLORS["negative"]}
)
ax1.bar(feature_drift_df["name"], feature_drift_df["psi"], color=colors)
ax1.axhline(PSI_WATCH, color=COLORS["neutral"], linestyle="--", linewidth=1, label="Watch")
ax1.axhline(PSI_ALERT, color=COLORS["negative"], linestyle=":", linewidth=1, label="Alert")
add_message_title(
    ax1, "Population stability index by feature", subtitle="Watch and alert thresholds dashed"
)
ax1.set_ylabel("PSI")
ax1.tick_params(axis="x", rotation=35)
ax1.legend()

ax2 = axes[0, 1]
# Explicit edges avoid numpy 2.2.x histogram regression with integer bins
_lo = min(baseline_predictions.min(), recent_predictions.min())
_hi = max(baseline_predictions.max(), recent_predictions.max())
_edges = np.linspace(_lo, _hi, 41)
sns.histplot(
    baseline_predictions,
    bins=_edges,
    stat="density",
    color=COLORS["neutral"],
    alpha=0.45,
    ax=ax2,
)
sns.histplot(
    recent_predictions,
    bins=_edges,
    stat="density",
    color=COLORS["blue"],
    alpha=0.55,
    ax=ax2,
)
add_message_title(
    ax2,
    "Prediction score density, launch against latest window",
    subtitle=f"{LOOKBACK_DAYS} sessions each, shared bins",
)
ax2.set_xlabel("Score")
ax2.legend(["Launch baseline", f"Latest {LOOKBACK_DAYS} sessions"])

ax3 = axes[1, 0]
ax3.plot(daily_metrics["timestamp"], daily_metrics[ROLLING_IC], color=COLORS["blue"], linewidth=2)
ax3.axhline(
    baseline_ic, color=COLORS["neutral"], linestyle="--", linewidth=1, label="Launch baseline"
)
add_message_title(
    ax3,
    "Rolling cross-sectional information coefficient",
    subtitle=f"{LOOKBACK_DAYS}-session mean, launch baseline dashed",
)
ax3.set_ylabel("Cross-sectional IC")
ax3.set_xlabel("Holdout date")
ax3.legend()

ax4 = axes[1, 1]
ax4.plot(
    daily_metrics["timestamp"],
    daily_metrics[ROLLING_HIT_RATE],
    color=COLORS["copper"],
    linewidth=2,
)
ax4.axhline(
    baseline_hit_rate,
    color=COLORS["neutral"],
    linestyle="--",
    linewidth=1,
    label="Launch baseline",
)
add_message_title(
    ax4,
    "Rolling directional hit rate",
    subtitle=f"{LOOKBACK_DAYS}-session mean, launch baseline dashed",
)
ax4.set_ylabel("Hit rate (%)")
ax4.set_xlabel("Holdout date")
format_pct_axis(ax4)
ax4.legend()

for axis in (ax3, ax4):
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

show_with_alt(
    fig,
    "Four-panel monitoring dashboard. Top left: bar chart of the population stability index "
    "for each monitored feature, with dashed watch and dotted alert reference lines. Top "
    "right: overlaid density histograms of the launch-window and latest-window prediction "
    "scores. Bottom left: the rolling information coefficient across the holdout, against a "
    "dashed launch-baseline line. Bottom right: the rolling hit rate on the same dates, "
    "against its own launch baseline.",
)


# %% [markdown]
# Read the panels against each other rather than one at a time. The top row asks whether the
# distributions moved: the left panel says which inputs did, the right whether that reached the
# model's output. The bottom row asks whether the model still works. The information coefficient
# here is the Spearman rank correlation between each session's scores and the returns they
# predicted, which is what a long-short book uses, and the hit rate is the share of names whose
# direction the model got right.
#
# The two rows compare against different reference periods, because the data allows nothing
# else. Feature values exist before the holdout, so their reference is the model's last
# validation window. Predictions exist only inside the holdout, so their baseline is its first
# `LOOKBACK_DAYS` sessions. A feature shift and a prediction shift measured here are therefore
# changes relative to different starting points, and only their direction is comparable.
#
# The rows can disagree, and what a disagreement narrows down is worth being precise about.
# Features moving while the output distribution sits still is consistent with the model not
# weighting those features heavily, and it is also consistent with offsetting moves, so it says
# to look rather than that nothing happened. The output moving while every monitored feature
# looks stable is the more urgent direction: whatever moved is either outside the monitored set
# or is a change in how the features relate to each other, which no comparison of one feature
# at a time can see.

# %% [markdown]
# ### Persist the dashboard's inputs
#
# The four panels above are drawn a second time, at print size, for Figure 26.2 in the book.
# Writing their inputs out here is what lets that happen from stored arrays.

# %%
ARTIFACT_DIR = get_output_dir(26, "figure_26_2")

feature_drift_df[["name", "psi", "status"]].to_parquet(
    ARTIFACT_DIR / "feature_drift.parquet", index=False
)
np.save(ARTIFACT_DIR / "baseline_predictions.npy", baseline_predictions)
np.save(ARTIFACT_DIR / "recent_predictions.npy", recent_predictions)
daily_metrics[["timestamp", ROLLING_IC, ROLLING_HIT_RATE]].to_parquet(
    ARTIFACT_DIR / "daily_metrics.parquet", index=False
)
_ = (ARTIFACT_DIR / "scalars.json").write_text(
    json.dumps(
        {"baseline_ic": float(baseline_ic), "baseline_hit_rate": float(baseline_hit_rate)}, indent=2
    )
)


# %% [markdown]
# ### Feature drift summary

# %%
feature_drift_df[["name", "psi", "ks_pvalue", "status", "reference_mean", "current_mean"]]

# %% [markdown]
# ### Alert table

# %%
alert_table


# %% [markdown]
# The alert table turns the diagnostics into an operating state. Each row carries the launch
# baseline, the current window, and the threshold between them, so a status is traceable to the
# two numbers that produced it rather than being an opinion.
#
# What the table is not is a retraining trigger. A degraded status says something changed, and
# the three candidate explanations - the data broke, the market moved, the model decayed - call
# for different responses and are told apart by different evidence. Retraining on a broken feed
# fits the model to the break. So this is the evidence package for a decision:
# [`03_safe_model_rollout`](03_safe_model_rollout.ipynb) is where the decision is acted on, and
# only after data integrity has been confirmed.

# %% [markdown]
# ## Key Takeaways
#
# 1. Validate the incoming feature panel before computing any drift statistic. A broken join or
#    a null-filled column produces the same distribution shift a market regime change does, and
#    the two call for opposite responses.
# 2. Read the stability index and the Kolmogorov-Smirnov test together rather than either alone.
#    The index has no null distribution, so its thresholds are conventions; the test has one and
#    will reject on differences too small to act on at production sample sizes.
# 3. Measure drift only over sessions the model was scored on out of sample. Comparing a live
#    window against sessions the model trained on measures the training set, not the drift.
# 4. Distribution drift and performance decay are separate measurements and can move
#    independently. Track the input distributions, the output distribution and the rolling
#    signal quality, and read the disagreements between them.
#
# **Known limitations**
#
# - A holdout window is a stand-in for live data. It is a single period on a single universe, so
#   the thresholds that look reasonable here are not calibrated for another market.
# - The stability index is computed on the reference sample's own bin edges. A feature whose
#   support extends past the reference range piles into the outer bins, and the index understates
#   how far that feature moved.
# - Every statistic here compares two windows and says nothing about when in between the change
#   happened. `02_online_drift_detection` covers detectors that answer that.
# - The alert levels are a review trigger, not an action. What to do about a degraded status
#   depends on which of the three causes produced it, and this notebook does not identify which.
#
# **Next**: See `02_online_drift_detection` for sequential drift detectors on the same validation streams.
