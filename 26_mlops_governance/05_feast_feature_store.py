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
# # Feature Store Patterns on Real Case-Study Artifacts
#
# **Chapter 26: MLOps and Governance**
# **Docker image**: `ml4t`
# **Book Reference**: Chapter 26, Section 26.6
# **Prerequisites**: Familiarity with feature engineering and Chapter 25 deployment verification.
#
# **Learning Objectives**:
# - Define feature views on real Parquet sources with an explicit entity key,
#   event timestamp, and TTL.
# - Perform a point-in-time offline join that respects the sealed-holdout
#   boundary as a fail-closed governance guard.
# - Retrieve an online-style as-of snapshot for inference and quantify the
#   training-serving skew that an incorrect timestamp rule introduces.
#
# The notebook demonstrates the core feature-store tasks on the actual
# `us_equities_panel` artifacts. A production feature store such as Feast would
# automate these steps; here the same controls stay visible and reproducible
# inside the repo.

# %%
"""Feature Store Patterns on Real Case-Study Artifacts — demonstrate core feature-store tasks on real case-study artifacts."""

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"
PRIMARY_LABEL = "fwd_ret_1d"
# Left unset so the offline window and the serving date are derived from the fixture's
# own fold geometry. The literals that stood here - 2015-10-01 / 2015-12-30 / 2016-01-04
# - were not independent choices: TRAINING_END equalled the then-final validation fold's
# `val_end` to the day. #819 restored this case study to 16 folds and moved every
# boundary, so a date that used to sit at the edge of a window now sits wherever the new
# geometry puts it. Set any of them to a date string to pin it by hand.
TRAINING_START = None
TRAINING_END = None
AS_OF_DATE = None
# How much of the final validation window the offline join draws on, when derived.
TRAINING_LOOKBACK_DAYS = 91
N_SAMPLE_ASSETS = 8

# %%
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import yaml
from IPython.display import Markdown, display

from utils.cv_splits import generate_cv_splits
from utils.paths import get_case_study_dir
from utils.style import COLORS, FIGSIZE, add_message_title

warnings.filterwarnings("ignore")

CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
CODE_ROOT = CASE_DIR.parent.parent  # repo root (case_studies lives at repo root)
SETUP_PATH = CASE_DIR / "config" / "setup.yaml"

print("Feature Store Patterns on Real Case-Study Artifacts")
print("=" * 60)

# %% [markdown]
# ## 1. Define feature views from the real feature tables
#
# The `financial.parquet` and `model_based.parquet` files act as the offline
# store. The entity key is `symbol`, and the event timestamp is `timestamp`.


# %%
@dataclass
class FeatureViewSpec:
    name: str
    source_path: Path
    entity_key: str
    event_timestamp: str
    ttl_days: int
    feature_columns: list[str]


FINANCIAL_FEATURES = ["past_ret_21d", "vol_21d", "rsi_14", "sharpe_21d"]
MODEL_FEATURES = ["garch_cond_vol", "ffd_log_price", "ffd_log_volume"]

feature_views = [
    FeatureViewSpec(
        name="financial_features",
        source_path=CASE_DIR / "features" / "financial.parquet",
        entity_key="symbol",
        event_timestamp="timestamp",
        ttl_days=1,
        feature_columns=FINANCIAL_FEATURES,
    ),
    FeatureViewSpec(
        name="model_based_features",
        source_path=CASE_DIR / "features" / "model_based.parquet",
        entity_key="symbol",
        event_timestamp="timestamp",
        ttl_days=1,
        feature_columns=MODEL_FEATURES,
    ),
]

# %%
feature_registry = pd.DataFrame(
    [
        {
            "feature_view": spec.name,
            "source_path": spec.source_path.relative_to(CODE_ROOT),
            "entity_key": spec.entity_key,
            "event_timestamp": spec.event_timestamp,
            "ttl_days": spec.ttl_days,
            "n_features": len(spec.feature_columns),
        }
        for spec in feature_views
    ]
)
feature_registry

# %%
setup = yaml.safe_load(SETUP_PATH.read_text())
holdout_start = pd.Timestamp(setup["evaluation"]["holdout_start"])
holdout_end = pd.Timestamp(setup["evaluation"]["holdout_end"])
print(f"Sealed holdout starts on {holdout_start.date()}")

# %%
timeline = pl.scan_parquet(feature_views[0].source_path).select("timestamp").unique().collect()
cv_splits = generate_cv_splits(timeline, case_study_id=CASE_STUDY_ID, label_buffer="1D")


# A window decides the date range served to the offline store: a feature value is only
# servable for a session the model was evaluated on. Whether it *also* names a fold
# depends on which artifact vintage is on disk, and both are in circulation. Stage 04
# now writes one row per stock-date, so a date's value is the same whichever fold reads
# it. A fixture built before that conversion writes one row per (key, fold) with
# genuinely different values per fold, so there the fold is still how a date is
# addressed and cannot be collapsed away.
# %%
def folds_by_coverage(path, windows: list[tuple]) -> list[object]:
    """Pair each evaluation window with the artifact fold that covers it, by date.

    Never by fold id. `ml4t-diagnostic` 0.1.4 reversed what a fold number means, and a
    legacy artifact was written under the older convention, so pairing a stored id with
    a freshly generated one joins each date against the wrong vintage - invisibly, since
    both ids exist and the join succeeds. Ordering both sides by date is the one pairing
    that does not depend on which convention wrote the file.
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


MODEL_BASED_PATH = feature_views[1].source_path
FOLD_KEYED_ARTIFACT = "fold" in pl.scan_parquet(MODEL_BASED_PATH).collect_schema().names()

validation_spans = [
    (pd.Timestamp(split["val_start"]).date(), pd.Timestamp(split["val_end"]).date())
    for split in cv_splits
]
model_windows = [*validation_spans, (holdout_start.date(), holdout_end.date())]
model_folds = (
    folds_by_coverage(MODEL_BASED_PATH, model_windows)
    if FOLD_KEYED_ARTIFACT
    else [None] * len(model_windows)
)
# By date rather than by position: `ml4t-diagnostic` 0.1.4 reversed fold numbering, so
# which end of the list holds the latest window is the thing that moved.
last_validation_span = max(validation_spans)

# The offline join trains on the tail of the last validation window, and serves as of
# the first session inside the sealed holdout - "the model is live, today is after it
# was fitted". Both derived from the geometry rather than pinned.
if TRAINING_END is None:
    TRAINING_END = str(last_validation_span[1])
if TRAINING_START is None:
    TRAINING_START = str(
        max(
            last_validation_span[0],
            pd.Timestamp(TRAINING_END).date() - pd.Timedelta(days=TRAINING_LOOKBACK_DAYS),
        )
    )
if AS_OF_DATE is None:
    AS_OF_DATE = str(
        pl.scan_parquet(feature_views[0].source_path)
        .filter(pl.col("timestamp") >= holdout_start.date())
        .select(pl.min("timestamp"))
        .collect()
        .item()
    )

# Fail-closed governance guard: the training window must end before the sealed holdout
# starts. A misconfigured TRAINING_END would otherwise silently mix pre- and
# post-holdout data into the offline join. It runs after the derivation so that it
# covers a hand-pinned override, which is the case it exists for.
assert pd.Timestamp(TRAINING_END) < holdout_start, (
    f"TRAINING_END {TRAINING_END} must precede sealed holdout {holdout_start.date()}"
)
assert pd.Timestamp(TRAINING_START) <= pd.Timestamp(TRAINING_END), (
    f"TRAINING_START {TRAINING_START} must not follow TRAINING_END {TRAINING_END}"
)
print(f"Offline training window {TRAINING_START} to {TRAINING_END}; serving as of {AS_OF_DATE}")

# %% [markdown]
# ## 2. Offline training retrieval with point-in-time correctness
#
# The offline join uses feature values observed on the decision date and the
# forward-return label generated after that date. This is the contract a feature
# store must preserve. Here features are available after the close on session
# $t$, the label is the next-session return, and any position acts no earlier
# than the next tradable bar.


# %%
def window_terms(clipped: list[tuple]) -> list[pl.Expr]:
    """One predicate per evaluation window, to be OR-ed into a single filter.

    Against a fold-free artifact a window is a date range and nothing else. Against a
    fold-keyed one the same window also names the fold whose fitted state that range is
    evaluated under, and the two must be applied together: those vintages hold different
    values for the same stock-date, so a date range alone would return one row per fold
    and no rule for choosing between them.
    """
    if FOLD_KEYED_ARTIFACT:
        return [
            (pl.col("fold") == fold) & pl.col("timestamp").is_between(start, end)
            for fold, start, end in clipped
        ]
    return [pl.col("timestamp").is_between(start, end) for _, start, end in clipped]


# %%
def load_training_events(start: str, end: str) -> pl.DataFrame:
    labels = (
        pl.scan_parquet(CASE_DIR / "labels" / f"{PRIMARY_LABEL}.parquet")
        .filter(
            (pl.col("timestamp") >= pl.lit(pd.Timestamp(start).date()))
            & (pl.col("timestamp") <= pl.lit(pd.Timestamp(end).date()))
        )
        .select(
            pl.col("timestamp").cast(pl.Date).alias("timestamp"),
            "symbol",
            pl.col(PRIMARY_LABEL).alias("label"),
        )
        .collect()
    )
    return labels


# %%
def load_model_vintage(
    start: str | object,
    end: str | object,
    columns: list[str],
    assets: list[str] | None = None,
) -> pl.DataFrame:
    """Load the model features servable for each decision date in the range.

    One filter over the union of the evaluation windows, not one frame per window
    concatenated. A concat double-counts any date two windows cover, and would do it
    silently; under a single filter the only duplication that can reach the result is
    the artifact's own, which `collapse_fold_replication` resolves or refuses.
    """
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date()
    clipped = [
        (fold, max(start_date, window_start), min(end_date, window_end))
        for fold, (window_start, window_end) in zip(model_folds, model_windows, strict=True)
        if window_start <= end_date and window_end >= start_date
    ]
    if not clipped:
        raise ValueError(
            f"No validation window and not the sealed holdout covers {start}..{end}; "
            "the requested range lies outside every window this model was evaluated on"
        )
    frame = pl.scan_parquet(MODEL_BASED_PATH).filter(pl.any_horizontal(*window_terms(clipped)))
    if assets is not None:
        frame = frame.filter(pl.col("symbol").is_in(assets))
    result = frame.select(["symbol", "timestamp", *columns]).collect().sort(["timestamp", "symbol"])
    assert not result.select(pl.struct("symbol", "timestamp").is_duplicated().any()).item(), (
        "a stock-date resolved to more than one model-based row; the evaluation windows "
        "this notebook derived are not disjoint over the requested range"
    )
    return result


# %% [markdown]
# ### Point-in-time join
# Join feature values to training events using exact timestamp matching.


# %%
def offline_join(events: pl.DataFrame) -> pl.DataFrame:
    assert not events.select(pl.struct("symbol", "timestamp").is_duplicated().any()).item()
    financial = (
        pl.scan_parquet(feature_views[0].source_path)
        .select(["symbol", "timestamp", *FINANCIAL_FEATURES])
        .join(
            events.lazy().select(["symbol", "timestamp"]),
            on=["symbol", "timestamp"],
            how="inner",
        )
        .collect()
    )
    model_based = load_model_vintage(TRAINING_START, TRAINING_END, MODEL_FEATURES).join(
        events.select(["symbol", "timestamp"]), on=["symbol", "timestamp"], how="inner"
    )
    joined = (
        events.join(financial, on=["symbol", "timestamp"], how="left")
        .join(model_based, on=["symbol", "timestamp"], how="left")
        .sort(["timestamp", "symbol"])
    )
    assert joined.height == events.height
    assert not joined.select(pl.struct("symbol", "timestamp").is_duplicated().any()).item()
    return joined.drop_nulls(FINANCIAL_FEATURES + MODEL_FEATURES)


training_events = load_training_events(TRAINING_START, TRAINING_END)
offline_training_set = offline_join(training_events)

print(f"Training events:      {training_events.height:,}")
print(f"Offline joined rows:  {offline_training_set.height:,}")
print(f"Excluded incomplete:  {training_events.height - offline_training_set.height:,}")
offline_training_set.head(5)

# %% [markdown]
# ## 3. Online-style as-of retrieval
#
# At inference time the system needs the latest known features for each asset at
# or before the decision timestamp. The notebook uses the same source tables and
# resolves the latest valid snapshot directly.


# %%
def sample_assets(n_assets: int) -> list[str]:
    from data import load_us_equities

    # Rank on prior dollar liquidity, not nominal share volume. Ranked over the tail of
    # the offline training window rather than a pinned calendar month, so the sample
    # follows the derived window instead of silently drifting away from it.
    rank_from = pd.Timestamp(TRAINING_END).date() - pd.Timedelta(days=30)
    prices = load_us_equities(start_date=TRAINING_START, end_date=TRAINING_END)
    universe = (
        prices.lazy()
        .sort("symbol", "timestamp")
        .with_columns((pl.col("adj_close") * pl.col("adj_volume")).alias("dollar_volume"))
        .with_columns(pl.col("dollar_volume").rolling_mean(21).over("symbol").alias("adv_21d"))
        .filter(
            (pl.col("timestamp") >= rank_from)
            & (pl.col("timestamp") <= pd.Timestamp(TRAINING_END).date())
        )
        .group_by("symbol")
        .agg(pl.col("adv_21d").mean().alias("avg_adv_21d"))
        .sort("avg_adv_21d", descending=True)
        .head(n_assets)
        .collect()
    )
    return universe.get_column("symbol").to_list()


# %% [markdown]
# ### Latest-known snapshot retrieval
# Retrieve the most recent valid feature row for each asset at or before the decision date.


# %%
def latest_snapshot(
    source_path: Path, as_of_date: str, assets: list[str], columns: list[str]
) -> pl.DataFrame:
    cutoff = pd.Timestamp(as_of_date).date()
    return (
        pl.scan_parquet(source_path)
        .filter((pl.col("timestamp") <= pl.lit(cutoff)) & pl.col("symbol").is_in(assets))
        .select(["symbol", "timestamp", *columns])
        .sort(["symbol", "timestamp"])
        .group_by("symbol")
        .tail(1)
        .collect()
    )


# %%
def latest_model_snapshot(as_of_date: str, assets: list[str]) -> pl.DataFrame:
    start = holdout_start if pd.Timestamp(as_of_date) >= holdout_start else pd.Timestamp(as_of_date)
    panel = load_model_vintage(start, as_of_date, MODEL_FEATURES, assets)
    return panel.group_by("symbol").tail(1).sort("symbol")


sampled_assets = sample_assets(N_SAMPLE_ASSETS)
online_financial = latest_snapshot(
    feature_views[0].source_path, AS_OF_DATE, sampled_assets, FINANCIAL_FEATURES
)
online_model = latest_model_snapshot(AS_OF_DATE, sampled_assets)
online_snapshot = online_financial.join(online_model, on=["symbol", "timestamp"], how="inner").sort(
    "symbol"
)

# %% [markdown]
# ### Online snapshot

# %%
online_snapshot

# %% [markdown]
# ## 4. Quantify training-serving skew
#
# The failure mode is simple: serve the *next* available snapshot instead of the
# last known snapshot. That is only one day of look-ahead, but it still changes
# the feature vector and leaks future information into inference.


# %%
def leaked_snapshot(as_of_date: str, assets: list[str]) -> pl.DataFrame:
    cutoff = pd.Timestamp(as_of_date).date()
    future_end = min((pd.Timestamp(cutoff) + pd.Timedelta(days=7)).date(), holdout_end.date())
    financial = (
        pl.scan_parquet(feature_views[0].source_path)
        .filter((pl.col("timestamp") > pl.lit(cutoff)) & pl.col("symbol").is_in(assets))
        .select(["symbol", "timestamp", *FINANCIAL_FEATURES])
        .sort(["symbol", "timestamp"])
        .group_by("symbol")
        .head(1)
    )
    model_based = (
        load_model_vintage(cutoff, future_end, MODEL_FEATURES, assets)
        .lazy()
        .filter(pl.col("timestamp") > pl.lit(cutoff))
        .sort(["symbol", "timestamp"])
        .group_by("symbol")
        .head(1)
    )
    return (
        financial.join(model_based, on=["symbol", "timestamp"], how="inner")
        .collect()
        .sort("symbol")
    )


future_snapshot = leaked_snapshot(AS_OF_DATE, sampled_assets)
comparison = (
    online_snapshot.rename({col: f"{col}_correct" for col in FINANCIAL_FEATURES + MODEL_FEATURES})
    .join(
        future_snapshot.rename(
            {col: f"{col}_leaked" for col in FINANCIAL_FEATURES + MODEL_FEATURES}
        ),
        on="symbol",
        how="inner",
    )
    .to_pandas()
)

# %%
skew_rows = []
for column in FINANCIAL_FEATURES + MODEL_FEATURES:
    skew_rows.append(
        {
            "feature": column,
            "mean_abs_delta": np.abs(
                comparison[f"{column}_leaked"] - comparison[f"{column}_correct"]
            ).mean(),
            "max_abs_delta": np.abs(
                comparison[f"{column}_leaked"] - comparison[f"{column}_correct"]
            ).max(),
        }
    )
skew_table = pd.DataFrame(skew_rows).sort_values("mean_abs_delta", ascending=False)
skew_table

# %%
largest_skew = skew_table.iloc[0]
display(
    Markdown(
        f"**Finding**: `{largest_skew['feature']}` moves most under the deliberately "
        f"leaked timestamp rule (mean absolute delta {largest_skew['mean_abs_delta']:.4g}). "
        "A feature store prevents even a one-session look-ahead from reaching production."
    )
)

# %% [markdown]
# ## 5. Source-lineage view
#
# A feature registry needs more than names. Operators need to know where a view
# came from, how many rows it contains, and what date range it covers.

# %%
lineage_rows = []
for spec in feature_views:
    stats = (
        pl.scan_parquet(spec.source_path)
        .select(
            pl.len().alias("rows"),
            pl.min("timestamp").alias("min_date"),
            pl.max("timestamp").alias("max_date"),
            pl.struct("symbol", "timestamp").n_unique().alias("unique_keys"),
        )
        .collect()
        .row(0)
    )
    lineage_rows.append(
        {
            "feature_view": spec.name,
            "rows": stats[0],
            "start": pd.Timestamp(stats[1]).date(),
            "end": pd.Timestamp(stats[2]).date(),
            "unique_keys": stats[3],
            "features": ", ".join(spec.feature_columns),
        }
    )
lineage_table = pd.DataFrame(lineage_rows)
lineage_table

# %%
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["dual_h_tall"])

ax1 = axes[0]
ax1.barh(skew_table["feature"], skew_table["mean_abs_delta"], color=COLORS["negative"])
add_message_title(ax1, "Look-ahead moves served features")
ax1.set_xlabel("Absolute feature delta")

ax2 = axes[1]
ax2.barh(
    lineage_table["feature_view"],
    lineage_table["unique_keys"] / 1_000_000,
    color=COLORS["blue"],
)
add_message_title(ax2, "Lineage counts unique keys")
ax2.set_xlabel("Unique entity-time keys (millions)")

plt.tight_layout()
fig.show()

# %% [markdown]
# ### Feature registry

# %%
feature_registry

# %% [markdown]
# ### Lineage table

# %%
lineage_table

# %% [markdown]
# **Trading implication**: The operational contract is simple. Training joins
# use the last valid feature values at the decision timestamp, serving uses the
# same rule online, and the registry makes the source tables auditable. A tool
# like Feast automates these controls, but the control itself is what matters.

# %% [markdown]
# ## Key Takeaways
#
# 1. Feature stores enforce point-in-time correctness by joining only the feature vintage fitted for each decision timestamp.
# 2. Training-serving skew from using the wrong timestamp rule is quantifiable — even one day of look-ahead changes the feature vector.
# 3. A source-lineage registry makes each feature view auditable — operators know the source, date range, and row count.
#
# **Next**: See `05b_feast_live` for the same workflow automated with Feast, or `06_mlflow_experiments` for experiment tracking.
