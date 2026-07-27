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
# # Feature Store with Feast — Live Integration
#
# **Chapter 26: MLOps and Governance**
# **Docker image**: `ml4t`
# **Book Reference**: Chapter 26, Section 26.6
# **Companion to**: `05_feast_feature_store` (manual Polars implementation)
#
# **Learning Objectives**:
# - Bring up a local Feast repository from the same `us_equities_panel`
#   feature artifacts NB05 uses, declaring entity and feature views in code.
# - Run an offline historical retrieval against an event DataFrame and use the
#   same API for a single-as-of snapshot that simulates an online request.
# - Validate exact parity with the manual Polars implementation after selecting
#   the fitted feature vintage that was valid at each decision time.
#
# Notebook `05` demonstrates feature-store concepts — offline joins,
# as-of retrieval, training-serving skew — using pure Polars. This companion
# notebook shows how **Feast** automates the same workflow against the same
# `us_equities_panel` artifacts.
#
# **Prerequisites**: `pip install 'feast>=0.40'` (included in the `[mlops]` extra).

# %%
"""Feature Store with Feast — Live Integration."""

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"
PRIMARY_LABEL = "fwd_ret_1d"
TRAINING_START = "2015-10-01"
TRAINING_END = "2015-12-30"
AS_OF_DATE = "2016-01-04"
N_SAMPLE_ASSETS = 8

# %%
import shutil
import tempfile
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml

from utils.cv_splits import generate_cv_splits
from utils.paths import get_case_study_dir

warnings.filterwarnings("ignore")

CODE_ROOT = Path.cwd()
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)

FINANCIAL_FEATURES = ["past_ret_21d", "vol_21d", "rsi_14", "sharpe_21d"]
MODEL_FEATURES = ["garch_cond_vol", "ffd_log_price", "ffd_log_volume"]
ALL_FEATURES = FINANCIAL_FEATURES + MODEL_FEATURES

print("Feature Store with Feast — Live Integration")
print("=" * 60)


# %% [markdown]
# ## 1. Import Feast
#
# Feast provides `Entity`, `FeatureView`, `FileSource`, and `FeatureStore` — the
# building blocks of a feature store declaration. The `Field` class defines
# feature schemas with typed columns.

# %%
from feast import Entity, FeatureStore, FeatureView, Field, FileSource
from feast.data_format import ParquetFormat
from feast.types import Float64
from feast.value_type import ValueType

print("Feast imported successfully")


# %% [markdown]
# ## 2. Prepare Feast-compatible source files
#
# Our case study parquet files store `timestamp` as a `Date` column. Feast's
# offline store requires datetime timestamps for point-in-time joins. We create
# temporary copies with the timestamp cast to `Datetime` — in production this
# conversion would happen at materialization time.

# %%
feast_tmp = tempfile.mkdtemp(prefix="feast_ml4t_")
feast_data_dir = Path(feast_tmp) / "data"
feast_data_dir.mkdir()

financial_src = CASE_DIR / "features" / "financial.parquet"
model_src = CASE_DIR / "features" / "model_based.parquet"

financial_feast_path = feast_data_dir / "financial.parquet"
model_feast_path = feast_data_dir / "model_based.parquet"

train_start = pd.Timestamp(TRAINING_START).date()
train_end = pd.Timestamp(TRAINING_END).date()
# Pad 30 days before training start so TTL lookback has data
filter_start = train_start - timedelta(days=30)
# Pad 30 days after to cover as-of retrieval demo
filter_end = train_end + timedelta(days=30)

(
    pl.scan_parquet(financial_src)
    .select(["symbol", "timestamp", *FINANCIAL_FEATURES])
    .filter((pl.col("timestamp") >= filter_start) & (pl.col("timestamp") <= filter_end))
    .with_columns(pl.col("timestamp").cast(pl.Datetime("ns")))
    .collect()
    .write_parquet(financial_feast_path)
)

# %%
setup = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())
holdout_start = pd.Timestamp(setup["evaluation"]["holdout_start"])
holdout_end = pd.Timestamp(setup["evaluation"]["holdout_end"])
timeline = pl.scan_parquet(financial_src).select("timestamp").unique().collect()
cv_splits = generate_cv_splits(timeline, case_study_id=CASE_STUDY_ID, label_buffer="1D")
holdout_fold = pl.scan_parquet(model_src).select(pl.max("fold")).collect().item()
model_windows = [
    (
        split["fold"],
        pd.Timestamp(split["val_start"]).date(),
        pd.Timestamp(split["val_end"]).date(),
    )
    for split in cv_splits
]
model_windows.append((holdout_fold, holdout_start.date(), holdout_end.date()))

# %%
model_scan = pl.scan_parquet(model_src)
model_frames = [
    model_scan.filter(
        (pl.col("fold") == fold)
        & pl.col("timestamp").is_between(
            max(pd.Timestamp(filter_start).date(), start),
            min(pd.Timestamp(filter_end).date(), end),
        )
    ).select(["symbol", "timestamp", *MODEL_FEATURES])
    for fold, start, end in model_windows
    if start <= pd.Timestamp(filter_end).date() and end >= pd.Timestamp(filter_start).date()
]
model_features = pl.concat(model_frames).collect().sort(["timestamp", "symbol"])
assert not model_features.select(pl.struct("symbol", "timestamp").is_duplicated().any()).item()
model_features.with_columns(pl.col("timestamp").cast(pl.Datetime("ns"))).write_parquet(
    model_feast_path
)

fin_rows = pl.scan_parquet(financial_feast_path).select(pl.len()).collect().item()
mod_rows = pl.scan_parquet(model_feast_path).select(pl.len()).collect().item()
print(f"Financial features: {fin_rows:,} rows → {financial_feast_path.name}")
print(f"Model-based features: {mod_rows:,} rows → {model_feast_path.name}")


# %% [markdown]
# ## 3. Create the Feast repository
#
# A Feast repo needs a `feature_store.yaml` that declares the project name,
# provider, registry location, and store backends. We use the local file
# provider with SQLite for both the registry and online store.

# %%
feast_config = {
    "project": "ml4t_feature_store",
    "provider": "local",
    "registry": {"path": str(Path(feast_tmp) / "registry.db")},
    "online_store": {"type": "sqlite", "path": str(Path(feast_tmp) / "online.db")},
    "offline_store": {"type": "file"},
    "entity_key_serialization_version": 3,
}
config_path = Path(feast_tmp) / "feature_store.yaml"
config_path.write_text(yaml.dump(feast_config))
print(f"Feast repo: {feast_tmp}")
print(yaml.dump(feast_config, default_flow_style=False))


# %% [markdown]
# ## 4. Define entity and feature views
#
# The entity is `symbol` — the universal identifier across all case study
# artifacts. Each feature view maps to one parquet source file and declares a
# typed schema. The `ttl` (time-to-live) of 2 days prevents stale features
# from leaking into point-in-time joins.

# %%
symbol_entity = Entity(
    name="symbol",
    join_keys=["symbol"],
    value_type=ValueType.STRING,
    description="Stock ticker symbol",
)

financial_source = FileSource(
    path=str(financial_feast_path.resolve()),
    timestamp_field="timestamp",
    file_format=ParquetFormat(),
)

model_source = FileSource(
    path=str(model_feast_path.resolve()),
    timestamp_field="timestamp",
    file_format=ParquetFormat(),
)

financial_fv = FeatureView(
    name="financial_features",
    entities=[symbol_entity],
    schema=[Field(name=col, dtype=Float64) for col in FINANCIAL_FEATURES],
    source=financial_source,
    ttl=timedelta(days=2),
)

model_fv = FeatureView(
    name="model_features",
    entities=[symbol_entity],
    schema=[Field(name=col, dtype=Float64) for col in MODEL_FEATURES],
    source=model_source,
    ttl=timedelta(days=2),
)

print("Defined:")
print(f"  Entity:  {symbol_entity.name} (join_key={symbol_entity.join_key})")
print(f"  View 1:  {financial_fv.name} ({len(FINANCIAL_FEATURES)} features)")
print(f"  View 2:  {model_fv.name} ({len(MODEL_FEATURES)} features)")


# %% [markdown]
# ## 5. Apply the feature store
#
# `store.apply()` registers the entity and feature views in the Feast registry.
# After this call, the feature store knows where to find each feature, what
# entity key to join on, and what timestamp field governs point-in-time
# correctness.

# %%
store = FeatureStore(repo_path=feast_tmp)
store.apply([symbol_entity, financial_fv, model_fv])

registered_views = store.list_feature_views()
registered_entities = store.list_entities()
print(f"Registered {len(registered_entities)} entities, {len(registered_views)} feature views")
for fv in registered_views:
    print(f"  {fv.name}: {[f.name for f in fv.features]}")


# %% [markdown]
# ## 6. Offline retrieval — point-in-time join
#
# This is the core feature-store operation: given an entity DataFrame with
# `(symbol, event_timestamp)` pairs, retrieve the latest valid feature values
# at or before each timestamp. Feast handles the point-in-time join internally
# using the TTL constraint. Features become available after the session close;
# the forward label resolves on the next session, and a position can act no
# earlier than that next tradable bar.

# %%
# Consecutive validation windows are separated by the label embargo, so the dates
# inside a gap carry no fitted model-feature vintage at all, and the same holds
# after the last window ends. Asking for a decision date there is answerable only
# by carrying the previous fold forward: Feast does so within its TTL, the manual
# exact-key join does not, and the two stop agreeing. Restrict the event set to
# the (symbol, timestamp) keys the producer actually fitted, which removes the
# trailing edge and every interior gap rather than only the last date.
requested = (
    pl.scan_parquet(CASE_DIR / "labels" / f"{PRIMARY_LABEL}.parquet")
    .filter(
        (pl.col("timestamp") >= pl.lit(pd.Timestamp(TRAINING_START).date()))
        & (pl.col("timestamp") <= pl.lit(pd.Timestamp(TRAINING_END).date()))
    )
    .select("symbol", "timestamp", pl.col(PRIMARY_LABEL).alias("label"))
    .collect()
)
# Both feature views carry a TTL, so a key missing from either source can be
# answered by Feast from an earlier row while the exact-key join drops it. The
# event set has to be restricted to the keys present in both.
financial_keys = (
    pl.scan_parquet(financial_feast_path)
    .select(pl.col("symbol"), pl.col("timestamp").cast(pl.Date))
    .unique()
    .collect()
)
labels = requested.join(
    model_features.select("symbol", "timestamp"), on=["symbol", "timestamp"]
).join(financial_keys, on=["symbol", "timestamp"])
if labels.is_empty():
    raise ValueError(
        f"No event in {TRAINING_START}..{TRAINING_END} has a row in both feature sources. "
        "Both join paths would return nothing, and the parity check below reads an "
        "empty difference as a match, so it would report every feature matching."
    )
if labels.height < requested.height:
    # Split by source: a key missing from the model features is the embargo gap
    # this clip exists for, while one missing from the financial features is a
    # different problem wearing the same total.
    dropped = requested.join(
        labels.select("symbol", "timestamp"), on=["symbol", "timestamp"], how="anti"
    )
    no_model = dropped.join(
        model_features.select("symbol", "timestamp"), on=["symbol", "timestamp"], how="anti"
    ).height
    no_financial = dropped.join(financial_keys, on=["symbol", "timestamp"], how="anti").height
    # An event can be missing from both, so the two counts overlap. Report the
    # overlap rather than letting the parts add up to more than the whole.
    both = no_model + no_financial - dropped.height
    print(
        f"Dropped {dropped.height:,} of {requested.height:,} events: "
        f"{no_model:,} with no fitted model-feature vintage, "
        f"{no_financial:,} with no financial-feature row, "
        f"{both:,} missing from both."
    )

entity_df = labels.select(
    "symbol",
    pl.col("timestamp").cast(pl.Datetime("ns")).alias("event_timestamp"),
).to_pandas()

print(f"Entity DataFrame: {len(entity_df):,} rows")
print(f"Date range: {entity_df['event_timestamp'].min()} → {entity_df['event_timestamp'].max()}")
print(f"Symbols: {entity_df['symbol'].nunique()}")


# %%
feature_refs = [f"financial_features:{col}" for col in FINANCIAL_FEATURES] + [
    f"model_features:{col}" for col in MODEL_FEATURES
]

feast_training = store.get_historical_features(
    entity_df=entity_df,
    features=feature_refs,
).to_df()

feast_training = feast_training.dropna(subset=ALL_FEATURES)
feast_training = feast_training.sort_values(["event_timestamp", "symbol"]).reset_index(drop=True)

print(f"Feast offline join: {len(feast_training):,} rows × {len(feast_training.columns)} columns")
feast_training.head()


# %% [markdown]
# ## 7. Compare with Polars offline join
#
# The manual Polars join in notebook `05` produces the same result. We verify
# that the Feast retrieval matches: same rows and same feature values. Any
# discrepancy would indicate a point-in-time join misconfiguration.


# %%
def polars_offline_join() -> pl.DataFrame:
    """Reproduce the manual offline join from notebook 05."""
    events = labels.select("symbol", "timestamp")

    financial = (
        pl.scan_parquet(financial_feast_path)
        .with_columns(pl.col("timestamp").cast(pl.Date))
        .select(["symbol", "timestamp", *FINANCIAL_FEATURES])
        .join(events.lazy(), on=["symbol", "timestamp"], how="inner")
    )
    model_based = (
        pl.scan_parquet(model_feast_path)
        .with_columns(pl.col("timestamp").cast(pl.Date))
        .select(["symbol", "timestamp", *MODEL_FEATURES])
        .join(events.lazy(), on=["symbol", "timestamp"], how="inner")
    )
    return (
        financial.join(model_based, on=["symbol", "timestamp"], how="inner")
        .drop_nulls(ALL_FEATURES)
        .collect()
        .sort(["timestamp", "symbol"])
    )


polars_result = polars_offline_join()
print(f"Polars offline join: {polars_result.height:,} rows")
print(f"Feast offline join:  {len(feast_training):,} rows")
print(f"Row count match:     {polars_result.height == len(feast_training)}")


# %%
# Compare the complete one-row-per-event results.
polars_pd = polars_result.to_pandas().rename(columns={"timestamp": "event_timestamp"})
# Align timestamp dtype/tz with Feast (ns + UTC) for the join key.
polars_pd["event_timestamp"] = pd.to_datetime(polars_pd["event_timestamp"], utc=True)
feast_training["event_timestamp"] = pd.to_datetime(feast_training["event_timestamp"], utc=True)

merged = polars_pd.merge(
    feast_training,
    on=["symbol", "event_timestamp"],
    how="inner",
    suffixes=("_polars", "_feast"),
)
assert len(merged) > 0, "Nothing to compare - a parity check over zero rows proves nothing."
assert len(polars_pd) == len(feast_training) == len(merged)

mismatches = []
for col in ALL_FEATURES:
    diff = (merged[f"{col}_polars"] - merged[f"{col}_feast"]).abs()
    max_diff = float(diff.max()) if len(diff) else 0.0
    mismatches.append({"feature": col, "max_abs_diff": max_diff, "match": max_diff < 1e-10})

match_df = pd.DataFrame(mismatches)
match_df

# %%
n_match = int(match_df["match"].sum())
n_total = len(match_df)
mismatch_cols = match_df.loc[~match_df["match"], "feature"].tolist()
print(f"Parity: {n_match}/{n_total} features match within 1e-10.")
if mismatch_cols:
    print(f"Differing columns: {mismatch_cols}")
assert n_match == n_total, f"Feast parity failed for {mismatch_cols}"


# %% [markdown]
# **Finding**: Feast and the manual Polars join reproduce all seven features
# exactly. The source contains one fitted model-feature vintage per walk-forward
# fold, so both paths first select the fold valid at each decision date. The
# explicit selection prevents duplicate-key fan-out and future-fitted state.


# %% [markdown]
# ## 8. Online-style as-of retrieval
#
# In production, `get_online_features` serves the latest feature vector from a
# materialized online store. This notebook does not create or query that store.
# It uses `get_historical_features` with one as-of timestamp per entity to
# simulate the request shape while retaining point-in-time historical semantics.


# %%
def sample_assets(n_assets: int) -> list[str]:
    from data import load_us_equities

    # Rank on prior dollar liquidity, not nominal share volume.
    prices = load_us_equities(start_date="2015-10-01", end_date="2015-12-31")
    return (
        prices.lazy()
        .sort("symbol", "timestamp")
        .with_columns((pl.col("adj_close") * pl.col("adj_volume")).alias("dollar_volume"))
        .with_columns(pl.col("dollar_volume").rolling_mean(21).over("symbol").alias("adv_21d"))
        .filter(
            (pl.col("timestamp") >= pl.date(2015, 12, 1))
            & (pl.col("timestamp") <= pl.date(2015, 12, 31))
        )
        .group_by("symbol")
        .agg(pl.col("adv_21d").mean().alias("avg_adv_21d"))
        .sort("avg_adv_21d", descending=True)
        .head(n_assets)
        .collect()
        .get_column("symbol")
        .to_list()
    )


sampled = sample_assets(N_SAMPLE_ASSETS)
as_of_entity_df = pd.DataFrame(
    {
        "symbol": sampled,
        "event_timestamp": pd.Timestamp(AS_OF_DATE),
    }
)

online_style = store.get_historical_features(
    entity_df=as_of_entity_df,
    features=feature_refs,
).to_df()

print(f"Online-style snapshot for {len(sampled)} assets as of {AS_OF_DATE}:")
online_style.sort_values("symbol")


# %% [markdown]
# ## 9. Feature registry introspection
#
# Feast maintains a registry of all entities, feature views, and their metadata.
# This is the library equivalent of the lineage table built manually in NB05.

# %%
registry_rows = []
for fv in store.list_feature_views():
    registry_rows.append(
        {
            "feature_view": fv.name,
            "entity": ", ".join(str(e) for e in fv.entity_columns),
            "features": len(fv.features),
            "ttl": str(fv.ttl),
            "source_type": type(fv.batch_source).__name__,
        }
    )

registry_df = pd.DataFrame(registry_rows)
print("Feast Feature Registry:")
registry_df


# %% [markdown]
# ## 10. Clean up
#
# Remove the temporary Feast repository. In production, the registry and online
# store would persist across sessions.

# %%
shutil.rmtree(feast_tmp, ignore_errors=True)
print(f"Cleaned up temporary Feast repo: {feast_tmp}")


# %% [markdown]
# ## Key Takeaways
#
# 1. **Feast automates the historical workflow**: entity, FeatureView, and TTL declarations replace bespoke timestamp-rule plumbing, while `get_historical_features` supplies both the training set and the single-as-of simulation shown here. A deployed online store would instead be materialized and queried with `get_online_features`, which this notebook does not execute.
# 2. **Parity is a vintage-tracking diagnostic**: all seven features reproduce exactly after both paths select the fitted fold valid at each decision date.
# 3. **Registry as governance**: `store.list_feature_views()` is the library equivalent of the lineage table built manually in NB05; both make feature sources auditable.
# 4. **Trade-off**: Feast requires infrastructure setup (registry, online store) and datetime-typed timestamps; the manual approach in NB05 works directly with the existing parquet files. Pick the library when you need multi-team isolation, on-demand transforms, or an online store.
#
# **Next**: See `06_mlflow_experiments` for experiment tracking with MLflow.
