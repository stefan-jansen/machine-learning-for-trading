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
# # Feature Store with Feast: Live Integration
#
# **Chapter 26: MLOps and Governance**
# **Docker image**: `ml4t`
# **Book Reference**: Chapter 26, Section 26.6
# **Companion to**: [`05_feast_feature_store`](05_feast_feature_store.ipynb), which does the
# same work in Polars
#
# **Learning Objectives**:
# - Stand up a local Feast repository from Parquet files already on disk, declaring the
#   entity, the sources and the feature views in code.
# - Ask Feast for a training set by handing it the entity-timestamp pairs it should answer
#   for, and ask it for a single moment the way a live system would.
# - Check Feast's answers against the hand-written join from `05`, feature by feature, and
#   read a disagreement as a configuration defect rather than a rounding difference.
#
# `05` builds the offline join, the as-of retrieval and the skew measurement by hand, which
# is what makes the rules visible. This notebook hands the same job to **Feast**, an
# open-source feature store, and then checks that the two agree.
#
# The check is the point. A feature store is configured, not written: the timestamp field,
# the entity key and the time to live are declarations, and a wrong one produces a store that
# runs, answers every query, and answers them wrong. Comparing against an implementation whose
# rules you can read is how you find that out.
#
# **Prerequisites**: `pip install 'feast>=0.40'` (included in the `[mlops]` extra).

# %%
"""Feature Store with Feast: the same retrievals as notebook 05, checked against it."""

# %% [markdown]
# ## Settings
#
# `TRAINING_START`, `TRAINING_END` and `AS_OF_DATE` bound the two retrievals. Left unset they
# are derived from the case study's fold geometry, as in `05` and for the same reason: a date
# typed here is a claim about where a fold boundary falls, and rebuilding the case study moves
# every boundary while the literal does not.
#
# `FEATURE_TTL_DAYS` is the time to live declared on both Feast views, and it is worth reading
# carefully. Two days on a daily feature table means a value from the previous session is
# still servable, so Feast answers a query for a date with no row of its own by carrying the
# previous row forward. That behaviour is what a store is for and it is also what makes the
# parity check below non-trivial: the hand-written join matches on exact keys and returns
# nothing for such a date, so the two paths only agree where the event set is restricted to
# keys both sources actually hold.
#
# `SOURCE_PAD_DAYS` widens the Parquet copies handed to Feast on both sides of the training
# window, so the time-to-live lookback and the as-of query have rows to reach.
#
# `PARITY_TOLERANCE` is how far apart the two implementations may be per feature. It is set
# well below anything a correct join could produce, because the difference being looked for is
# a wrong row, not a rounding error: a store that picked a different date is off by whatever
# the feature moved that day, which is many orders of magnitude larger than this.
#
# `N_SAMPLE_ASSETS`, `LIQUIDITY_WINDOW_DAYS` and `LIQUIDITY_RANK_DAYS` pick the names the as-of
# retrieval asks about, by average dollar volume over the tail of the training window.

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"
PRIMARY_LABEL = "fwd_ret_1d"
TRAINING_START = None
TRAINING_END = None
AS_OF_DATE = None
TRAINING_LOOKBACK_DAYS = 91
N_SAMPLE_ASSETS = 8
FEATURE_TTL_DAYS = 2
SOURCE_PAD_DAYS = 30
LIQUIDITY_WINDOW_DAYS = 21
LIQUIDITY_RANK_DAYS = 30
PARITY_TOLERANCE = 1e-10

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

# Named, not blanket: a bare ignore would also hide the convergence and numerical
# warnings a reader needs to see.
warnings.filterwarnings("ignore", category=FutureWarning, module="feast")

CASE_DIR = get_case_study_dir(CASE_STUDY_ID)

FINANCIAL_FEATURES = ["past_ret_21d", "vol_21d", "rsi_14", "sharpe_21d"]
MODEL_FEATURES = ["garch_cond_vol", "ffd_log_price", "ffd_log_volume"]
ALL_FEATURES = FINANCIAL_FEATURES + MODEL_FEATURES

print("Feature Store with Feast: Live Integration")
print("=" * 60)


# %% [markdown]
# ## 1. Import Feast
#
# Four classes carry the declaration. `Entity` names the thing rows are about and the column
# that identifies it. `FileSource` points at a table and says which of its columns is the
# event timestamp. `FeatureView` binds a source to a typed list of features and a time to
# live. `FeatureStore` is the repository those declarations are registered into and the object
# every query goes through.

# %%
from feast import Entity, FeatureStore, FeatureView, Field, FileSource
from feast.data_format import ParquetFormat
from feast.types import Float64
from feast.value_type import ValueType

print("Feast imported successfully")


# %% [markdown]
# ## 2. Prepare Feast-compatible source files
#
# Feast's offline store needs a datetime timestamp for its point-in-time joins, and these
# Parquet files carry a `Date`. Temporary copies are written with the column cast, and only
# over the padded window, so nothing here writes into the case study's own artifacts. In
# production the cast happens in the job that materializes the source, once, rather than
# beside every query.

# %%
feast_tmp = tempfile.mkdtemp(prefix="feast_ml4t_")
feast_data_dir = Path(feast_tmp) / "data"
feast_data_dir.mkdir()


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


financial_src = CASE_DIR / "features" / "financial.parquet"
model_src = CASE_DIR / "features" / "model_based.parquet"

financial_feast_path = feast_data_dir / "financial.parquet"
model_feast_path = feast_data_dir / "model_based.parquet"

setup = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())
holdout_start = pd.Timestamp(setup["evaluation"]["holdout_start"])
holdout_end = pd.Timestamp(setup["evaluation"]["holdout_end"])
timeline = pl.scan_parquet(financial_src).select("timestamp").unique().collect()
cv_splits = generate_cv_splits(timeline, case_study_id=CASE_STUDY_ID, label_buffer="1D")
# A feature value is servable for a session the model was evaluated on, so the date range
# handed to Feast is the union of the validation windows and the holdout.
validation_spans = [
    (pd.Timestamp(split["val_start"]).date(), pd.Timestamp(split["val_end"]).date())
    for split in cv_splits
]
model_windows = [*validation_spans, (holdout_start.date(), holdout_end.date())]
# Where the table carries a fold column each stock-date appears once per fold, so the
# window and the fold are applied together; where it does not, the date range selects it.
FOLD_KEYED_ARTIFACT = "fold" in pl.scan_parquet(model_src).collect_schema().names()
model_folds = (
    folds_by_coverage(model_src, model_windows)
    if FOLD_KEYED_ARTIFACT
    else [None] * len(model_windows)
)
# By date rather than by list position: which end of the list holds the latest window
# depends on the numbering convention, and the dates do not.
last_validation_span = max(validation_spans)

training_end = TRAINING_END or str(last_validation_span[1])
training_start = TRAINING_START or str(
    max(
        last_validation_span[0],
        pd.Timestamp(training_end).date() - pd.Timedelta(days=TRAINING_LOOKBACK_DAYS),
    )
)
as_of_date = AS_OF_DATE or str(
    pl.scan_parquet(financial_src)
    .filter(pl.col("timestamp") >= holdout_start.date())
    .select(pl.min("timestamp"))
    .collect()
    .item()
)
assert pd.Timestamp(training_end) < holdout_start, (
    f"training window ends {training_end}, on or after the holdout opens "
    f"{holdout_start.date()}; the offline join would mix data the model may not see"
)
print(f"Offline training window {training_start} to {training_end}; serving as of {as_of_date}")

train_start = pd.Timestamp(training_start).date()
train_end = pd.Timestamp(training_end).date()
# Padded on both sides so the lookback and the as-of query have rows to reach.
filter_start = train_start - timedelta(days=SOURCE_PAD_DAYS)
filter_end = train_end + timedelta(days=SOURCE_PAD_DAYS)

(
    pl.scan_parquet(financial_src)
    .select(["symbol", "timestamp", *FINANCIAL_FEATURES])
    .filter((pl.col("timestamp") >= filter_start) & (pl.col("timestamp") <= filter_end))
    .with_columns(pl.col("timestamp").cast(pl.Datetime("ns")))
    .collect()
    .write_parquet(financial_feast_path)
)


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


# %% [markdown]
# The model-based copy is built with one filter over the union of the evaluation windows,
# rather than one frame per window concatenated: a concat would double-count any date two
# windows cover, and would do it silently.

# %%
_span_lo = pd.Timestamp(filter_start).date()
_span_hi = pd.Timestamp(filter_end).date()
model_spans = [
    (fold, max(_span_lo, start), min(_span_hi, end))
    for fold, (start, end) in zip(model_folds, model_windows, strict=True)
    if start <= _span_hi and end >= _span_lo
]
if not model_spans:
    raise ValueError(
        f"No validation window and not the holdout covers {_span_lo}..{_span_hi}; "
        "the materialization range lies outside every window this model was evaluated on"
    )
model_features = (
    pl.scan_parquet(model_src)
    .filter(pl.any_horizontal(*window_terms(model_spans)))
    .select(["symbol", "timestamp", *MODEL_FEATURES])
    .collect()
    .sort(["timestamp", "symbol"])
)
assert not model_features.select(pl.struct("symbol", "timestamp").is_duplicated().any()).item(), (
    "a stock-date resolved to more than one model-based row; the evaluation windows "
    "this notebook derived are not disjoint over the materialization range"
)
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
# The entity is `symbol`, which is the identifier every case-study artifact in this repository
# uses. Each view maps to one Parquet source and declares its features with types, because
# Feast validates the schema at registration rather than at query time.
#
# The time to live is what bounds how far back a query may reach. At `FEATURE_TTL_DAYS` a
# request for a date with no row of its own is answered from the most recent row within that
# many days, and a request older than that returns nothing rather than a stale value. Set it
# too short and legitimate queries come back empty; too long and the store answers with data
# that no longer describes the moment asked about.

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
    ttl=timedelta(days=FEATURE_TTL_DAYS),
)

model_fv = FeatureView(
    name="model_features",
    entities=[symbol_entity],
    schema=[Field(name=col, dtype=Float64) for col in MODEL_FEATURES],
    source=model_source,
    ttl=timedelta(days=FEATURE_TTL_DAYS),
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
# ## 6. Offline retrieval: the point-in-time join
#
# The query Feast is built around. It is handed a frame of `(symbol, event_timestamp)` pairs -
# the moments a training set needs answered - and returns, for each pair, the most recent
# feature values at or before that moment, within the time to live. The features here are
# known after the session close, the label is the next session's return, and a position acts
# no earlier than that next session, so the pairing is a next-bar decision.
#
# ### Restricting the event set, and why the parity check needs it
#
# The two implementations answer differently for a date that has no feature row of its own.
# Feast carries the previous row forward, within its time to live; the exact-key join in `05`
# returns nothing. Both are right for what they are, so comparing them on such a date would
# report a disagreement that is a property of the question rather than of either
# implementation.
#
# Dates like that exist wherever the training window spans an embargo gap between consecutive
# validation windows, or runs past the last one. Whether this particular window contains any
# is a fact about the fold geometry, and the counts printed below say so: the events dropped
# are split by which source was missing, because a key absent from the model features is the
# gap this restriction exists for, while one absent from the financial features is a
# different problem arriving in the same total.

# %%
requested = (
    pl.scan_parquet(CASE_DIR / "labels" / f"{PRIMARY_LABEL}.parquet")
    .filter(
        (pl.col("timestamp") >= pl.lit(pd.Timestamp(training_start).date()))
        & (pl.col("timestamp") <= pl.lit(pd.Timestamp(training_end).date()))
    )
    .select("symbol", "timestamp", pl.col(PRIMARY_LABEL).alias("label"))
    .collect()
)
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
        f"No event in {training_start}..{training_end} has a row in both feature sources. "
        "Both join paths would return nothing, and the parity check below reads an "
        "empty difference as a match, so it would report every feature matching."
    )
if labels.height < requested.height:
    dropped = requested.join(
        labels.select("symbol", "timestamp"), on=["symbol", "timestamp"], how="anti"
    )
    no_model = dropped.join(
        model_features.select("symbol", "timestamp"), on=["symbol", "timestamp"], how="anti"
    ).height
    no_financial = dropped.join(financial_keys, on=["symbol", "timestamp"], how="anti").height
    # The two counts overlap where an event is missing from both sources.
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
# ## 7. Check Feast against the hand-written join
#
# The same event set, answered twice: once by Feast's point-in-time join and once by the exact
# key join `05` builds. Both should return the same rows with the same values.
#
# A disagreement here is a configuration defect, not a numerical one. The two paths read the
# same Parquet files, so a difference means one of them selected a different row: a wrong
# timestamp field, a time to live reaching further than intended, or a fold vintage picked by
# id rather than by date. Each of those produces a store that runs and answers wrongly, which
# is why the comparison is per feature and the tolerance is set far below anything a correct
# join could produce.


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
    mismatches.append(
        {"feature": col, "max_abs_diff": max_diff, "match": max_diff < PARITY_TOLERANCE}
    )

match_df = pd.DataFrame(mismatches)
match_df

# %%
n_match = int(match_df["match"].sum())
n_total = len(match_df)
mismatch_cols = match_df.loc[~match_df["match"], "feature"].tolist()
print(f"Parity: {n_match}/{n_total} features match within {PARITY_TOLERANCE:g}.")
if mismatch_cols:
    print(f"Differing columns: {mismatch_cols}")
assert n_match == n_total, f"Feast parity failed for {mismatch_cols}"


# %% [markdown]
# The assertion above is what makes this a check rather than a display: the notebook stops if
# any feature disagrees, so a reader who sees the cells below knows the two paths agreed.
#
# What the agreement establishes is narrower than it looks. The source holds one fitted
# model-feature vintage per walk-forward fold, and both paths select the vintage valid at each
# decision date before comparing. That selection is the part that had to be got right in both;
# without it, one stock-date resolves to several rows and the join fans out, or resolves to a
# vintage fitted on later data. Agreement says the two implementations made the same
# selection. It does not say the selection is the correct one - that is what the fold-by-date
# pairing above is for.


# %% [markdown]
# ## 8. Online-style as-of retrieval
#
# A live system calls `get_online_features`, which reads a key-value store that a
# materialization job keeps current. Standing that up needs infrastructure this notebook does
# not have, so the request shape is reproduced with `get_historical_features` and one
# timestamp per entity: the same query, answered from the offline store.
#
# The shape is what matters here - a handful of names, one moment, one row each - because it
# is the shape a model server issues per decision. What it does not reproduce is the latency
# or the consistency question, which is whether the online store holds what the offline store
# would have answered.


# %%
def sample_assets(n_assets: int) -> list[str]:
    from data import load_us_equities

    rank_from = pd.Timestamp(training_end).date() - pd.Timedelta(days=LIQUIDITY_RANK_DAYS)
    prices = load_us_equities(start_date=training_start, end_date=training_end)
    return (
        prices.lazy()
        .sort("symbol", "timestamp")
        .with_columns((pl.col("adj_close") * pl.col("adj_volume")).alias("dollar_volume"))
        .with_columns(
            pl.col("dollar_volume")
            .rolling_mean(LIQUIDITY_WINDOW_DAYS)
            .over("symbol")
            .alias("avg_dollar_volume")
        )
        .filter(
            (pl.col("timestamp") >= rank_from)
            & (pl.col("timestamp") <= pd.Timestamp(training_end).date())
        )
        .group_by("symbol")
        .agg(pl.col("avg_dollar_volume").mean().alias("mean_dollar_volume"))
        .sort("mean_dollar_volume", descending=True)
        .head(n_assets)
        .collect()
        .get_column("symbol")
        .to_list()
    )


sampled = sample_assets(N_SAMPLE_ASSETS)
as_of_entity_df = pd.DataFrame(
    {
        "symbol": sampled,
        "event_timestamp": pd.Timestamp(as_of_date),
    }
)

online_style = store.get_historical_features(
    entity_df=as_of_entity_df,
    features=feature_refs,
).to_df()

print(f"Online-style snapshot for {len(sampled)} assets as of {as_of_date}:")
online_style.sort_values("symbol")


# %% [markdown]
# ## 9. Feature registry introspection
#
# The registry is Feast's own record of what has been declared, and reading it back is how an
# operator answers "where does this served value come from" without opening the code that
# declared it. It is the library's version of the lineage table `05` assembles by hand.

# %%
registry_rows = []
for fv in store.list_feature_views():
    registry_rows.append(
        {
            "feature_view": fv.name,
            "entity": ", ".join(field.name for field in fv.entity_columns),
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
# The repository was written under a temporary directory, so removing it leaves nothing
# behind. A production registry and online store persist across sessions and across the
# services that read them, which is most of what makes them shared infrastructure.

# %%
shutil.rmtree(feast_tmp, ignore_errors=True)
print(f"Cleaned up temporary Feast repo: {feast_tmp}")


# %% [markdown]
# ## Key Takeaways
#
# 1. A feature store is configured rather than written. The entity, the timestamp field and
#    the time to live are declarations, and a wrong one produces a store that runs, answers
#    every query and answers them wrongly. That is why the check below the retrieval exists.
# 2. Check a store against an implementation whose rules you can read, feature by feature, and
#    set the tolerance far below anything a correct join could produce. The failure being
#    looked for is a wrong row, which is off by whatever the feature moved that day, not a
#    rounding difference.
# 3. Compare the two on questions they answer the same way. A store carries the last value
#    forward within its time to live and an exact-key join does not, so a date with no row of
#    its own makes them disagree by construction; restrict the event set first, and print what
#    was dropped and why.
# 4. Reading agreement correctly is as important as getting it. Agreement says both paths
#    selected the same fitted vintage for each decision date. It does not say the selection
#    was the right one, which is a separate argument the fold-by-date pairing makes.
# 5. The library buys shared infrastructure, not correctness. Feast supplies a registry, an
#    online store and multi-team isolation, and needs them stood up and their timestamps typed
#    for it; `05` needs none of that and does the same retrievals on the Parquet files
#    directly. Choose on which of those you need.
#
# **Known limitations**
#
# - The online retrieval here is `get_historical_features` with one timestamp per entity. It
#   reproduces the request shape and not the online store, so nothing here exercises the
#   materialization job or the consistency between the two tiers, which is where a live
#   feature store's own failures are.
# - The registry, the online store and the source copies all live in a temporary directory
#   that this notebook deletes. A production registry persists and is shared, and most of what
#   makes it governance is that other services read the same one.
# - The parity check covers one training window on one case study. It establishes that the two
#   implementations agree on those events, not that the Feast declaration is correct for a
#   window with a different fold geometry.
#
# **Next**: See `06_mlflow_experiments` for experiment tracking with MLflow.
