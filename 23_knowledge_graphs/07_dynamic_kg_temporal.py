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
# # Temporal Knowledge Graphs with Leakage Controls
#
# **Chapter 23: Knowledge Graphs for Financial AI** | Section 23.6
#
# **Docker image**: `ml4t`
#
# > **Neo4j required**: This notebook queries a Neo4j graph database.
# > Start Neo4j first, then run the notebook:
# > ```bash
# > docker compose --profile kg up -d neo4j
# > docker compose run --rm ml4t python 23_knowledge_graphs/07_dynamic_kg_temporal.py
# > ```
#
#
# This notebook uses the real 8-K event graph loaded by `08_8k_event_extraction.py`
# and demonstrates leakage-safe temporal KG analysis with explicit event,
# public-disclosure, and extraction timestamps.
#
# **Learning Objectives**:
# - Build cutoff-safe temporal snapshots from a knowledge graph with three timestamps
# - Compute relationship churn across consecutive snapshots as a dynamic feature
# - Verify that post-cutoff events are excluded from the visible graph
# - Understand the distinction between event time, disclosure time, and extraction time
#
# **Book Reference**: Chapter 23, Section 23.6 (Temporal Integrity and Leakage-Safe Evaluation)
#
# **Prerequisites**: Run `08_8k_event_extraction.py` first to populate Neo4j with timestamped events.

# %%
"""Temporal Knowledge Graphs with Leakage Controls — leakage-safe temporal KG analysis with event timestamps."""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from utils.paths import get_output_dir
from utils.style import COLORS

# %% tags=["parameters"]
WINDOW_DAYS = 90
CUTOFF_LAG_DAYS = 60

# %%
OUTPUT_DIR = get_output_dir(23, "dynamic_kg_temporal")
print(f"Output directory: {OUTPUT_DIR}")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


# %% [markdown]
# ## 1. Load the Real Temporal Event Graph
#
# Query the event KG directly from Neo4j. Each relationship carries three dates:
# the event date extracted from the filing text, the filing date when the
# information became public, and the extraction date when the pipeline produced
# the KG edge.


# %%
TEMPORAL_EVENT_QUERY = """
MATCH (snapshot:GraphSnapshot)
WITH snapshot ORDER BY snapshot.extraction_time DESC LIMIT 1
MATCH (company:Company)-[r]->(target)
WHERE r.run_id = snapshot.run_id
  AND type(r) IN ['APPOINTED', 'ACQUIRED', 'ANNOUNCED', 'VALUED_AT']
  AND r.event_time IS NOT NULL
  AND r.public_time IS NOT NULL
  AND r.extraction_time IS NOT NULL
RETURN company.name AS subject,
       type(r) AS relation,
       coalesce(target.name, target.description, target.value_text) AS object,
       r.event_time AS event_time,
       r.public_time AS public_time,
       date(r.extraction_time) AS extraction_time,
       snapshot.run_id AS source_run_id,
       snapshot.source_sha256 AS source_sha256,
       snapshot.model AS extractor_model,
       snapshot.model_revision AS extractor_model_revision
ORDER BY public_time, subject, relation, object
"""


# %% [markdown]
# Convert Neo4j temporal values to ISO strings before Polars applies a strict
# date schema.


# %%
def to_iso_date(value) -> str | None:
    """Convert a Neo4j temporal value to its ISO representation."""
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


# %% [markdown]
# Query the latest complete extractor snapshot and preserve its immutable
# source and model identities on every returned edge.


# %%
def load_temporal_events() -> pl.DataFrame:
    """Load real temporal KG edges from Neo4j."""
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError(
            "Neo4j support requires the `neo4j` Python driver. Install project dependencies first."
        ) from exc

    try:
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            driver.verify_connectivity()
            with driver.session() as session:
                rows = [record.data() for record in session.run(TEMPORAL_EVENT_QUERY)]
    except Exception as exc:
        raise RuntimeError(
            f"Neo4j is required at {NEO4J_URI}. Run 08_8k_event_extraction.py first."
        ) from exc

    if not rows:
        raise RuntimeError(
            "No temporal event edges with public/extraction timestamps found. "
            "Re-run 08_8k_event_extraction.py after the timestamp persistence update."
        )

    normalized_rows = [
        {
            **row,
            "event_time": to_iso_date(row["event_time"]),
            "public_time": to_iso_date(row["public_time"]),
            "extraction_time": to_iso_date(row["extraction_time"]),
        }
        for row in rows
    ]

    return pl.DataFrame(normalized_rows).with_columns(
        [
            pl.col("event_time").cast(pl.Date),
            pl.col("public_time").cast(pl.Date),
            pl.col("extraction_time").cast(pl.Date),
        ]
    )


# %%
events = load_temporal_events()
events.head(10)
print(f"Temporal events loaded: {len(events)}")
print(f"Public date range: {events['public_time'].min()} to {events['public_time'].max()}")


# %% [markdown]
# ## 2. Lag Diagnostics
#
# The event graph now exposes the disclosure and extraction delays directly.


# %%
events = events.with_columns(
    [
        (pl.col("public_time") - pl.col("event_time"))
        .dt.total_days()
        .alias("raw_disclosure_lag_days"),
        (pl.col("extraction_time") - pl.col("public_time"))
        .dt.total_days()
        .alias("extraction_lag_days"),
        (pl.col("subject") + "|" + pl.col("relation") + "|" + pl.col("object")).alias("edge_key"),
    ]
).with_columns(
    [
        pl.max_horizontal("raw_disclosure_lag_days", pl.lit(0)).alias("disclosure_lag_days"),
        (pl.col("raw_disclosure_lag_days") < 0).alias("post_disclosure_effective_date"),
    ]
)

lag_summary = events.select(
    [
        pl.len().alias("events"),
        pl.col("subject").n_unique().alias("companies"),
        pl.col("relation").n_unique().alias("relation_types"),
        pl.col("post_disclosure_effective_date").sum().alias("future_effective_dates"),
        pl.col("disclosure_lag_days").mean().alias("avg_disclosure_lag_days"),
        pl.col("disclosure_lag_days").max().alias("max_disclosure_lag_days"),
        pl.col("extraction_lag_days").mean().alias("avg_extraction_lag_days"),
    ]
)
print("Lag summary:")
print(lag_summary)

# %% [markdown]
# ### Temporal Event Distribution
#
# Visualize when events occurred vs when they became publicly known.

# %%
if len(events) > 0 and "public_time" in events.columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")

    # Panel (a): event timeline by relation type
    for rtype in events["relation"].unique().to_list():
        subset = events.filter(pl.col("relation") == rtype)
        dates_plot = subset["public_time"].to_list()
        axes[0].scatter(dates_plot, [rtype] * len(dates_plot), alpha=0.7, s=50)
    axes[0].set_xlabel("Public Disclosure Date")
    axes[0].set_title("(a) Events by Relation Type")
    axes[0].tick_params(axis="x", rotation=30)

    # Panel (b): disclosure-lag distribution (event date -> public date).
    lags = events["raw_disclosure_lag_days"].drop_nulls().to_numpy()
    if len(lags) > 0:
        axes[1].hist(lags, bins=max(5, len(lags) // 30), color=COLORS["amber"], alpha=0.7)
        axes[1].axvline(
            float(lags.mean()),
            color=COLORS["negative"],
            linestyle="--",
            label=f"Mean: {lags.mean():.0f}d",
        )
        axes[1].legend()
        figure_title = f"Mean Disclosure Lag Is {lags.mean():.0f} Days in This Fixed Cohort"
    else:
        axes[1].text(0.5, 0.5, "No disclosure lags", transform=axes[1].transAxes, ha="center")
        figure_title = "The Fixed Cohort Contains No Measurable Disclosure Lags"
    axes[1].set_xlabel("Disclosure Lag (days)")
    axes[1].set_title("(b) Disclosure Lag Distribution")

    fig.suptitle(figure_title, fontsize=13)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
        fig.show()

# %% [markdown]
# **Finding**: Panel (a) scatters events by relation type over public-disclosure
# time; panel (b) shows the disclosure-lag distribution (event date to public
# date). The lag summary reports the observed center and tail for this extraction
# run. Section 5 then measures edge additions and removals across fixed windows
# without assuming that the run represents the population of 8-K events.

# %% [markdown]
# ## 3. Cutoff-Date Filtering
#
# Build a point-in-time slice using public disclosure time rather than event time.


# %%
def visible_at_cutoff(df: pl.DataFrame, cutoff_date: date) -> pl.DataFrame:
    """Return only edges that were public by the cutoff date."""
    return df.filter(pl.col("public_time") <= cutoff_date)


latest_public = events["public_time"].max()
cutoff = latest_public - timedelta(days=CUTOFF_LAG_DAYS)
visible = visible_at_cutoff(events, cutoff)
hidden = events.filter(pl.col("public_time") > cutoff)

print(f"Cutoff date: {cutoff}")
print(f"Visible at cutoff: {len(visible)}")
print(f"Hidden after cutoff: {len(hidden)}")


# %% [markdown]
# ## 4. Temporal Snapshots
#
# Roll the public event stream into fixed windows so each snapshot respects the
# same leakage guard.


# %%
def build_snapshots(df: pl.DataFrame, window_days: int) -> pl.DataFrame:
    """Aggregate temporal KG activity into fixed-width windows."""
    start = df["public_time"].min()
    end = df["public_time"].max()
    if start is None or end is None:
        return pl.DataFrame()

    rows = []
    window_start = start
    window = 0
    while window_start <= end:
        window_end = window_start + timedelta(days=window_days)
        window_df = df.filter(
            (pl.col("public_time") >= window_start) & (pl.col("public_time") < window_end)
        )
        rows.append(
            {
                "window": window,
                "start": window_start,
                "end": window_end,
                "n_edges": len(window_df),
                "n_subjects": window_df["subject"].n_unique() if len(window_df) else 0,
                "n_objects": window_df["object"].n_unique() if len(window_df) else 0,
                "n_relations": window_df["relation"].n_unique() if len(window_df) else 0,
            }
        )
        window += 1
        window_start = window_end

    return pl.DataFrame(rows)


# %%
snapshots = build_snapshots(visible, WINDOW_DAYS)
print("Snapshot summary:")
print(snapshots)


# %% [markdown]
# ## 5. Relationship Churn
#
# Compare consecutive public snapshots to measure how quickly the graph changes.


# %%
def relationship_churn(df: pl.DataFrame, window_days: int) -> pl.DataFrame:
    """Compute edge additions and removals across consecutive windows."""
    snapshots = build_snapshots(df, window_days)
    if snapshots.is_empty():
        return pl.DataFrame()

    edge_sets = {}
    for row in snapshots.iter_rows(named=True):
        current = df.filter(
            (pl.col("public_time") >= row["start"]) & (pl.col("public_time") < row["end"])
        )
        edge_sets[row["window"]] = set(current["edge_key"].to_list())

    rows = []
    previous_edges: set[str] = set()
    for row in snapshots.iter_rows(named=True):
        current_edges = edge_sets[row["window"]]
        added = current_edges - previous_edges
        dropped = previous_edges - current_edges
        union = current_edges | previous_edges
        rows.append(
            {
                "window": row["window"],
                "start": row["start"],
                "n_edges": len(current_edges),
                "added_edges": len(added),
                "dropped_edges": len(dropped),
                "relationship_churn": (len(added) + len(dropped)) / len(union) if union else 0.0,
            }
        )
        previous_edges = current_edges

    return pl.DataFrame(rows)


# %%
churn = relationship_churn(visible, WINDOW_DAYS)
print("Relationship churn:")
print(churn)


# %% [markdown]
# ## 6. Leakage Test
#
# Validate that the visible snapshot never contains post-cutoff disclosures.


# %%
def leakage_test(df: pl.DataFrame, cutoff_date: date) -> bool:
    """Confirm the slice contains no events disclosed after the cutoff."""
    return len(df.filter(pl.col("public_time") > cutoff_date)) == 0


passed = leakage_test(visible, cutoff)
print(f"Leakage test passed: {passed}")


# %% [markdown]
# ## 7. Persist Outputs
#
# Alongside the parquet artifacts, write a `snapshot_manifest.json` recording
# the cutoff, window, upstream extractor identity, and per-parquet SHA-256.
# The manifest is the audit anchor for the §23.6 protocol bullet
# *Log snapshot hash and extractor version*: it lets a reader reproduce the
# exact visible-graph state and verify that downstream features were computed
# from this snapshot.


# %%
def write_snapshot_manifest(
    out_dir: Path,
    cutoff_date: date,
    window_days: int,
    cutoff_lag_days: int,
    source_graph: dict[str, str],
    parquet_frames: dict[str, pl.DataFrame],
) -> Path:
    """Write snapshot_manifest.json with per-parquet row counts and SHA-256."""
    artifacts = {}
    for filename, df in parquet_frames.items():
        path = out_dir / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifacts[filename] = {"rows": df.height, "sha256": digest}
    manifest = {
        "manifest_version": "1.0",
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cutoff_date": cutoff_date.isoformat(),
        "window_days": window_days,
        "cutoff_lag_days": cutoff_lag_days,
        "source_graph": source_graph,
        "artifacts": artifacts,
    }
    manifest_path = out_dir / "snapshot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


# %%
events.write_parquet(OUTPUT_DIR / "temporal_events.parquet")
visible.write_parquet(OUTPUT_DIR / "visible_at_cutoff.parquet")
snapshots.write_parquet(OUTPUT_DIR / "temporal_snapshots.parquet")
churn.write_parquet(OUTPUT_DIR / "relationship_churn.parquet")

source_run_ids = events["source_run_id"].unique().to_list()
if len(source_run_ids) != 1:
    raise RuntimeError(f"Expected one extractor run, found {source_run_ids}")

manifest_path = write_snapshot_manifest(
    out_dir=OUTPUT_DIR,
    cutoff_date=cutoff,
    window_days=WINDOW_DAYS,
    cutoff_lag_days=CUTOFF_LAG_DAYS,
    source_graph={
        "neo4j_uri": NEO4J_URI,
        "upstream_extractor": "08_8k_event_extraction.py",
        "upstream_run_id": source_run_ids[0],
        "source_sha256": events["source_sha256"].unique().item(),
        "extractor_model": events["extractor_model"].unique().item(),
        "extractor_model_revision": events["extractor_model_revision"].unique().item(),
    },
    parquet_frames={
        "temporal_events.parquet": events,
        "visible_at_cutoff.parquet": visible,
        "temporal_snapshots.parquet": snapshots,
        "relationship_churn.parquet": churn,
    },
)

print(f"Saved: {OUTPUT_DIR / 'temporal_events.parquet'}")
print(f"Saved: {OUTPUT_DIR / 'visible_at_cutoff.parquet'}")
print(f"Saved: {OUTPUT_DIR / 'temporal_snapshots.parquet'}")
print(f"Saved: {OUTPUT_DIR / 'relationship_churn.parquet'}")
print(f"Saved: {manifest_path}")


# %% [markdown]
# ## 8. Verification


# %%
print("\n" + "=" * 70)
print("NOTEBOOK EXECUTION COMPLETE")
print("=" * 70)
print(f"Temporal events: {len(events)}")
print(f"Visible at cutoff: {len(visible)}")
print(f"Hidden after cutoff: {len(hidden)}")
print(f"Cutoff date: {cutoff}")
print(f"Future effective dates: {lag_summary['future_effective_dates'][0]}")
print(f"Average disclosure lag (days): {lag_summary['avg_disclosure_lag_days'][0]:.1f}")
print(f"Average extraction lag (days): {lag_summary['avg_extraction_lag_days'][0]:.1f}")
print(f"Leakage test passed: {passed}")


# %% [markdown]
# ## Key Takeaways
#
# 1. The three-timestamp model (event, disclosure, extraction) is essential for
#    leakage-safe temporal KG analysis in finance.
# 2. Cutoff filtering on **disclosure time** — not event time — prevents
#    lookahead bias when constructing historical feature snapshots.
# 3. Relationship churn across consecutive windows captures network dynamics
#    that static snapshots miss, such as restructuring periods or emerging
#    competitive relationships.
# 4. The leakage test verifies that no post-cutoff information enters the
#    visible graph, providing an automated guard for backtesting pipelines.
#
# **Next**: See `09_knowledge_graph_features.py` for converting graph structure
# into ML-ready features, and Chapter 23.6 for the full temporal integrity
# framework.
