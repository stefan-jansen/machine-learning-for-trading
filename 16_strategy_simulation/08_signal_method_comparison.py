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
# # Turning prediction scores into trading signals
#
# **Docker image**: `ml4t`
#
# ## Purpose
# A model trained to predict returns emits a number. A portfolio needs a decision. The step in
# between - the rule that turns a score into "hold this or do not" - is usually written in one line
# and rarely examined, and it decides how often the strategy trades before any cost model is
# involved.
#
# Three rules are compared here on two real prediction sets, one from daily ETF predictions and one
# from eight-hourly crypto perpetual predictions. They differ in what the score is compared to: a
# constant, the symbol's own recent scores, or every symbol's score that day. Two things are
# measured for each: how often the rule is active, and how often a symbol changes state.
#
# ## Learning objectives
#
# - Convert one set of model scores into positions three different ways, and measure how far apart
#   the resulting signal streams are before any return is computed.
# - Explain why a constant cutoff on an uncalibrated regression score is a rule about the model's
#   arbitrary scale rather than about the market.
# - Choose a lookback and a cutoff for a trailing percentile rule, and say what each one controls.
# - Distinguish how often a position changes from how much is traded, and say why the first cannot
#   stand in for the second.
#
# ## Book reference
# Chapter 16, Section 16.2 (specifying a trading protocol).
#
# ## Prerequisites
#
# - A registered gradient-boosting validation prediction set for the ETF and crypto case studies.
#   Nothing here fits a model; the scores are read from the registry.
#
# The scores are signed return predictions, not probabilities, so a cutoff of zero has a plain
# reading: hold when the model predicts a positive return. That is what makes the fixed rule worth
# including even though it does badly - its failure is legible.

# %% [markdown]
# ## Setup

# %%
"""Compare fixed, rolling-percentile, and cross-sectional signal conversion."""

import sqlite3
import warnings
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import yaml
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# %%
from case_studies.utils.signals import (
    cross_sectional_percentile_signal,
    fixed_threshold_signal,
    rolling_percentile_signal,
)
from utils.paths import get_case_study_dir, get_output_dir
from utils.style import COLORS

CASE_STUDIES = {
    "crypto_perps_funding": "Crypto perpetuals",
    "etfs": "ETFs",
}
CASE_STUDY_COLORS = {
    "crypto_perps_funding": COLORS["blue"],
    "etfs": COLORS["amber"],
}

# %% [markdown]
# ### What each setting decides
#
# **Lookbacks.** A trailing percentile rule needs a window of past scores to take a percentile of.
# A short window tracks a drifting score distribution closely and changes its mind often; a long
# one is stable and slow to notice that the distribution has moved. The grid spans roughly a month
# to a quarter of trading observations so that both ends of that trade-off are visible.
#
# **Percentiles.** How selective the relative rules are. The 75th percentile admits the top quarter
# of recent scores, the 95th only the top twentieth. This is the setting that most directly
# controls how often the strategy is in the market.
#
# **Operating percentile.** One cutoff at which all three methods are compared side by side. It has
# to be a member of the grid above, which the run asserts rather than trusts.
#
# The lookbacks are counted in observations, not in days. For the daily ETF predictions the two
# coincide; for the eight-hourly crypto predictions three observations make a day.

# %% tags=["parameters"]
ROLLING_WINDOWS = [21, 42, 63]
PERCENTILES = [75, 80, 85, 90, 95]
OPERATING_PERCENTILE = 90

# %%
OPERATING_WINDOW = max(ROLLING_WINDOWS)
assert OPERATING_PERCENTILE in PERCENTILES, "the operating percentile must be one of the grid"

print(f"Trailing lookbacks compared: {ROLLING_WINDOWS} observations")
print(f"Percentile cutoffs compared: {PERCENTILES}")
print(f"Side-by-side operating point: {OPERATING_WINDOW} observations, p{OPERATING_PERCENTILE}")
print(f"Percentile sensitivity read at the shortest lookback: {min(ROLLING_WINDOWS)}")


# %% [markdown]
# Which label each case study predicts is the case study's decision, recorded in its own
# `setup.yaml`, so this notebook reads it rather than repeating it. A label pinned here would go
# stale the first time a case study renamed its target or dropped a variant, and the notebook would
# then fail with a missing-predictions error that says nothing about the cause.


# %%
def primary_label(case_study: str) -> str:
    """Read the case study's declared primary label from its own configuration."""
    setup_path = get_case_study_dir(case_study, create=False) / "config" / "setup.yaml"
    return str(yaml.safe_load(setup_path.read_text())["labels"]["primary"])


LABELS = {case_study: primary_label(case_study) for case_study in CASE_STUDIES}


# %% [markdown]
# ### Pick one registered prediction set per case study
#
# Each case study contributes one gradient-boosting prediction set from its validation window. The
# question is which one, and the answer has to be a rule rather than a judgement, because a
# notebook that quietly picks a favourable configuration is measuring the favour, not the method.
#
# The rule here is the widest date coverage, with the prediction hash breaking ties. That is
# deliberately not a rule about how good the predictions are. Ranking candidates by information
# coefficient would select on the same statistic the comparison is meant to be indifferent to, and
# a signal-conversion rule should be judged on what it does to *a* prediction stream, not on
# whether it was handed the strongest one. What the comparison needs from the artifact is length,
# because a rolling percentile cannot be computed on a short one.

# %%
SELECTOR_SQL = """
    SELECT t.config_name, p.prediction_hash, p.checkpoint_value, pm.ic_n_days
    FROM training_runs AS t
    JOIN prediction_sets AS p USING (training_hash)
    JOIN prediction_metrics AS pm USING (prediction_hash)
    WHERE t.family = ? AND t.label = ? AND p.split = 'validation'
      AND pm.ic_n_days IS NOT NULL
    ORDER BY pm.ic_n_days DESC, p.prediction_hash
    LIMIT 1
"""


# %%
def select_registered_prediction(db_path: Path, family: str, label: str) -> dict[str, object]:
    """Select the widest-coverage validation prediction set, ties broken by hash."""
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        row = connection.execute(SELECTOR_SQL, (family, label)).fetchone()
    if row is None:
        raise RuntimeError(f"No registered {family} validation predictions for {label}")

    keys = ["config_name", "prediction_hash", "checkpoint_value", "registered_n_days"]
    return dict(zip(keys, row, strict=True))


# %% [markdown]
# Before any signal rule runs, the artifact has to carry what the rules need. Three of the checks
# below are about integrity - the columns exist, no timestamp-symbol pair appears twice, nothing
# required is null - and two are preconditions of the analysis itself. A trailing percentile over
# the longest lookback needs at least that many observations of each symbol, and a cross-sectional
# rank needs at least two symbols quoted on a date to have anything to rank. Both would otherwise
# produce empty or constant signals and a comparison that measures nothing.
#
# The artifact's own daily rank IC is computed here and reported next to the coverage the registry
# recorded for it. Those two are written by the same production run and should agree; where they do
# not, the artifact on disk is not the one the registry describes, and the printed pair is what
# makes that visible.


# %%
def validate_prediction_set(
    pred_path: Path, metadata: dict[str, object], min_observations: int
) -> pl.DataFrame:
    """Check integrity and the analysis's preconditions, and measure the artifact's rank IC."""
    raw_predictions = pl.read_parquet(pred_path)
    required = {"timestamp", "symbol", "fold", "prediction", "actual"}
    if not required.issubset(raw_predictions.columns):
        raise RuntimeError(f"Prediction schema is incomplete in {pred_path}")
    predictions = raw_predictions.rename(
        {"prediction": "y_score", "actual": "y_true", "fold": "fold_id"}
    )
    if predictions.select("timestamp", "symbol").is_duplicated().any():
        raise RuntimeError(f"Duplicate timestamp-symbol rows in {pred_path}")
    if (
        predictions.select("timestamp", "symbol", "y_score", "y_true", "fold_id")
        .null_count()
        .sum_horizontal()
        .item()
        != 0
    ):
        raise RuntimeError(f"Null required values in {pred_path}")
    shortest_symbol = predictions.group_by("symbol").len()["len"].min()
    if shortest_symbol < min_observations:
        raise RuntimeError(
            f"{pred_path} has a symbol with {shortest_symbol} observations; a "
            f"{min_observations}-observation trailing window needs at least that many"
        )
    thinnest_date = predictions.group_by("timestamp").len()["len"].min()
    if thinnest_date < 2:
        raise RuntimeError(
            f"{pred_path} has a date carrying {thinnest_date} symbol; a cross-sectional "
            "rank needs at least two"
        )
    daily_ic = (
        predictions.group_by("timestamp")
        .agg(pl.corr("y_score", "y_true", method="spearman").alias("rank_ic"))
        .filter(pl.col("rank_ic").is_not_null())
    )
    metadata["artifact_n_days"] = daily_ic.height
    metadata["daily_rank_ic"] = float(daily_ic["rank_ic"].mean())
    return predictions


# %% [markdown]
# The registry is opened read-only, once per case study, to resolve one hash into one path. The
# notebook never scans a case study for files, and never writes to a run log.


# %%
def load_registered_predictions(
    case_study: str, label: str, family: str = "gbm"
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Load the complete validation prediction set with the highest daily rank IC."""
    run_log = get_case_study_dir(case_study, create=False) / "run_log"
    db_path = run_log / "registry.db"
    metadata = select_registered_prediction(db_path, family, label)
    pred_path = run_log / "predictions" / str(metadata["prediction_hash"]) / "predictions.parquet"
    predictions = validate_prediction_set(pred_path, metadata, max(ROLLING_WINDOWS))
    return predictions, metadata


# %% [markdown]
# One row per case study records exactly which artifact the comparison ran on, so any number below
# can be traced back to a hash. `registered_n_days` is what the registry recorded when the
# predictions were produced and `artifact_n_days` is what the file on disk carries now.


# %%
def prediction_summary(
    case_study: str, predictions: pl.DataFrame, metadata: dict[str, object]
) -> dict[str, object]:
    """Summarize the exact registered prediction set used by the comparison."""
    return {
        "case_study": case_study,
        "label": LABELS[case_study],
        "config": metadata["config_name"],
        "prediction_hash": metadata["prediction_hash"],
        "daily_rank_ic": metadata["daily_rank_ic"],
        "registered_n_days": metadata["registered_n_days"],
        "artifact_n_days": metadata["artifact_n_days"],
        "rows": predictions.height,
        "symbols": predictions["symbol"].n_unique(),
        "dates": predictions["timestamp"].n_unique(),
    }


# %% [markdown]
# ## 1. Three ways to turn a score into a position
#
# A model emits a number per symbol per date. A portfolio needs to know whether to hold that symbol.
# The step between them is a choice, and the three below differ in what they compare the score to.
#
# | | Compares the score to | Works when | Breaks when |
# |---|---|---|---|
# | Fixed threshold | A constant | Scores are calibrated to a fixed scale, so the constant means something | The score distribution drifts, and the rule silently becomes always-on or never-on |
# | Trailing percentile | The same symbol's recent scores | The scale drifts but the shape is stable | The symbol has little history, or its recent past is unlike its present |
# | Cross-sectional | Every symbol's score that day | The universe is broad enough to rank | The universe is one symbol, or the whole cross-section moves together |
#
# What is measured here is the signal stream itself: how often each rule is active, and how often a
# symbol changes state. No returns are computed and no costs are charged, so nothing below says
# which rule earns more.

# %% [markdown]
# ## 2. Load one prediction set per case study

# %%
predictions = {}
summary_rows = []

for case_study, label in LABELS.items():
    frame, metadata = load_registered_predictions(case_study, label)
    predictions[case_study] = {"data": frame, "metadata": metadata}
    summary_rows.append(prediction_summary(case_study, frame, metadata))

prediction_sets = pl.DataFrame(summary_rows).sort("case_study")
prediction_sets

# %% [markdown]
# ## 3. The two diagnostics


# %%
def signal_diagnostics(signals: pl.DataFrame) -> dict[str, float | int]:
    """Compute signal frequency and per-symbol state-transition frequency."""
    ordered = signals.sort("symbol", "timestamp").with_columns(
        (pl.col("signal") != pl.col("signal").shift(1).over("symbol"))
        .fill_null(False)
        .alias("changed")
    )
    return {
        "n_signals": ordered.filter(pl.col("signal") != 0).height,
        "signal_rate": ordered["signal"].ne(0).mean(),
        "transition_rate": ordered["changed"].mean(),
        "n_dates": ordered["timestamp"].n_unique(),
    }


# %% [markdown]
# A zero cutoff is the economically neutral fixed rule for signed return
# predictions: activate when the predicted return is positive.


# %%
def fixed_threshold_rows(predictions: pl.DataFrame, case_study: str) -> list[dict]:
    """Evaluate the zero fixed threshold for one prediction set."""
    diagnostics = signal_diagnostics(fixed_threshold_signal(predictions, threshold=0.0))
    return [
        {
            "case_study": case_study,
            "method": "fixed_threshold",
            "threshold": 0.0,
            "window": None,
            "percentile": None,
            **diagnostics,
        }
    ]


# %% [markdown]
# A trailing rule reads each symbol's own recent scores and asks where today's sits among them.
# That makes the cutoff move with the symbol's own score distribution, which is what a fixed
# threshold cannot do. It also means the rule is blind to the cross-section: two symbols can both
# be in their own top decile on the same day, or neither can.


# %%
def rolling_percentile_rows(predictions: pl.DataFrame, case_study: str) -> list[dict]:
    """Evaluate trailing percentile rules for one prediction set."""
    rows = []
    for window in ROLLING_WINDOWS:
        for percentile in PERCENTILES:
            signals = rolling_percentile_signal(
                predictions, window=window, percentile=float(percentile)
            )
            rows.append(
                {
                    "case_study": case_study,
                    "method": "rolling_percentile",
                    "threshold": None,
                    "window": window,
                    "percentile": percentile,
                    **signal_diagnostics(signals),
                }
            )
    return rows


# %% [markdown]
# Cross-sectional rules rank the contemporaneous universe and therefore need
# no fitted time-series threshold.


# %%
def cross_sectional_rows(predictions: pl.DataFrame, case_study: str) -> list[dict]:
    """Evaluate cross-sectional percentile rules for one prediction set."""
    rows = []
    for percentile in PERCENTILES:
        signals = cross_sectional_percentile_signal(predictions, percentile=float(percentile))
        rows.append(
            {
                "case_study": case_study,
                "method": "cross_sectional",
                "threshold": None,
                "window": None,
                "percentile": percentile,
                **signal_diagnostics(signals),
            }
        )
    return rows


# %% [markdown]
# The three result blocks share a schema so they can be compared without
# conflating signal transitions with portfolio turnover.


# %%
def compare_signal_methods(predictions: pl.DataFrame, case_study: str) -> pl.DataFrame:
    """Combine fixed, rolling, and cross-sectional diagnostics."""
    rows = [
        *fixed_threshold_rows(predictions, case_study),
        *rolling_percentile_rows(predictions, case_study),
        *cross_sectional_rows(predictions, case_study),
    ]
    return pl.DataFrame(rows)


# %% [markdown]
# ## 4. Evaluate the whole grid

# %%
all_results = []

for case_study, pred_info in predictions.items():
    preds = pred_info["data"]
    all_results.append(compare_signal_methods(preds, case_study))

comparison_df = pl.concat(all_results)
comparison_df.group_by("case_study", "method").len().sort("case_study", "method")

# %% [markdown]
# ## 5. The fixed threshold

# %%
fixed_results = (
    comparison_df.filter(pl.col("method") == "fixed_threshold")
    .select(["case_study", "threshold", "signal_rate", "transition_rate", "n_signals"])
    .sort("case_study")
)

# %% [markdown]
# The zero cutoff activates on a positive predicted return. The two rates differ
# because prediction scales and horizons differ across the registered models.

# %%
fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=[CASE_STUDIES[cs] for cs in fixed_results["case_study"]],
        y=(fixed_results["signal_rate"] * 100).to_list(),
        marker_color=[CASE_STUDY_COLORS[case_study] for case_study in fixed_results["case_study"]],
        text=[f"{value:.1f}%" for value in fixed_results["signal_rate"] * 100],
        textposition="outside",
        showlegend=False,
    )
)

fig.update_layout(
    title="A zero cutoff produces model-specific activation rates",
    xaxis_title="Registered prediction set",
    yaxis_title="Signal rate (%)",
    height=400,
)
fig.show()

# %% [markdown]
# ## 6. Trailing rules across the grid
#
# Every lookback-and-percentile pair from the grid is one marker. The horizontal axis is how often
# the rule is active, the vertical axis is how often a symbol flips between active and inactive.
# Hovering a marker names the pair that produced it.
#
# What to look for is the shape rather than any one point. A rule that is active more often has
# more opportunities to change its mind, so the two rates rise together; the question worth asking
# of a candidate rule is whether it sits above or below that trend, because a rule that transitions
# more than its activation rate implies is one whose cutoff the scores keep crossing.

# %%
fig = go.Figure()

for case_study in CASE_STUDIES:
    data = comparison_df.filter(
        (pl.col("case_study") == case_study) & (pl.col("method") == "rolling_percentile")
    )
    fig.add_trace(
        go.Scatter(
            x=(data["signal_rate"] * 100).to_list(),
            y=(data["transition_rate"] * 100).to_list(),
            name=CASE_STUDIES[case_study],
            mode="markers",
            marker=dict(size=10, color=CASE_STUDY_COLORS[case_study]),
            text=[
                f"window={window}, percentile={percentile}"
                for window, percentile in zip(data["window"], data["percentile"], strict=True)
            ],
            hovertemplate="%{text}<br>Signal: %{x:.1f}%<br>Transitions: %{y:.1f}%",
        )
    )

fig.update_layout(
    title="Crypto rules change state more often than their activation rate implies",
    xaxis_title="Share of observations with a signal (%)",
    yaxis_title="Share of observations that change state (%)",
    height=450,
)
fig.show()

# %% [markdown]
# ## 7. The three methods at one cutoff
#
# Comparing methods at whatever settings each happens to like is not a comparison. All three are
# read here at the same operating percentile, which the settings block above printed: the trailing
# rule takes it over each symbol's own history at the longest lookback in the grid, the
# cross-sectional rule takes it over the universe quoted that day, and the fixed rule has no
# percentile at all - it fires whenever the predicted return is positive.
#
# These are diagnostics computed on validation predictions. Nothing here is a selection, and no
# holdout has been touched.


# %%
operating_points = comparison_df.filter(
    (pl.col("method") == "fixed_threshold")
    | (
        (pl.col("method") == "rolling_percentile")
        & (pl.col("window") == OPERATING_WINDOW)
        & (pl.col("percentile") == OPERATING_PERCENTILE)
    )
    | ((pl.col("method") == "cross_sectional") & (pl.col("percentile") == OPERATING_PERCENTILE))
).select("case_study", "method", "signal_rate", "transition_rate", "n_dates")

# %% [markdown]
# The two panels share a method axis, so a method's activation rate and its transition rate can be
# read off together. The pairing is the point: two rules can be active equally often and disagree
# completely about how often to change position.

# %%
method_order = ["fixed_threshold", "rolling_percentile", "cross_sectional"]
method_labels = [
    "Fixed > 0",
    f"Trailing p{OPERATING_PERCENTILE}",
    f"Cross-sectional p{OPERATING_PERCENTILE}",
]
operating_plot_data = {
    case_study: (
        operating_points.filter(pl.col("case_study") == case_study)
        .with_columns(pl.col("method").replace_strict(method_order, [0, 1, 2]).alias("order"))
        .sort("order")
    )
    for case_study in CASE_STUDIES
}

# %% [markdown]
# The ordered plotting view keeps each method aligned across the two panels.

# %%
fig = make_subplots(rows=1, cols=2, subplot_titles=["Signal rate", "State-transition rate"])

for case_study in CASE_STUDIES:
    data = operating_plot_data[case_study]
    fig.add_trace(
        go.Bar(
            x=method_labels,
            y=(data["signal_rate"] * 100).to_list(),
            name=CASE_STUDIES[case_study],
            marker_color=CASE_STUDY_COLORS[case_study],
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=method_labels,
            y=(data["transition_rate"] * 100).to_list(),
            name=CASE_STUDIES[case_study],
            marker_color=CASE_STUDY_COLORS[case_study],
            showlegend=False,
        ),
        row=1,
        col=2,
    )
fig.update_layout(
    title="Relative rules sharply reduce activation versus a zero cutoff",
    barmode="group",
    height=430,
    margin=dict(t=100, r=140),
    legend=dict(x=1.02, y=1.0, xanchor="left", yanchor="top"),
)
fig.update_xaxes(title_text="Signal method", row=1, col=1)
fig.update_xaxes(title_text="Signal method", row=1, col=2)
fig.update_yaxes(title_text="Signal rate (%)", row=1, col=1)
fig.update_yaxes(title_text="State-transition rate (%)", row=1, col=2)
fig.show()

# %% [markdown]
# ## 8. What the lookback length buys
#
# Holding the percentile fixed and sweeping the lookback isolates one setting. The left panel is
# activation frequency and the right is how often a symbol changes state.
#
# Neither is turnover. Turnover is traded notional, which needs position sizes, and this notebook
# never assigns one: a state transition tells you a position changed, not how large the trade was.
# Two rules with identical transition rates can have turnover an order of magnitude apart if one
# sizes by conviction and the other equally.

# %%
lookback_plot_data = {
    case_study: comparison_df.filter(
        (pl.col("case_study") == case_study)
        & (pl.col("method") == "rolling_percentile")
        & (pl.col("percentile") == OPERATING_PERCENTILE)
    ).sort("window")
    for case_study in CASE_STUDIES
}

# %% [markdown]
# Both panels use the same ordered lookback grid for each prediction set.

# %%
fig = make_subplots(rows=1, cols=2, subplot_titles=["Signal rate", "State-transition rate"])
for case_study in CASE_STUDIES:
    data = lookback_plot_data[case_study]
    fig.add_trace(
        go.Scatter(
            x=data["window"].to_list(),
            y=(data["signal_rate"] * 100).to_list(),
            name=CASE_STUDIES[case_study],
            mode="lines+markers",
            line=dict(color=CASE_STUDY_COLORS[case_study]),
            marker=dict(color=CASE_STUDY_COLORS[case_study]),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data["window"].to_list(),
            y=(data["transition_rate"] * 100).to_list(),
            name=CASE_STUDIES[case_study],
            mode="lines+markers",
            line=dict(color=CASE_STUDY_COLORS[case_study]),
            marker=dict(color=CASE_STUDY_COLORS[case_study]),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
fig.update_layout(
    title="Longer lookbacks reduce state-transition frequency",
    height=400,
    margin=dict(t=100, r=140),
    legend=dict(x=1.02, y=1.0, xanchor="left", yanchor="top"),
)
fig.update_xaxes(title_text="Lookback (observations)", row=1, col=1)
fig.update_xaxes(title_text="Lookback (observations)", row=1, col=2)
fig.update_yaxes(title_text="Signal rate (%)", row=1, col=1)
fig.update_yaxes(title_text="State-transition rate (%)", row=1, col=2)
fig.show()

# %% [markdown]
# ## 9. What the cutoff buys
#
# The same sweep in the other direction: lookback held at the shortest in the grid, percentile
# varying. The shortest lookback is the one where the cutoff moves most, so this is the setting
# pair that produces the widest range of behaviour.

# %%
fig = make_subplots(rows=1, cols=2, subplot_titles=["Signal rate", "State-transition rate"])

for case_study in CASE_STUDIES:
    data = comparison_df.filter(
        (pl.col("case_study") == case_study)
        & (pl.col("method") == "rolling_percentile")
        & (pl.col("window") == min(ROLLING_WINDOWS))
    ).sort("percentile")

    fig.add_trace(
        go.Scatter(
            x=data["percentile"].to_list(),
            y=(data["signal_rate"] * 100).to_list(),
            name=CASE_STUDIES[case_study],
            mode="lines+markers",
            line=dict(color=CASE_STUDY_COLORS[case_study]),
            marker=dict(color=CASE_STUDY_COLORS[case_study]),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data["percentile"].to_list(),
            y=(data["transition_rate"] * 100).to_list(),
            name=CASE_STUDIES[case_study],
            mode="lines+markers",
            line=dict(color=CASE_STUDY_COLORS[case_study]),
            marker=dict(color=CASE_STUDY_COLORS[case_study]),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

fig.update_layout(
    title="Higher percentile cutoffs reduce signals and state changes",
    height=400,
    margin=dict(t=100, r=140),
    legend=dict(x=1.02, y=1.0, xanchor="left", yanchor="top"),
)
fig.update_xaxes(title_text="Percentile", row=1, col=1)
fig.update_xaxes(title_text="Percentile", row=1, col=2)
fig.update_yaxes(title_text="Signal rate (%)", row=1, col=1)
fig.update_yaxes(title_text="State-transition rate (%)", row=1, col=2)
fig.show()

# %% [markdown]
# ## 10. The grid, summarized
#
# Averaging across a grid describes the grid, not the method: the mean depends on which settings
# were put in it. What the spread is good for is seeing which method's behaviour is most sensitive
# to its own settings, which is a property of the method.

# %%
method_summary = (
    comparison_df.group_by("method")
    .agg(
        [
            pl.col("signal_rate").mean().alias("avg_signal_rate"),
            pl.col("signal_rate").std().alias("std_signal_rate"),
            pl.col("transition_rate").mean().alias("avg_transition_rate"),
            pl.col("transition_rate").std().alias("std_transition_rate"),
            pl.len().alias("n_settings"),
        ]
    )
    .sort("method")
)
method_summary

# %%
rolling_at_operating_point = operating_points.filter(pl.col("method") == "rolling_percentile")
print(f"Trailing rule at {OPERATING_WINDOW} observations, p{OPERATING_PERCENTILE}, across the")
print("two prediction sets:")
print(
    f"  signal rate      {rolling_at_operating_point['signal_rate'].min():.1%}"
    f" to {rolling_at_operating_point['signal_rate'].max():.1%}"
)
print(
    f"  transition rate  {rolling_at_operating_point['transition_rate'].min():.1%}"
    f" to {rolling_at_operating_point['transition_rate'].max():.1%}"
)

# %% [markdown]
# One rule, one setting, two prediction sets, and the rates are not close. That range is the reason
# a signal-conversion setting cannot be carried from one strategy to another: the percentile is a
# statement about a score distribution, and two models trained on different assets at different
# horizons do not share one.

# %% [markdown]
# ## 11. Save the diagnostics for the backtest that follows
#
# The comparison frame is written where the downstream notebooks read it, so a rule chosen later can
# be traced back to the signal-stream measurements behind it.

# %%
OUTPUT_DIR = get_output_dir(16, "signal_method_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

comparison_df.write_parquet(OUTPUT_DIR / "method_comparison.parquet")
method_summary.write_parquet(OUTPUT_DIR / "method_summary.parquet")
print(f"Wrote {comparison_df.height} method-setting rows and {method_summary.height} summaries")

# %% [markdown]
# ## Key takeaways
#
# 1. **The conversion rule is part of the strategy, not a formatting step.** One prediction set
#    produced signal streams that differ in how often they are active and how often they flip. Any
#    return a backtest reports is a return on the stream, not on the model.
# 2. **A constant cutoff on an uncalibrated score is a bet on the score's scale.** Nothing trains a
#    regression to put its zero anywhere in particular, so a fixed threshold can be always-on for
#    one model and never-on for another trained on the same data. Relative rules sidestep this by
#    construction: a percentile always admits its share.
# 3. **Trailing and cross-sectional rules are relative to different things,** and the difference is
#    not cosmetic. A trailing rule can hold every symbol at once, when all of them are strong
#    against their own history. A cross-sectional rule cannot: it always holds a fixed share of the
#    universe, whatever the universe is doing.
# 4. **A state transition is not a trade.** It says a position changed, not how much was bought or
#    sold. Cost estimates need notional, and notional needs position sizing, which happens after
#    this notebook. Reading transition rates as turnover overstates a concentrated strategy and
#    understates a diffuse one.
# 5. **A setting does not travel between strategies.** The same lookback and cutoff produced very
#    different activation rates on the two prediction sets. Re-measure on your own scores rather
#    than inheriting a number from a notebook.
#
# ### Known limitations
#
# - No returns and no costs. Nothing here establishes that any rule earns more than another, and a
#   rule with attractive diagnostics can still lose money once it is traded.
# - Both prediction sets are validation-window output from one model family. The comparison holds
#   the model fixed on purpose, so it says nothing about how these rules behave on a model with a
#   different score distribution.
# - Signals are binary. A rule that sizes by conviction is a different object and would change the
#   transition-rate reading entirely.
#
# **Next:** `09_performance_reporting` evaluates realized returns once the signal, the position
# sizing and the cost protocol are all fixed. Section 16.2 covers the trading-protocol
# specification that turns signals into positions.
