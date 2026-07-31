"""Shared construction, auditing and figures for the ``03_financial_features`` stage.

The stage writes ``features/financial.parquet``, the matrix every later stage reads.
Three things are shared across the nine case studies and live here rather than in
nine notebooks:

* the **feature-specification register** - one row per family, carrying the lookback
  and the information lag the notebook claims, which the timing figure draws and the
  warmup audit asserts against;
* the **construction primitives** that were copied between notebooks - the
  cross-sectional percentile, the trailing momentum / volatility / risk-adjusted
  block, and the rolling z-score with one denominator guard;
* the **six figures** the stage shows the reader.

Every rolling primitive here reads a trailing window that ends at the row it is
computed for, and every cross-sectional statistic is taken within one decision
timestamp. Nothing in this module fits a parameter across dates.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatch

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage, set_link_color_palette
from scipy.spatial.distance import squareform

from utils.style import COLOR_CYCLER, COLORS, FIGSIZE, add_message_title, show_with_alt

__all__ = [
    "FeatureFamily",
    "assert_values_agree",
    "assign_families",
    "cross_sectional_percentile",
    "drawdown_block",
    "families_from_config",
    "family_coverage",
    "plot_coverage_through_time",
    "plot_cross_sectional_dispersion",
    "plot_feature_distributions",
    "plot_persistence",
    "plot_redundancy_clusters",
    "momentum_volatility_block",
    "plot_timing_contract",
    "register_frame",
    "relative_volume_block",
    "rolling_zscore",
    "trailing_return",
    "trailing_sharpe",
    "trailing_volatility",
    "warmup_audit",
]

# One denominator guard for every ratio in the stage. Five different ones shipped
# across the nine notebooks, which made otherwise identical features incomparable.
EPS = 1e-8


# ---------------------------------------------------------------------------
# The feature-specification register
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureFamily:
    """One row of the feature-specification register.

    ``lookback`` is the number of bars the family's longest window spans, counted back
    from the decision timestamp, and it is the floor the warmup audit holds each column
    to. ``lookback`` and ``lag`` are counted in bars of the case study's own frequency,
    so that the timing figure and the warmup audit read the same numbers the
    construction code used.

    ``role`` separates a *signal* - something the thesis says ranks assets against
    each other - from a *state* variable, which describes the environment a signal
    is read in. The distinction decides how a feature is used downstream, and it is
    a judgement the notebook has to record because no assertion can recover it.
    """

    name: str
    pattern: str
    role: str
    hypothesis: str
    inputs: str
    lookback: int
    lag: int
    frame: str
    representation: str
    failure_mode: str

    def matches(self, column: str) -> bool:
        """True when *column* matches any of the family's ``|``-separated patterns."""
        return any(fnmatch(column, p) for p in self.pattern.split("|"))


def assign_families(
    columns: Iterable[str],
    families: Sequence[FeatureFamily],
    *,
    strict: bool = True,
) -> dict[str, str]:
    """Map each column to the first register family whose pattern matches it.

    With ``strict`` set, a column no family claims raises: a feature written to the
    matrix without a register row is a feature whose timing contract nobody stated.
    """
    assignment: dict[str, str] = {}
    unclaimed: list[str] = []
    for column in columns:
        for family in families:
            if family.matches(column):
                assignment[column] = family.name
                break
        else:
            unclaimed.append(column)
    if unclaimed and strict:
        raise AssertionError(
            f"{len(unclaimed)} columns have no register row: {sorted(unclaimed)[:12]}"
        )
    return assignment


def register_frame(
    families: Sequence[FeatureFamily],
    columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Render the register as a table, with the realized column count per family."""
    counts: dict[str, int] = dict.fromkeys((f.name for f in families), 0)
    if columns is not None:
        for _, name in assign_families(columns, families).items():
            counts[name] += 1
    return pl.DataFrame(
        [
            {
                "family": f.name,
                "columns": counts[f.name],
                "role": f.role,
                "driver hypothesis": f.hypothesis,
                "inputs": f.inputs,
                "lookback (bars)": f.lookback,
                "lag (bars)": f.lag,
                "frame": f.frame,
                "representation": f.representation,
                "failure mode": f.failure_mode,
            }
            for f in families
        ]
    )


def families_from_config(setup: Mapping) -> list[FeatureFamily]:
    """Read the register out of ``config/setup.yaml::features.families``.

    The register belongs in the configuration rather than in the notebook, for the
    same reason the label name and the holdout boundary do: it is the statement of
    what a case study's feature set *is*, and a statement only the notebook holds
    cannot be read by a test, by a later stage, or by anyone asking what changed.
    """
    return [FeatureFamily(**row) for row in setup["features"]["families"]]


# ---------------------------------------------------------------------------
# Construction primitives
# ---------------------------------------------------------------------------


def momentum_volatility_block(
    df: pl.DataFrame,
    *,
    entity: str | Sequence[str],
    price: str = "close",
    log_return: str = "log_return",
    return_windows: Sequence[int],
    volatility_windows: Sequence[int],
    periods_per_year: float = 252.0,
    return_prefix: str = "ret",
    time: str = "timestamp",
) -> pl.DataFrame:
    """Trailing returns, volatility and risk-adjusted returns, over one entity.

    Three case studies computed this block character for character and a fourth
    almost did, with five different denominator guards between them. Every window
    is trailing and every statistic is taken within one entity, so a shift means
    "the previous bar for this entity" and never "the previous row in the file".

    Only the three families whose column names already agree across the case studies
    are produced here. The derived columns built on top of them - skip-recent
    momentum, acceleration, volatility ratios - are named differently in each case
    study, and renaming a written column is a schema change for every later stage.
    """
    keys = [entity] if isinstance(entity, str) else list(entity)
    df = df.sort([*keys, time])
    if log_return not in df.columns:
        df = df.with_columns(pl.col(price).log().diff().over(keys).alias(log_return))
    return df.with_columns(
        [trailing_return(price, w, keys).alias(f"{return_prefix}_{w}d") for w in return_windows]
        + [
            trailing_volatility(log_return, w, keys, periods_per_year=periods_per_year).alias(
                f"vol_{w}d"
            )
            for w in volatility_windows
        ]
        + [
            trailing_sharpe(log_return, w, keys, periods_per_year=periods_per_year).alias(
                f"sharpe_{w}d"
            )
            for w in return_windows
        ]
    )


def drawdown_block(
    df: pl.DataFrame,
    *,
    entity: str | Sequence[str],
    price: str = "close",
    windows: Sequence[int],
) -> pl.DataFrame:
    """Distance below the trailing peak, per window.

    This is the *current* drawdown - where price sits now relative to the highest
    close of the window - and not the worst peak-to-trough decline inside it. The
    two are different statistics and shipped under one column name across the case
    studies; they are named apart here.
    """
    keys = [entity] if isinstance(entity, str) else list(entity)
    peak = {w: pl.col(price).rolling_max(w).over(keys) for w in windows}
    return df.with_columns(
        ((pl.col(price) - peak[w]) / peak[w].clip(lower_bound=EPS)).alias(f"max_dd_{w}d")
        for w in windows
    )


def relative_volume_block(
    df: pl.DataFrame,
    *,
    entity: str | Sequence[str],
    volume: str = "volume",
    windows: Sequence[int],
    clip_quantiles: tuple[float, float] = (0.01, 0.99),
    time: str = "timestamp",
) -> pl.DataFrame:
    """Volume against its own trailing mean, clipped within the decision date.

    An index rebalance puts one entity's volume orders of magnitude above its own
    average. Clipping at percentiles taken **within the date** removes that from the
    scale a model sees while reading no other date's volume - which a clip fitted
    over the whole column would.
    """
    return clip_within_date(
        trailing_volume_ratio(df, entity=entity, volume=volume, windows=windows),
        columns=[f"vol_ratio_{w}d" for w in windows],
        quantiles=clip_quantiles,
        time=time,
    )


def trailing_volume_ratio(
    df: pl.DataFrame,
    *,
    entity: str | Sequence[str],
    volume: str = "volume",
    windows: Sequence[int],
) -> pl.DataFrame:
    """Volume over its own trailing mean, per entity, unclipped.

    Separate from the clip because the two need different row sets. The trailing
    mean is a property of the entity's whole history and has to read every bar it
    traded, including bars a downstream eligibility gate will drop; taking it after
    the gate makes a newly eligible entity's first year read a mean of a few days,
    and lets an entity that re-enters after a gap average across the gap.
    """
    keys = [entity] if isinstance(entity, str) else list(entity)
    return df.with_columns(
        (pl.col(volume) / pl.col(volume).rolling_mean(w).over(keys).clip(lower_bound=EPS)).alias(
            f"vol_ratio_{w}d"
        )
        for w in windows
    )


def clip_within_date(
    df: pl.DataFrame,
    *,
    columns: Sequence[str],
    quantiles: tuple[float, float] = (0.01, 0.99),
    time: str = "timestamp",
) -> pl.DataFrame:
    """Winsorize *columns* at quantiles taken within each decision date.

    The bounds are a property of one cross-section, so they read no other date -
    and, like any within-date statistic, they must be taken over the rows that are
    actually tradable on that date.
    """
    lo, hi = quantiles
    return df.with_columns(
        pl.col(c)
        .clip(pl.col(c).quantile(lo).over(time), pl.col(c).quantile(hi).over(time))
        .alias(c)
        for c in columns
    )


def cross_sectional_percentile(column: str, over: str | Sequence[str]) -> pl.Expr:
    """Percentile position of *column* within each decision timestamp, in (0, 100).

    The ``+ 1`` in the denominator is what keeps the top asset off the boundary, so
    the mapping is a percentile rather than a rank divided by its own maximum. The
    partition is the decision timestamp - and, where a case study carries several
    instruments per entity-date, the timestamp together with that second key.
    """
    partition = [over] if isinstance(over, str) else list(over)
    return (
        pl.col(column).rank(method="min").over(partition)
        / (pl.col(column).count().over(partition) + 1)
        * 100
    ).alias(f"{column}_pct")


def rolling_zscore(column: str, window: int, over: str | Sequence[str]) -> pl.Expr:
    """Standardize *column* against its own trailing *window*, within one entity.

    This is a trailing statistic: the mean and standard deviation at each row are
    computed from that row's own history, so nothing is estimated across the sample
    and there is no boundary to seal.
    """
    partition = [over] if isinstance(over, str) else list(over)
    mean = pl.col(column).rolling_mean(window).over(partition)
    std = pl.col(column).rolling_std(window).over(partition)
    return ((pl.col(column) - mean) / std.clip(lower_bound=EPS)).alias(f"{column}_z{window}")


def trailing_return(price: str, window: int, over: str | Sequence[str]) -> pl.Expr:
    """Simple return over the trailing *window* bars, within one entity."""
    partition = [over] if isinstance(over, str) else list(over)
    past = pl.col(price).shift(window).over(partition)
    return (pl.col(price) / past.clip(lower_bound=EPS) - 1).alias(f"ret_{window}")


def trailing_volatility(
    log_return: str,
    window: int,
    over: str | Sequence[str],
    *,
    periods_per_year: float = 252.0,
) -> pl.Expr:
    """Annualized close-to-close volatility over the trailing *window* bars."""
    partition = [over] if isinstance(over, str) else list(over)
    return (
        pl.col(log_return).rolling_std(window).over(partition) * np.sqrt(periods_per_year)
    ).alias(f"vol_{window}")


def trailing_sharpe(
    log_return: str,
    window: int,
    over: str | Sequence[str],
    *,
    periods_per_year: float = 252.0,
) -> pl.Expr:
    """Annualized trailing Sharpe ratio: mean log return over its own dispersion.

    Both terms are per period and the annualization is the usual root-time factor,
    so the result is on the scale a Sharpe ratio is read on and horizons of one
    family are comparable.

    Four case studies previously divided the rolling **sum** by the per-period
    standard deviation and scaled by ``sqrt(periods_per_year / window)``. That is
    a mean/std ratio inflated by ``sqrt(window)`` - a factor of about 16 at a
    one-year window - which is why the shipped one-year values ranged past 50.
    """
    partition = [over] if isinstance(over, str) else list(over)
    mean = pl.col(log_return).rolling_mean(window).over(partition)
    std = pl.col(log_return).rolling_std(window).over(partition)
    return (mean / std.clip(lower_bound=EPS) * np.sqrt(periods_per_year)).alias(f"sharpe_{window}")


# ---------------------------------------------------------------------------
# The warmup audit
# ---------------------------------------------------------------------------


def warmup_audit(
    df: pl.DataFrame,
    expected: Mapping[str, int],
    *,
    entity: str | Sequence[str],
    time: str = "timestamp",
) -> pl.DataFrame:
    """Assert that every column's leading nulls match its declared lookback.

    For each column in *expected*, the first bar at which it can hold a value is
    counted within each entity and compared with the declared number of warmup
    bars. A column that is populated **earlier** than its lookback allows is
    reading rows it cannot see, so the check raises rather than reporting.

    Returns the per-column census so the notebook can show it.
    """
    keys = [entity] if isinstance(entity, str) else list(entity)
    ranked = df.sort([*keys, time]).with_columns(pl.col(time).cum_count().over(keys).alias("_bar"))
    rows = []
    for column, bars in expected.items():
        observed = ranked.filter(pl.col(column).is_not_null())["_bar"].min()
        observed = None if observed is None else int(observed)
        rows.append(
            {
                "column": column,
                "declared warmup (bars)": int(bars),
                "first populated bar": observed,
                "populated": observed is not None,
            }
        )
    census = pl.DataFrame(rows)
    early = census.filter(
        pl.col("populated") & (pl.col("first populated bar") < pl.col("declared warmup (bars)"))
    )
    if early.height:
        raise AssertionError(
            "columns populated from fewer bars than their window spans, so the window "
            f"produced a value it cannot have: {early['column'].to_list()}"
        )
    empty = census.filter(~pl.col("populated"))
    if empty.height:
        raise AssertionError(f"columns are null everywhere: {empty['column'].to_list()}")
    return census


def assert_values_agree(
    full: pl.DataFrame,
    withheld: pl.DataFrame,
    *,
    columns: Sequence[str],
    keys: Sequence[str],
) -> pl.DataFrame:
    """Assert two builds of the same rows produced the same feature values.

    *full* is the panel built from everything and then cut back to the rows before
    the boundary; *withheld* is the panel built from those rows alone. A trailing
    statistic reads only its own row's history and a within-date statistic reads only
    its own timestamp, so both are unchanged by the later rows being absent. A
    parameter fitted over a whole column is not: truncating the column moves the
    parameter, and with it every row it was applied to.

    This tests the whole construction at once and does not depend on anyone having
    remembered to flag the transform that fits. Returns the per-column comparison.
    """
    order = list(keys)
    a_frame, b_frame = full.sort(order), withheld.sort(order)
    if a_frame.height != b_frame.height:
        raise AssertionError(
            f"withholding later rows changed the row count ({a_frame.height} -> "
            f"{b_frame.height}), so the row set itself depends on rows after the boundary"
        )
    rows = []
    for column in columns:
        a = a_frame[column].cast(pl.Float64).to_numpy()
        b = b_frame[column].cast(pl.Float64).to_numpy()
        missing_a, missing_b = np.isnan(a), np.isnan(b)
        # A value on one side and a null on the other is the loudest form of this
        # failure, and it is exactly what a nan-skipping maximum hides.
        both = missing_a & missing_b
        gap = np.where(both, 0.0, np.abs(a - b))
        gap = np.where(missing_a ^ missing_b, np.inf, gap)
        rows.append(
            {
                "column": column,
                "rows compared": int(a.size),
                "null only on one side": int((missing_a ^ missing_b).sum()),
                "max abs difference": float(gap.max()) if gap.size else 0.0,
            }
        )
    census = pl.DataFrame(rows)
    moved = census.filter(
        (pl.col("max abs difference") > 1e-12) | (pl.col("null only on one side") > 0)
    )
    if moved.height:
        raise AssertionError(
            "these features move when later rows are withheld, so their transform was "
            f"fitted across the sample: {moved['column'].to_list()}"
        )
    return census


def family_coverage(
    df: pl.DataFrame,
    assignment: Mapping[str, str],
    *,
    time: str = "timestamp",
    every: str | None = None,
) -> pl.DataFrame:
    """Non-null share per family per decision timestamp.

    *every* buckets the time axis (``"1mo"``) where the panel has more decision
    timestamps than a chart can resolve.
    """
    frame = df
    if every is not None:
        frame = frame.with_columns(pl.col(time).dt.truncate(every).alias(time))
    families: dict[str, list[str]] = {}
    for column, family in assignment.items():
        families.setdefault(family, []).append(column)
    aggs = [
        pl.mean_horizontal([pl.col(c).is_not_null() for c in cols]).mean().alias(family)
        for family, cols in families.items()
    ]
    return frame.group_by(time).agg(aggs).sort(time)


# ---------------------------------------------------------------------------
# The six figures
# ---------------------------------------------------------------------------


def _cycle(n: int) -> list[tuple[str, str]]:
    """Colour and line style per series, so more series than hues stay separable.

    The palette has six hues. Beyond that, cycling colour alone puts two series in
    the same navy and the reader cannot tell which line is which; the style cycles
    at a different rate, so each pair is distinct.
    """
    styles = ["-", "--", ":", "-."]
    return [
        (COLOR_CYCLER[i % len(COLOR_CYCLER)], styles[(i // len(COLOR_CYCLER)) % len(styles)])
        for i in range(n)
    ]


def plot_coverage_through_time(
    coverage: pl.DataFrame,
    *,
    time: str = "timestamp",
    warmup_boundary: object | None = None,
    title: str,
    alt: str,
    subtitle: str | None = None,
) -> None:
    """F1. Non-null share per family against date, with the warmup boundary drawn."""
    families = [c for c in coverage.columns if c != time]
    fig, ax = plt.subplots(figsize=FIGSIZE["single"])
    x = coverage[time].to_list()
    for (color, style), family in zip(_cycle(len(families)), families, strict=False):
        ax.plot(x, coverage[family].to_list(), label=family, color=color, ls=style, linewidth=1.1)
    if warmup_boundary is not None:
        ax.axvline(warmup_boundary, color=COLORS["neutral"], linestyle="--", linewidth=1)
        ax.annotate(
            "warmup ends",
            xy=(warmup_boundary, 1.0),
            xytext=(4, -9),
            textcoords="offset points",
            fontsize=7,
            color=COLORS["neutral"],
        )
    # Scaled to the data rather than pinned to zero. A matrix that is 99% dense everywhere
    # draws as one flat line at the top of a 0-1 axis, which hides the only thing the figure
    # is for: where, and by how much, a family is actually thin.
    minima = [coverage[f].min() for f in families]
    lowest = min([float(v) for v in minima if v is not None], default=1.0)
    ax.set_ylim(min(lowest - 0.02 * (1 - lowest) - 0.002, 0.999), 1.0005)
    ax.set_ylabel("non-null share")
    ax.legend(fontsize=6, ncol=3, frameon=False, loc="lower right")
    add_message_title(ax, title, subtitle=subtitle)
    fig.tight_layout()
    show_with_alt(fig, alt)


def plot_feature_distributions(
    df: pl.DataFrame,
    columns: Sequence[str],
    *,
    title: str,
    alt: str,
    subtitle: str | None = None,
    bins: int = 60,
    ncols: int = 3,
    clip_quantiles: tuple[float, float] | None = (0.005, 0.995),
) -> None:
    """F2. Small multiples of the primary signal family, one panel per feature.

    The tails are clipped for display only, so that one outlier cannot compress the
    body of a distribution the reader is being asked to judge the shape of.
    """
    columns = list(columns)
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(FIGSIZE["grid_2x3"][0], 1.55 * nrows + 0.85),
        squeeze=False,
    )
    flat = axes.ravel()
    for ax, column in zip(flat, columns, strict=False):
        values = df[column].cast(pl.Float64).drop_nulls().drop_nans().to_numpy()
        if clip_quantiles is not None and values.size:
            lo, hi = np.quantile(values, clip_quantiles)
            values = values[(values >= lo) & (values <= hi)]
        ax.hist(values, bins=bins, color=COLORS["blue"], edgecolor="none")
        ax.set_xlabel(column, fontsize=7, color=COLORS["neutral"], labelpad=1)
        ax.tick_params(labelsize=6)
        ax.set_yticks([])
    for ax in flat[len(columns) :]:
        ax.set_visible(False)
    # A grid has no single axes to hang the message title on, so it goes on the figure with
    # a band reserved for it; `add_message_title` would put it over the top-left panel.
    height = 1.55 * nrows + 0.85
    band = 1.0 - 0.78 / height
    fig.tight_layout(rect=(0, 0, 1, band))
    fig.text(
        0.01,
        band + 0.18 / height,
        title,
        ha="left",
        va="bottom",
        color=COLORS["blue"],
        fontweight="semibold",
        fontsize=11,
    )
    if subtitle:
        fig.text(0.01, band, subtitle, ha="left", va="bottom", fontsize=8, color=COLORS["neutral"])
    show_with_alt(fig, alt)


def plot_cross_sectional_dispersion(
    df: pl.DataFrame,
    column: str,
    *,
    time: str = "timestamp",
    title: str,
    alt: str,
    subtitle: str | None = None,
    every: str | None = None,
) -> None:
    """F3. Per decision date, the 10th-90th percentile band of *column* with its median."""
    # The quantiles are taken within one decision date and only then averaged over
    # the period. Truncating the timestamp first and taking a quantile of the pooled
    # month measures the spread of a month of entity-days, which is a different and
    # always wider quantity than the cross-section a strategy actually ranks on.
    daily = (
        df.group_by(time)
        .agg(
            pl.col(column).quantile(0.10).alias("p10"),
            pl.col(column).median().alias("p50"),
            pl.col(column).quantile(0.90).alias("p90"),
        )
        .sort(time)
        .drop_nulls()
    )
    band = daily
    if every is not None:
        band = (
            daily.with_columns(pl.col(time).dt.truncate(every).alias(time))
            .group_by(time)
            .agg(pl.col("p10").mean(), pl.col("p50").mean(), pl.col("p90").mean())
            .sort(time)
        )
    fig, ax = plt.subplots(figsize=FIGSIZE["single"])
    x = band[time].to_list()
    ax.fill_between(
        x,
        band["p10"].to_list(),
        band["p90"].to_list(),
        color=COLORS["blue"],
        alpha=0.20,
        linewidth=0,
        label="10th-90th percentile",
    )
    ax.plot(x, band["p50"].to_list(), color=COLORS["blue"], linewidth=1.2, label="median")
    ax.set_ylabel(column)
    ax.legend(fontsize=7, frameon=False)
    add_message_title(ax, title, subtitle=subtitle)
    fig.tight_layout()
    show_with_alt(fig, alt)


def plot_timing_contract(
    families: Sequence[FeatureFamily],
    *,
    bar_unit: str,
    title: str,
    alt: str,
    subtitle: str | None = None,
) -> None:
    """F4. The register's lookback and information lag, drawn on a shared axis.

    Time runs left to right and ends at the decision timestamp at zero. A family's
    bar spans the window it reads; a gap between the bar's right edge and zero is
    the lag with which that input becomes knowable.
    """
    families = list(families)
    fig, ax = plt.subplots(figsize=(FIGSIZE["single"][0], max(2.0, 0.32 * len(families) + 1.0)))
    for i, family in enumerate(reversed(families)):
        start = -(family.lookback + family.lag)
        ax.barh(
            i,
            width=family.lookback,
            left=start,
            height=0.55,
            color=COLORS["blue"],
            edgecolor="none",
        )
        if family.lag:
            # Hatched and unfilled. A solid bar over the lag makes the family look like
            # it reads right up to the decision, which is the opposite of what a lag is.
            ax.barh(
                i,
                width=family.lag,
                left=-family.lag,
                height=0.55,
                facecolor="none",
                edgecolor=COLORS["amber"],
                hatch="////",
                linewidth=0.8,
                label="_nolegend_",
            )
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels([f.name for f in reversed(families)], fontsize=7)
    ax.axvline(0, color=COLORS["negative"], linewidth=1)
    ax.set_xlabel(f"{bar_unit} before the decision timestamp")
    if any(f.lag for f in families):
        ax.barh(
            0,
            width=0,
            left=0,
            facecolor="none",
            edgecolor=COLORS["amber"],
            hatch="////",
            linewidth=0.8,
            label="published but not yet available",
        )
        ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.annotate(
        "decision",
        xy=(0, -0.45),
        xytext=(-3, 0),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=7,
        color=COLORS["negative"],
    )
    add_message_title(ax, title, subtitle=subtitle)
    fig.tight_layout()
    show_with_alt(fig, alt)


def plot_redundancy_clusters(
    df: pl.DataFrame,
    columns: Sequence[str],
    *,
    cut: float = 0.7,
    title: str,
    alt: str,
    subtitle: str | None = None,
    max_rows: int = 200_000,
    seed: int = 42,
) -> dict[str, int]:
    """F5. Hierarchical clustering on distance :math:`1 - |\\rho|`, with the cut drawn.

    Returns the cluster each column falls in, which is what ``05_evaluation`` needs
    in order to pick one representative per cluster on a fold-aware criterion.
    """
    columns = list(columns)
    frame = df.select(columns)
    if frame.height > max_rows:
        frame = frame.sample(max_rows, seed=seed)
    # Ranked before correlating, so the distance is Spearman. The claim the figure makes
    # is that two features carry the same *ordering*, and Pearson on raw values answers a
    # narrower question: it misses a monotone but curved relation, which would leave two
    # interchangeable features in separate clusters and inflate the reported count.
    ranked = frame.with_columns(pl.col(c).rank().alias(c) for c in columns)
    matrix = ranked.to_numpy().astype(float)
    corr = np.ma.corrcoef(np.ma.masked_invalid(matrix), rowvar=False).filled(0.0)
    corr = np.clip(np.nan_to_num(corr, nan=0.0), -1.0, 1.0)
    distance = 1.0 - np.abs(corr)
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2.0
    tree = linkage(squareform(distance, checks=False), method="average")
    height = 1.0 - cut
    labels = fcluster(tree, t=height, criterion="distance")

    fig, ax = plt.subplots(figsize=(FIGSIZE["single"][0], max(2.4, 0.13 * len(columns) + 1.2)))
    set_link_color_palette([c for c, _ in _cycle(6)])
    dendrogram(
        tree,
        labels=columns,
        orientation="left",
        color_threshold=height,
        above_threshold_color=COLORS["neutral"],
        ax=ax,
    )
    ax.axvline(height, color=COLORS["amber"], linestyle="--", linewidth=1)
    ax.set_xlabel(r"distance $1 - |\rho_s|$")
    ax.tick_params(axis="y", labelsize=6)
    add_message_title(ax, title, subtitle=subtitle)
    fig.tight_layout()
    show_with_alt(fig, alt)
    set_link_color_palette(None)
    return dict(zip(columns, (int(v) for v in labels), strict=True))


def _bootstrap_median_interval(
    values: np.ndarray, *, seed: int, draws: int = 500, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap interval for the median of *values*, resampling entities."""
    if values.size < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    medians = np.median(rng.choice(values, size=(draws, values.size), replace=True), axis=1)
    lo, hi = np.quantile(medians, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def plot_persistence(
    df: pl.DataFrame,
    columns: Sequence[str],
    *,
    entity: str | Sequence[str],
    time: str = "timestamp",
    max_lag: int,
    decision_dates: Sequence,
    title: str,
    alt: str,
    subtitle: str | None = None,
    seed: int = 42,
) -> Figure | None:
    """F6. Feature autocorrelation with bootstrap intervals, plus rank stability.

    The left panel is the autocorrelation of the feature itself, estimated per entity
    on pairs of decision dates exactly *k* apart, summarized by the median over
    entities and shown with a percentile bootstrap interval over entities. It runs to
    at least one decision cycle: a feature whose value has decayed before the next
    rebalance cannot support that cadence, however well it predicts the day it is
    computed.

    The right panel asks the same question of the ordering rather than the level, and
    is per date rather than per entity: one cross-sectional rank correlation for each
    consecutive pair in ``decision_dates`` - the schedule the strategy rebalances on -
    summarized by the median over those pairs.

    Both panels read every entity in *df*. An earlier version sampled 40 of them,
    which put a number in front of the reader that was not the universe the notebook
    had just described - and sampled an unsorted ``unique()``, so it was not even the
    same 40 across runs.
    """
    keys = [entity] if isinstance(entity, str) else list(entity)
    columns = list(columns)
    frame = df.select([*keys, time, *columns]).sort([*keys, time])

    lags = np.unique(np.linspace(1, max_lag, min(max_lag, 24)).astype(int))
    # Lags are counted along the panel's decision dates, not along each entity's own
    # rows. Slicing an entity's rows positionally makes "21 bars ago" mean "21 rows
    # ago", which is a different date for every entity and crosses any stretch the
    # entity was absent for - after an eligibility gate, by months or years.
    dates = frame[time].unique().sort()
    acf: dict[str, list[float]] = {c: [] for c in columns}
    lower: dict[str, list[float]] = {c: [] for c in columns}
    upper: dict[str, list[float]] = {c: [] for c in columns}
    counts: list[int] = []
    for lag in lags:
        back = dict(zip(dates[lag:].to_list(), dates[: -int(lag)].to_list(), strict=True))
        pairs = frame.with_columns(
            pl.col(time).replace_strict(back, default=None).alias("_then")
        ).join(
            frame.select(
                *keys,
                pl.col(time).alias("_then"),
                *[pl.col(c).alias(f"_{c}_then") for c in columns],
            ),
            on=[*keys, "_then"],
            how="inner",
        )
        # One estimate per entity, then the median over entities. Pooling every
        # entity-date pair into a single correlation measures something else: two ETFs
        # that sit at different levels make the pooled pairs line up whether or not
        # either one's value persists, so the pooled number is high for a feature with
        # no temporal persistence at all.
        per_entity = (
            pairs.group_by(keys)
            .agg(
                pl.len().alias("_pairs"),
                *[pl.corr(pl.col(c), pl.col(f"_{c}_then")).alias(c) for c in columns],
            )
            .filter(pl.col("_pairs") > 10)
        )
        counts.append(per_entity.height)
        for column in columns:
            rho = per_entity[column].drop_nulls().drop_nans().to_numpy()
            acf[column].append(float(np.median(rho)) if rho.size else np.nan)
            lower[column], upper[column] = (
                bound + [value]
                for bound, value in zip(
                    (lower[column], upper[column]),
                    _bootstrap_median_interval(rho, seed=seed),
                    strict=True,
                )
            )
    acf["_n"] = counts

    width, height = FIGSIZE["dual_h_tall"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(width, height + 0.5))
    # An interval around each curve, at each lag. A single width drawn around zero is
    # a white-noise significance band, which is a different statement and not one this
    # estimator supports: the quantity plotted is a median over ETFs, so its
    # uncertainty is a bootstrap over ETFs and it belongs around the median.
    for (color, style), column in zip(_cycle(len(columns)), columns, strict=False):
        left.fill_between(lags, lower[column], upper[column], color=color, alpha=0.18, linewidth=0)
        left.plot(lags, acf[column], label=column, color=color, ls=style, linewidth=1.2, ms=2.5)
    left.axhline(0, color=COLORS["neutral"], linewidth=0.8)
    left.set_xlabel("lag (bars)")
    left.set_ylabel("autocorrelation")
    # Below the axes, not inside them. A persistent feature fills the upper half and a
    # decaying one fills the lower left, so every in-axes position covers either a
    # curve or the y-axis label depending on the data - which is not something the
    # caller should have to tune per notebook.
    left.legend(
        fontsize=6,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        frameon=False,
    )

    # One cross-sectional rank correlation per consecutive pair of decision dates, then
    # the median over pairs. Pooling every entity-date row into one correlation instead
    # measures how stable an entity's rank is against the whole panel, which is high
    # whenever entities differ from each other at all, and a within-entity shift
    # silently bridges dates that entity was absent for.
    #
    # The pairs come from the schedule the strategy rebalances on, not a fixed number of
    # sessions. `monthly_month_end` leaves a varying number of sessions between
    # decisions, so a fixed lag correlates dates the strategy never compares - and at a
    # one-session lag it answers a question nobody rebalancing monthly is asking, far
    # too favourably, because a daily ordering barely moves.
    schedule = sorted(decision_dates)
    step = dict(zip(schedule[1:], schedule[:-1], strict=True))
    stability = []
    for column in columns:
        ranked = frame.select(
            time, *keys, pl.col(column).rank().over(time).alias("_r")
        ).drop_nulls()
        joined = (
            ranked.with_columns(pl.col(time).replace_strict(step, default=None).alias("_prev"))
            .join(
                ranked.select(*keys, pl.col(time).alias("_prev"), pl.col("_r").alias("_r_prev")),
                on=[*keys, "_prev"],
                how="inner",
            )
            .group_by(time)
            .agg(pl.corr(pl.col("_r"), pl.col("_r_prev"), method="spearman").alias("rho"))
        )
        rho = joined["rho"].drop_nulls().drop_nans()
        stability.append(float(rho.median()) if rho.len() else np.nan)
    right.barh(range(len(columns)), stability, color=COLORS["blue"], height=0.6)
    right.set_yticks(range(len(columns)))
    right.set_yticklabels(columns, fontsize=6)
    # The lower bound follows the data and is never clipped at zero: a negative value is
    # rank reversal between rebalances, which is the most interesting thing this panel can
    # show and the one an axis pinned at zero hides.
    lowest = float(np.nanmin(stability))
    right.set_xlim(min(lowest, 0.0) - 0.08, 1.0)
    right.axvline(0, color=COLORS["neutral"], linewidth=0.8)
    right.set_xlabel("rank correlation, consecutive rebalances", fontsize=8)
    add_message_title(left, title, subtitle=subtitle)
    fig.tight_layout(w_pad=2.5)
    show_with_alt(fig, alt)
    return None
