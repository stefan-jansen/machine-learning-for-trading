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
from scipy.stats import spearmanr

from utils.style import COLOR_CYCLER, COLORS, FIGSIZE, add_message_title, show_with_alt

__all__ = [
    "FeatureFamily",
    "QuantileProfile",
    "assert_values_agree",
    "assign_families",
    "cross_sectional_percentile",
    "drawdown_block",
    "families_from_config",
    "family_coverage",
    "has_value",
    "plot_coverage_through_time",
    "plot_cross_sectional_dispersion",
    "plot_feature_distributions",
    "plot_persistence",
    "plot_redundancy_clusters",
    "momentum_volatility_block",
    "plot_timing_contract",
    "quantile_profile",
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


def has_value(dtype: pl.DataType, column: str) -> pl.Expr:
    """True where the column holds a number: neither null nor NaN.

    Polars counts NaN as a value, so ``is_not_null()`` is True for it, and the
    feature calls do not agree on which one a warm-up head is written with -
    polars' own ``rolling_mean`` emits null, ``ml4t.engineer.features.momentum``
    emits NaN. Reading a NaN as covered breaks both directions: ``warmup_audit``
    takes the first bar as populated for any NaN-warmup column and raises the
    look-ahead assertion on a column that is correctly warmed up, which is what
    ``rsi_14d`` did in fx_pairs, and ``family_coverage`` draws a stretch holding
    no values as fully covered, which is silent.

    A notebook should not have to normalize its frame before a helper can count
    it, so both helpers apply this rather than expecting a ``fill_nan(None)``.
    """
    populated = pl.col(column).is_not_null()
    if dtype.is_float():
        populated = populated & pl.col(column).is_not_nan()
    return populated


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

    A NaN is not a value here: see :func:`has_value`.

    Returns the per-column census so the notebook can show it.
    """
    keys = [entity] if isinstance(entity, str) else list(entity)
    ranked = df.sort([*keys, time]).with_columns(pl.col(time).cum_count().over(keys).alias("_bar"))
    schema = ranked.schema
    rows = []
    for column, bars in expected.items():
        observed = ranked.filter(has_value(schema[column], column))["_bar"].min()
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
    """Share of a family's columns holding a value, per decision timestamp.

    A NaN is not a value here: see :func:`has_value`. Counting one as covered
    draws a warm-up head, or any stretch a library call filled with NaN, as a
    dense line.

    *every* buckets the time axis (``"1mo"``) where the panel has more decision
    timestamps than a chart can resolve.
    """
    frame = df
    if every is not None:
        frame = frame.with_columns(pl.col(time).dt.truncate(every).alias(time))
    schema = frame.schema
    families: dict[str, list[str]] = {}
    for column, family in assignment.items():
        families.setdefault(family, []).append(column)
    aggs = [
        pl.mean_horizontal([has_value(schema[c], c) for c in cols]).mean().alias(family)
        for family, cols in families.items()
    ]
    return frame.group_by(time).agg(aggs).sort(time)


@dataclass(frozen=True)
class QuantileProfile:
    """What each quantile of a feature earned, one decision time at a time."""

    means: list[float]
    medians: list[float]
    monotonicity: float
    spread: float
    periods_used: int
    periods_available: int


def quantile_profile(
    frame: pl.DataFrame,
    feature: str,
    label: str,
    *,
    date_col: str,
    n_quantiles: int = 5,
    min_cross_section: int | None = None,
    demean_within_date: bool = False,
) -> QuantileProfile | None:
    """Average and median label by feature quantile, averaged over decision times.

    Sorting on a feature and reading what each quantile went on to earn is the
    shape behind a rank correlation, and the two have to be built on the same
    inferential unit or they answer different questions while appearing to agree.
    Both averages here are therefore taken twice: first across the assets in one
    quantile at one decision time, and then across decision times. Every decision
    time counts once however many assets it quoted, which is what a portfolio
    rebalanced at every decision time earns and what the information coefficient
    already measures.

    Pooling every row into one average instead is the failure this exists to
    prevent, and it is silent: the profile comes out looking the same, weighted by
    how many assets happened to be quoted rather than by time.

    Quantiles are assigned inside each decision time, so a quantile means "high
    relative to the assets quoted alongside it" rather than "high relative to the
    whole history". Cutting the pooled sample instead lets a period when the whole
    market sat high place all of its assets in the top quantile, which mixes
    movement over time into a diagnostic that is meant to be purely across assets.

    A decision time enters only when it quotes at least *min_cross_section* assets
    (the number of quantiles, where none is given) and at least *n_quantiles*
    distinct feature values. A feature taking two values cannot be split five ways,
    and forcing it produces boundaries that fall inside a tie, where the split
    reports the order the rows sit in rather than the feature.

    Set *demean_within_date* where every quantile earns whatever the market earned
    that period and the difference between quantiles is the only quantity a
    long-short book collects: the label is then taken relative to its own decision
    time's average before the quantiles are formed.

    Returns ``None`` where nothing survives the floor, or where the surviving rows
    do not fill all *n_quantiles* quantiles - both mean the feature has no profile
    to read, not that its profile is flat.
    """
    valid = frame.select(date_col, feature, label).drop_nulls()
    periods_available = valid[date_col].n_unique()
    floor = max(min_cross_section or n_quantiles, n_quantiles)
    valid = valid.filter(
        (pl.len().over(date_col) >= floor)
        & (pl.col(feature).n_unique().over(date_col) >= n_quantiles)
    )
    periods_used = valid[date_col].n_unique()
    if not valid.height:
        return None

    scored = label
    if demean_within_date:
        scored = "_excess"
        valid = valid.with_columns(
            (pl.col(label) - pl.col(label).mean().over(date_col)).alias(scored)
        )

    # Imported here rather than at module scope. `test-unit` builds a deliberately small
    # environment for this module's figure tests - its own comment says matplotlib and
    # hmmlearn are the whole import surface and that nothing here pulls an `ml4t-*`
    # package, which is what keeps it a fast per-commit gate. A module-level import of
    # `ml4t.diagnostic` fails that job at collection, which is exactly what it did.
    from ml4t.diagnostic.signal import quantize_factor

    binned = quantize_factor(valid, n_quantiles=n_quantiles, factor_col=feature, date_col=date_col)
    profile = (
        binned.group_by([date_col, "quantile"])
        .agg(
            pl.col(scored).mean().alias("_mean"),
            pl.col(scored).median().alias("_median"),
        )
        .group_by("quantile")
        .agg(
            pl.col("_mean").mean().alias("mean"),
            pl.col("_median").mean().alias("median"),
        )
        .sort("quantile")
    )
    means = profile["mean"].to_list()
    medians = profile["median"].to_list()
    if len(means) < n_quantiles or any(value is None for value in means):
        return None

    return QuantileProfile(
        means=[float(value) for value in means],
        medians=[float(value) for value in medians],
        monotonicity=float(spearmanr(range(len(means)), means).statistic),
        spread=float(means[-1] - means[0]),
        periods_used=int(periods_used),
        periods_available=int(periods_available),
    )


# ---------------------------------------------------------------------------
# The six figures
# ---------------------------------------------------------------------------


def _cycle(n: int) -> list[tuple[str, str]]:
    """Colour and line style per series, so more series than hues stay separable.

    The style turns over once per full pass through the palette, so the first six
    series are told apart by hue alone and only a seventh onwards needs a dash to
    part it from the series six earlier.

    It used to turn over every *five*. ``COLOR_CYCLER``'s sixth entry was ``slate``,
    a second navy, so a solid first and a solid sixth series were one line, and
    parting them by style was the only lever this helper had. The palette now ends
    in ``recede``, which is a hue away from navy in colour and in gray, so the
    workaround is gone and with it the combinations it cost.
    """
    styles = ["-", "--", ":", "-."]
    hues = len(COLOR_CYCLER)
    return [(COLOR_CYCLER[i % hues], styles[(i // hues) % len(styles)]) for i in range(n)]


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
    # The headroom scales with the range too, for the same reason the floor does. The
    # stage-03 notebooks pass the panel before the null policy, so every family starts at
    # zero and the axis spans the full unit; a fixed 1.0005 ceiling then leaves the dense
    # stretch half a thousandth below the top, drawn underneath the spine. That stretch is
    # what the figure claims - the families fill and never thin again - so it has to be
    # visible as a line rather than as the frame.
    ax.set_ylim(
        min(lowest - 0.02 * (1 - lowest) - 0.002, 0.999),
        1.0 + max(0.0005, 0.03 * (1 - lowest)),
    )
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
    legend = None
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
        # Below the axes, and owned by the figure. Inside them at the bottom left it
        # sat on the last family's bar - the register declares the families and the
        # axes grow one row per family, so the more a case study declares the further
        # the bottom row reaches under the legend, and at the eight of
        # us_firm_characteristics and cme_futures the entry crossed the `interaction`
        # bar and touched the tick labels under it. There is no in-axes corner that is
        # safe here: a lag bar can reach any of them, because which families are
        # lagged and how far is exactly what the figure is drawn to show.
        legend = fig.legend(
            *ax.get_legend_handles_labels(),
            fontsize=7,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
        )
    # A strip under the bottom row that belongs to the "decision" label and to nothing
    # else, and the label anchored to the axes' floor rather than to a data coordinate
    # guessed once. At -0.45 it was inside the bottom row's bar - which spans 0.55 - so
    # it was drawn over the last family's lookback and its lag hatch, the same thing the
    # legend was doing a few lines above and in the same corner. How far the label
    # reached into the bar varied with the family count, because that sets the axes
    # height and so what 7pt is worth in data units.
    ax.set_ylim(-1.0, len(families) - 0.5)
    ax.annotate(
        "decision",
        xy=(0, 0),
        xycoords=ax.get_xaxis_transform(),
        xytext=(-3, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=7,
        color=COLORS["negative"],
    )
    add_message_title(ax, title, subtitle=subtitle)
    # The strip the legend needs is measured after it is drawn, not guessed from the
    # entry count, so a longer label cannot clip and a shorter one cannot leave a band
    # of dead space. Same reservation `plot_persistence` makes for its own legend.
    if legend is not None:
        fig.canvas.draw()
        strip = legend.get_window_extent(fig.canvas.get_renderer()).transformed(
            fig.transFigure.inverted()
        )
        fig.tight_layout(rect=(0.0, strip.height + 0.02, 1.0, 1.0))
    else:
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

    Returns the cluster each column falls in, which the calling notebook uses to state
    how many distinct orderings its matrix carries. Nothing downstream reads it: the
    feature screens in ``05_evaluation`` test one column at a time, and the one case
    study that does pick a representative per cluster builds its own clusters from its
    own fold ICs rather than from this tree.
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
    # Five hues, not six: the sixth is `slate`, a second navy, and a cluster drawn in it
    # is indistinguishable from one drawn in `blue`. Above the cut the links are the
    # figure's background - they say only "these two clusters eventually join" - so they
    # recede to a light slate. They were `neutral`, #334155, which is the same dark
    # blue-grey as `blue` and `slate` at the same weight, so all three read as one thing
    # and the cluster structure the figure exists to show was not visible.
    set_link_color_palette([c for c, _ in _cycle(5)])
    dendrogram(
        tree,
        labels=columns,
        orientation="left",
        color_threshold=height,
        above_threshold_color=COLORS["recede"],
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
    """Percentile bootstrap interval for the median of *values*, resampling entities.

    *values* is sorted first. The median does not care what order it arrives in, but
    ``rng.choice`` draws by index, so the same entities in a different order give the
    same seed a different resample and a different interval. The callers read their
    values out of a ``group_by``, whose row order Polars does not guarantee, so the
    ribbon moved between runs on byte-identical input - a seed that fixed the sampling
    and not the ordering (#329, #333).
    """
    if values.size < 3:
        return float("nan"), float("nan")
    values = np.sort(values)
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
    # Both lag maps below are built from Python objects and handed to
    # `replace_strict`, which types its output from the values it was given, not from
    # the column it replaces. A Python `datetime` carries microseconds, so on a panel
    # stamped in anything else the new column came back `datetime[us]` and the join
    # that follows failed outright:
    #
    #   SchemaError: `_then`: datetime[us, UTC] does not match `_then`: datetime[ms, UTC]
    #
    # Binance stamps in milliseconds, so the whole crypto_perps_funding pipeline is
    # `datetime[ms]` and this figure could not be drawn for it at all; its notebook
    # cast the frame it passed here and nothing else, which fixes one caller and
    # leaves the helper broken for the next. Pinning the return type to the column's
    # own dtype fixes it for every panel, at whatever precision it is stamped in.
    when = frame.schema[time]
    acf: dict[str, list[float]] = {c: [] for c in columns}
    lower: dict[str, list[float]] = {c: [] for c in columns}
    upper: dict[str, list[float]] = {c: [] for c in columns}
    counts: list[int] = []
    for lag in lags:
        back = dict(zip(dates[lag:].to_list(), dates[: -int(lag)].to_list(), strict=True))
        pairs = frame.with_columns(
            pl.col(time).replace_strict(back, default=None, return_dtype=when).alias("_then")
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
    # The left panel carries a full lag axis and every curve the prose reads off; the right
    # is a horizontal bar per feature on a 0-1 axis. Equal columns compressed the panel that
    # holds the information and left a band of white between the two.
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(width, height + 0.5), gridspec_kw={"width_ratios": [2, 1]}
    )
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
    # decaying one fills the lower left, so every in-axes position covers either a curve
    # or the y-axis label depending on the data - which is not something the caller
    # should have to tune per notebook.
    #
    # It belongs to the FIGURE, not to the left panel. `tight_layout` packs each axes
    # together with its decorations, and a legend of feature names centred under a
    # panel is far wider than the panel - so the whole column was sized to the legend
    # and the axes inside it shrank to whatever was left. Measured on four names of
    # `premium_vol_ratio_7d_30d` length: the lag panel held 27% of the figure with the
    # legend attached and 42% without it. A figure legend is outside that packing, so
    # the space it needs is reserved once, in the `rect` below.

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
            ranked.with_columns(
                pl.col(time).replace_strict(step, default=None, return_dtype=when).alias("_prev")
            )
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
    # Feature names on the OUTER edge. `width_ratios` divides the axes, not the figure,
    # and these labels are drawn outside the right panel - so on the inner edge they take
    # their width out of the gap between the panels, and `tight_layout` pays for it by
    # shrinking both. Names in this corpus run to `premium_vol_ratio_7d_30d`, which left
    # the autocorrelation panel - the one the prose reads four curves off - at about a
    # third of the figure. On the outer edge they grow into the right margin, which
    # nothing else is using.
    right.yaxis.set_ticks_position("right")
    right.yaxis.set_label_position("right")
    # The lower bound follows the data and is never clipped at zero: a negative value is
    # rank reversal between rebalances, which is the most interesting thing this panel can
    # show and the one an axis pinned at zero hides.
    lowest = float(np.nanmin(stability))
    right.set_xlim(min(lowest, 0.0) - 0.08, 1.0)
    right.axvline(0, color=COLORS["neutral"], linewidth=0.8)
    # Wrapped, and anchored to the panel's right edge rather than centred under it. On one
    # centred line this label is wider than the panel, so it ran past the right edge of the
    # figure and was cut mid-word - "consecutive rebala" - in every stage-03 render. How far
    # past depends on how long the feature names are, since those are the panel's y-tick
    # labels and they set how much width is left; anchoring makes it grow into the gap
    # between the panels instead, which no case study can exhaust.
    right.set_xlabel("rank correlation,\nconsecutive rebalances", fontsize=8, ha="right", x=1.0)
    add_message_title(left, title, subtitle=subtitle)
    # The legend is drawn, measured, and only then is the strip it needs reserved. A
    # guess from the row count leaves a band of dead space under one notebook's figure
    # and clips another's, because how tall it is depends on the feature names.
    legend = fig.legend(
        *left.get_legend_handles_labels(),
        fontsize=6,
        ncol=min(3, len(columns)),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
    )
    fig.canvas.draw()
    strip = legend.get_window_extent(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted()
    )
    fig.tight_layout(rect=(0.0, strip.height + 0.02, 1.0, 1.0))
    show_with_alt(fig, alt)
    return None
