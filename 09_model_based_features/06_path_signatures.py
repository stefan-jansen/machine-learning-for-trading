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
# # Path Signatures for Time Series Feature Engineering
#
# **Chapter 9 | Section 9.2**
#
# **Docker image**: `ml4t-py312`
#
# > **Docker required**: this notebook uses `esig`, an x86-only package that is not in the
# > default environment. Run it with:
# > ```bash
# > docker compose --profile py312 run --rm py312 python 09_model_based_features/06_path_signatures.py
# > ```
#
# A window of prices is a sequence, and most features throw the sequence away: a mean, a
# standard deviation and a total return are the same whichever order the days arrived in.
# The **path signature** is a way of summarising a sequence that keeps the order. It
# produces a fixed-length vector from a window of any length, and two windows with the same
# summary statistics but different shapes get different vectors.
#
# **Learning objectives**
#
# - Describe what a path signature computes and why its first level is the total change
#   while its second level records which coordinate moved first.
# - Build the path a signature is computed from, including the time coordinate that makes
#   rising-then-falling distinguishable from falling-then-rising.
# - Choose a truncation depth and a signature form, and say what each costs in the number
#   of features it produces.
# - Compare signature features against hand-built shape features on the same windows,
#   across several assets, and read the comparison without over-reading one split.
#
# **Book reference**
#
# Chapter 9, Section 9.2 (Transforming signals to uncover hidden structure).
#
# **Prerequisites**
#
# `04_kalman_filter` and `05_spectral_features` for the other two ways this chapter
# summarises a sequence. `06_strategy_definition/02_cv_foundations` for why a split
# between overlapping windows needs a gap.

# %% [markdown]
# ## Setup

# %%
"""Path Signatures - sequence features that keep the order of a window."""

import warnings

# The py312 image pairs kaleido 0.2.1 with plotly 6, which announces six deprecations at
# import. They repeat every run, say nothing about the data, and would otherwise land in
# the first cell's output; named by category and by the two modules that raise them.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"(plotly\.io\._kaleido|kaleido\.scopes\.plotly)",
)

import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from IPython.display import display
from ml4t.diagnostic.metrics import pooled_ic
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

from data import load_etfs
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, FIGSIZE, show_with_alt

# %% tags=["parameters"]
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
WINDOW_SIZE = 20
FORECAST_HORIZON = 5
SIGNATURE_DEPTH = 3
SEED = 42

# %%
set_global_seeds(SEED)

try:
    esig = importlib.import_module("esig")
except ImportError as exc:
    raise ImportError(
        "`esig` is not available in the current image.\n"
        "This notebook runs in the `ml4t-py312` image:\n"
        "  docker compose --profile py312 run --rm py312 \\\n"
        "      python 09_model_based_features/06_path_signatures.py"
    ) from exc

# %% [markdown]
# ## What a signature is
#
# Take a path $X$ with $d$ coordinates, running over an interval. Its **signature** is the
# collection of all iterated integrals along it:
#
# $$S(X)^{i_1, \ldots, i_k} = \int_{0 < t_1 < \cdots < t_k < T} dX_{t_1}^{i_1} \cdots dX_{t_k}^{i_k}$$
#
# The definition is short and the first two levels are readable without it.
#
# **Level one** has $d$ terms, one per coordinate, and each is the total change in that
# coordinate over the window. For a price coordinate that is the window's return.
#
# **Level two** has $d^2$ terms. The pair $S^{i,j}$ and its mirror $S^{j,i}$ differ by twice
# the signed area the path sweeps out in the plane of coordinates $i$ and $j$, and the sign
# of that area is the direction the path went round that loop. This is the level that
# carries information no summary statistic has, because a summary statistic of each
# coordinate separately cannot see a loop at all.
#
# **Level three and beyond** carry finer path geometry at a cost that grows as $d^k$.
#
# Two properties matter for using them as features. A signature does not change if the path
# is traversed faster or slower, only if its shape changes, so the features do not depend on
# how many observations the window happened to contain. And different paths give different
# signatures, so nothing is lost by construction; what is lost is what truncation discards.

# %% [markdown]
# ## Depth, and the two forms
#
# | Depth | Terms for a $d$-dimensional path | What the level adds |
# |---|---|---|
# | 1 | $d$ | the total change in each coordinate |
# | 2 | $d^2$ | which coordinate moved first, and by how much |
# | 3 | $d^3$ | the order of three movements, so reversals and acceleration |
#
# `esig` returns the constant level-0 term as well, so a depth-$K$ signature of a
# $d$-dimensional path has $1 + \sum_{k=1}^{K} d^k$ entries. At $d = 3$ and depth 3 that is
# 40; at $d = 5$ it is 156.
#
# The **log-signature** is a smaller set carrying the same information. The full signature
# satisfies algebraic identities, so many of its terms are determined by the others; the
# log-signature keeps one term per independent direction and drops the rest. It is the
# form to use for a model, because the redundant terms add columns without adding anything
# to learn from.

# %% [markdown]
# ## Why the path needs a time coordinate
#
# Without time, a window that rises to a peak and falls back to where it started has the
# same level-one signature as a window that never moved: total change zero, in every
# coordinate. Adding a coordinate that increases steadily from zero to one over the window
# fixes this. The two level-two terms pairing time with price differ by twice the signed
# area between the price path and the straight line from its start to its end. That is an
# integrated displacement from the chord, so it separates a window that spent its time
# above the line from one that spent it below, and its size says how far and for how long.
#
# It does not say when the move happened. A triangular path of a given height that starts
# and ends where it began encloses the same area whether it peaks early or late, so the
# term is the same for both. Timing shows up at level three, which is one reason a depth
# past two is ever worth its columns.
#
# The path built below has three coordinates: time, the price relative to where the window
# opened, and the traded volume relative to where the window opened. Both of the last two
# are normalised inside the window, so nothing outside it enters, and the level-two term
# between them says whether volume moved ahead of price.

# %% [markdown]
# ## Reading one term out of a signature
#
# `esig` returns a depth-two signature as a flat array: the constant one, then the $d$
# level-one terms, then the $d^2$ level-two terms in row-major order. So the term pairing
# coordinate $i$ then $j$ sits at position $1 + d + i\,d + j$, and `esig.sigkeys` prints
# the same layout as labels. One helper below reads a term by its pair of coordinates, and
# every claim about level two in this notebook goes through it.

# %%
TIME, PRICE, VOLUME = 0, 1, 2  # the coordinates of the path built below
PATH_DIMENSION = 3


def level_two(signature: np.ndarray, first: int, second: int, dimension: int) -> float:
    """The level-two term pairing coordinate *first* then *second*."""
    return float(signature[1 + dimension + first * dimension + second])


def signed_area(signature: np.ndarray, first: int, second: int, dimension: int) -> float:
    """Twice the signed area the path sweeps in the plane of the two coordinates.

    The sign is the orientation of the loop the two coordinates trace together, so it
    combines the order of their moves with their directions. Paired with a monotone time
    coordinate as the *second* argument, that reduces to something simpler: positive when
    the first coordinate spent the window above the straight line joining its own endpoints.
    """
    return level_two(signature, first, second, dimension) - level_two(
        signature, second, first, dimension
    )


# %% [markdown]
# ## The data

# %%
SYMBOLS = ["SPY", "QQQ", "IWM", "EFA"]

etfs = (
    load_etfs(symbols=SYMBOLS)
    .filter(pl.col("timestamp") >= pl.lit(START_DATE).str.to_date())
    .filter(pl.col("timestamp") <= pl.lit(END_DATE).str.to_date())
    .sort(["symbol", "timestamp"])
)
print(f"Symbols: {etfs['symbol'].unique().sort().to_list()}")
print(f"Sessions: {etfs['timestamp'].min()} to {etfs['timestamp'].max()}, {etfs.height:,} rows")


# %%
def build_paths(
    frame: pl.DataFrame, window_size: int, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Overlapping windows as (time, price, volume) paths, with each window's forward return.

    Every coordinate is normalised against the first observation *inside* its own window,
    so a path uses that window and nothing else. A statistic taken over the whole sample,
    such as a z-score of volume, would put every later session into every earlier path.
    """
    frame = frame.sort("timestamp").with_columns(
        forward_return=pl.col("close").pct_change(horizon).shift(-horizon)
    )

    closes = frame["close"].to_numpy()
    volumes = frame["volume"].to_numpy().astype(float)
    forward = frame["forward_return"].to_numpy()
    stamps = frame["timestamp"].to_numpy()

    time_axis = np.linspace(0, 1, window_size).reshape(-1, 1)
    paths, targets, ends = [], [], []

    for start in range(len(frame) - window_size - horizon + 1):
        stop = start + window_size
        price = closes[start:stop]
        volume = np.maximum(volumes[start:stop], 1.0)  # a zero-volume session would break the log

        paths.append(
            np.hstack(
                [
                    time_axis,
                    ((price - price[0]) / price[0]).reshape(-1, 1),
                    (np.log(volume) - np.log(volume[0])).reshape(-1, 1),
                ]
            )
        )
        targets.append(forward[stop - 1])
        ends.append(stamps[stop - 1])

    paths, targets, ends = np.array(paths), np.array(targets), np.array(ends)
    usable = ~np.isnan(targets)
    return paths[usable], targets[usable], ends[usable]


spy_paths, spy_targets, spy_ends = build_paths(
    etfs.filter(pl.col("symbol") == "SPY"), WINDOW_SIZE, FORECAST_HORIZON
)
print(f"SPY: {len(spy_paths):,} windows of shape {spy_paths.shape[1:]}")
print(f"Forward returns: mean {spy_targets.mean():.4f}, standard deviation {spy_targets.std():.4f}")

# %% [markdown]
# ## Two paths that end in the same place
#
# The clearest way to see what the signature adds is to draw two windows whose start and
# end agree and whose middles do not, and read their level-one and level-two terms. Both
# are real SPY windows: among those that ended within `FLAT_RETURN` of where they opened,
# these are the two that had travelled furthest in each direction by their midpoint.

# %%
window_returns = np.array(
    [path[-1, 1] for path in spy_paths]
)  # the window return, which is the level-one price term
midpoint_returns = np.array([path[WINDOW_SIZE // 2, 1] for path in spy_paths])

FLAT_RETURN = 0.002  # how close to unchanged a window has to end to qualify

flat = np.where(np.abs(window_returns) < FLAT_RETURN)[0]
# Among those, the two that went furthest in each direction at the midpoint, so the figure
# shows the contrast rather than the first pair that happened to qualify.
example = {
    "Rose then fell": spy_paths[flat[np.argmax(midpoint_returns[flat])]],
    "Fell then rose": spy_paths[flat[np.argmin(midpoint_returns[flat])]],
}

# %%
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["dual_h_tall"], sharey=True)

for ax, (label, path) in zip(axes, example.items()):
    ax.plot(path[:, 0], path[:, 1] * 100, linewidth=1.5, color=COLORS["blue"])
    ax.plot(
        [path[0, 0], path[-1, 0]],
        [path[0, 1] * 100, path[-1, 1] * 100],
        linestyle="--",
        linewidth=1,
        color=COLORS["negative"],
        label="Straight line from open to close",
    )
    ax.set_title(label)
    ax.set_xlabel("Position in the window")
axes[0].set_ylabel("Price relative to the window open, percent")
axes[0].legend(fontsize=7, loc="lower left")

fig.suptitle("Same start, same end, opposite level-two terms")
show_with_alt(
    fig,
    "Two panels showing the price path of one twenty-session window each, measured "
    "relative to where the window opened, on a shared vertical scale. Both start and end "
    "at about zero, marked by a dashed line joining each window's open to its close. The "
    "left path rises several percent above that line through the middle of the window and "
    "comes back; the right path falls further below it and comes back.",
)

# %%
comparison = []
for label, path in example.items():
    depth_two = esig.stream2sig(path, 2)
    comparison.append(
        {
            "window": label,
            "level one, price": float(esig.stream2sig(path, 1)[1 + PRICE]),
            "time then price": level_two(depth_two, TIME, PRICE, PATH_DIMENSION),
            "price then time": level_two(depth_two, PRICE, TIME, PATH_DIMENSION),
            "signed area": signed_area(depth_two, PRICE, TIME, PATH_DIMENSION),
        }
    )
display(pd.DataFrame(comparison))

# %% [markdown]
# The level-one term is near zero for both, which is the whole point: a feature built from
# the window's return cannot tell them apart. The two level-two terms pairing time with
# price swap places between the windows, and their difference, twice the signed area
# between the path and the dashed line in the figure, is positive for the window that
# spent its time above that line and negative for the one that spent it below.

# %% [markdown]
# ## Computing the features


# %%
def signature_features(paths: np.ndarray, depth: int, log_form: bool) -> np.ndarray:
    """Signatures, or log-signatures, one row per path."""
    transform = esig.stream2logsig if log_form else esig.stream2sig
    return np.array([transform(path, depth) for path in paths])


spy_signatures = signature_features(spy_paths, SIGNATURE_DEPTH, log_form=False)
spy_logsignatures = signature_features(spy_paths, SIGNATURE_DEPTH, log_form=True)

print(f"Full signature at depth {SIGNATURE_DEPTH}: {spy_signatures.shape[1]} terms per window")
print(f"Log-signature at depth {SIGNATURE_DEPTH}: {spy_logsignatures.shape[1]} terms per window")

# %% [markdown]
# ## What to compare against
#
# Signatures are worth something only against features a person would otherwise build from
# the same window. The seven below are the obvious ones, and between them they cover the
# return, the variability, the change in direction, how far the window travelled and how
# far it moved from its extremes.


# %%
def shape_features(paths: np.ndarray) -> np.ndarray:
    """Seven hand-built summaries of the same windows, for comparison."""
    rows = []
    for path in paths:
        price = path[:, 1]
        steps = np.diff(price)
        midpoint = len(price) // 2
        rows.append(
            [
                price[-1] - price[0],  # window return
                np.std(steps),  # variability of the daily moves
                (price[-1] - price[midpoint]) - (price[midpoint] - price[0]),  # change of pace
                np.max(price) - price[-1],  # distance below the window high
                price[-1] - np.min(price),  # distance above the window low
                np.mean(price) - np.median(price),  # asymmetry of the visited levels
                np.sum(np.abs(steps)),  # total distance travelled
            ]
        )
    return np.array(rows)


spy_shape = shape_features(spy_paths)
print(f"Hand-built features: {spy_shape.shape[1]} per window")

# %% [markdown]
# ## Splitting overlapping windows
#
# Consecutive windows share all but one of their observations, and every target reaches
# `FORECAST_HORIZON` sessions past its own window. A split made at a single index therefore
# puts windows on the training side whose targets are drawn from sessions on the test side.
# The gap that removes this is the window length plus the horizon, less one, dropped at
# every boundary.

# %%
PURGE = WINDOW_SIZE + FORECAST_HORIZON - 1
TRAIN_SHARE, VALIDATION_SHARE = 0.7, 0.15


def split_indices(n: int) -> dict[str, slice]:
    """Training, validation and test slices with a purge gap at each boundary."""
    train_end = int(TRAIN_SHARE * n)
    validation_start = train_end + PURGE
    validation_end = validation_start + int(VALIDATION_SHARE * n)
    return {
        "train": slice(0, train_end),
        "validation": slice(validation_start, validation_end),
        "test": slice(validation_end + PURGE, n),
    }


spy_splits = split_indices(len(spy_paths))
print(
    "SPY windows: "
    + ", ".join(
        f"{name} {len(range(*part.indices(len(spy_paths)))):,}" for name, part in spy_splits.items()
    )
    + f"; {2 * PURGE} dropped at the two boundaries"
)

# %% [markdown]
# ## The comparison, over four symbols
#
# One split on one symbol produces three numbers with nothing to measure their spread
# against, and the difference between two of them cannot be read. Running the same
# comparison on four symbols gives four paired differences instead, which is still a small
# number and is at least a number with a spread.
#
# The learner is a gradient-boosted tree ensemble with fixed settings, used as an
# instrument for measuring how much the features carry rather than as a model anyone would
# ship. Its tuning is Chapter 12's subject.


# %%
def evaluate(features: np.ndarray, targets: np.ndarray, splits: dict[str, slice]) -> dict:
    """Fit on the training slice; report fit and rank agreement on the test slice."""
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=SEED
    )
    model.fit(features[splits["train"]], targets[splits["train"]])
    predictions = model.predict(features[splits["test"]])
    return {
        "train_r2": r2_score(targets[splits["train"]], model.predict(features[splits["train"]])),
        "test_r2": r2_score(targets[splits["test"]], predictions),
        "test_ic": pooled_ic(predictions, targets[splits["test"]]),
        "model": model,
    }


# %%
FEATURE_SETS = ["hand-built", "log-signature", "both"]

per_symbol = []
fitted = {}
for symbol in SYMBOLS:
    paths, targets, _ = build_paths(
        etfs.filter(pl.col("symbol") == symbol), WINDOW_SIZE, FORECAST_HORIZON
    )
    hand = shape_features(paths)
    signature = signature_features(paths, SIGNATURE_DEPTH, log_form=True)
    splits = split_indices(len(paths))

    for name, matrix in zip(FEATURE_SETS, [hand, signature, np.hstack([hand, signature])]):
        outcome = evaluate(matrix, targets, splits)
        per_symbol.append(
            {
                "symbol": symbol,
                "features": name,
                "train R2": outcome["train_r2"],
                "test R2": outcome["test_r2"],
                "test IC": outcome["test_ic"],
            }
        )
        fitted[(symbol, name)] = outcome["model"]

results = pd.DataFrame(per_symbol)
display(results.pivot(index="symbol", columns="features", values="test IC").round(4))

# %%
paired = results.pivot(index="symbol", columns="features", values="test IC")
difference = paired["both"] - paired["hand-built"]
print("Test IC, both feature sets minus the hand-built ones alone:")
print(f"  per symbol: {', '.join(f'{s} {d:+.4f}' for s, d in difference.items())}")
print(
    f"  mean {difference.mean():+.4f}, standard deviation of the differences {difference.std():.4f}"
)
print(
    f"  symbols where adding signatures raised the IC: {(difference > 0).sum()} of {len(difference)}"
)

# %% [markdown]
# Read the last line before the mean. What the comparison establishes is how often adding
# signature terms moved the rank agreement up, over four symbols and one split each, and
# the spread of the differences says how far a single symbol's answer can sit from the
# average of the four. It does not establish a general ranking: four paired observations
# with this much spread would not separate two feature sets even if the comparison were
# designed to, and it was not, because each symbol contributes one split rather than a
# distribution over splits.
#
# The fit statistics say the other half of it. Out-of-sample $R^2$ near or below zero
# across every symbol and every feature set is the normal state of daily return
# prediction, and it is why rank agreement rather than fit is the quantity being read.

# %%
display(results.pivot(index="symbol", columns="features", values="test R2").round(4))

# %% [markdown]
# ## Where the model looks
#
# Impurity importance is the total reduction in the split criterion that a column achieved,
# weighted by how many samples reached each split. It is not a count of splits: a column
# used once at the root can outscore one used ten times deep in the trees. It is a statement
# about this fitted model and these windows, not about which features cause returns, and a
# column can score low because another column carries the same information.

# %%
combined_model = fitted[("SPY", "both")]
n_hand = spy_shape.shape[1]
importances = combined_model.feature_importances_
labels = [f"hand {i}" for i in range(n_hand)] + [
    f"signature {i}" for i in range(spy_logsignatures.shape[1])
]

fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["dual_h_tall"])

ax = axes[0]
ax.bar(
    ["Hand-built", "Log-signature"],
    [importances[:n_hand].sum(), importances[n_hand:].sum()],
    color=[COLORS["blue"], COLORS["copper"]],
)
ax.set_ylabel("Total impurity reduction")
ax.set_title("Share of the impurity reduction by family")

ax = axes[1]
TOP_TERMS = 15
order = np.argsort(importances)[-TOP_TERMS:]
ax.barh(
    [labels[i] for i in order],
    importances[order],
    color=[COLORS["blue"] if i < n_hand else COLORS["copper"] for i in order],
)
ax.set_xlabel("Impurity reduction")
ax.set_title(f"The {TOP_TERMS} highest-importance columns")
ax.tick_params(axis="y", labelsize=6)

fig.suptitle("Both families are used, and neither dominates the importance")
per_column = {
    "hand-built": importances[:n_hand].sum() / n_hand,
    "log-signature": importances[n_hand:].sum() / (len(importances) - n_hand),
}
show_with_alt(
    fig,
    "Two panels for the SPY model fitted on both feature families. The left panel has two "
    "bars, one per family, of comparable height with the signature family somewhat taller. "
    "The right panel is a horizontal bar chart of the fifteen highest-importance columns, "
    "with hand-built and signature columns interleaved rather than separated.",
)

# %%
print(
    "Mean impurity reduction per column: "
    + ", ".join(f"{name} {value:.4f}" for name, value in per_column.items())
)

# %% [markdown]
# Read the left panel against the column counts before reading it as a ranking. The
# log-signature family contributes twice as many columns as the hand-built one, so a larger
# share of the total is partly a statement about how many columns it brought. The mean per
# column above is the comparison that does not have the count in it.

# %% [markdown]
# ## Across assets, not just across time
#
# The same construction applied to several assets at once puts each asset's cumulative
# return in its own coordinate. Level two then pairs two assets, and the same signed area
# describes the loop the pair traces in the plane of their two cumulative returns.
#
# Its sign is not simply which asset moved first. Orientation combines the order of the two
# moves with their directions: a window in which the first asset falls and then the second
# rises traces the loop the opposite way round from one in which both rise, order
# unchanged. The four cases below make that concrete, and they are why the count reported
# afterwards is a count of positive areas rather than a count of windows one asset led.

# %%
CASE_STEPS = 40
case_time = np.linspace(0, 1, CASE_STEPS).reshape(-1, 1)


def two_step_path(first_move: float, second_move: float) -> np.ndarray:
    """A path where coordinate one moves over the first half and coordinate two over the second."""
    half = CASE_STEPS // 2
    one, two = np.zeros(CASE_STEPS), np.zeros(CASE_STEPS)
    one[:half] = np.linspace(0, first_move, half)
    one[half:] = first_move
    two[half:] = np.linspace(0, second_move, CASE_STEPS - half)
    return np.hstack([case_time, one.reshape(-1, 1), two.reshape(-1, 1)])


display(
    pd.DataFrame(
        [
            {
                "first coordinate moves": "up" if a > 0 else "down",
                "second coordinate moves": "up" if b > 0 else "down",
                "order": "first, then second",
                "signed area": signed_area(
                    esig.stream2sig(two_step_path(a, b), 2), 1, 2, PATH_DIMENSION
                ),
            }
            for a in (0.05, -0.05)
            for b in (0.05, -0.05)
        ]
    )
)


# %%
def cross_asset_paths(frame: pl.DataFrame, symbols: list[str], window_size: int):
    """Windows whose coordinates are time and one cumulative return per symbol."""
    wide = (
        frame.pivot(on="symbol", index="timestamp", values="close").sort("timestamp").drop_nulls()
    )
    wide = wide.with_columns(
        [(pl.col(symbol).pct_change()).alias(f"{symbol}_return") for symbol in symbols]
    ).drop_nulls()

    returns = wide.select([f"{symbol}_return" for symbol in symbols]).to_numpy()
    stamps = wide["timestamp"].to_numpy()
    time_axis = np.linspace(0, 1, window_size).reshape(-1, 1)

    paths, ends = [], []
    for start in range(len(wide) - window_size + 1):
        block = returns[start : start + window_size]
        paths.append(np.hstack([time_axis, np.cumsum(block, axis=0)]))
        ends.append(stamps[start + window_size - 1])
    return np.array(paths), np.array(ends)


CROSS_ASSET_DEPTH = 2  # level two is where the lead-lag term lives, and d is larger here

cross_paths, cross_ends = cross_asset_paths(etfs, SYMBOLS, WINDOW_SIZE)
print(f"Cross-asset windows: {len(cross_paths):,}, path dimension {cross_paths.shape[2]}")

# %%
cross_dimension = cross_paths.shape[2]
# Coordinate 0 is time, so each symbol's coordinate is its position in SYMBOLS plus one.
LEADER, FOLLOWER = SYMBOLS.index("QQQ") + 1, SYMBOLS.index("SPY") + 1

lead_lag = np.array(
    [
        signed_area(esig.stream2sig(path, CROSS_ASSET_DEPTH), LEADER, FOLLOWER, cross_dimension)
        for path in cross_paths
    ]
)

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.plot(cross_ends, lead_lag, linewidth=0.6, color=COLORS["blue"])
ax.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.5)
ax.set_ylabel("Signed area, QQQ against SPY")
ax.set_xlabel("Session the window ends")
ax.set_title("The signed area changes sign constantly and has no persistent direction")
show_with_alt(
    fig,
    "A single series over the whole sample showing the signed area between the QQQ and SPY "
    "cumulative return paths inside each twenty-session window. It oscillates around zero "
    "with no visible trend, with its widest swings in early 2020.",
)

# %%
print(f"Windows with a positive signed area: {(lead_lag > 0).sum():,} of {len(lead_lag):,}")
print(f"Mean signed area: {lead_lag.mean():+.5f}, standard deviation {lead_lag.std():.5f}")

# %% [markdown]
# The areas split almost exactly evenly and average to nearly nothing, which is what two
# broad equity indices that move together should produce: their paths cross and recross
# rather than tracing a loop in one direction. The term is a conditioning input that
# changes sign window by window, and reading a single window's sign as "QQQ led" requires
# knowing the direction both assets moved, which the sign alone does not carry.

# %% [markdown]
# ## When a signature is worth the trouble
#
# **The shape of the path has to matter.** Signatures separate windows by geometry, so they
# pay where geometry is the signal: execution timing inside a session, hedging flows that
# produce a characteristic intraday shape, the difference between a slow trend and a sharp
# reversal that ends in the same place.
#
# **The window has to be long enough to have a shape and short enough to have one shape.**
# Twenty daily sessions is at the short end. Intraday windows are where the method is
# usually applied.
#
# **The cost is in the dimension, not the length.** Signature terms grow as $d^K$ in the
# number of coordinates, and hardly at all in the number of observations. Adding an asset
# to a path is expensive; adding observations is nearly free, and reparameterisation
# invariance means unevenly spaced observations are not a problem.
#
# **Start at depth two in the log form.** Depth three multiplies the columns for terms
# whose meaning is hard to state, and a model with more columns than it can use is the
# usual outcome.
#
# **What signatures do not fix.** If the signal is in the level or the total return, a
# signature is a longer way of writing a number you already had, and its individual terms
# are much harder to explain to anyone than a moving average.

# %% [markdown]
# ## The features this notebook produces
#
# | Column group | What it holds | Causal |
# |---|---|---|
# | `log-signature` terms | the shape of the trailing window, normalised inside it | yes |
# | level-two time-price term | how far and how long the path sat off its own chord | yes |
# | level-two price-volume term | the orientation of the loop price and volume traced | yes |
# | cross-asset level-two term | the orientation of the loop two assets traced together | yes |
#
# Each is computed from a trailing window whose every coordinate is normalised against that
# window's own first observation, so a value stamped on a session is computable from that
# session's close.

# %% [markdown]
# ## Key takeaways
#
# 1. **A signature keeps the joint geometry a summary statistic throws away.** Its first
#    level is the total change in each coordinate, which any feature set already has; its
#    second level is the signed area between pairs of coordinates, which nothing computed
#    coordinate by coordinate can reach.
# 2. **The time coordinate is not optional.** Without it, a window that rose and fell back
#    is indistinguishable from one that never moved.
# 3. **Use the log form.** It carries the same information in far fewer columns, because
#    the full signature's terms satisfy identities that make most of them redundant.
# 4. **Depth costs columns exponentially in the number of coordinates.** Two coordinates
#    and depth three is cheap; five coordinates and depth three is 156 columns per window.
# 5. **A single split on a single symbol cannot rank two feature sets.** Four symbols give
#    four paired differences and a spread to read them against, which is enough to say how
#    often one set helped and not enough to say that it does.
#
# **Known limitations.** The comparison uses one fixed learner with fixed settings, so it
# measures what that learner can extract rather than what the features contain. Four
# symbols from one asset class and one split each is a small and correlated sample. The
# windows overlap, so neither the fit statistics nor the rank agreement carries the
# independence a significance test would need. And no cost, turnover or capacity enters
# any of it, so nothing here says a difference in rank agreement is worth trading.
#
# **Previous**: `04_kalman_filter` and `05_spectral_features` for the other two summaries
# of a sequence.
# **Next**: `07_arima_features`, which fits a model to the window instead of transforming
# it.
