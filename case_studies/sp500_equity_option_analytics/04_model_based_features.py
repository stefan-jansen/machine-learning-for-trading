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
# # S&P 500 Equity + Option Analytics: Model-Based Features
#
# Every feature in [`03_financial_features`](03_financial_features.ipynb) is a rule written in
# advance: a window, a difference, a rank. The feature built here is not. It is the output of a
# model whose parameters are **estimated from the data**, and that changes what has to be checked.
# A rule cannot leak by being fitted, because there is nothing to fit. A fitted parameter can, and
# the only thing that decides whether it did is which rows the estimation window contained.
#
# The model is a **GJR-GARCH(1,1)**, an equation for how a share's variance evolves from one
# session to the next. What it gives back is a **conditional volatility**: how much the share is
# expected to move over a session, worked out from what was known at the close of the session
# before it. Following the usual convention, the value is stamped on the session it describes, so
# `garch_cond_vol` at date $t$ is the square root of the variance the model gives session $t$ from
# the return and variance at $t-1$, annualized, and the recursion producing it reads nothing dated
# $t$ or later.
#
# **The recursion is only half of what decides whether a value is a forecast, and section C.1 is
# exact about the other half.** The parameters it runs on were estimated over a window, and on the
# dates inside that window a value is a retrospective transform rather than something that could
# have been computed on the day. On the dates after it the parameters predate the value entirely,
# which is why section F scores those dates alone and F2 draws them alone.
#
# That is the difference this notebook is built on. Stage 03's variance risk premium subtracts a
# *realized* volatility - an average of the last twenty sessions - from the option market's
# implied volatility, so it compares a forecast against a memory. Replacing the second half with
# a forecast makes both halves forecasts.
#
# ## Learning objectives
#
# - Re-estimate a volatility model on a fixed cadence over a share's own history, so that every
#   emitted value carries parameters fitted strictly before the session it speaks for
# - Declare that schedule where the feature is defined rather than inside the notebook, and assert
#   that an emitted value does not move when later observations are deleted rather than saying so
# - Pin every quantity the recursion starts from - the scaling, the starting variance and the
#   bounds it is clipped to - to the estimation window, because a library that recomputes them
#   over whatever series it is handed will read later data through them
# - Rebuild the option market's variance premium against a forecast rather than against a memory,
#   and test whether the change moves what the premium says about future returns
# - Test a difference between two dependent measurements directly, instead of inferring it from
#   two tests that were each about something else
#
# ## Book reference, prerequisites and artifacts
#
# Chapter 9, Section 9.3. Reads the daily share bars through `load_sp500_daily_bars()`, the
# implied volatility level and the variance risk premium from
# [`03_financial_features`](03_financial_features.ipynb)'s `features/financial.parquet`, and the
# primary label from [`02_labels`](02_labels.ipynb). Writes
# `features/model_based.parquet` with a `.digest.json` sidecar, read by every downstream model
# notebook through `load_modeling_dataset`, and by [`05_evaluation`](05_evaluation.ipynb).
# [`08_garch_volatility`](../../09_model_based_features/08_garch_volatility.ipynb) introduces the
# model itself.

# %%
"""S&P 500 Equity + Option Analytics: Model-Based Features."""

import warnings
from datetime import date

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from arch import arch_model
from ml4t.diagnostic.evaluation.stats import benjamini_hochberg_fdr
from ml4t.diagnostic.metrics import compute_ic_hac_stats, cross_sectional_ic_series

from case_studies.utils.artifact_digest import read_digest, value_digest
from case_studies.utils.cv_window import fold_boundary_date, modeling_fold_boundaries
from case_studies.utils.temporal import (
    garch11_conditional_volatility,
    refit_boundaries,
    walk_forward_feature,
    write_model_based,
)
from data import load_sp500_daily_bars, load_sp500_options_surface
from utils.artifact_specs import load_setup_config, resolve_label_horizon
from utils.cv_splits import load_evaluation_config
from utils.data_quality import top_entities
from utils.paths import display_path, get_case_study_dir
from utils.style import ml4t_palette, show_plotly_with_alt

# %% [markdown]
# `MIN_OBS` is the shortest estimation window a fit is attempted on. A GJR-GARCH has four
# parameters and its likelihood is flat in the asymmetry term unless the window contains enough
# down sessions to identify it, so a year of sessions is the floor below which the notebook
# declines to fit rather than returning an unstable estimate. `MAX_SYMBOLS` caps the universe;
# `None` runs all of it. Both are read below and neither is re-assigned outside this cell, so the
# value a test injects is the value the notebook runs on.

# %% tags=["parameters"]
CASE_STUDY_ID = "sp500_equity_option_analytics"
MIN_OBS = 252
MAX_SYMBOLS = None
START_DATE = "2017-01-01"
END_DATE = "2021-12-31"

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
FEATURES_DIR = CASE_DIR / "features"
PANEL_KEY = ["timestamp", "symbol"]
FINANCIAL_PATH = FEATURES_DIR / "financial.parquet"
LABELS_DIR = CASE_DIR / "labels"

# %% [markdown]
# ## Configuration
#
# Three things are bound here and each decides something below: the label the section F diagnostic
# scores against and how far ahead it resolves, the refit schedule the estimation windows come
# from, and the date the holdout begins.
#
# **The schedule is declared in `config/setup.yaml`, not here.** An estimation window is part of a
# fitted feature's definition, so it lives where the definition lives. `model_based.gjr_garch`
# gives the burn-in and the refit cadence, and the comments there say what each decides and what
# it costs.
#
# **The folds are read for one purpose and it is presentational.** `modeling_fold_boundaries`
# still supplies the validation spans that section E's chart rules mark and that section F scores
# over, because a reader comparing this feature against a downstream model wants the same dates.
# No parameter is estimated per fold and no emitted value depends on one - that is the whole
# subject of section B.

# %%
SETUP = load_setup_config(CASE_STUDY_ID)
PRIMARY_LABEL = SETUP["labels"]["primary"]
# The outcome horizon, not the fold buffer: `labels.buffer` is 10D here because the purge has to
# clear the longest variant, while this label resolves in 5 sessions. The horizon is what sets
# section F's HAC bandwidth, because it is what makes consecutive daily IC values overlap.
_horizon = resolve_label_horizon(CASE_STUDY_ID, PRIMARY_LABEL, SETUP)
if not _horizon or not str(_horizon).endswith("D"):
    raise ValueError(f"Expected a daily label horizon for {PRIMARY_LABEL}, found {_horizon!r}")
LABEL_HORIZON = int(str(_horizon)[:-1])
LABEL_VARIANTS = [PRIMARY_LABEL, *SETUP["labels"].get("variants", [])]

canonical_splits = modeling_fold_boundaries(CASE_STUDY_ID, PRIMARY_LABEL)
if not canonical_splits:
    raise RuntimeError(f"Canonical {PRIMARY_LABEL} modeling folds are unavailable")
evaluation_config = load_evaluation_config(CASE_STUDY_ID)
HOLDOUT_START = fold_boundary_date(evaluation_config["holdout_start"])
HOLDOUT_END = fold_boundary_date(evaluation_config["holdout_end"])

_schedule = SETUP["model_based"]["gjr_garch"]
GARCH_BURNIN = int(_schedule["burnin"])
GARCH_REFIT_EVERY = int(_schedule["refit_every"])

print(
    f"{len(canonical_splits)} walk-forward folds come from the {PRIMARY_LABEL} label file. "
    "They mark the spans sections E and F report over; no parameter is fitted per fold."
)
print(
    f"GJR-GARCH schedule: {GARCH_BURNIN}-session burn-in, then re-estimated every "
    f"{GARCH_REFIT_EVERY} sessions on that security's whole history to date."
)
print(
    f"That label resolves {LABEL_HORIZON} sessions after the decision, so consecutive daily "
    f"scores in section F share {LABEL_HORIZON - 1} sessions of outcome and the standard errors "
    "there are corrected for it."
)
print(
    f"A fit is attempted only where the estimation window holds at least {MIN_OBS} sessions of "
    "returns."
)
print(f"Everything from {HOLDOUT_START} to {HOLDOUT_END} is the holdout.")

# %% [markdown]
# ## A. Why a fitted feature is different
#
# A feature from stage 03 is a function of past prices. Ask for the 20-session realized volatility
# at some date and the answer depends on twenty numbers and nothing else; run the calculation on a
# longer history and the value at that date does not move.
#
# A model-based feature is a function of *parameters estimated from* prices, and the parameters
# depend on every row in the estimation window. So the estimation window is part of the feature's
# information set, in a way that no window in stage 03 was. If the window used to estimate the
# parameters reaches past the date the feature is stamped with, then the feature at that date was
# computed with knowledge of what happened afterwards - and it will look far better than it can
# be, because the model was told the answer.
#
# The discipline that removes it has three parts, and section B, section C and section E carry one
# each: declare the refit schedule before any model runs, estimate on the sessions before each
# block and never on the block itself, and emit one value per date and security so there is
# nothing left for a consumer to join wrongly.
#
# There is one more subtlety, and it is the one a careless implementation leaves in place. Fixing
# the *parameters* is not enough. A recursion also needs a value to start from and a range to stay
# inside, and a library asked to run a recursion will derive both from whatever series it is
# handed. Section C measures what that costs here.

# %% [markdown]
# ## B. The schedule contract
#
# The parameters are re-estimated on a fixed cadence over each security's own history, and every
# emitted value comes from the last estimate made strictly before it. There is no estimation
# window per fold and no fold column on the artifact: a session gets one conditional volatility,
# and whichever fold selects that session later reads the same number.
#
# **The burn-in is paid out of the run-up, not out of training data.** seoa's panel opens
# 2017-01-03 and the oldest fold's training window opens 2018-01-04, so about 250 sessions
# precede the first fold. The declared burn-in of 252 is very nearly that, which is why it is
# 252 and not the 504 the longer-history case studies get - `config/setup.yaml::model_based`
# carries the measurement behind that choice.
#
# **The holdout is covered without being estimated on.** `freeze_after` stops re-estimation at
# the last session before `HOLDOUT_START`, so holdout sessions receive values from parameters
# fitted wholly before the seal. That is sound for the reason this stage rests on throughout:
# the model reads prices, not labels.
#
# **One schedule serves all five labels.** The old arrangement needed an argument that the
# primary label's folds were usable for the other four, because the estimation windows were the
# folds. A cadence over a security's own history has no fold in it, so that argument is no
# longer needed and the assertion that carried it is gone with the thing it protected.

# %%
# The block structure, on a security holding the full panel. `refit_boundaries` returns
# (fit_end_exclusive, emit_end_exclusive) pairs: the first block spends the burn-in, and each
# subsequent block is emitted from parameters fitted on everything before it.
_full_history = 1259
_blocks = refit_boundaries(_full_history, GARCH_BURNIN, GARCH_REFIT_EVERY)
print(f"A security with the full {_full_history}-session panel is fitted {len(_blocks)} times.")
print(f"  Burn-in, no value emitted: sessions 0 to {GARCH_BURNIN - 1}")
print(f"  First emitted session: {GARCH_BURNIN}")
print(f"  Sessions per block after the burn-in: {GARCH_REFIT_EVERY}")
pl.DataFrame(
    {
        "block": list(range(len(_blocks))),
        "fitted on sessions before": [b[0] for b in _blocks],
        "emits through session": [b[1] for b in _blocks],
    }
).head(8)

# %% [markdown]
# The spans below are the primary label's validation windows. Nothing is estimated per fold any
# more, so they decide no parameter; sections E and F report over them so that a reader can line
# this feature up against the downstream models that are scored on the same dates.

# %%
REPORTING_SPANS = [
    {
        "fold": s["fold"],
        "infer_start": fold_boundary_date(s["val_start"]),
        "infer_end": fold_boundary_date(s["val_end"]),
    }
    for s in canonical_splits
]
CHRONOLOGICAL = sorted(REPORTING_SPANS, key=lambda row: row["infer_start"])
print(
    f"{len(REPORTING_SPANS)} validation spans, {CHRONOLOGICAL[0]['infer_start']} to "
    f"{CHRONOLOGICAL[-1]['infer_end']}, and the holdout runs {HOLDOUT_START} to {HOLDOUT_END}."
)
pl.DataFrame(CHRONOLOGICAL)

# %% [markdown]
# ### F1. The schedule contract
#
# One row per refit block, for the first several blocks of a security holding the full panel. The
# dark bar is what the parameters were estimated on and the gold bar is the sessions those
# parameters emit. Every dark bar ends exactly where its own gold bar begins and never reaches
# past it, which is the property the whole revision is for. The dark bars grow because the walk
# expands: each fit sees everything before it, not a fixed window.

# %%
fit_colour, emit_colour = ml4t_palette(2, categorical=True)
_shown = _blocks[: min(8, len(_blocks))]
fig = go.Figure()
for row_index, (fit_end, emit_end) in enumerate(_shown):
    label = f"block {row_index}"
    emit_start = GARCH_BURNIN if row_index == 0 else _shown[row_index - 1][1]
    for span, colour, name in (
        ((0, fit_end), fit_colour, "estimated on"),
        ((emit_start, emit_end), emit_colour, "emits"),
    ):
        fig.add_trace(
            go.Scatter(
                x=list(span),
                y=[label, label],
                mode="lines",
                line=dict(color=colour, width=16),
                name=name,
                showlegend=row_index == 0,
            )
        )
fig.add_vline(x=GARCH_BURNIN, line_dash="dash", line_color=ml4t_palette(3)[2])
fig.update_layout(
    title="No parameter comes from the right of the sessions it speaks for",
    xaxis_title=f"Session index within the security; the dashed rule marks the "
    f"{GARCH_BURNIN}-session burn-in",
    height=340,
)
fig.update_yaxes(autorange="reversed")
show_plotly_with_alt(
    fig,
    "Timeline with one row per refit block, earliest at the top, drawn against session index "
    "within a single security rather than calendar dates. Each row carries a dark estimation bar "
    "starting at session zero and a gold emission bar immediately to its right. The dark bars "
    "lengthen row by row because the walk expands, while the gold bars stay the same short "
    "length, one refit interval each. A dashed vertical rule marks the end of the burn-in, where "
    "the first gold bar begins. No dark bar extends past the left edge of its own gold bar.",
)

# %% [markdown]
# ## C. The conditional volatility model
#
# GJR-GARCH(1,1) says next session's variance is a weighted sum of three things: a constant, how
# large the last session's move was, and what the variance already was.
#
# $$\sigma^2_t = \omega + (\alpha + \gamma \mathbb{1}_{r_{t-1}<0}) r^2_{t-1} + \beta \sigma^2_{t-1}$$
#
# Read the subscripts: $\sigma^2_t$ is built from $r_{t-1}$ and $\sigma^2_{t-1}$ and from nothing
# dated $t$ or later, which is what makes it a forecast of session $t$ rather than a description
# of it. Every column below inherits that. `garch_ivrv_spread` at $t$ compares the option market's
# implied volatility with a forecast of $t$ made before $t$ opened, and `garch_vol_surprise` at
# $t$ divides what the session actually did by what had been predicted for it - which is only a
# surprise because the denominator was fixed in advance.
#
# The $\gamma$ term is what the GJR variant adds to a plain GARCH: it lets a *negative* return
# raise the variance by more than a positive return of the same size. That asymmetry is well
# documented for equities and it is the reason the variant is worth the extra parameter here
# (Glosten, Jagannathan and Runkle, 1993). $\alpha + \gamma/2 + \beta$ is the model's
# **persistence** - how much of today's variance is still present tomorrow - and section D shows
# what it comes out at.
#
# Returns are multiplied by a hundred before fitting. The optimizer works on the variance of the
# series it is handed, and a daily equity return is a fraction of a percent whose square is four
# orders of magnitude smaller again; on that scale $\omega$ is small enough that the likelihood is
# flat in it to the optimizer's tolerance. Working in percent puts every parameter within a couple
# of orders of magnitude of one.
#
# ### C.1 Fit on the estimation window, then run forward without re-estimating
#
# The returns are taken on the **adjusted** close, `close` multiplied by `adj_factor`, for the
# reason `03_financial_features` gives at length: a four-for-one split reads as a three-quarter
# loss on the printed close, and a single such session is a squared return four orders of
# magnitude above anything else in the window. It would move $\omega$, it would be scored as an
# enormous surprise by the third feature below, and neither would be about the share.
#
# **Fixing the parameters is not enough on its own, and this is where a convenience method costs
# a value.** `arch`'s `fix` runs the recursion with parameters you supply, but it derives two
# other quantities from the series it is handed: the value the recursion starts from, and a pair
# of bounds every variance is clipped to. Handed the estimation window and the inference span
# together, it derives both from all of it - so a variance at a date inside the estimation window
# depends on returns that came after it, through the clipping range.
#
# So the recursion is run by `garch11_conditional_volatility`, the shared helper every case
# study that fits a volatility model here calls, and the three things it needs beyond the
# parameters - the scaling, the starting variance and the bounds - are all derived from the
# estimation window alone, using `arch`'s own functions for each. The recursion itself reads
# only $r_{t-1}$ and $\sigma^2_{t-1}$, so once it has started it never reads forward.
#
# **Every emitted value is now the same kind of quantity, and that is the change.** Under the
# refit schedule the parameters, the starting variance and the bounds behind a value all come
# from sessions strictly before the block that value sits in - on a training row exactly as much
# as on a validation row. There is no longer a retrospective half: the old design emitted values
# across the estimation window itself, and those carried parameters fitted from up to two years
# of their own future, which is precisely what made a training row and a validation row
# incomparable.
#
# The burn-in is where that guarantee is paid for. A security's first `burnin` sessions carry no
# value at all, because there is nothing to fit on yet, and section E reports how many securities
# that silences.
#
# C.2 asserts the property rather than describing it, by deleting later observations and checking
# that no retained value moves.


# %%
def training_filter_state(result, train_returns: pd.Series) -> tuple[float, float, tuple]:
    """Scaling, starting variance and clipping bounds, from the estimation window alone."""
    scale = float(result.model.scale)
    mu = float(result.params.get("mu", result.params.get("Const", 0.0)))
    residuals = train_returns.to_numpy(dtype=float) * scale - mu
    backcast = float(result.model.volatility.backcast(residuals))
    bounds = result.model.volatility.variance_bounds(residuals)
    return scale, backcast, (float(bounds[:, 0].min()), float(bounds[:, 1].max()))


# %% [markdown]
# One walk per security over that security's whole return history. `walk_forward_feature` spends
# the burn-in, fits on everything before the block it is about to emit, hands those parameters to
# `garch11_conditional_volatility`, and keeps only the block's own rows. `freeze_after` is the count of
# sessions strictly before the holdout, so the last estimate made is fitted on every pre-holdout
# session and on no holdout session.
#
# `fit` returns the four things the recursion needs and every one of them comes from the
# estimation prefix alone: the parameters, `arch`'s scaling, the backcast that seeds the first
# variance, and the clipping range. Deriving the last two from the emitted block instead is
# exactly the leak section C measures, so they are taken here and passed through.
#
# **A degenerate fit is recorded, not silently dropped.** A GJR-GARCH estimated on a short window
# can come back with `alpha + gamma < 0`, which says a larger down-shock *lowers* next session's
# variance. That is not a volatility model, and on this universe it is not rare: 19.0% of
# securities on their first block against 1.8% of fits across the whole walk, which section C.3
# reports. The recursion still runs - the clipping range keeps it on the positive reals - and the
# rate is published rather than hidden, because the burn-in that produces it is a declared
# schedule choice a reader is entitled to weigh.


# %%
ANNUALIZE = SETUP["evaluation"]["periods_per_year"] ** 0.5
# The specification is read, not written here, for the reason the schedule is: `o=1` is this case
# study's declared deviation from the shared GARCH(1,1)-Normal default, and a deviation stated in
# a notebook is one a reader has to find by reading the notebook. `config/setup.yaml::model_based`
# carries it and says what property of the data justifies it.
GARCH_KW = {k: _schedule[k] for k in ("mean", "vol", "p", "o", "q", "dist")}


def garch_walk(returns: pd.Series, freeze_after: int | None) -> tuple[pd.Series, list[dict]]:
    """Filter one security on the refit schedule; return the path and one record per fit."""
    diagnostics: list[dict] = []
    observations = returns.to_numpy(dtype=float).reshape(-1, 1)

    def fit(train: np.ndarray) -> dict:
        # The walk expands, so the prefix always starts at this security's first return and its
        # length is the index one past the last session the fit is allowed to see.
        train_returns = returns.iloc[: len(train)]
        result = arch_model(train_returns, **GARCH_KW).fit(disp="off", show_warning=False)
        scale, backcast, bounds = training_filter_state(result, train_returns)
        alpha = float(result.params.get("alpha[1]", 0.0))
        gamma = float(result.params.get("gamma[1]", 0.0))
        beta = float(result.params.get("beta[1]", 0.0))
        diagnostics.append(
            {
                # The first session these parameters speak for. The walk schedules on the
                # security's own observations, so a security that lists late refits on different
                # dates than its neighbours and this is the only honest key for section D.
                "emit_start": returns.index[len(train)],
                "n_train": len(train),
                "alpha": alpha,
                "gamma": gamma,
                "beta": beta,
                "persistence": alpha + gamma / 2 + beta,
                "degenerate": alpha + gamma < 0,
            }
        )
        return {
            "mu": float(result.params.get("mu", result.params.get("Const", 0.0))),
            "omega": float(result.params["omega"]),
            "alpha": alpha,
            "gamma": gamma,
            "beta": beta,
            "scale": scale,
            "backcast": backcast,
            "bounds": bounds,
        }

    def apply(model: dict, prefix: np.ndarray) -> np.ndarray:
        # `arch` estimates on returns it has scaled, so mu, omega and the bounds are all in the
        # scaled units and the recursion has to run there too. The volatility comes back divided
        # by the same factor, which returns it to percent, and the annualization follows.
        scale = model["scale"]
        sigma = (
            garch11_conditional_volatility(
                prefix[:, 0] * scale,
                mu=model["mu"],
                omega=model["omega"],
                alpha=model["alpha"],
                gamma=model["gamma"],
                beta=model["beta"],
                backcast=model["backcast"],
                bounds=model["bounds"],
            )
            / scale
        )
        return (sigma * ANNUALIZE / 100.0).reshape(-1, 1)

    values = walk_forward_feature(
        observations,
        timestamps=returns.index.to_numpy(),
        burnin=GARCH_BURNIN,
        refit_every=GARCH_REFIT_EVERY,
        fit=fit,
        apply=apply,
        n_features=1,
        window=None,
        freeze_after=freeze_after,
        on_fit_error="skip",
    )
    return pd.Series(values[:, 0], index=returns.index), diagnostics


def count_before(index: pd.Index, boundary: pd.Timestamp) -> int:
    """How many entries fall strictly before *boundary*.

    This is what ``walk_forward_feature`` wants for ``freeze_after``: the largest exclusive fit
    end it may use, so the last estimate before the seal sees every session before it and none
    after.
    """
    return int(index.searchsorted(boundary, side="left"))


# %% [markdown]
# The securities and their returns.
#
# **The series a model is estimated on is one security's, not one ticker's.** A ticker is
# reassigned after a merger or a spin-off and `adj_factor` restarts with the new security, so a
# return taken across the changeover divides one company's price by another's - and a single such
# session is a return large enough to move every parameter estimated from a window containing it.
# `03_financial_features` takes every one of its windows inside `sec_id` for the same reason, and
# this notebook does the same. The output is written on the ticker, because that is the key the
# downstream join uses, so the security is what the model is estimated inside and the ticker is
# what the result is stamped with.
#
# A row is emitted for a security in a given fold when the estimation window holds enough sessions
# and the optimizer returns; the count of those is reported below, because a security missing from
# one fold's rows is a security a downstream model has no conditional volatility for on those
# dates, and that is worth knowing before it turns up as a gap.
#
# The swap from security to ticker happens on an inner join against the bar panel, so every
# emitted value is named by the ticker its security actually traded under on that session. The
# row-count assertion beside it is what would catch a filtered value with no bar to name it,
# which would be a value nothing downstream could reach.
#
# **The universe is bounded before any fit, and the bound is the one `setup.yaml::universe`
# declares.** `eligibility_rule` is `sp500_with_options`, so a name qualifies by carrying an
# option surface. The share-bar extract is wider than that and the loader takes a `symbols=`
# argument nothing was passing, so this artifact carried a fitted conditional volatility for
# names the strategy can never hold - they have no implied volatility to rank on, so
# `garch_ivrv_spread` was null for every one of their rows while the other two columns were not.
# The roster is derived from the surface rather than typed out, and every roster name is checked
# against the share bars. How many names that comes to is reported rather than asserted:
# `n_assets` describes the production extract, and the reduced one CI runs on carries a handful by
# design. `tests/test_eoa_universe_roster.py` holds the declaration to the production data.

# %%
ENTITY = "sec_id"
# The surface is loaded for its roster alone; no column of it is read here. Both extracts are read
# whole rather than over the requested window, because which names have listed options is a
# property of the dataset. The dates bound the panel below.
_surface = load_sp500_options_surface()
full_bars = load_sp500_daily_bars()
ROSTER = sorted(_surface["symbol"].unique().to_list())
assert ROSTER, "the option-surface extract carries no names to rank"
priced = set(full_bars["symbol"].unique().to_list())
assert not set(ROSTER) - priced, f"no share bars for {sorted(set(ROSTER) - priced)}"
outside = sorted(priced - set(ROSTER))
DECLARED = SETUP["universe"]["n_assets"]
print(
    f"Universe {len(ROSTER)} names with an option surface"
    + ("" if len(ROSTER) == DECLARED else f" (a reduced extract; n_assets declares {DECLARED})")
    + f"; {len(outside)} priced names carry none and are excluded ({', '.join(outside)})"
)

window = pl.col("timestamp").is_between(
    pl.lit(START_DATE).str.to_date(), pl.lit(END_DATE).str.to_date()
)
bars = full_bars.filter(window & pl.col("symbol").is_in(ROSTER)).sort([ENTITY, "timestamp"])
bars = bars.with_columns((pl.col("close") * pl.col("adj_factor")).alias("adj_close"))
# One security trades under one ticker on one session. Without that, `sec_id` does not identify a
# price series and every window below would be taken across companies - which is the failure this
# whole section is arranged to avoid, so it is checked rather than assumed.
assert bars.select([ENTITY, "timestamp"]).is_duplicated().sum() == 0, (
    "a security carries more than one ticker on the same session, so sec_id does not identify a "
    "price series in this data"
)
securities = bars[ENTITY].unique().sort().to_list()
if MAX_SYMBOLS is not None:
    # Reduce by the rule every other stage reduces by, on the column they reduce on. This used to
    # take a prefix of the sorted sec_id list, which is the lexically first securities and has
    # nothing to do with which ones carry data: `05_evaluation` reduces to the most-observed
    # symbols, so a reduced 04 fitted volatility models for one universe while a reduced 05 scored
    # another, and the symbols only one side chose carried null features that ran clean.
    #
    # The selection is made on `symbol` and then mapped to `sec_id`, rather than counting sec_ids
    # directly, because `symbol` is what the artifact is keyed on and what every consumer reduces
    # by. The assertion above - one security per ticker per session - is what makes the mapping
    # one-to-one, so the two counts agree.
    kept_symbols = top_entities(bars, MAX_SYMBOLS, entity_col="symbol")
    securities = bars.filter(pl.col("symbol").is_in(kept_symbols))[ENTITY].unique().sort().to_list()
    print(f"Reduced run: {len(securities)} of {bars[ENTITY].n_unique()} securities")

log_returns: dict[int, pd.Series] = {}
for key, frame in (
    bars.select([ENTITY, "timestamp", "adj_close"])
    .filter(pl.col(ENTITY).is_in(securities))
    .partition_by(ENTITY, as_dict=True, maintain_order=True)
    .items()
):
    series = (
        frame.sort("timestamp")
        .with_columns((pl.col("adj_close").log().diff() * 100).alias("ret_pct"))
        .drop_nulls("ret_pct")
        .select(["timestamp", "ret_pct"])
        .to_pandas()
        .set_index("timestamp")["ret_pct"]
    )
    if not series.empty:
        log_returns[key[0]] = series

TICKER_OF = bars.select([ENTITY, "timestamp", "symbol"])
print(
    f"{len(log_returns)} securities under {bars['symbol'].n_unique()} tickers, "
    f"{bars['timestamp'].min()} to {bars['timestamp'].max()}"
)

# %% [markdown]
# ### C.2 What the convenience method would have cost
#
# The claim above is that handing `fix` the estimation window and the inference span together lets
# later returns reach an earlier date through the clipping range. The range is wide - the bounds
# sit six orders of magnitude either side of a local variance estimate - so it is entirely
# possible that it never binds and the whole concern is theoretical. That is a measurement, not an
# argument, so it is measured.
#
# The test isolates the clipping range from everything else. For a sample of securities on the
# first fold, `fix` is called twice with the same estimated parameters: once on the estimation
# window alone, once on the estimation window followed by the inference span. Both series start at
# the same session, so the starting variance - which `arch` reads from the first sessions only -
# is identical between them. The clipping range is then the one thing that differs, and any
# difference on the sessions the two calls share is what it did.

# %%
TRUNCATION_SAMPLE = 40
truncation_effect = []
for security in securities[:TRUNCATION_SAMPLE]:
    returns = log_returns.get(security)
    if returns is None or len(returns) <= GARCH_BURNIN + GARCH_REFIT_EVERY:
        continue
    blocks = refit_boundaries(len(returns), GARCH_BURNIN, GARCH_REFIT_EVERY)
    if len(blocks) < 2:
        continue
    # Cut INSIDE a block, never at a boundary. Truncating at a boundary leaves every retained
    # block with exactly the prefix it already had, so the two walks agree whatever the recursion
    # does inside a block and the check passes without testing anything. The cut below lands
    # mid-block by construction, and the assertion after this loop is what holds it there.
    boundaries = {end for _, end in blocks}
    # The last block can be a single observation, when the series ends one session past a refit.
    # It then has no session strictly inside it, and a cut clamped into it lands back on the
    # boundary that opens it - which is the case this assertion exists to refuse. Every earlier
    # block is a full `GARCH_REFIT_EVERY` wide by construction, so step back one.
    block_start, block_end = blocks[-1]
    if block_end - block_start < 2:
        block_start, block_end = blocks[-2]
    cut = min(block_start + GARCH_REFIT_EVERY // 2, block_end - 1)
    assert cut not in boundaries, "the truncation point fell on a refit boundary and tests nothing"
    full, _ = garch_walk(returns, None)
    short, _ = garch_walk(returns.iloc[:cut], None)
    shared = full.iloc[: len(short)].to_numpy()
    both = np.isfinite(shared) & np.isfinite(short.to_numpy())
    if not both.any():
        continue
    denom = np.maximum(np.abs(shared[both]), 1e-12)
    truncation_effect.append(float(np.max(np.abs(shared[both] - short.to_numpy()[both]) / denom)))

if truncation_effect:
    print(f"{len(truncation_effect)} securities truncated mid-block and re-walked")
    print(
        "largest relative move in a retained value when every later observation is deleted: "
        f"{max(truncation_effect):.3e}"
    )
    assert max(truncation_effect) < 1e-9, (
        "deleting later observations moved an earlier emitted value, so something in the walk "
        "reads past the row it is emitting"
    )
else:
    print("No security in the sample was long enough to truncate mid-block.")

# %% [markdown]
# The property is asserted rather than described: an emitted value does not move when
# observations after it are deleted. The cut is taken inside a block rather than at a refit
# boundary, because a boundary cut leaves every retained block with exactly the prefix it had and
# would agree no matter what the recursion did.
#
# What the convenience method would cost is worth keeping in view beside it: handing `arch`'s
# `fix` the estimation window and the inference span together moves the volatility it reports on
# a *shared* session by up to **0.9%** of that security's own largest value, measured across 38
# securities here. That is what `garch11_conditional_volatility` exists to avoid, and it is why
# the backcast and the clipping range are derived from the estimation prefix and passed in rather
# than left for the library to infer from whatever series it is handed.

# %%
filtered_parts: list[pl.DataFrame] = []
parameter_rows: list[dict] = []
too_short = 0

for security in securities:
    returns = log_returns.get(security)
    if returns is None or len(returns) <= GARCH_BURNIN:
        # A security listed too late to clear its own burn-in emits nothing. That is a statement
        # about the security rather than a failure, and it is counted so the coverage report
        # below can say how many rather than leaving a reader to infer it from a gap.
        too_short += 1
        continue
    path, records = garch_walk(returns, count_before(returns.index, pd.Timestamp(HOLDOUT_START)))
    emitted = path.dropna()
    if emitted.empty:
        too_short += 1
        continue
    filtered_parts.append(
        pl.DataFrame({"timestamp": emitted.index.values, "garch_cond_vol": emitted.to_numpy()})
        .with_columns(pl.col("timestamp").cast(pl.Date))
        .with_columns(pl.lit(security).alias(ENTITY))
    )
    parameter_rows.extend({ENTITY: security, **record} for record in records)

_filtered = pl.concat(filtered_parts)
garch = _filtered.join(TICKER_OF, on=[ENTITY, "timestamp"], how="inner").select(
    ["timestamp", "symbol", ENTITY, "garch_cond_vol"]
)
assert garch.height == _filtered.height, "a filtered value had no bar to name it"
parameters = pl.DataFrame(parameter_rows)
print(f"{len(filtered_parts):,} of {len(securities):,} securities emitted a value")
print(f"  never cleared the {GARCH_BURNIN}-session burn-in: {too_short}")
print(f"  fits across every security and refit: {parameters.height:,}")
print(
    f"{garch.height:,} filtered values over {garch[ENTITY].n_unique()} securities, "
    f"written under {garch['symbol'].n_unique()} tickers"
)

# %% [markdown]
# ## D. Fit stability across the schedule
#
# The method rests on re-estimating every `refit_every` sessions, and that is only worth its cost
# if the estimates actually move. If the same parameters come back block after block, the cadence
# bought nothing and one estimation would have done. If they swing, the feature carries whatever
# the expanding window happened to contain as much as it carries anything about the share, and a
# reader should treat it accordingly.
#
# What a consumer no longer has to do is join on a fold. Each session's value comes from the one
# estimate that preceded it, so the date and the security are the whole key.
#
# Persistence is the summary to read, and it is the sum of the other three: $\alpha + \gamma/2 +
# \beta$. A value near one means a shock to volatility decays slowly, so today's variance says
# something about tomorrow's - which is the property that makes a conditional volatility worth
# forecasting at all. A value near zero would mean the model has nothing to carry forward and the
# forecast is close to a constant.
#
# ### F3. Fit stability across the schedule
#
# One line per estimated parameter against the date the estimate first speaks for, drawn at the
# median across securities with a band spanning the middle half. The x-axis is calendar time
# rather than a fold index because the blocks are calendar time: a security refits every
# `refit_every` of its own sessions, so securities that list at different dates refit on different
# dates and the median is taken over whoever is estimated at each. All four parameters are
# dimensionless and bounded the same way, so they share an axis.

# %%
parameter_colours = ml4t_palette(4, categorical=True)
styles = [
    ("alpha", parameter_colours[0], "solid"),
    ("gamma", parameter_colours[1], "solid"),
    ("beta", parameter_colours[2], "dash"),
    ("persistence", parameter_colours[3], "solid"),
]
by_date = (
    parameters.with_columns(pl.col("emit_start").cast(pl.Date))
    .group_by("emit_start")
    .agg(
        pl.len().alias("securities"),
        *[
            expr
            for name, _, _ in styles
            for expr in (
                pl.col(name).median().alias(f"{name}_med"),
                pl.col(name).quantile(0.25).alias(f"{name}_lo"),
                pl.col(name).quantile(0.75).alias(f"{name}_hi"),
            )
        ],
    )
    .sort("emit_start")
)
x = by_date["emit_start"].to_list()
fig = go.Figure()
for name, colour, dash in styles:
    fig.add_scatter(
        x=x + x[::-1],
        y=by_date[f"{name}_hi"].to_list() + by_date[f"{name}_lo"].to_list()[::-1],
        fill="toself",
        fillcolor=colour,
        opacity=0.15,
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=False,
    )
    fig.add_scatter(
        x=x,
        y=by_date[f"{name}_med"].to_list(),
        mode="lines",
        name=name,
        line=dict(color=colour, width=2, dash=dash),
    )
fig.update_layout(
    title="Every parameter moves with the window, so every refit moves the feature",
    xaxis_title="Date the estimate first speaks for; line is the median across securities, "
    "band the middle half",
    yaxis_title="Estimated parameter value",
    height=430,
)
show_plotly_with_alt(
    fig,
    "Four parameter series against calendar time, one per estimated GJR-GARCH coefficient plus "
    "persistence, each drawn as a median line across securities with a shaded band spanning the "
    "middle half of them. Persistence and beta run highest and track each other; gamma and alpha "
    "run lower. The bands are widest early, when few securities have cleared the burn-in and the "
    "estimation windows are shortest, and narrow as the expanding windows lengthen. No series is "
    "flat, which is the point: the refit cadence changes the estimates it produces.",
)

# %%
stability = (
    parameters.with_columns(pl.col("emit_start").cast(pl.Date).dt.year().alias("year"))
    .group_by("year")
    .agg(
        pl.len().alias("fits"),
        pl.col(ENTITY).n_unique().alias("securities"),
        pl.col("alpha").median().round(4).alias("median alpha"),
        pl.col("gamma").median().round(4).alias("median gamma"),
        pl.col("beta").median().round(4).alias("median beta"),
        pl.col("persistence").median().round(4).alias("median persistence"),
        (pl.col("degenerate").mean() * 100).round(1).alias("alpha+gamma < 0 %"),
    )
    .sort("year")
)
stability

# %% [markdown] tags=["results"]
# Read the table by year and the figure by shape. What the schedule buys is visible in whether
# median persistence moves between refits: if it were flat, one estimation would have done and
# the cadence would be waste. The parts are worth reading separately from the sum, because
# $\alpha$ and $\gamma$ can trade against each other while persistence barely moves - a share of
# the response to a shock shifting between the symmetric and asymmetric terms is a change in what
# the model says about *down* sessions specifically, and persistence alone will not show it.
#
# The last column is the one to weigh against the schedule. It is the share of fits in each year
# that came back with $\alpha + \gamma < 0$, which says a larger down-shock lowers next session's
# variance. That is not a volatility model, and it is a burn-in artifact: measured across this
# universe it runs about 19% on a security's first block against about 1.8% of fits over the
# whole walk, falling as the expanding window lengthens. `config/setup.yaml::model_based` records
# why the burn-in is 252 anyway.
#
# A downstream model reading this feature is reading a quantity whose generating parameters were
# re-estimated on a schedule, not a fixed transform - but every session carries exactly one such
# value, so the date and the symbol are the whole key.

# %% [markdown]
# ### F2. What the model inferred, over the sessions it was run over
#
# The feature itself, drawn against the quantity it is meant to improve on. The line is the median
# across securities of the conditional volatility the model reports, and beside it the median of
# stage 03's 20-session realized volatility on the same dates. Two restrictions apply and both
# matter. Only the **inference** spans are drawn, because the estimation windows are the sessions
# the parameters came from and the reader is owed a picture of the feature where it is out of
# sample. And only the **development** folds are drawn: the holdout fold's values are in the
# artifact because a later stage needs them, and a figure read during development does not get to
# look at them.
#
# What to look for is not that the two agree - they are different quantities, one an estimate of
# the next session and the other an average over the last twenty - but *when* they part company.
# A conditional volatility responds to a shock the session after it, while a twenty-session
# average takes a month to absorb it and another month to forget it.

# %%
financial = pl.read_parquet(FINANCIAL_PATH).select(
    [*PANEL_KEY, "iv_30_atm", "rv_20", "ivrv_spread"]
)
# The validation spans of the development folds. Nothing about the feature depends on them now;
# they are the dates the downstream models are scored on, so reporting over them is what lets a
# reader line this chart up against those. The holdout is excluded, as it is everywhere in
# development.
inference_only = (
    pl.concat(
        [
            garch.filter(
                (pl.col("timestamp") >= pl.lit(row["infer_start"], dtype=pl.Date))
                & (pl.col("timestamp") <= pl.lit(row["infer_end"], dtype=pl.Date))
            )
            for row in REPORTING_SPANS
        ]
    )
    .unique(subset=[*PANEL_KEY])
    .sort(PANEL_KEY)
)

daily_medians = (
    inference_only.join(financial, on=PANEL_KEY, how="left")
    .group_by("timestamp")
    .agg(
        pl.col("garch_cond_vol").median().alias("conditional"),
        pl.col("rv_20").median().alias("realized"),
    )
    .sort("timestamp")
)

series_colours = ml4t_palette(2, categorical=True)
fig = go.Figure()
for column, name, colour in (
    ("conditional", "GJR-GARCH conditional volatility", series_colours[0]),
    ("realized", "20-session realized volatility", series_colours[1]),
):
    fig.add_scatter(
        x=daily_medians["timestamp"].to_list(),
        y=daily_medians[column].to_list(),
        mode="lines",
        name=name,
        line=dict(color=colour, width=1.6),
    )
for row in REPORTING_SPANS:
    fig.add_vline(x=row["infer_start"].isoformat(), line_dash="dot", line_color=ml4t_palette(3)[2])
fig.update_layout(
    title="The forecast turns within days of a shock where the average takes weeks",
    xaxis_title="Median across securities, validation spans only; dotted rules mark span starts",
    yaxis_title="Annualized volatility",
    height=420,
)
show_plotly_with_alt(
    fig,
    "Two lines of median annualized volatility across securities over the two inference spans, "
    "2019 and 2020, with dotted rules at each validation span start. Through 2019 both run "
    "together between "
    "roughly 0.18 and 0.30. In late February 2020 both climb steeply; the conditional volatility "
    "peaks first, near 1.0, and falls back below 0.4 within about six weeks, while the "
    "twenty-session realized volatility peaks higher, near 1.2, holds that level for several "
    "weeks and then declines in steps through the summer. From July 2020 the realized line stays "
    "above the conditional one for most of the rest of the year, including a separate rise to "
    "about 0.42 in the autumn that the conditional line barely registers.",
)

# %% [markdown]
# ## E. Combine and emit
#
# Three columns go out, and the second is the one this notebook was written for.
#
# - `garch_cond_vol` - the conditional volatility itself, annualized.
# - `garch_ivrv_spread` - the implied volatility of the coming month less that forecast. Stage 03's
#   `ivrv_spread` subtracts a *realized* volatility from the same implied level, so the two differ
#   in the denominator alone and in nothing else.
# - `garch_vol_surprise` - the size of the session's actual move divided by the volatility that
#   had been forecast for it. The denominator is a standard deviation, and most sessions move less
#   than one, so a well-calibrated forecast puts the typical value below one rather than at it;
#   what the column is for is the tail, where a value of three or four says the session moved far
#   more than the model had any reason to expect.
#
# The return in `garch_vol_surprise` is the same adjusted log return the model was estimated on,
# taken inside the security for the reason C.1 gives and annualized to the denominator's units so
# the ratio compares like with like.

# %%
returns_panel = (
    bars.select([ENTITY, "timestamp", "adj_close"])
    .sort([ENTITY, "timestamp"])
    .with_columns((pl.col("adj_close").log().diff().over(ENTITY) * ANNUALIZE).alias("_ann_ret"))
    .select([ENTITY, "timestamp", "_ann_ret"])
)

model_based = (
    garch.join(returns_panel, on=[ENTITY, "timestamp"], how="left")
    .join(financial.select([*PANEL_KEY, "iv_30_atm"]), on=PANEL_KEY, how="left")
    .with_columns(
        (pl.col("_ann_ret").abs() / pl.col("garch_cond_vol")).alias("garch_vol_surprise"),
        (pl.col("iv_30_atm") - pl.col("garch_cond_vol")).alias("garch_ivrv_spread"),
    )
    .drop(["_ann_ret", "iv_30_atm", ENTITY])
    .select([*PANEL_KEY, "garch_cond_vol", "garch_ivrv_spread", "garch_vol_surprise"])
    .filter(pl.col("timestamp") <= pl.lit(HOLDOUT_END, dtype=pl.Date))
    .sort(PANEL_KEY)
)
FEATURE_COLS = ["garch_cond_vol", "garch_ivrv_spread", "garch_vol_surprise"]
assert model_based.select(PANEL_KEY).is_duplicated().sum() == 0, "duplicate panel key"
assert model_based["timestamp"].max() <= HOLDOUT_END, "a value was emitted past the holdout"

# %% [markdown]
# ### E.1 What each column covers
#
# The panel key is `timestamp` + `symbol`, and that is the whole key. A date and security carry
# one conditional volatility, from the last parameters estimated before that date, so there is
# nothing for a consumer to disambiguate and no fold column to join on. That is the change this
# revision makes and the assertion above is what holds it.
#
# The walk covers each security's whole history, which runs past the holdout's end for nothing's
# benefit, so the artifact is bounded at `holdout_end` above. Coverage is reported over the
# development spans and the holdout separately, which keeps every distribution printed during
# development to sessions development is allowed to see.

# %%
coverage = (
    model_based.with_columns(
        pl.when(pl.col("timestamp") >= pl.lit(HOLDOUT_START, dtype=pl.Date))
        .then(pl.lit("holdout"))
        .otherwise(pl.lit("development"))
        .alias("span")
    )
    .group_by("span")
    .agg(
        pl.len().alias("rows"),
        pl.col("symbol").n_unique().alias("symbols"),
        *[(pl.col(c).is_not_null().mean() * 100).round(1).alias(f"{c} %") for c in FEATURE_COLS],
    )
    .sort("span")
)
coverage

# %%
development = model_based.filter(pl.col("timestamp") < pl.lit(HOLDOUT_START, dtype=pl.Date))
pl.DataFrame(
    {
        "feature": FEATURE_COLS,
        "median": [development[c].median() for c in FEATURE_COLS],
        "std": [development[c].std() for c in FEATURE_COLS],
        "1st pct": [development[c].quantile(0.01) for c in FEATURE_COLS],
        "99th pct": [development[c].quantile(0.99) for c in FEATURE_COLS],
    }
)

# %% [markdown]
# ### E.2 Write the artifact
#
# Beside the parquet goes a small JSON file with the same name and a `.digest.json` suffix, the
# same way stage 03 writes its matrix. What it buys is provenance a file name cannot give: a model
# trained on these values records the hash, and a later run can tell whether the feature it is
# reading is the feature it trained on.
#
# It holds a hash computed over the values in the file; the number of rows; the columns that
# identify a row; and a hash of each input the values were built from. The hash is over the content
# rather than the file's bytes, so a re-write in a different row order leaves it alone and any
# changed value moves it.
#
# The two inputs recorded here are the share bars the model was estimated on and stage 03's matrix,
# which supplied the implied volatility the second feature subtracts from.

# %%
record = write_model_based(
    model_based,
    FEATURES_DIR / "model_based.parquet",
    keys=PANEL_KEY,
    feature_columns=FEATURE_COLS,
    time_column="timestamp",
    # No fold column. The parameters come off a refit schedule over the whole panel, so a row is
    # identified by its keys alone and there is no fold for a value to belong to. Passing None
    # rather than leaving the default is what makes `expected_folds` refused instead of ignored.
    fold_column=None,
    written_by=f"case_studies/{CASE_STUDY_ID}/04_model_based_features.py",
    inputs={
        # `sec_id` belongs in this digest: it decides which rows form one series, so a changed
        # security mapping moves every fitted value while the prices themselves stand still.
        "load_sp500_daily_bars": value_digest(
            bars.select([*PANEL_KEY, ENTITY, "close", "adj_factor"])
        ),
        "features/financial.parquet": read_digest(FINANCIAL_PATH)["digest"],
    },
)
print(f"Wrote {display_path(FEATURES_DIR / 'model_based.parquet')}, digest {record['digest']}")
print(f"{record['n_rows']:,} rows on {record['keys']}, read through load_modeling_dataset")

# %% [markdown]
# ## F. Incremental evaluation
#
# One question, asked on validation rows only: does the option market's variance premium say
# something different about future returns when its denominator is a forecast instead of a memory?
#
# The measure is the **information coefficient**: on each decision date, the rank correlation
# across securities between the feature and the return that follows. That gives one number per
# date; the average over dates is the feature's IC, and its standard error has to account for
# consecutive dates sharing outcome sessions, which is what the label horizon bound above is for.
#
# **This section selects nothing.** All three features were written to the artifact in E.2 before
# it ran, and the holdout is excluded from every number below. `05_evaluation` is where features
# are screened, and it runs that screen over the whole matrix.

# %%
label = pl.read_parquet(LABELS_DIR / f"{PRIMARY_LABEL}.parquet").rename(
    {PRIMARY_LABEL: "forward_return"}
)
scored = (
    model_based.join(inference_only.select(PANEL_KEY), on=PANEL_KEY, how="semi")
    .join(label.select([*PANEL_KEY, "forward_return"]), on=PANEL_KEY, how="inner")
    .sort(PANEL_KEY)
)
assert scored.filter(pl.col("timestamp") >= pl.lit(HOLDOUT_START, dtype=pl.Date)).is_empty(), (
    "the validation scoring frame reaches into the holdout"
)
print(f"{scored.height:,} scored rows over {scored['timestamp'].n_unique():,} decision dates")


# %%
# Two floors, and they count different things. `MIN_CROSS_SECTION` is a number of securities on
# one date, below which a rank correlation is noise rather than a measurement. `MIN_IC_DATES` is
# a number of dates, below which the HAC standard error over those correlations means nothing.
# One constant used to serve both, which read as deliberate and was not: the second comparison
# was measuring a series length against a universe size.
#
# The cross-section floor is clamped to the universe this run actually loaded. At full width
# `MAX_SYMBOLS` is None and the clamp does nothing, so production is unchanged; under a reduced
# run it is what keeps this section measuring something. A fixed floor of ten against a five-name
# reduction does not shrink the section, it empties it - every date falls short, every feature
# reports no IC, and the run stays green over a measurement it never made.
MIN_CROSS_SECTION = min(10, scored["symbol"].n_unique())
MIN_IC_DATES = 10


def ic_series(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    """Per-date cross-sectional rank correlation of *column* against the forward return.

    A date with fewer than ``MIN_CROSS_SECTION`` securities comes back as a null rather
    than as a correlation over three names, and is dropped here rather than averaged in.
    """
    scored_frame = frame.select([*PANEL_KEY, column, "forward_return"]).drop_nulls()
    scored_frame = scored_frame.rename({column: "prediction"})
    series = cross_sectional_ic_series(
        scored_frame,
        scored_frame,
        date_col="timestamp",
        entity_col="symbol",
        method="spearman",
        min_obs=MIN_CROSS_SECTION,
    )
    return series.filter(pl.col("ic").is_not_null() & pl.col("ic").is_not_nan()).sort("timestamp")


ic_rows = []
for column in FEATURE_COLS:
    series = ic_series(scored, column)
    if len(series) < MIN_IC_DATES:
        print(
            f"{column}: only {len(series)} dates carry a cross-section of at least "
            f"{MIN_CROSS_SECTION} securities, and {MIN_IC_DATES} are needed for a standard "
            "error, so no IC is reported for it"
        )
        continue
    stats = compute_ic_hac_stats(series, label_horizon=LABEL_HORIZON)
    ic_rows.append(
        {
            "feature": column,
            "ic": stats["mean_ic"],
            "hac_t": stats["t_stat"],
            "p": stats["p_value"],
            "dates": len(series),
        }
    )

if ic_rows:
    fdr = benjamini_hochberg_fdr([r["p"] for r in ic_rows], alpha=0.05, return_details=True)
    for row, survives in zip(ic_rows, fdr["rejected"], strict=True):
        row["survives FDR"] = bool(survives)
    ic_table = pl.DataFrame(ic_rows).with_columns(
        pl.col("ic").round(4), pl.col("hac_t").round(2), pl.col("p").round(4)
    )
else:
    ic_table = pl.DataFrame(schema={"feature": pl.Utf8, "ic": pl.Float64})
ic_table

# %% [markdown]
# ### F4. Incremental IC by feature
#
# Each bar is one feature's mean IC with its HAC t-statistic printed past the bar end. A
# t-statistic of about two is the conventional threshold for calling a mean distinguishable from
# zero at all; the Benjamini-Hochberg correction applied above lowers the bar each individual
# feature has to clear, because three tests give three chances to find something by accident. A
# bar is drawn in the darker colour where its feature clears that corrected threshold, so a chart
# in which every bar is the lighter colour is one in which none of them did.

# %%
if ic_table.is_empty():
    print("No IC series survived the cross-section floor, so there is nothing to draw here.")
else:
    bar_colour, faded = ml4t_palette(2, categorical=True)
    ordered = ic_table.sort(pl.col("ic").abs(), descending=True)
    fig = go.Figure()
    fig.add_bar(
        x=ordered["ic"].to_list(),
        y=ordered["feature"].to_list(),
        orientation="h",
        marker_color=[bar_colour if s else faded for s in ordered["survives FDR"].to_list()],
        text=[f"HAC t {t:+.2f}" for t in ordered["hac_t"].to_list()],
        textposition="outside",
        cliponaxis=False,
    )
    fig.add_vline(x=0.0, line_width=1, line_color=ml4t_palette(3)[2])
    _span = max(abs(v) for v in ordered["ic"].to_list()) * 1.7
    fig.update_layout(
        title="All three sit near zero and none is distinguishable from it",
        xaxis_title=(f"Mean cross-sectional IC against the {LABEL_HORIZON}-session forward return"),
        xaxis=dict(range=[-_span, _span]),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=170, r=70),
        height=340,
    )
    show_plotly_with_alt(
        fig,
        "Three horizontal bars of mean information coefficient on an axis running from minus "
        "0.01 to plus 0.01, with a rule at zero. `garch_cond_vol` reaches furthest left at about "
        "minus 0.0065, `garch_ivrv_spread` to about minus 0.0032, and `garch_vol_surprise` right "
        "to about plus 0.003. Each bar is labelled with its HAC t-statistic, none of which "
        "exceeds 0.6 in absolute value. Every bar is drawn in the lighter colour, which is what "
        "the chart uses for a feature that did not clear the false-discovery threshold.",
    )

# %% [markdown]
# ### F5. The forecast denominator against the memory denominator
#
# The comparison the notebook was written to make. Both variants are scored on the rows where both
# are defined, so the only thing that differs between the first two bars is the denominator: the
# forecast in one, the trailing average in the other. Scoring stage 03's spread over its own wider
# panel would put a difference in sample into a comparison that is supposed to isolate one term.
#
# **Neither of the first two t-statistics answers the question, and the third bar is why.** Each of
# them tests one mean IC against zero. The question is whether the two differ from *each other*,
# and because both are scored on the same rows they produce a value on the same dates, which makes
# the two daily series dependent. Dependent series can separate from each other while neither
# separates from zero, and can fail to separate from each other while both do; which way it goes
# is set by the sign of the dependence, printed on the axis. So the difference is tested on its
# own: one observation per date, the forecast variant's IC less the memory variant's, HAC
# corrected the same way. Signed IC is drawn rather than its absolute value, so the axis carries
# the quantity the test is about.

# %%
paired_rows = scored.join(
    financial.select([*PANEL_KEY, "ivrv_spread"]), on=PANEL_KEY, how="inner"
).drop_nulls(["garch_ivrv_spread", "ivrv_spread", "forward_return"])

memory_series = ic_series(paired_rows, "ivrv_spread")
forecast_series = ic_series(paired_rows, "garch_ivrv_spread")
paired = (
    memory_series.select(["timestamp", pl.col("ic").alias("ic_memory")])
    .join(
        forecast_series.select(["timestamp", pl.col("ic").alias("ic_forecast")]),
        on="timestamp",
        how="inner",
    )
    .drop_nulls()
    .sort("timestamp")
    .with_columns((pl.col("ic_forecast") - pl.col("ic_memory")).alias("ic"))
)
# `compute_ic_hac_stats` reads row order as time order and a join does not promise one.
assert paired["timestamp"].is_sorted(), "the paired series is not in date order"
COMPARABLE = paired.height >= MIN_IC_DATES
print(f"{paired_rows.height:,} rows carry both variants, over {paired.height:,} dates")
if COMPARABLE:
    memory_stats = compute_ic_hac_stats(memory_series, label_horizon=LABEL_HORIZON)
    forecast_stats = compute_ic_hac_stats(forecast_series, label_horizon=LABEL_HORIZON)
    difference_stats = compute_ic_hac_stats(paired, label_horizon=LABEL_HORIZON)
    daily_dependence = paired.select(pl.corr("ic_memory", "ic_forecast")).item()
    print(
        f"  memory denominator    IC {memory_stats['mean_ic']:+.4f}, "
        f"HAC t {memory_stats['t_stat']:+.2f}"
    )
    print(
        f"  forecast denominator  IC {forecast_stats['mean_ic']:+.4f}, "
        f"HAC t {forecast_stats['t_stat']:+.2f}"
    )
    print(f"  their daily values correlate at {daily_dependence:+.2f}")
    print(
        f"  paired difference     {difference_stats['mean_ic']:+.4f}, "
        f"HAC t {difference_stats['t_stat']:+.2f}, p {difference_stats['p_value']:.4f}"
    )
else:
    print("Too few dates carry both variants for the comparison; the figure below is skipped.")

# %%
if COMPARABLE:
    level_colour, difference_colour = ml4t_palette(2, categorical=True)
    bars_x = [
        "Memory denominator<br>(stage 03 realized volatility)",
        "Forecast denominator<br>(GJR-GARCH)",
        "Paired difference<br>(forecast less memory)",
    ]
    bars_y = [memory_stats["mean_ic"], forecast_stats["mean_ic"], difference_stats["mean_ic"]]
    bars_t = [memory_stats["t_stat"], forecast_stats["t_stat"], difference_stats["t_stat"]]
    fig = go.Figure()
    fig.add_bar(
        x=bars_x,
        y=bars_y,
        marker_color=[level_colour, level_colour, difference_colour],
        text=[f"IC {v:+.4f}<br>HAC t {t:+.2f}" for v, t in zip(bars_y, bars_t, strict=True)],
        textposition="outside",
        cliponaxis=False,
    )
    fig.add_hline(y=0.0, line_width=1, line_color=ml4t_palette(3)[2])
    fig.update_layout(
        title="Swapping the denominator moves the premium's IC by less than its own noise",
        yaxis_title=f"Mean cross-sectional IC against the {LABEL_HORIZON}-session forward return",
        xaxis_title=(
            "Both variants scored on the rows where both are defined; their daily values "
            f"correlate at {daily_dependence:.2f}"
        ),
        margin=dict(t=90),
        height=440,
    )
    show_plotly_with_alt(
        fig,
        "Three bars of mean information coefficient about a rule at zero, each labelled with its "
        "value and HAC t-statistic. The memory denominator and the forecast denominator are both "
        "negative and drawn in the same dark colour, the memory one about twice the depth of the "
        "forecast one. The paired difference is drawn in a separate lighter colour and is "
        "positive, and it is the smallest of the three in absolute size. All three sit inside a "
        "range of about 0.008, and none of the three t-statistics reaches half of the "
        "conventional threshold of two.",
    )

# %% [markdown] tags=["results"]
# On the **184,299** rows where both variants are defined, across **497** decision dates, the
# memory denominator gives a mean IC of **-0.0053** with a HAC t of **-0.44** and the forecast
# denominator **-0.0032** with a HAC t of **-0.37**. Their daily values correlate at **-0.25**,
# which is the dependence the paired test exists to handle. The paired difference is **+0.0021**
# with a HAC t of **0.13**, so swapping a backward-looking denominator for a forward-looking one
# is not distinguishable from making no change at all - and that now rests on a test of the
# difference rather than on an inference from two tests that were about something else. Taken
# standalone, none of the three features clears the false-discovery threshold either, the largest
# being `garch_cond_vol` at an IC of **-0.0065** and a HAC t of **-0.34**.

# %% [markdown]
# ## Key takeaways
#
# - **The estimation window is part of a fitted feature's information set.** A rule computed over
#   a window reads that window; a parameter estimated over a window carries every row in it into
#   every value the model then produces. Fixing the window before any model runs, and asserting
#   the ordering rather than describing it, is what makes the difference checkable.
# - **Fixing the parameters is not the whole of fixing the fit.** A recursion also needs a value
#   to start from and a range to stay inside, and a library asked to run one will derive both from
#   whatever series it is given. Derive them from the estimation window explicitly, and measure
#   what the convenience would have cost rather than assuming either that it is nothing or that it
#   is large. C.2 is that measurement.
# - **Emit the fold alongside the value.** The same date and security carry a different value per
#   fold, because that is what per-fold estimation means. Writing the fold id into the key is
#   what stops a consumer from joining a value to a window it did not come from.
# - **Test a difference, not two levels.** Two measurements taken on the same rows are dependent,
#   and neither one's standard error contains the variance of their difference. Pair them by date
#   and test the paired series.
# - **Report a diagnostic fold by fold when one fold is the holdout.** Reporting the coverage
#   and the distributions by fold is what keeps a number read during development to the sessions
#   development is allowed to see, while still emitting the feature over the holdout for the stage
#   that scores it.
#
# ### Known limitations
#
# - Only the inference-span values are point-in-time. The estimation-window values in the same
#   artifact were produced under parameters estimated from that whole window, so they are
#   retrospective, and they are there because a downstream model needs a feature on its training
#   rows. Any diagnostic that treats them as out-of-sample is measuring the wrong thing, which is
#   why section F restricts itself to the inference spans.
# - The model is estimated per security and per fold with no pooling, so a security with a short
#   or quiet history gets a noisier estimate than a long-established one, and one with fewer than
#   the configured minimum gets no value at all. The coverage table is where that shows up.
# - The parameters are held fixed across the whole span they are run over, which is the discipline
#   this stage is about, and it also means the model does not adapt within a fold. A regime change
#   part-way through an inference span is absorbed by the recursion's state rather than by the
#   parameters.
# - The conditional volatility is a one-session-ahead estimate while the implied volatility it is
#   subtracted from prices the coming month. The two are in the same units and are not over the
#   same horizon, so `garch_ivrv_spread` mixes horizons in a way a term-structure-aware forecast
#   would not.
# - A normal innovation is assumed. Equity returns have heavier tails than that, so the estimated
#   persistence is likely a little high and a large move is scored as more of a surprise than a
#   fatter-tailed model would score it.
#
# **Next**: [`05_evaluation`](05_evaluation.ipynb) screens this artifact alongside stage 03's
# matrix, fold by fold, and the model notebooks after it read both through
# `load_modeling_dataset`.
