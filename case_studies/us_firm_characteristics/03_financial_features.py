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
# # US Firm Characteristics: Feature Engineering
#
# This notebook assembles the model matrix for the cross-sectional factor study. Its input is the
# released Chen-Pelger-Zhu characteristic panel, which arrives already cross-sectionally
# rank-transformed, and its output is `features/financial.parquet` keyed on the persistent anonymous
# firm axis that `02_labels` keys its labels on.
#
# Because the characteristics arrive ranked, this notebook fits nothing and looks back over nothing.
# Every column it adds is a function of other columns *in the same row*. That makes the timing
# contract in Section D unusually strong and unusually cheap to demonstrate: rebuilding the panel
# with the holdout withheld reproduces every pre-holdout value exactly, and Section D.3 asserts it.
#
# ## Learning objectives
#
# - Read a register that states, per family, what is expected to carry information and when it is
#   knowable
# - Separate the provider's update conventions from windows this notebook computes, and audit only
#   what this notebook is in a position to audit
# - Build composites and interactions without introducing a fitted transform
# - Demonstrate a timing contract by rebuild-and-compare rather than by assertion in prose
# - Emit a keyed artifact with a digest sidecar that the next stage reads
#
# ## Book reference, prerequisites and artifacts
#
# Chapter 8, Section 8.4 (Contextual and Slow-Moving Features).
#
# | | |
# |---|---|
# | Requires | [`02_labels`](02_labels.ipynb) - the label panel and its firm identity |
# | Reads | `load_firm_characteristics()`; `labels/fwd_ret_1m.parquet` for the key check only |
# | Writes | `features/financial.parquet` + `financial.parquet.digest.json` |
# | Read by | [`04_evaluation`](04_evaluation.ipynb), and the modeling stages downstream of it |
#
# The feature *screen* - information coefficients, HAC uncertainty, false-discovery control - is
# not here. `04_evaluation` owns it, and it reaches one decision per feature with three things
# around the correlation: how uncertain that correlation is once monthly overlap is accounted for,
# whether its sign holds across the walk-forward folds, and how many features were tested at once.
# A bare correlation computed here, as an appendix to construction, would carry none of them.

# %%
"""US Firm Characteristics: Feature Engineering."""

import warnings
from datetime import date

import polars as pl
import yaml

from case_studies.utils.artifact_digest import value_digest, write_artifact
from case_studies.utils.feature_engineering import (
    assert_values_agree,
    families_from_config,
    plot_coverage_through_time,
    plot_cross_sectional_dispersion,
    plot_feature_distributions,
    plot_persistence,
    plot_redundancy_clusters,
    plot_timing_contract,
    register_frame,
    warmup_audit,
)
from data import load_firm_characteristics
from utils.paths import display_path, get_case_study_dir

warnings.filterwarnings("ignore")

CASE_DIR = get_case_study_dir("us_firm_characteristics")
FEATURES_DIR = CASE_DIR / "features"
LABELS_DIR = CASE_DIR / "labels"

# %% [markdown]
# Production runs both dates as `None` and takes the window from `setup.yaml`; CI overrides them
# to shorten it. There is no firm cap: every family below is cross-sectional, so capping the
# cross-section would change the ranks rather than the runtime.

# %% tags=["parameters"]
START_DATE = None
END_DATE = None

# %% [markdown]
# ## Configuration
#
# The window, the register, the label name and the holdout boundary are declared in
# `config/setup.yaml` and bound here. A window retyped in the notebook is a second source of truth
# for a decision that the register, the warmup audit and the timing figure all have to agree on.

# %%
setup = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())
FAMILIES = families_from_config(setup)
WINDOW = setup["features"]["window"]
PRIMARY_LABEL = setup["labels"]["primary"]
HOLDOUT_START = date.fromisoformat(str(setup["evaluation"]["holdout_start"]))
PERIODS_PER_YEAR = setup["evaluation"]["periods_per_year"]

WINDOW_START = date.fromisoformat(START_DATE) if START_DATE else WINDOW["start"]
WINDOW_END = date.fromisoformat(END_DATE) if END_DATE else WINDOW["end"]

# The panel key, and the partition every within-date statistic is taken over.
ENTITY = "symbol"
PANEL_KEY = ["symbol", "timestamp"]

# Two features whose orderings agree this closely are read as one piece of evidence in F5.
# It is a reading aid for the dendrogram and not a screen: nothing is dropped at this level,
# here or anywhere. `04_evaluation` decides what survives, from its own correlation matrix.
REDUNDANCY_CUT = 0.7

# Every table below is meant to be read in full. Polars shows ten rows and cuts a string cell
# at 32 characters by default, which would hide three of the register's families and clip the
# longest failure mode, so both limits are taken from what the register actually holds.
WIDEST_REGISTER_CELL = max(len(str(v)) for row in register_frame(FAMILIES).iter_rows() for v in row)
pl.Config.set_tbl_rows(max(50, len(FAMILIES)))
pl.Config.set_fmt_str_lengths(WIDEST_REGISTER_CELL)

print(f"Feature window: {WINDOW_START} to {WINDOW_END}, both months included.")
print(f"Register: {len(FAMILIES)} families - the whole of what the matrix may contain.")
print(f"Primary label: {PRIMARY_LABEL}, the one-month forward return Section E keys against.")
print(f"Holdout: {HOLDOUT_START} onward. In the artifact, and in D and E's integrity checks.")
print("  Section F, where a reader forms a view of the feature set, reads only what precedes it.")
print(f"Decision dates per year: {PERIODS_PER_YEAR}, which is how far F6 runs its lag axis.")

# %% [markdown]
# ## A. What the thesis says should carry information
#
# The thesis is cross-sectional and it is not one claim but six: among the several thousand firms
# the release covers in a month, the ones that are cheap against fundamentals, more profitable,
# growing their asset base less aggressively, and trending, earn the higher subsequent return, read
# against a risk and liquidity state that says how much of that ordering is tradable.
#
# The register below is declared in `config/setup.yaml::features.families` rather than here, for the
# same reason the label name and the holdout boundary are: it states what the feature set *is*, and
# a statement only the notebook holds cannot be read by a test, by a later stage, or by anyone
# asking what changed between two runs.
#
# It comes in two halves. The first says **when** each family is knowable, which is what decides
# whether a feature is usable at all; the second says **why** it should carry information, which is
# what the rest of the pipeline is going to test.
#
# A **bar** here is one month, because this panel is monthly: a lookback of 12 bars is a window a
# year long, and a lag of 6 bars means six months pass between the period a number describes and
# the decision allowed to read it. Of the two columns, `lag` is the one that decides look-ahead and
# it is fully sourced - the release publishes its update conventions, and an annual variable is
# published at the end of June against a December fiscal year end. `lookback` is not equally
# sourced: the release does not publish a per-characteristic estimation window. Where a
# characteristic names its own window it is used (`r36_13` reads 36 months back to 13); otherwise
# the entry is the span of one provider observation. Section B says what follows from that, and
# Section D audits only what this notebook is in a position to audit.

# %%
register_frame(FAMILIES).select(
    ["family", "role", "inputs", "lookback (bars)", "lag (bars)", "frame", "representation"]
)

# %% [markdown]
# The second half is the claim itself. `driver hypothesis` is what the family is expected to say
# about subsequent returns, and `failure mode` is the way that claim is most likely to be wrong -
# written down here, before anything is measured, so that a later stage finding nothing has a
# stated alternative to weigh rather than an absence to explain.

# %%
register_frame(FAMILIES).select(["family", "driver hypothesis", "failure mode"])

# %% [markdown]
# ## B. Inputs and their observability
#
# `load_firm_characteristics(split="all")` returns the released panel: one row per firm-month,
# 46 characteristics, a realized return and the authors' split label. This notebook consumes the
# 46 characteristics and the panel key. It does not consume `ret` - the label panel is built by
# `02_labels`, and reading a realized return here would put the outcome in the feature matrix.
#
# **The firm identity is the load-bearing input, and it has to be recovered.** The release
# publishes three blocks of a three-dimensional array - date by firm by characteristic - and within
# a block the firm axis is fixed, so position *j* on that axis is the same company at every date.
# That position is what `symbol` holds, offset per block because the archive publishes no mapping
# between them, and it is what `02_labels` keys its labels on.
#
# What is not a firm is a row's place in a month's cross-section. A firm appears in a month only
# where its return is present, so flattening the array to one row per firm-month leaves a different
# number of rows each month and renumbers them as firms enter and leave. Keying a matrix on that
# position produces something that looks like an identifier, joins to no label, and raises nothing
# while doing it - which is why Section E asserts the join against the label panel.
#
# **What observability means for a panel that arrives ranked.** The provider computed each
# characteristic's window and applied its own publication convention before releasing the panel, and
# it releases complete cases only. Two consequences follow, and both constrain what Section D can
# claim. The leading nulls that a rolling window leaves behind were removed upstream, so this panel
# cannot evidence the windows in the register. And no null policy of this notebook's own is needed
# for the released columns, because there are no nulls in them - Section E asserts that rather than
# assuming it.

# %%
firm_chars = load_firm_characteristics(split="all").with_columns(pl.col("timestamp").cast(pl.Date))
firm_chars = firm_chars.filter(
    pl.col("timestamp").is_between(pl.lit(WINDOW_START), pl.lit(WINDOW_END))
)

CONSUMED = ["timestamp", "symbol", "ret", "split"]
RELEASED = [c for c in firm_chars.columns if c not in CONSUMED]

print(f"Loaded {len(firm_chars):,} firm-months over {firm_chars['timestamp'].n_unique()} months")
print(f"Period {firm_chars['timestamp'].min()} to {firm_chars['timestamp'].max()}")
print(f"Persistent anonymous firms: {firm_chars['symbol'].n_unique():,}")
print(f"Released characteristics consumed: {len(RELEASED)}")

# %% [markdown]
# ### What the released panel actually contains
#
# The table below is the whole input: every characteristic the release publishes, under the
# register family whose pattern claims it. The names are the provider's own abbreviations, and the
# ones this notebook goes on to use by themselves are worth reading off before it does.
#
# `BEME` is book value of equity over market value of equity - the classic measure of cheapness,
# high when the market prices a firm below what its accounts say it owns. `CF2P` is cash flow over
# price and `E2P` earnings over price, the same question asked of two different accounting lines.
# `PROF` is gross profitability and `ROE` return on equity, two ways of asking how much a firm earns
# on what it holds. `LME` is log market equity, which is size. `IdioVol` is the volatility of the
# part of a firm's return that its market exposure does not explain. `r12_2` is the cumulative
# return from twelve months back to two months back and `r12_7` from twelve back to seven: momentum
# measured over roughly a year, both stopping short of the most recent month, which tends to
# reverse rather than continue.

# %%
MEMBERS = {family.name: [c for c in RELEASED if family.matches(c)] for family in FAMILIES}

pl.DataFrame(
    [
        {"family": name, "n": len(columns), "the release's names": ", ".join(sorted(columns))}
        for name, columns in MEMBERS.items()
        if columns
    ]
)

# %% [markdown]
# ## C. Feature construction
#
# Three subsections, and none of them fits a parameter or reads another row.

# %% [markdown]
# ### C.1 The released characteristics, carried through unchanged
#
# The 46 characteristics arrive already cross-sectionally rank-transformed onto the symmetric
# interval the register records under `representation`, and Section E prints the range they
# actually span. They are carried through as they arrive. Re-ranking them
# here would be a second cross-sectional transform over a quantity that is already one, and a
# rolling z-score over a bounded rank is a statistic of the ranking, not of the firm.

# %%
features = firm_chars.select(PANEL_KEY + RELEASED)
print(f"Released characteristics carried through: {len(RELEASED)}")

# %% [markdown]
# ### C.2 Within-family and cross-family composites
#
# Seven equal-weight means: one per accounting family, one over the two 12-month momentum
# characteristics, and three cross-family pairs. Averaging ranks cancels characteristic-specific
# noise while keeping the members' scale, so a composite is comparable with the columns it averages.
#
# Each mean divides by the number of members present in that row rather than by the family size, so
# a composite is null only where every member is null. On this release that distinction never bites,
# because the panel is complete; it is written this way so the same code gives a composite rather
# than whichever members happened to be published on a release that is not.


# %%
def family_mean(df: pl.DataFrame, columns: list[str], alias: str) -> pl.DataFrame:
    """Row-wise mean over *columns*, dividing by the per-row non-null count."""
    present = [c for c in columns if c in df.columns]
    total = pl.sum_horizontal(pl.col(c) for c in present)
    count = pl.sum_horizontal(pl.col(c).is_not_null().cast(pl.Int32) for c in present)
    return df.with_columns(
        pl.when(count > 0).then(total / count).otherwise(None).alias(alias),
    )


# %%
MOMENTUM_12M = [c for c in ("r12_2", "r12_7") if c in RELEASED]

features = family_mean(features, MEMBERS["value"], "composite_value")
features = family_mean(features, MEMBERS["quality"], "composite_quality")
features = family_mean(features, MEMBERS["investment"], "composite_investment")
features = family_mean(features, MOMENTUM_12M, "composite_momentum")

features = features.with_columns(
    ((pl.col("composite_value") + pl.col("composite_quality")) / 2).alias(
        "composite_value_quality"
    ),
    ((pl.col("composite_value") + pl.col("composite_momentum")) / 2).alias(
        "composite_value_momentum"
    ),
    ((pl.col("composite_quality") + pl.col("composite_momentum")) / 2).alias(
        "composite_quality_momentum"
    ),
)

COMPOSITES = [c for c in features.columns if c.startswith("composite_")]
print(f"Composites: {len(COMPOSITES)} - {', '.join(COMPOSITES)}")

# %% [markdown]
# ### C.3 Interactions
#
# Four products of two ranks each, for theses that are conditional rather than additive: cheap is
# worth more when the firm is also profitable, and momentum reads differently at high idiosyncratic
# volatility. Because the members are centred on zero, the product encodes *agreement* - it is
# positive when both ranks sit on the same side of the cross-section and negative when they
# disagree, and it is large at both extremes, which is the failure mode the register records.

# %%
features = features.with_columns(
    (pl.col("BEME") * pl.col("PROF")).alias("interaction_value_x_quality"),
    (pl.col("BEME") * pl.col("ROE")).alias("interaction_value_x_roe"),
    (pl.col("r12_2") * pl.col("IdioVol")).alias("interaction_momentum_x_ivol"),
    (pl.col("LME") * pl.col("BEME")).alias("interaction_size_x_value"),
)

INTERACTIONS = [c for c in features.columns if c.startswith("interaction_")]
CONSTRUCTED = COMPOSITES + INTERACTIONS
FEATURE_COLS = RELEASED + CONSTRUCTED
print(f"Interactions: {len(INTERACTIONS)}")
print(
    f"Feature matrix: {len(FEATURE_COLS)} columns ({len(RELEASED)} released, "
    f"{len(CONSTRUCTED)} constructed)"
)

# %% [markdown]
# ## D. The timing contract

# %% [markdown]
# ### D.1 Every operation, and the window it sees
#
# | Operation | Where | Rows it reads | Window |
# |---|---|---|---|
# | Window filter | B | its own row | none |
# | Carry released characteristic | C.1 | its own row | none |
# | Family mean | C.2 | its own row, member columns | none |
# | Product of two ranks | C.3 | its own row | none |
#
# No feature is built by a rolling operation, an expanding operation, a ranking or a
# cross-sectional aggregate: every value in the matrix is a function of its own row. Sections E
# and F do group by timestamp, to count coverage and to take within-month percentiles for the
# figures, but nothing they compute is written to the artifact. The provider's windows are
# upstream and are recorded in the register; the figure below draws them so that a reader of the
# model matrix knows what each column spans, even though the span was not computed here.

# %%
plot_timing_contract(
    FAMILIES,
    bar_unit="months",
    title="Accounting families become knowable six months after their period",
    subtitle=("Register lookback and lag, in months; zero is the decision date"),
    alt=(
        "Horizontal bars, one per register family, spanning the months each family's window "
        "reads and ending at a gap equal to the months before it becomes knowable. Thirteen "
        "bars. Seven stop six months short of the decision timestamp, their gap drawn hatched: "
        "value, quality, investment, other, composite accounting, composite investment and "
        "interaction accounting. Six reach the decision: momentum, risk, composite momentum, "
        "interaction momentum, and the two mixed groups, which pair an accounting member with "
        "a price member and so span 18 months with no gap. Momentum spans the longest window "
        "at 36 months; the investment rows span 24."
    ),
)

# %% [markdown]
# ### D.2 Warmup, asserted
#
# `warmup_audit` holds each constructed column to the number of bars its window spans and raises if
# a column is populated earlier than that, or is null everywhere.
#
# **What this audit reaches, and what it cannot.** The released characteristics are complete cases:
# the provider removed their warmup nulls before publishing, so holding them to the register's
# lookbacks would fail on every one of them, and what it would be measuring is the provider's screen
# rather than this notebook's construction. They are therefore not in the audit. The constructed
# columns are functions of their own row and span no window of their own, so their declared floor is
# zero, and the branch that fires for them is the one that catches a column null everywhere - which
# is what a mistyped member list produces.
#
# What can actually go wrong here is the member list, and two assertions cover it. The register's
# patterns must partition the released columns, each claimed exactly once and no family left empty,
# which a mistyped pattern breaks. And each composite must equal the mean of its declared members,
# recomputed by a separate route, which a fault in `family_mean` breaks. Neither is satisfied by the
# panel being complete, so neither passes for free.

# %%
census = warmup_audit(features, dict.fromkeys(CONSTRUCTED, 0), entity=ENTITY)
print(census)

# All seven composites, not just the four built straight from released columns: the three
# cross-family ones average two composites, so their declared member list is those two.
COMPOSED_OF = {
    "composite_value": MEMBERS["value"],
    "composite_quality": MEMBERS["quality"],
    "composite_investment": MEMBERS["investment"],
    "composite_momentum": MOMENTUM_12M,
    "composite_value_quality": ["composite_value", "composite_quality"],
    "composite_value_momentum": ["composite_value", "composite_momentum"],
    "composite_quality_momentum": ["composite_quality", "composite_momentum"],
}
assert set(COMPOSED_OF) == set(COMPOSITES), (
    f"composites built but not audited: {sorted(set(COMPOSITES) - set(COMPOSED_OF))}"
)

# 1. the register partitions the released columns: claimed exactly once, no family empty
claimed = [c for family in FAMILIES for c in RELEASED if family.matches(c)]
assert sorted(claimed) == sorted(RELEASED), (
    "the register does not claim the released columns exactly once: "
    f"{sorted(set(claimed) ^ set(RELEASED))} claimed twice or not at all"
)
resolved = {f.name: [c for c in FEATURE_COLS if f.matches(c)] for f in FAMILIES}
empty = sorted(name for name, cols in resolved.items() if not cols)
assert not empty, f"register families that claim no column in the matrix: {empty}"

# 2. each composite equals the mean of its declared members, recomputed separately
for name, members in COMPOSED_OF.items():
    independent = features.select(members).mean_horizontal()
    gap = (features[name] - independent).abs().max()
    assert gap is not None and gap < 1e-12, (
        f"{name} does not equal the mean of {members}: max gap {gap}"
    )
print(f"Register claims all {len(RELEASED)} released columns exactly once")
print(f"All {len(COMPOSED_OF)} composites equal the mean of their declared members")

# %% [markdown]
# ### D.3 Rebuild without the holdout, and compare
#
# The strongest available statement about look-ahead is not that no transform was fitted, it is that
# withholding the holdout changes nothing. The panel is rebuilt from the pre-holdout rows alone and
# compared value by value against the full build cut back to the same rows. A trailing statistic
# reads only its own row's history and a within-row statistic reads only its own row, so both are
# unchanged; a parameter fitted over a whole column is not, and this comparison finds it without
# anyone having remembered to flag it.


# %%
def build(panel: pl.DataFrame) -> pl.DataFrame:
    """Run C.1 to C.3 over *panel*, so the rebuild is the same code path as the build."""
    out = panel.select(PANEL_KEY + RELEASED)
    out = family_mean(out, MEMBERS["value"], "composite_value")
    out = family_mean(out, MEMBERS["quality"], "composite_quality")
    out = family_mean(out, MEMBERS["investment"], "composite_investment")
    out = family_mean(out, MOMENTUM_12M, "composite_momentum")
    out = out.with_columns(
        ((pl.col("composite_value") + pl.col("composite_quality")) / 2).alias(
            "composite_value_quality"
        ),
        ((pl.col("composite_value") + pl.col("composite_momentum")) / 2).alias(
            "composite_value_momentum"
        ),
        ((pl.col("composite_quality") + pl.col("composite_momentum")) / 2).alias(
            "composite_quality_momentum"
        ),
    )
    return out.with_columns(
        (pl.col("BEME") * pl.col("PROF")).alias("interaction_value_x_quality"),
        (pl.col("BEME") * pl.col("ROE")).alias("interaction_value_x_roe"),
        (pl.col("r12_2") * pl.col("IdioVol")).alias("interaction_momentum_x_ivol"),
        (pl.col("LME") * pl.col("BEME")).alias("interaction_size_x_value"),
    )


# %%
agreement = assert_values_agree(
    features.filter(pl.col("timestamp") < pl.lit(HOLDOUT_START)),
    build(firm_chars.filter(pl.col("timestamp") < pl.lit(HOLDOUT_START))),
    columns=FEATURE_COLS,
    keys=PANEL_KEY,
)
print(
    f"Compared {len(FEATURE_COLS)} columns over "
    f"{agreement['rows compared'].max():,} pre-holdout rows."
)
print(f"Largest disagreement in any of them: {agreement['max abs difference'].max():g}")
print(
    f"Values present on one side and null on the other: {agreement['null only on one side'].sum()}"
)

# %% [markdown]
# ## E. Matrix assembly and coverage
#
# The panel key is `symbol` + `timestamp`, and it is asserted unique. Two columns of the released
# panel are excluded, each for a stated reason:
#
# | Excluded | Reason |
# |---|---|
# | `ret` | the realized return, which is the outcome; `02_labels` owns it |
# | `split` | the authors' own train/test partition, superseded by this study's `evaluation` block |
#
# **One null policy, applied once.** The release publishes complete cases, so the policy is to
# assert completeness rather than to impose a rule: every released characteristic must be fully
# populated over the window, and every constructed column must be populated wherever its members
# are. Nothing is dropped, so the row set the artifact carries is the row set the release
# published, and the assertions below say so loudly the first time that stops being true.

# %%
assert features.select(PANEL_KEY).n_unique() == len(features), "panel key is not unique"

null_counts = {c: features[c].null_count() for c in FEATURE_COLS}
populated = {c: n for c, n in null_counts.items() if n}
assert not populated, f"released panel is not complete: {populated}"

labels = (
    pl.read_parquet(LABELS_DIR / f"{PRIMARY_LABEL}.parquet")
    .select(PANEL_KEY)
    .filter(pl.col("timestamp").is_between(pl.lit(WINDOW_START), pl.lit(WINDOW_END)))
)
joined = features.select(PANEL_KEY).join(labels, on=PANEL_KEY, how="inner")
assert joined.height == len(features) == labels.height, (
    f"feature and label identities do not align one to one: {len(features):,} features, "
    f"{labels.height:,} labels, {joined.height:,} joined - the two panels are keyed on "
    "different firm identities"
)

coverage = (
    features.group_by("timestamp")
    .agg(
        [
            pl.mean_horizontal(pl.col(c).is_not_null() for c in members).mean().alias(family)
            for family, members in MEMBERS.items()
            if members
        ]
        + [
            pl.mean_horizontal(pl.col(c).is_not_null() for c in COMPOSITES)
            .mean()
            .alias("composite"),
            pl.mean_horizontal(pl.col(c).is_not_null() for c in INTERACTIONS)
            .mean()
            .alias("interaction"),
        ]
    )
    .sort("timestamp")
)

plot_coverage_through_time(
    coverage,
    title="Every family is complete in every month the release publishes",
    subtitle=("Non-null share per family per month; watch for any dip below one"),
    alt=(
        "Non-null share per feature family plotted monthly. All eight families sit flat at "
        "1.0 across the whole window, with no dips."
    ),
)

# %% [markdown] tags=["results"]
# ### E. What the matrix contains
#
# The cell below is the matrix as it will be written: its row count, its key, its per-month
# cross-section, and the number of columns in each register family.

# %%
per_month = features.group_by("timestamp").len()["len"]
print(f"Rows: {len(features):,}   key: {' + '.join(PANEL_KEY)}")
print(f"Firms: {features['symbol'].n_unique():,}   months: {features['timestamp'].n_unique()}")
print(
    f"Cross-section per month: min {per_month.min():,}  median {int(per_month.median()):,}  "
    f"max {per_month.max():,}"
)
print(f"Features: {len(FEATURE_COLS)}")
released_min = min(features[c].min() for c in RELEASED)
released_max = max(features[c].max() for c in RELEASED)
print(f"Released characteristics span [{released_min:.3f}, {released_max:.3f}]")

register_frame(FAMILIES, columns=FEATURE_COLS).select(
    ["family", "columns", "role", "representation"]
)

# %% [markdown]
# ## F. What the features look like
#
# Four descriptive views: shape, spread through time, redundancy, and persistence. None of them
# states an information coefficient - `04_evaluation` owns predictive strength and measures it with
# fold stability and false-discovery control around it. What these establish is whether the columns
# are shaped the way the register says they
# are, and whether they are distinct enough and stable enough to be worth screening at all.
#
# **Every view below reads development rows only.** Section E wrote the full panel, because the
# artifact has to carry the holdout rows for a later stage to score them. This section is
# different: it is where a reader forms a judgement about which features are worth carrying
# forward, and a judgement about the feature set is selection whether or not a metric is
# attached to it. So it is formed on the rows before the boundary.
#
# The cluster assignment below is **descriptive and local**: it stays in this notebook, and
# `04_evaluation` builds its own pairwise Spearman matrix and triages every feature rather than
# one representative per cluster. What this view is for is telling a reader how many distinct
# orderings the matrix actually carries, before any column is scored. It is computed on a seeded
# 200,000-row sample of the development panel rather than all of it, which is enough to place a
# correlation to two decimals and keeps a full pairwise rank matrix over every column affordable.
#
# Read the dendrogram by where branches meet rather than by how they are coloured: two columns are
# in the same cluster when the branch joining them meets to the right of the dashed cut. The cell
# after the figure lists the memberships that matter, one line per composite.

# %%
development = features.filter(pl.col("timestamp") < pl.lit(HOLDOUT_START))
print(
    f"Section F reads {len(development):,} of {len(features):,} rows "
    f"({development['timestamp'].max()} is the last, holdout starts {HOLDOUT_START})"
)

# %%
plot_feature_distributions(
    development,
    MEMBERS["value"],
    title="The released ranks are uniform on the provider's interval, not bell-shaped",
    subtitle="Value family, pooled across development months; tails clipped at 0.5%",
    alt=(
        "Six histograms, one per value characteristic. Each is flat across the interval "
        "minus 0.5 to 0.5 rather than peaked in the middle, which is what a cross-sectional "
        "rank transform produces."
    ),
)

# %%
plot_cross_sectional_dispersion(
    development,
    "composite_value",
    title="Spread in the value composite is stable across three decades",
    subtitle="10th-90th percentile band with the median, taken within each month",
    alt=(
        "A band between the 10th and 90th percentile of the value composite computed within "
        "each month, with the median through it. The band holds a near-constant width across "
        "the window."
    ),
)

# %%
clusters = plot_redundancy_clusters(
    development,
    FEATURE_COLS,
    cut=REDUNDANCY_CUT,
    title="Only the momentum composite clusters with all the columns it averages",
    subtitle=(
        r"Distance is $1-|\rho_s|$ over Spearman ranks, so the dashed cut at "
        rf"{1 - REDUNDANCY_CUT:.1f} is $|\rho_s|={REDUNDANCY_CUT}$"
    ),
    alt=(
        "A dendrogram over every column of the matrix, drawn from a distance of 1.0 on the left "
        "down to 0.0 on the right, with each column named along the right edge. A dashed "
        "vertical line marks the cut. Two columns belong to the same cluster when the branch "
        "joining them meets to the right of that line; most branches meet well to the left of "
        "it, so most columns stand alone. One of the two largest groups to the right of the cut "
        "holds the momentum composite together with the two return characteristics it averages "
        "and the two cross-family composites built from it. The cell below counts the clusters and "
        "lists, for each composite, which of its own members ended up beside it."
    ),
)

# %%
decision_dates = development["timestamp"].unique().sort().to_list()
plot_persistence(
    development,
    ["composite_value", "composite_quality", "r12_2"],
    entity=ENTITY,
    max_lag=PERIODS_PER_YEAR,
    decision_dates=decision_dates,
    title="Accounting composites persist at 12 months; momentum has reversed",
    subtitle=("Left: per-firm autocorrelation by lag. Right: month-to-month rank correlation"),
    alt=(
        "Two panels. On the left, autocorrelation against lag in months for two accounting "
        "composites and 12-month momentum; the accounting series decay slowly and momentum "
        "decays faster. On the right, month-to-month rank correlation for the same three."
    ),
)

# %% [markdown] tags=["results"]
# ### F. Redundancy, as numbers
#
# A dendrogram shows structure but not counts, so the counts go here: how many groups the matrix
# falls into once closely-agreeing features are read together, and how many of the columns a
# composite averages end up beside it. Both are over development rows only.
#
# The linkage is average, so a group is one whose *average* distance to its neighbours is below the
# cut. Two columns can therefore share a group without their own correlation reaching 0.7, which is
# why the count below is reported as clusters at a cut and not as a pairwise threshold.

# %%
n_clusters = len(set(clusters.values()))
largest = max(sum(1 for v in clusters.values() if v == c) for c in set(clusters.values()))
shared = sum(1 for c in set(clusters.values()) if sum(1 for v in clusters.values() if v == c) > 1)
print(
    f"Average-linkage clusters cut at distance {1 - REDUNDANCY_CUT:.1f}: "
    f"{n_clusters} over {len(FEATURE_COLS)} columns"
)
print(f"Clusters holding more than one column: {shared}; the largest holds {largest}")

for name, members in COMPOSED_OF.items():
    together = sorted(m for m in members if clusters[m] == clusters[name])
    print(f"  {name}: {len(together)} of {len(members)} members in its cluster - {together}")

# %% [markdown]
# ## G. Emit
#
# The matrix is written with `write_artifact`, which puts a digest sidecar beside the parquet. The
# sidecar records the panel key, this notebook as the writer, and the digest of the input panel
# restricted to the columns and window actually consumed - so a later stage can tell whether it is
# reading the matrix that was built from the data it thinks it was.
#
# | Artifact | Read by |
# |---|---|
# | `features/financial.parquet` | [`04_evaluation`](04_evaluation.ipynb), which screens the columns and writes one decision per feature, and the modeling stages downstream of it |
# | `features/financial.parquet.digest.json` | a reader, or a maintainer, checking by hand which build of the matrix a downstream result came from |

# %%
matrix = features.select(PANEL_KEY + FEATURE_COLS).sort(PANEL_KEY)

record = write_artifact(
    matrix,
    FEATURES_DIR / "financial.parquet",
    keys=PANEL_KEY,
    written_by="case_studies/us_firm_characteristics/03_financial_features.py",
    inputs={
        "load_firm_characteristics": value_digest(
            firm_chars.select(PANEL_KEY + RELEASED).sort(PANEL_KEY)
        ),
    },
)

print(f"Wrote {display_path(FEATURES_DIR / 'financial.parquet')}")
print(f"  rows {record['n_rows']:,}   digest {record['digest']}")

# %% [markdown]
# ## Key takeaways
#
# 1. A panel that arrives cross-sectionally ranked moves the timing question upstream. The
#    construction here fits nothing and reads no other row, so the contract can be demonstrated by
#    rebuilding without the holdout and comparing values rather than argued for in prose.
# 2. State what an audit can reach. The provider releases complete cases, so its warmup nulls are
#    gone and no assertion made here can recover the windows the register records; saying so is
#    worth more than an assertion that passes because there is nothing left for it to catch.
# 3. A composite is only a composite if it averages the member list it declares. What establishes
#    that is D.2 recomputing each one from its declared members by a separate route, and asserting
#    the register's patterns claim every released column exactly once. Choose checks that can fail
#    on the data in front of you: on a panel with no nulls anywhere, a nullity check passes whatever
#    the composites contain.
# 4. Where a release already guarantees complete cases, the null policy is to assert that guarantee
#    rather than to impose a rule on top of it. Dropping rows on whichever columns a frame happens
#    to list first is a screen nobody chose, and it stays invisible until the data changes.
# 5. The firm identity is part of the artifact's contract, not an implementation detail. Flattening
#    an array to a panel offers two positions that both look like identifiers - the fixed axis,
#    which is one, and the row's place in a filtered cross-section, which is not - so Section E
#    asserts the one-to-one join against the labels instead of assuming it.
#
# **Known limitations.** The register's lookbacks are the provider's stated conventions and the
# windows some characteristics name, not per-characteristic estimation windows, which the release
# does not publish. The `other` family is a residual grouping with no single thesis, so a
# family-level reading of it means less than it does for the five named families.
#
# **Next**: [`04_evaluation`](04_evaluation.ipynb) screens these features, with HAC uncertainty,
# fold-level sign stability and false-discovery control.
