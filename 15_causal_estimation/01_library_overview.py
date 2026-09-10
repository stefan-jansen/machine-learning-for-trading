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
# # Causal Inference Library Decision Guide
#
# **Chapter 15: Causal Estimation with ML**
# **Docker image**: `ml4t`
# **Section Reference**: See Section 15.1 (Table 15.1) for the method selection guide
#
# ## Purpose
# Chapter 15 uses five Python libraries, and they do not compete with one another.
# This notebook maps a causal question to the library that answers it, and shows the
# API each one expects: what it takes as treatment, outcome, controls and graph.
#
# ## Why the estimates below are not a benchmark
#
# The same synthetic sample runs through several libraries, which invites a ranking.
# Four differences prevent one:
#
# 1. **Different problems.** Effect estimation (EconML, DoWhy, CausalML) starts from a
#    causal structure and quantifies an effect. Structure discovery (Tigramite,
#    causal-learn) starts from data and proposes a structure.
# 2. **Different treatment types.** EconML and DoWhy handle a continuous treatment
#    directly; CausalML is built for binary and discrete interventions.
# 3. **Different estimands.** The **estimand** is the quantity the analysis targets.
#    Binarizing a continuous treatment to suit CausalML changes the estimand from a
#    marginal effect to a contrast between a high and a low group.
# 4. **Different assumptions.** Each library asks for a different account of how the
#    data were generated and what confounding remains.
#
# ## Learning Objectives
# After completing this notebook, you will be able to:
# - LO1: Match a causal question to the library that answers it
# - LO2: Apply the selection guide in Table 15.1 to your own problem
# - LO3: Read the API each library expects, and which argument carries the controls
# - LO4: Separate effect estimation from structure discovery
#
# ## Cross-References
# - **Upstream**: None (synthetic data with a known data-generating process)
# - **Downstream**: Each library has a notebook of its own
#   - EconML: [`03_econml_dml`](03_econml_dml.ipynb), [`04_dml_crypto_regime`](04_dml_crypto_regime.ipynb)
#   - DoWhy: [`02_dowhy_causal_graph`](02_dowhy_causal_graph.ipynb)
#   - BSTS: [`06_fed_announcement_bsts`](06_fed_announcement_bsts.ipynb)
#   - Tigramite: [`07_tigramite_time_series`](07_tigramite_time_series.ipynb)
#   - causal-learn: [`08_neural_causal_discovery`](08_neural_causal_discovery.ipynb)
#
# ## Libraries Covered
# **Effect estimation**, given a causal structure:
# 1. **EconML** (Microsoft) - double machine learning (DML) and metalearners for a
#    continuous or binary treatment
# 2. **DoWhy** (Microsoft/Amazon) - explicit graph, identification, refutation tests
#
# **Structure discovery**, to propose a causal graph:
# 3. **Tigramite** - PCMCI for time series
# 4. **causal-learn** (CMU) - PC, FCI, GES, LiNGAM and Granger, for contemporaneous
#    and lagged structure
#
# **Also referenced**: CausalML (binary treatment, not used in this chapter),
# tfp-causalimpact (Bayesian structural time series for event studies, used in
# `06_fed_announcement_bsts`)
#
# **Prerequisites**: None
#
# ## Causal Design Contract
#
# | Element                   | This notebook                                                                          |
# |---------------------------|----------------------------------------------------------------------------------------|
# | Unit                      | Synthetic observation (1 of 1,000 i.i.d. rows generated from a known DGP)               |
# | Treatment                 | `momentum` (continuous), partly driven by `volatility` and `regime`                    |
# | Outcome                   | `returns` (continuous), driven by treatment and confounders; the true average treatment effect (ATE) is bound to `TRUE_ATE` below |
# | Controls (W in EconML)    | `volatility`, `regime` - both confound the treatment-outcome path                      |
# | Effect modifiers (X)      | None; the target here is a constant ATE, and `04_dml_crypto_regime` uses X for regime heterogeneity |
# | Identification assumption | The synthetic DGP is fully observed, so there is no unobserved confounding by design    |
# | Main failure mode         | None in identification; this is an API smoke test, not an empirical finding             |

# %% [markdown]
# ## Setup

# %%
"""Causal Inference Library Decision Guide - choose the right causal library for your problem."""

import warnings
from importlib.metadata import PackageNotFoundError, version

import networkx as nx
import numpy as np
import pandas as pd

from utils.reproducibility import set_global_seeds

# Two third-party import-time warnings, each silenced by category and module: DoWhy 0.14
# compiles regexes and docstrings with unescaped backslashes, and pydot calls pyparsing
# methods pyparsing has renamed. Convergence and numerical warnings stay visible.
warnings.filterwarnings("ignore", category=SyntaxWarning, module=".*dowhy")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydot")

# networkx 3.x removed nx.algorithms.d_separated; DoWhy 0.14 still calls it.
# Bind the renamed replacement to the old name before importing DoWhy.
if not hasattr(nx.algorithms, "d_separated"):
    nx.algorithms.d_separated = nx.d_separation.is_d_separator


# %% tags=["parameters"]
# Production defaults - Papermill injects overrides for CI
SEED = 42

# %%
set_global_seeds(SEED)
rng = np.random.default_rng(SEED)

# %% [markdown]
# ## Check Library Availability
#
# A library counts as available when it imports. The version comes from the installed
# distribution metadata rather than from the package itself, because three of these five
# packages expose no version attribute.

# %%
lib_checks = [
    ("EconML", "econml", "econml"),
    ("DoWhy", "dowhy", "dowhy"),
    ("CausalML", "causalml", "causalml"),
    ("Tigramite", "tigramite", "tigramite"),
    ("causal-learn", "causallearn", "causal-learn"),
]

libraries = {}
for name, module_name, dist_name in lib_checks:
    try:
        __import__(module_name)
    except ImportError:
        libraries[name] = None
        continue
    try:
        libraries[name] = version(dist_name)
    except PackageNotFoundError:
        libraries[name] = "installed"

status_df = pd.DataFrame(
    [
        {"Library": k, "Version": v or "not installed", "Available": v is not None}
        for k, v in libraries.items()
    ]
).set_index("Library")
status_df

# %% [markdown]
# ## Generate Common Demonstration Data
#
# One synthetic dataset runs through every library so the API calls sit side by side.
# `volatility` and `regime` drive both the treatment and the outcome, which makes them
# confounders: an unadjusted regression of `returns` on `momentum` picks up their effect
# as well as the treatment's.

# %%
n = 1000

# Confounders
volatility = rng.exponential(0.02, n)
regime = rng.binomial(1, 0.6, n)

# Treatment (momentum) - depends on the confounders
momentum = 0.5 * regime - 2 * volatility + rng.normal(0, 0.1, n)

# Outcome (returns) - depends on the confounders and on the treatment
TRUE_ATE = 0.02  # the coefficient on momentum, which every estimator below targets
returns = TRUE_ATE * momentum + 0.1 * regime - 3 * volatility + rng.normal(0, 0.02, n)

df = pd.DataFrame(
    {
        "momentum": momentum,
        "returns": returns,
        "volatility": volatility,
        "regime": regime,
    }
)

print(f"Test data: {df.shape[0]:,} obs, true ATE = {TRUE_ATE}")
df.head()

# %% [markdown]
# The treatment carries very little independent variation: once `volatility` and `regime`
# are partialled out, what is left of `momentum` is its own noise term. That, not the size
# of the effect, is what sets how precisely any of these estimators can measure the ATE,
# which is why each estimate below is reported with an interval.

# %% [markdown]
# ## 1. EconML (Microsoft)
#
# EconML fits **double machine learning** (DML): one model predicts the outcome from the
# controls, another predicts the treatment from the controls, and the treatment effect is
# estimated from what is left over in both. It takes a continuous or a binary treatment and
# can model heterogeneous effects, so it is the tool when the question is how large an
# effect is and how it varies.
#
# Use it when the adjustment set is known but a graph is not written down.

# %%
if libraries["EconML"]:
    from econml.dml import LinearDML
    from sklearn.ensemble import GradientBoostingRegressor

    Y = df["returns"].to_numpy()
    T = df["momentum"].to_numpy()
    W = df[["volatility", "regime"]].to_numpy()  # controls used for residualization

    dml = LinearDML(
        model_y=GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=SEED),
        model_t=GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=SEED),
        cv=3,
        random_state=SEED,
    )
    dml.fit(Y, T, W=W)

    econml_ate = float(dml.ate())
    econml_lo, econml_hi = (float(b) for b in dml.ate_interval(alpha=0.05))
    print(f"EconML ATE: {econml_ate:.6f}  95% CI [{econml_lo:.6f}, {econml_hi:.6f}]")
else:
    econml_ate = econml_lo = econml_hi = None
    print("EconML not available")

# %% [markdown]
# Confounders enter EconML's DML interface as `W`, the controls used for residualization,
# not as `X`, the effect modifiers that model treatment-effect heterogeneity. For a single
# adjusted ATE the distinction does not change the answer, but it does as soon as the
# question is whether the effect differs across regimes; `04_dml_crypto_regime` uses the
# `X` role.
#
# `cv=3` splits the sample at random, which is valid here because the rows are i.i.d. by
# construction. A time series needs splits that respect the ordering, or the outcome model
# learns from the future; `03_econml_dml` walks forward instead.

# %% [markdown]
# ## 2. DoWhy (Microsoft/Amazon)
#
# DoWhy asks for the causal graph up front, derives an estimand from it, and only then
# estimates. Writing the graph down is the point: it makes the adjustment set a consequence
# of stated assumptions rather than a choice, and it gives the refutation tests something to
# perturb.
#
# Use it when the assumptions have to be visible and testable, which is most of the time in
# a research setting.

# %%
if libraries["DoWhy"]:
    from dowhy import CausalModel

    # Specify the causal graph
    causal_graph = """
    digraph {
        volatility -> momentum;
        volatility -> returns;
        regime -> momentum;
        regime -> returns;
        momentum -> returns;
    }
    """

    model = CausalModel(
        data=df,
        treatment="momentum",
        outcome="returns",
        graph=causal_graph,
    )

    identified_estimand = model.identify_effect()
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
    )

    dowhy_ate = float(estimate.value)
    dowhy_lo, dowhy_hi = (float(b) for b in np.asarray(estimate.get_confidence_intervals()).ravel())
    print(f"DoWhy ATE: {dowhy_ate:.6f}  95% CI [{dowhy_lo:.6f}, {dowhy_hi:.6f}]")
else:
    dowhy_ate = dowhy_lo = dowhy_hi = None
    print("DoWhy not available")

# %% [markdown]
# ### Estimand vs. estimator
#
# The **estimand** is the quantity being targeted; the **estimator** is the procedure that
# produces a number for it. Two libraries can report different numbers because they use
# different estimators for the same estimand, or because they target different estimands
# altogether, and only the first of those is a question about accuracy.
#
# | Estimand | Description | Libraries |
# |---|---|---|
# | $E[Y \mid do(T = t+1)] - E[Y \mid do(T = t)]$ | Marginal effect of a continuous treatment | EconML, DoWhy |
# | $E[Y \mid do(T = 1)] - E[Y \mid do(T = 0)]$ | Binary treatment effect, high group against low | CausalML |
#
# If the estimand is not identified from the data at hand, no estimator recovers it. A
# flexible model fitted to a confounded comparison returns a precise answer to the wrong
# question.

# %% [markdown]
# ## 3. Other Libraries Referenced in the Chapter
#
# **CausalML** (Uber) targets uplift with S-, T-, X- and R-learners, and its support is
# strongest for the binary or discrete interventions typical of experiments and marketing.
# Continuous treatments are possible but are not where the library is aimed. Chapter 15
# works with continuous treatments and event studies, so CausalML appears in the selection
# guide without a worked example; the
# [CausalML documentation](https://causalml.readthedocs.io/) covers the uplift setting.
#
# **tfp-causalimpact** fits a Bayesian structural time-series model and needs a pre-period
# of the outcome series to build a counterfactual, which the cross-sectional sample here
# does not provide. `06_fed_announcement_bsts` runs it on a Fed announcement.

# %% [markdown]
# ## 4. Library Comparison Summary
#
# The table lists every library the chapter references and where each one is used.


# %%
def build_library_comparison():
    """Build library comparison table showing Chapter 15 coverage."""
    comparison_data = [
        {
            "Library": "EconML",
            "Use": "Effect estimation",
            "Treatment": "Continuous/Binary",
            "Key Feature": "DML, metalearners",
            "Notebook": "03_econml_dml, 04_dml_crypto_regime",
        },
        {
            "Library": "DoWhy",
            "Use": "Effect estimation",
            "Treatment": "Continuous/Binary",
            "Key Feature": "Graph, identification, refutation",
            "Notebook": "02_dowhy_causal_graph",
        },
        {
            "Library": "CausalML",
            "Use": "Uplift/CATE",
            "Treatment": "Binary/discrete (focus)",
            "Key Feature": "S/T/X/R learners",
            "Notebook": "(not used in this chapter)",
        },
        {
            "Library": "tfp-causalimpact",
            "Use": "Event study",
            "Treatment": "Binary event",
            "Key Feature": "BSTS counterfactual",
            "Notebook": "06_fed_announcement_bsts",
        },
        {
            "Library": "Tigramite",
            "Use": "Time-series discovery",
            "Treatment": "N/A",
            "Key Feature": "PCMCI",
            "Notebook": "07_tigramite_time_series",
        },
        {
            "Library": "causal-learn",
            "Use": "Discovery",
            "Treatment": "N/A",
            "Key Feature": "PC, FCI, GES, LiNGAM, Granger",
            "Notebook": "08_neural_causal_discovery",
        },
    ]
    return pd.DataFrame(comparison_data).set_index("Library")


comparison_df = build_library_comparison()
comparison_df

# %% [markdown]
# Tigramite and causal-learn learn a graph; EconML and DoWhy quantify an effect once a
# graph or an adjustment set is settled. Discovery therefore comes before estimation rather
# than replacing it, and its output is a set of hypotheses to test rather than a structure
# to trust.
#
# Two of the chapter's discovery methods are not library calls. NOTEARS, which learns a
# contemporaneous graph by continuous optimization under a differentiable acyclicity
# constraint, is implemented directly in `08_neural_causal_discovery` following Zheng et al.
# (2018); causal-learn does not ship it. Granger causality serves there as a predictive
# baseline. causal-learn supplies the VAR-LiNGAM fit, which recovers lagged and
# contemporaneous structure from non-Gaussian residuals.

# %% [markdown]
# ## 5. Selection Guide
#
# Section 15.1 arranges the choice as a table, reproduced here. The first question is
# whether the causal graph is known, because that decides whether the problem is estimation
# or discovery.
#
# | Graph known? | Treatment type | Method | Library |
# |---|---|---|---|
# | Yes | Binary or continuous | Backdoor adjustment | DoWhy |
# | Yes | Continuous | DML | EconML |
# | Yes | Binary, at a point in time | BSTS | tfp-causalimpact |
# | Yes | Binary or discrete, uplift | Metalearners | CausalML |
# | No | Not applicable | PCMCI | Tigramite |
# | No | Not applicable | VAR-LiNGAM | causal-learn |
# | No | Not applicable | NOTEARS | implemented in `08_neural_causal_discovery` |
#
# Reading the top half: DoWhy and EconML both need an adjustment set that identifies the
# effect, and they differ in how it is supplied. DoWhy takes a graph and derives the
# adjustment set from it; EconML takes the controls directly and fits them with machine
# learning models. Neither escapes the identification assumption, so "no graph" is a reason to
# run discovery first, not a reason to prefer one estimator over the other.
#
# Reading the bottom half: the split is contemporaneous against lagged structure, not
# cross-sectional against time series. PCMCI and VAR-LiNGAM both read time-lagged
# dependence, and `08_neural_causal_discovery` runs them on the same universe; NOTEARS
# targets structure within a single cross-section.

# %% [markdown]
# ## 6. Effect Estimates, and What They Support
#
# EconML and DoWhy target the same estimand here, so their estimates are comparable with
# each other and with the known true ATE. Each is reported with a confidence interval, and
# the last column records whether that interval covers the true value.

# %%
rows = [
    {
        "Method": "True ATE",
        "Estimate": TRUE_ATE,
        "95% CI": "-",
        "Abs. error": 0.0,
        "Covers true ATE": "-",
    }
]
if econml_ate is not None:
    rows.append(
        {
            "Method": "EconML (DML)",
            "Estimate": econml_ate,
            "95% CI": f"[{econml_lo:.4f}, {econml_hi:.4f}]",
            "Abs. error": abs(econml_ate - TRUE_ATE),
            "Covers true ATE": bool(econml_lo <= TRUE_ATE <= econml_hi),
        }
    )
if dowhy_ate is not None:
    rows.append(
        {
            "Method": "DoWhy (Backdoor)",
            "Estimate": dowhy_ate,
            "95% CI": f"[{dowhy_lo:.4f}, {dowhy_hi:.4f}]",
            "Abs. error": abs(dowhy_ate - TRUE_ATE),
            "Covers true ATE": bool(dowhy_lo <= TRUE_ATE <= dowhy_hi),
        }
    )

estimates_df = pd.DataFrame(rows).set_index("Method")
estimates_df

# %% [markdown]
# Read the coverage column before the error column. The absolute error mixes sampling noise
# with bias and a single estimate cannot separate the two, so a gap between the estimate and
# the true ATE is not by itself evidence that the adjustment failed. The interval is what the
# estimator says about its own precision: an estimate whose interval covers the true value is
# consistent with it, however far the point estimate sits away, and one whose interval
# excludes it points at something the adjustment did not remove.
#
# Coverage on one sample is also the most these two rows can support. They are fitted to the
# same 1,000 observations, so their errors move together, and a single draw ranks nothing.
# CausalML would answer a different question on a binarized treatment, so its number is not
# comparable and the table leaves it out.

# %% [markdown]
# ## Key Takeaways
#
# ### Choosing a library
#
# 1. **The question decides the family.** Effect estimation given a structure: EconML,
#    DoWhy, CausalML. Structure discovery: Tigramite, causal-learn.
# 2. **The treatment decides the tool within the family.** Continuous: EconML DML or DoWhy.
#    Binary or discrete uplift: CausalML, or EconML. A single dated event: BSTS.
# 3. **The data's shape decides the discovery method.** Lagged dependence across a
#    multivariate series: PCMCI or VAR-LiNGAM. Contemporaneous structure within a
#    cross-section: NOTEARS.
#
# ### Methodological notes
#
# - **Different estimands do not compare.** An ATE from a continuous treatment and an
#   uplift estimate from a binarized one answer different questions.
# - **Discovery precedes estimation.** A discovered graph is a hypothesis; the estimator
#   that follows inherits whatever the discovery step got wrong.
# - **The two combine.** Specifying the graph in DoWhy and estimating with EconML is a
#   common pattern, and DoWhy can call EconML estimators directly.
#
# ### Going deeper
#
# This notebook shows API shape. The notebooks that follow add the parts it skips: temporal
# splits, standard errors that account for panel structure, refutation tests and sensitivity
# analysis.
# - **DML**: [`03_econml_dml`](03_econml_dml.ipynb), [`04_dml_crypto_regime`](04_dml_crypto_regime.ipynb)
# - **BSTS event studies**: [`06_fed_announcement_bsts`](06_fed_announcement_bsts.ipynb)
# - **DoWhy refutation**: [`02_dowhy_causal_graph`](02_dowhy_causal_graph.ipynb)
# - **Time-series discovery**: [`07_tigramite_time_series`](07_tigramite_time_series.ipynb)
# - **NOTEARS and VAR-LiNGAM**: [`08_neural_causal_discovery`](08_neural_causal_discovery.ipynb)
