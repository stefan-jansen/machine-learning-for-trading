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
# **Section Reference**: See Section 15.1 (Table 15.1) for library overview
#
# ## Purpose
# This notebook helps practitioners choose the right causal inference library
# for their specific problem. It provides a decision framework and demonstrates
# basic usage patterns.
#
# ## IMPORTANT: This is NOT a Fair Comparison
#
# **Why direct comparison is problematic:**
# 1. **Different problem domains**: Effect estimation (EconML, DoWhy, CausalML)
#    vs. structure discovery (Tigramite, causal-learn) solve different problems
# 2. **Different treatment types**: EconML/DoWhy handle continuous treatments;
#    CausalML is optimized for binary treatments
# 3. **Different estimands**: When we binarize continuous treatment for CausalML,
#    we change what we're estimating (effect of "high vs low" not marginal effect)
# 4. **Different assumptions**: Each library makes different assumptions about
#    data structure, confounding, and causal relationships
#
# **What this notebook provides:**
# - Decision framework for library selection
# - API usage examples for each library
# - Discussion of when each tool is appropriate
#
# ## Learning Objectives
# After completing this notebook, you will be able to:
# - LO1: Select the appropriate causal library for your specific problem
# - LO2: Apply the decision framework to match methods to questions
# - LO3: Understand the API patterns for each major library
# - LO4: Recognize when different libraries solve fundamentally different problems
#
# ## Cross-References
# - **Upstream**: None (uses synthetic data for illustration)
# - **Downstream**: Each library has dedicated notebooks for deeper coverage
#   - EconML: [`03_econml_dml`](03_econml_dml.ipynb), [`04_dml_crypto_regime`](04_dml_crypto_regime.ipynb)
#   - DoWhy: [`02_dowhy_causal_graph`](02_dowhy_causal_graph.ipynb)
#   - BSTS: [`06_fed_announcement_bsts`](06_fed_announcement_bsts.ipynb)
#   - Tigramite: [`07_tigramite_time_series`](07_tigramite_time_series.ipynb)
#   - Discovery: [`08_neural_causal_discovery`](08_neural_causal_discovery.ipynb)
#
# ## Libraries Covered
# **Effect Estimation** (given causal structure):
# 1. **EconML** (Microsoft) - DML, metalearners, continuous treatment
# 2. **DoWhy** (Microsoft/Amazon) - DAG specification, refutation tests
#
# **Structure Discovery** (learn causal graph):
# 3. **Tigramite** - Time series causal discovery (PCMCI)
# 4. **causal-learn** (CMU) - Cross-sectional discovery (PC/FCI/GES)
#
# **Also mentioned**: CausalML (binary treatment, not used in this chapter),
# tfp-causalimpact (BSTS event studies, demonstrated in `06_fed_announcement_bsts`)
#
# **Prerequisites**: None (uses synthetic data for illustration)
#
# ## Causal Design Contract
#
# | Element                   | This notebook                                                                          |
# |---------------------------|----------------------------------------------------------------------------------------|
# | Unit                      | Synthetic observation (1 of 1,000 i.i.d. rows generated from a known DGP)              |
# | Treatment                 | `momentum` (continuous), partially driven by `volatility` and `regime`                 |
# | Outcome                   | `returns` (continuous), driven by treatment and confounders with known true ATE = 0.02 |
# | Controls (W in EconML)    | `volatility`, `regime` - both confound the treatment-outcome path                      |
# | Effect modifiers (X)      | None (this NB targets a constant ATE; later NBs use X for regime heterogeneity)        |
# | Identification assumption | The synthetic DGP is fully observed - there is no unobserved confounding by design     |
# | Main failure mode         | None in identification; this is an API smoke test, not an empirical finding            |

# %% [markdown]
# ## Setup

# %%
"""Causal Inference Library Decision Guide - choose the right causal library for your problem."""

import warnings

import networkx as nx
import numpy as np
import pandas as pd

from utils.reproducibility import set_global_seeds

warnings.filterwarnings("ignore")

# networkx 3.x removed nx.algorithms.d_separated; DoWhy 0.12 still calls it.
# Monkey-patch with the renamed replacement before importing DoWhy.
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

# %%
libraries = {}
lib_checks = [
    ("EconML", "econml"),
    ("DoWhy", "dowhy"),
    ("CausalML", "causalml"),
    ("Tigramite", "tigramite"),
    ("causal-learn", "causallearn"),
]

for name, module_name in lib_checks:
    try:
        mod = __import__(module_name)
        libraries[name] = getattr(mod, "__version__", "installed")
    except ImportError:
        libraries[name] = None

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
# We use one synthetic dataset across all libraries to illustrate comparable
# API patterns. This is not a benchmark - the goal is to show how each library
# expresses a causal question, not to rank estimator accuracy on this sample.

# %%
# Generate synthetic data with known causal structure
n = 1000

# Confounders
volatility = rng.exponential(0.02, n)
regime = rng.binomial(1, 0.6, n)

# Treatment (momentum) - depends on confounders
momentum = 0.5 * regime - 2 * volatility + rng.normal(0, 0.1, n)

# Outcome (returns) - depends on confounders AND treatment
TRUE_ATE = 0.02  # Known true effect
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
# ## 1. EconML (Microsoft)
#
# **Strengths**:
# - Best for DML and metalearners
# - Production-ready, well-documented
# - Supports heterogeneous treatment effects (CATE)
#
# **Best for**: Continuous treatment effects with rich confounders

# %%
if libraries["EconML"]:
    from econml.dml import LinearDML
    from sklearn.ensemble import GradientBoostingRegressor

    Y = df["returns"].values.reshape(-1, 1)
    T = df["momentum"].values.reshape(-1, 1)
    W = df[["volatility", "regime"]].values  # Controls used for residualization

    dml = LinearDML(
        model_y=GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42),
        model_t=GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42),
        cv=3,
        random_state=42,
    )
    dml.fit(Y, T, W=W)

    econml_ate = float(dml.ate())
    print(
        f"EconML ATE: {econml_ate:.6f} (true: {TRUE_ATE}, error: {abs(econml_ate - TRUE_ATE):.6f})"
    )
else:
    econml_ate = None
    print("EconML not available")

# %% [markdown]
# Confounders enter EconML's DML interface as `W` (controls used for
# residualization), not `X` (effect modifiers used to model treatment-effect
# heterogeneity). For a simple adjusted ATE the distinction is invisible,
# but it matters as soon as the question is whether the effect varies across
# regimes - `04_dml_crypto_regime` demonstrates the `X` role.
#
# This example uses i.i.d. cross-validation (`cv=3`) on synthetic data. For
# time-series applications, temporal splitting is required to avoid lookahead
# bias - see `03_econml_dml` for a walk-forward implementation.

# %% [markdown]
# ## 2. DoWhy (Microsoft/Amazon)
#
# **Strengths**:
# - Forces explicit DAG specification
# - Built-in refutation tests
# - Good for communicating assumptions
#
# **Best for**: When you want explicit causal assumptions and sensitivity analysis

# %%
if libraries["DoWhy"]:
    from dowhy import CausalModel

    # Specify causal graph
    causal_graph = """
    digraph {
        volatility -> momentum;
        volatility -> returns;
        regime -> momentum;
        regime -> returns;
        momentum -> returns;
    }
    """

    # Create model
    model = CausalModel(
        data=df,
        treatment="momentum",
        outcome="returns",
        graph=causal_graph,
    )

    # Identify and estimate
    identified_estimand = model.identify_effect()
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
    )

    dowhy_ate = estimate.value
    print(f"DoWhy ATE: {dowhy_ate:.6f} (true: {TRUE_ATE}, error: {abs(dowhy_ate - TRUE_ATE):.6f})")
else:
    dowhy_ate = None
    print("DoWhy not available")

# %% [markdown]
# ### Estimand vs. Estimator
#
# A key lesson: **different libraries can target different estimands**, even
# on the same data. Comparing estimates is only valid when they answer the
# same causal question.
#
# | Estimand | Description | Libraries |
# |---|---|---|
# | $E[Y \mid do(T = t+1)] - E[Y \mid do(T = t)]$ | Marginal effect of continuous treatment | EconML, DoWhy |
# | $E[Y \mid do(T = 1)] - E[Y \mid do(T = 0)]$ | Binary treatment effect (high vs. low) | CausalML |
#
# A sophisticated estimator cannot rescue an identification failure, and
# comparing estimates across different estimands is meaningless.

# %% [markdown]
# ## 3. Other Libraries (Not Covered in This Chapter)
#
# **CausalML** (Uber) is strongest for uplift-style treatment-effect
# estimation with S/T/X/R meta-learners, especially binary or discrete
# interventions common in experiments and marketing applications. Continuous
# treatments are possible, but the library's strongest support is for the
# binary case. This chapter's main trading examples use continuous treatments
# and event-study designs, so CausalML is not in the critical path; see the
# [CausalML documentation](https://causalml.readthedocs.io/) for uplift and
# meta-learner applications.
#
# **tfp-causalimpact** (BSTS) is omitted here because it requires
# time-series structure incompatible with this cross-sectional test set.
# See `06_fed_announcement_bsts` for a full event-study demonstration.

# %% [markdown]
# ## 4. Library Comparison Summary
#
# The table below summarizes all libraries referenced in this chapter, with
# pointers to the dedicated notebooks where each is demonstrated in depth.


# %%
def build_library_comparison():
    """Build library comparison table showing Chapter 15 coverage."""
    comparison_data = [
        {
            "Library": "EconML",
            "Use": "Effect estimation",
            "Treatment": "Continuous/Binary",
            "Key Feature": "DML, Metalearners",
            "Notebook": "03_econml_dml",
        },
        {
            "Library": "DoWhy",
            "Use": "Effect estimation",
            "Treatment": "Continuous/Binary",
            "Key Feature": "Refutation tests",
            "Notebook": "02_dowhy_causal_graph",
        },
        {
            "Library": "CausalML",
            "Use": "Uplift/CATE",
            "Treatment": "Binary/discrete (focus)",
            "Key Feature": "S/T/X/R learners",
            "Notebook": "(not demonstrated in chapter)",
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
            "Use": "TS discovery",
            "Treatment": "N/A",
            "Key Feature": "PCMCI algorithm",
            "Notebook": "07_tigramite_time_series",
        },
        {
            "Library": "causal-learn",
            "Use": "Causal discovery",
            "Treatment": "N/A",
            "Key Feature": "PC, FCI, GES, LiNGAM, Granger",
            "Notebook": "08_neural_causal_discovery",
        },
    ]
    return pd.DataFrame(comparison_data).set_index("Library")


comparison_df = build_library_comparison()
comparison_df

# %% [markdown]
# Tigramite and causal-learn solve **discovery** (learning causal graphs),
# not **estimation** (quantifying effects). These are different problem domains:
# discovery is often a prerequisite to estimation, not an alternative.

# %% [markdown]
# ## 5. Decision Framework
#
# **Start: What is your causal question?**
#
# 1. **"What is the effect of treatment T on outcome Y?"**
#    - Do you have a DAG?
#      - **Yes**: DoWhy (specify graph, get sensitivity analysis)
#      - **No**: What is your treatment type?
#        - Continuous (e.g., momentum score): EconML DML
#        - Binary (e.g., treated/control): CausalML or EconML
#
# 2. **"What is the causal structure among variables?"**
#    - Time series: Tigramite (PCMCI)
#    - Cross-sectional: causal-learn (PC/FCI)
#
# 3. **"What was the impact of a discrete event?"**
#    - tfp-causalimpact (BSTS)
#
# **Quick Reference:**
#
# | Scenario | Recommended Library |
# |---|---|
# | Factor causal effect | EconML (LinearDML) |
# | Event study | tfp-causalimpact |
# | Explicit DAG needed | DoWhy |
# | A/B test analysis | CausalML |
# | Discover TS graph | Tigramite |
# | Discover CS graph | causal-learn |

# %% [markdown]
# ## 6. Effect Estimates (With Caveats)
#
# **Only comparable estimates**: EconML and DoWhy estimate the same quantity
# (marginal ATE for continuous treatment). CausalML estimates a different
# quantity (binary treatment effect) and should NOT be compared numerically.

# %%
rows = [{"Method": "True ATE", "Estimate": TRUE_ATE, "Error": 0.0}]
if econml_ate is not None:
    rows.append(
        {"Method": "EconML (DML)", "Estimate": econml_ate, "Error": abs(econml_ate - TRUE_ATE)}
    )
if dowhy_ate is not None:
    rows.append(
        {"Method": "DoWhy (Backdoor)", "Estimate": dowhy_ate, "Error": abs(dowhy_ate - TRUE_ATE)}
    )

estimates_df = pd.DataFrame(rows).set_index("Method")
estimates_df

# %% [markdown]
# Both EconML and DoWhy recover the sign and order of magnitude of the true
# ATE, but each overestimates the marginal effect on this small synthetic
# sample - recovering an effect of around 0.030 against a true value of
# 0.020. The point is that the APIs work as advertised, not that estimator
# accuracy is comparable across libraries on a single 1,000-row toy sample.
# CausalML estimates a binary treatment effect and is on a different scale.

# %% [markdown]
# ## Key Takeaways
#
# ### Library Selection Principles
#
# 1. **Match library to question type**:
#    - Effect estimation (given graph): EconML, DoWhy, CausalML
#    - Structure discovery (learn graph): Tigramite, causal-learn
#
# 2. **Match library to treatment type**:
#    - Continuous treatment: EconML (DML), DoWhy
#    - Binary treatment: CausalML (uplift), EconML (also works)
#
# 3. **Match library to data structure**:
#    - Time series: Tigramite (PCMCI) for discovery
#    - Cross-sectional: causal-learn (PC/FCI) for discovery
#
# ### Important Methodological Notes
#
# - **Different libraries, different estimands**: Don't compare ATE estimates
#   across libraries that solve different problems
# - **Discovery vs estimation**: Structure discovery (Tigramite, causal-learn)
#   is a prerequisite to effect estimation, not an alternative
# - **Complementary use**: DoWhy refutation + EconML estimation is a common pattern
#
# ### For Deeper Coverage
#
# This notebook provides API examples only. For methodologically rigorous
# applications, see the dedicated notebooks:
# - **DML**: [`03_econml_dml`](03_econml_dml.ipynb), [`04_dml_crypto_regime`](04_dml_crypto_regime.ipynb)
# - **BSTS event studies**: [`06_fed_announcement_bsts`](06_fed_announcement_bsts.ipynb)
# - **DoWhy refutation**: [`02_dowhy_causal_graph`](02_dowhy_causal_graph.ipynb)
# - **Time series discovery**: [`07_tigramite_time_series`](07_tigramite_time_series.ipynb)
# - **Neural discovery**: [`08_neural_causal_discovery`](08_neural_causal_discovery.ipynb)
#
# These dedicated notebooks include proper train/test splits, standard errors,
# refutation tests, and sensitivity analysis that this comparison skips.
