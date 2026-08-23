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
# # Holdout Predictions: Out-of-Sample Validation
#
# **Docker image**: `ml4t`
#
# **Chapter 20 — Strategy Synthesis**
#
# This notebook performs the final out-of-sample test for each case study:
#
# 1. **Select** the signal-stage model with the highest validation Sharpe from the registry
# 2. **Retrain** on all data before the holdout window
# 3. **Predict** on the holdout period (never seen during model selection)
# 4. **Backtest** the holdout predictions using the same strategy specification
#
# The holdout test is the last line of defense against overfitting. A model
# that performed well in walk-forward validation but fails on the holdout
# window was likely the beneficiary of selection bias across many model
# configurations and hyperparameter choices.
#
# **Learning Objectives**:
# - Understand the role of the holdout period in the ML4T workflow
# - Implement full retrain-on-all-history for final out-of-sample evaluation
# - Compare validation vs holdout Sharpe ratios to surface overfit
# - Read the validation→holdout decay pattern per case study
#
# **Book Reference**: Chapter 20, Section 20.6 (Stability Across Time and Regimes)
#
# **Prerequisites**: A case study appears in the table below only once its signal-stage backtests
# are in the registry. One whose earlier stages have not been run reports the reason it could not
# be evaluated rather than being omitted, so the table always accounts for all nine.

# %%
"""Ch20 Holdout Predictions — out-of-sample validation for all 9 case studies."""

import warnings

import polars as pl

warnings.filterwarnings("ignore")

from holdout import generate_holdout, has_holdout_predictions, select_best_model

# %% tags=["parameters"]
# Which case studies to run (empty list = all 9)
CASE_STUDIES = []
# Force regeneration of existing holdout predictions
FORCE = False
# A validation Sharpe smaller than this in absolute value makes the decay ratio meaningless,
# because the denominator rather than the decay determines its size.
MIN_VAL_SHARPE_FOR_DECAY = 0.05

# %%
ALL_CASE_STUDIES = [
    "cme_futures",
    "fx_pairs",
    "nasdaq100_microstructure",
    "us_firm_characteristics",
    "etfs",
    "crypto_perps_funding",
    "sp500_options",
    "us_equities_panel",
    "sp500_equity_option_analytics",
]

DISPLAY_NAMES = {
    "etfs": "ETFs",
    "crypto_perps_funding": "Crypto Perps",
    "nasdaq100_microstructure": "NASDAQ-100",
    "sp500_equity_option_analytics": "S&P 500 Eq+Opt",
    "us_firm_characteristics": "US Firms",
    "fx_pairs": "FX Pairs",
    "cme_futures": "CME Futures",
    "sp500_options": "S&P 500 Options",
    "us_equities_panel": "US Equities",
}

cs_list = CASE_STUDIES if CASE_STUDIES else ALL_CASE_STUDIES

# %% [markdown]
# ## Best Model Selection
#
# For each case study, `BacktestExplorer.best()` ranks all signal-stage backtests by validation
# Sharpe and the top-ranked one is retrained for the holdout. That ranking is the whole of the
# selection, and it is what makes the holdout a genuine out-of-sample test: the choice was made
# without reference to the holdout window.
#
# The top-ranked configuration may use a label other than the case study's primary one. That is
# not a defect. Several label horizons are part of the experimental design, and the ranking is
# free to prefer any of them - but it does mean a row of the table below can describe a different
# forecasting problem from the row beneath it.

# %%
# Preview: which model will be selected for each case study
selection_rows = []
for cs_id in cs_list:
    try:
        best = select_best_model(cs_id, min_ic=None)
        selection_rows.append(
            {
                "Case Study": DISPLAY_NAMES.get(cs_id, cs_id),
                "Family": best["family"],
                "Config": best["config_name"],
                "Label": best["training_spec"]["label"],
                "Val Sharpe": round(best["val_sharpe"], 3),
                "Has Holdout": has_holdout_predictions(cs_id),
            }
        )
    except Exception as e:
        selection_rows.append(
            {
                "Case Study": DISPLAY_NAMES.get(cs_id, cs_id),
                "Family": "ERROR",
                "Config": str(e)[:40],
                "Label": "",
                "Val Sharpe": float("nan"),
                "Has Holdout": False,
            }
        )

pl.DataFrame(selection_rows)

# %% [markdown]
# ## Generate Holdout Predictions
#
# For each case study, we:
#
# 1. Load the full modeling dataset (features + labels)
# 2. Create a single holdout split: train on everything before `holdout_start`,
#    predict on `[holdout_start, holdout_end]`
# 3. Retrain the exact same model configuration (same hyperparameters,
#    same checkpoint) on the larger training set
# 4. Register predictions with `split="holdout"` in the registry
# 5. Run a backtest using the same strategy specification as the top-ranked
#    validation backtest
#
# The holdout backtest uses `register=False` initially to avoid fold-metric
# computation (the holdout window doesn't align with CV folds), then manually
# registers the headline metrics.

# %%
results = []
for cs_id in cs_list:
    print(f"\n{'=' * 60}")
    print(f"  {cs_id}")
    print(f"{'=' * 60}")
    # Recorded before the call: with FORCE off, an existing holdout is loaded rather than
    # regenerated, and the row then describes the model selected whenever it was first built.
    was_cached = has_holdout_predictions(cs_id) and not FORCE
    try:
        result = generate_holdout(cs_id, force=FORCE, verbose=True)
        result["from_cache"] = was_cached
        results.append(result)
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({"cs_id": cs_id, "error": str(e)})

# %% [markdown]
# ## Validation vs Holdout Comparison
#
# The key question: does the signal survive out of sample?
#
# - **Sharpe ratio degradation** is expected — validation benefits from
#   selection across models, the holdout does not.
# - A **decline of roughly half** suggests genuine signal with some overfitting to the
#   validation window.
# - A **sign flip** (positive → negative) suggests the signal was
#   largely spurious or regime-dependent.
# - **Constant predictions** (IC = NaN) mean the model zeroes out on
#   the holdout training set — the regularization path has shifted.


# %%
def build_holdout_summary(results):
    """Build summary table reporting validation and holdout performance."""
    summary_rows = []
    for r in results:
        if "error" in r:
            summary_rows.append(
                {
                    "Case Study": DISPLAY_NAMES.get(r["cs_id"], r["cs_id"]),
                    "Family": "ERROR",
                    "Label": "",
                    "Val Sharpe": float("nan"),
                    "HO Sharpe": float("nan"),
                    "HO IC": float("nan"),
                    "HO CAGR": float("nan"),
                    "HO MaxDD": float("nan"),
                    "HO Decay": float("nan"),
                    "Status": r["error"][:30],
                }
            )
        elif r.get("skipped"):
            summary_rows.append(
                {
                    "Case Study": DISPLAY_NAMES.get(r["cs_id"], r["cs_id"]),
                    "Family": "",
                    "Label": "",
                    "Val Sharpe": float("nan"),
                    "HO Sharpe": float("nan"),
                    "HO IC": float("nan"),
                    "HO CAGR": float("nan"),
                    "HO MaxDD": float("nan"),
                    "HO Decay": float("nan"),
                    "Status": "cached",
                }
            )
        else:
            val_sr = r["val_sharpe"]
            ho_sr = r["holdout_sharpe"]
            ho_ic = r["holdout_ic"]

            # Decay is (HO - Val) / Val, which is only meaningful when Val is far enough from
            # zero to divide by. A validation Sharpe near zero sends the ratio to arbitrarily
            # large values that say nothing about how much the signal decayed - the denominator
            # is doing all the work. Below MIN_VAL_SHARPE_FOR_DECAY the ratio is left undefined
            # and the reader is directed to the two Sharpes themselves.
            if val_sr == val_sr and ho_sr == ho_sr and abs(val_sr) >= MIN_VAL_SHARPE_FOR_DECAY:
                ho_decay = (ho_sr - val_sr) / val_sr
            else:
                ho_decay = float("nan")

            summary_rows.append(
                {
                    "Case Study": DISPLAY_NAMES.get(r["cs_id"], r["cs_id"]),
                    "Family": r.get("family", ""),
                    "Label": r.get("label", ""),
                    "Val Sharpe": round(val_sr, 3),
                    "HO Sharpe": round(ho_sr, 3),
                    "HO IC": round(ho_ic, 4),
                    "HO CAGR": round(r.get("holdout_cagr", float("nan")), 3),
                    "HO MaxDD": round(r.get("holdout_maxdd", float("nan")), 3),
                    "HO Decay": round(ho_decay, 3),
                    "Status": "from registry" if r.get("from_cache") else "generated",
                }
            )
    return pl.DataFrame(summary_rows)


# %%
summary_df = build_holdout_summary(results)
summary_df

# %% [markdown]
# ## Reading the table
#
# **Read the `Status` column first.** A row marked `from registry` was not produced by this run:
# the holdout for that case study already existed, so it was loaded rather than regenerated. Its
# `Family`, `Label` and `Val Sharpe` describe the model that holdout was built from, which is not
# necessarily the model the selection table further up would choose today. Where the two tables
# disagree, the registry has gained or lost validation backtests since the holdout was written,
# and the holdout is the older statement.
#
# That is the intended behaviour and not a defect to route around: the holdout is used once, so a
# holdout that already exists is not silently rebuilt against a newer ranking. Regenerating one is
# a deliberate act, which is what the `FORCE` parameter is for.
#
# The table above reports, for each case study, the validation Sharpe of
# the model the holdout was built from, the
# holdout Sharpe of that *same model configuration* retrained on all
# pre-holdout data, the rank IC on holdout, and the arithmetic decay
# `(HO − Val) / Val`.
#
# The chapter prose (§20.6, *Stability Across Time and Regimes*) uses
# this output to discuss three *decay patterns*. In the first the prediction itself degrades and
# the rank IC falls toward zero. In the second the IC holds while the Sharpe halves or changes
# sign, because portfolio construction, costs or a regime shift stand between a correct ranking
# and a return. In the third the relationship breaks structurally. This notebook does not grade
# case studies; it produces the numbers that discussion rests on.
#
# Two design disciplines apply when reading the table:
#
# `HO Decay` is blank wherever the validation Sharpe is too close to zero to divide by. Read
# `Val Sharpe` and `HO Sharpe` directly on those rows: a signal that scored near zero in
# validation has no meaningful proportional decay, whatever it does out of sample.
#
# 1. **Measurement error**: holdout is a single out-of-sample window per
#    case study. Cross-case-study Sharpe gaps within roughly one pooled
#    standard error are not distinguishable (see `measurement_error.md`
#    and the paired-fold t-test in `01_aggregate_synthesis`).
# 2. **Single-metric Sharpe**: Sharpe-only ranking ignores CAGR,
#    MaxDD, IC direction, capacity, and cost sensitivity. A signal that turns a small positive IC
#    into a high Sharpe through concentrated leverage is a different proposition from one that
#    keeps its Sharpe under broad diversification, and Sharpe alone does not distinguish them.
#
# Readers should pair each row of this table with the corresponding
# signal-stage diagnostics in `03_signal_quality` and the cost and risk
# diagnostics in `06_cost_survival` / `07_regime_risk` before drawing
# any cross-case-study conclusion.

# %% [markdown]
# ### Known limitations
#
# - The holdout is one window per case study. Two case studies whose holdout Sharpes differ by less
#   than the measurement error on either are not distinguishable, and this table does not compute
#   that error - `01_aggregate_synthesis` does.
# - Selection is by validation Sharpe alone. A configuration that ranked second on Sharpe and far
#   better on drawdown or capacity is not the one retrained here.
# - Case studies whose upstream stages are absent from the registry report an error and contribute
#   no row of results. That is a statement about the registry on the day the notebook ran, not
#   about the case study.
# - Retraining on all pre-holdout data changes the training set the configuration was chosen on,
#   so a hyperparameter that suited the shorter window may not suit the longer one. That is the
#   intended procedure and it is a source of decay independent of overfitting.
#
# ## Key Takeaways
#
# 1. Out-of-sample holdout evaluation is the terminal step of the
#    workflow: walk-forward CV alone understates overfitting when the
#    hyperparameter and model-selection search is large.
# 2. The holdout uses the *same model specification* selected in-sample;
#    any re-tuning on the holdout window defeats its purpose.
# 3. Val-to-holdout decay is expected; the *magnitude* is informative,
#    not a pass/fail gate. Large positive-to-negative swings usually
#    signal regime sensitivity rather than outright overfitting.
# 4. The holdout rows this notebook registers are what `01_aggregate_synthesis` reads back out of
#    the registry, joins with signal-stage measurement error and cost sensitivity, and turns into
#    the cross-case comparison. Nothing is passed between them as a file.
#
# **Next**: `01_aggregate_synthesis` builds cross-case comparison tables
# and computes the paired-fold statistical tests that bound the
# measurement error on every Sharpe in the table above.
