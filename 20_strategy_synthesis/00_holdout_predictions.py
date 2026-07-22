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
# 1. **Select** the best signal-stage model from the validation registry
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
# **Prerequisites**: All case studies must have completed Ch16–19 backtests.

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
# For each case study, `BacktestExplorer.best()` ranks all signal-stage
# backtests by Sharpe ratio. The top model is selected for holdout
# retraining. Note that the best model may use a non-primary label —
# for example, CME Futures' best model uses `fwd_ret_21d` rather than the
# primary `fwd_ret_5d`. This is valid: multiple label horizons are part
# of the experimental design.

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
# 5. Run a backtest using the same strategy specification as the best
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
    try:
        result = generate_holdout(cs_id, force=FORCE, verbose=True)
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
# - A **moderate decline** (e.g., 1.0 → 0.5) suggests genuine signal
#   with some overfitting to the validation window.
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

            # Holdout decay = arithmetic (HO − Val) / Val when Val is finite
            # and non-zero. Left as NaN otherwise so downstream readers can
            # distinguish "not yet measured" from "measured and X%".
            if val_sr == val_sr and val_sr != 0 and ho_sr == ho_sr:
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
                    "Status": "done",
                }
            )
    return pl.DataFrame(summary_rows)


# %%
summary_df = build_holdout_summary(results)
summary_df

# %% [markdown]
# ## Reading the table
#
# The table above reports, for each case study, the validation Sharpe of
# the top signal-stage model selected by the in-sample registry, the
# holdout Sharpe of that *same model configuration* retrained on all
# pre-holdout data, the rank IC on holdout, and the arithmetic decay
# `(HO − Val) / Val`.
#
# The chapter prose (§20.6, *Stability Across Time and Regimes*) uses
# this output to discuss three *decay patterns* — prediction decay (IC
# collapses), translation decay (IC survives but Sharpe halves or
# reverses through portfolio construction, costs, or regime shifts),
# and structural breaks. The notebook does not grade case studies; it
# produces the numbers on which that discussion rests.
#
# Two design disciplines apply when reading the table:
#
# 1. **Measurement error**: holdout is a single out-of-sample window per
#    case study. Cross-case-study Sharpe gaps within roughly one pooled
#    standard error are not distinguishable (see `measurement_error.md`
#    and the paired-fold t-test in `01_aggregate_synthesis`).
# 2. **Single-metric Sharpe**: Sharpe-only ranking ignores CAGR,
#    MaxDD, IC direction, capacity, and cost sensitivity. A signal that
#    translates a small positive IC into a strong Sharpe through
#    concentrated leverage is not the same story as a signal that
#    survives broad diversification.
#
# Readers should pair each row of this table with the corresponding
# signal-stage diagnostics in `03_signal_quality` and the cost and risk
# diagnostics in `06_cost_survival` / `07_regime_risk` before drawing
# any cross-case-study conclusion.

# %% [markdown]
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
# 4. The table below is input to `01_aggregate_synthesis`, which joins
#    these numbers with signal-stage measurement error and cost
#    sensitivity to produce the Chapter 20 cross-case comparison.
#
# **Next**: `01_aggregate_synthesis` builds cross-case comparison tables
# and computes the paired-fold statistical tests that bound the
# measurement error on every Sharpe in the table above.
