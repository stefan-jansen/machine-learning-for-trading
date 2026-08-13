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
# # Deep Learning Suite for S&P 500 Options
#
# The Chapter 13 deep-learning models for the options case study now run in
# separate notebooks:
#
# - `09a_lstm`
# - `09b_patchtst`
#
# The supporting appendix notebook `90_ic_diagnostic` remains available for a
# deeper signal-attribution analysis, but it is not part of the main pipeline.
# Both producers below are the accepted explicit-CUDA cohort. This index is
# read-only and never launches training.

# %%
"""Deep learning notebook index for the S&P 500 options case study."""

import warnings

from case_studies.sp500_options.backtest_contract import (
    ACCEPTED_DEEP_PRODUCERS,
    validate_accepted_deep_predictions,
)
from case_studies.utils.registry import load_prediction_index

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "sp500_options"

# %% [markdown]
# ## Current Registered Result

# %%
accepted = validate_accepted_deep_predictions(
    load_prediction_index(
        CASE_STUDY_ID,
        label="ret_to_expiry",
        split="validation",
        family="deep_learning",
    )
)
if accepted.height != len(ACCEPTED_DEEP_PRODUCERS):
    raise RuntimeError(
        f"Expected {len(ACCEPTED_DEEP_PRODUCERS)} accepted deep predictions, "
        f"found {accepted.height}"
    )
print(
    accepted.select(
        "config_name",
        "training_hash",
        "prediction_hash",
        "checkpoint_value",
        "ic_mean",
    ).sort("ic_mean", descending=True)
)

# %% [markdown]
# LSTM has the higher point IC, but both accepted CUDA producers remain
# statistically unresolved in their producer notebooks. Downstream consumers
# must use these exact training and prediction hashes; this umbrella notebook
# performs no model selection and does not touch the holdout.
