"""Which reduction keys each model family accepts under the preview tier.

These sets are a contract between a notebook's ``PREVIEW_REDUCTIONS`` block and the family
resolver that reads it, so they are checked before anything runs: ``tests/test_pm_helpers.py``
reads them to reject a misspelled key at collection time rather than as a per-cell timeout
forty minutes into a smoke run.

They live here, apart from the resolvers that enforce them, because four of the five family
modules import ``torch`` at module scope and the job that runs that check has no torch
installed. Each resolver imports its own set from here, so the guard and the resolver read the
same object and neither can drift from the other.
"""

from __future__ import annotations

DML_PREVIEW_FIELDS = {"max_samples", "max_symbols", "n_folds", "n_placebo"}

GBM_PREVIEW_FIELDS = {
    "checkpoint_interval",
    "folds",
    "max_iterations",
    "max_symbols",
    "train_sample_frac",
}

LATENT_PREVIEW_FIELDS = {
    "folds",
    "max_iter",
    "max_symbols",
    "n_epochs",
    "n_epochs_cond",
    "n_epochs_moment",
    "n_epochs_unc",
    "n_factors",
}

LINEAR_PREVIEW_FIELDS = {"folds", "max_symbols", "train_sample_frac"}

SEQUENCE_PREVIEW_FIELDS = {
    "folds",
    "max_symbols",
    "max_train_sequences",
    "max_predict_sequences",
}

TABM_PREVIEW_FIELDS = {"checkpoint_interval", "folds", "max_symbols", "n_epochs"}
