"""Rounding for hyperparameters computed from the training data.

A derived hyperparameter - Lasso's alpha from the design matrix, Huber's delta from the label
standard deviation - carries floating-point path noise in its last digits. The noise is no
difference at all to the fit and two different training identities to the registry: the polars and
pandas fold paths produced ``0.004286799774493045`` and ``0.004286799774493049`` for the same
configuration, and the registry recorded two rows for one declared model.

Rounding happens before the value is recorded *and* before it is passed to the estimator, so the
identity always describes the model that was actually fitted.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["DERIVED_PARAM_SIGNIFICANT_DIGITS", "quantize_derived"]

# Far finer than any resolution a fit responds to, far coarser than the noise.
DERIVED_PARAM_SIGNIFICANT_DIGITS = 12


def quantize_derived(value: Any) -> Any:
    """Round a data-derived float to the digits that carry information."""
    if not isinstance(value, float) or value == 0.0 or not math.isfinite(value):
        return value
    magnitude = math.floor(math.log10(abs(value)))
    return round(value, DERIVED_PARAM_SIGNIFICANT_DIGITS - 1 - magnitude)
