"""Warning policy for notebooks that would otherwise silence every warning.

`warnings.filterwarnings("ignore")` at notebook import is a catch-all: it hides
third-party noise, and with it every diagnostic `case_studies.*` raises to reach
the person reading the executed notebook. `warnings.warn` is then not a channel,
and a guard written on it reports nothing whether or not it fired.

This module silences the third-party warnings by name and leaves everything else
audible. Each silenced entry records what was measured when it was added, so a
later reader can tell a measurement from a guess.
"""

from __future__ import annotations

import re
import warnings

# Message prefixes of third-party warnings observed in executed notebooks. The
# regex is start-anchored and case-insensitive, which is how `filterwarnings`
# compiles it. Category is deliberately left at `Warning`: `arch`'s
# DataScaleWarning subclasses `Warning` directly rather than `UserWarning`, so a
# filter narrowed to `UserWarning` would not reach it.
#
# Counted 2026-09-12 over the rendered `.ipynb` of the three case studies that
# install no filter (crypto_perps_funding, fx_pairs, us_equities_panel):
#   219  arch/univariate/base.py  DataScaleWarning  "y is poorly scaled, ..."
_THIRD_PARTY_NOISE: tuple[str, ...] = (r"y is poorly scaled",)

# A warning is issued with a `module` that `warnings` derives from the source
# filename with its `.py` suffix stripped, not from the dotted import path. The
# pattern therefore has to match a path segment, and `filterwarnings` anchors it
# at the start, so it needs the leading `.*`.
_CASE_STUDIES_MODULE = r".*[/\\]case_studies[/\\].*"


def apply_notebook_warning_policy() -> None:
    """Silence known third-party noise, keep `case_studies.*` diagnostics audible.

    Safe to call more than once. Replaces a bare
    ``warnings.filterwarnings("ignore")`` at the top of a notebook.
    """
    for message in _THIRD_PARTY_NOISE:
        warnings.filterwarnings("ignore", message=message)
    # Added last so it is matched first: `filterwarnings` inserts at the front
    # of the filter list. Without this, a broader ignore added later by a
    # notebook or a library would take precedence over it.
    warnings.filterwarnings("default", module=_CASE_STUDIES_MODULE)


def is_case_studies_module(filename: str) -> bool:
    """Whether `filename` is the source of a `case_studies.*` diagnostic.

    Exposed so a test can assert the pattern against a real path rather than
    restating it.
    """
    return re.compile(_CASE_STUDIES_MODULE).match(filename.removesuffix(".py")) is not None
