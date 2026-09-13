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

# `filterwarnings` matches `module` against the dotted `__name__` of the module
# that called `warnings.warn`, and anchors the regex at the start. The trailing
# group is what keeps the pattern from reaching a sibling package whose name
# merely begins the same way, such as `case_studies_notes`.
_CASE_STUDIES_MODULE = r"case_studies(\.|$)"


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


def is_case_studies_module(module_name: str) -> bool:
    """Whether a warning from `module_name` is one this policy keeps audible.

    `module_name` is a dotted import path, the same value `warnings` matches a
    filter's `module` against. Exposed so a test can assert the pattern without
    restating it.
    """
    return re.compile(_CASE_STUDIES_MODULE).match(module_name) is not None
