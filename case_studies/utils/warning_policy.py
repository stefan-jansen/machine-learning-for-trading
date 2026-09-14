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


# Diagnostics already reported, keyed by whatever the caller passed as `key`. A sweep calls
# one engine function once per scheme, and twelve copies of one diagnostic bury the output
# they are printed into.
_REPORTED: set[object] = set()


def reset_reader_warnings() -> None:
    """Forget what has been reported. For a test that runs the same condition twice."""
    _REPORTED.clear()


def warn_the_reader(
    message: str,
    *,
    source: str,
    key: object | None = None,
    category: type[Warning] = UserWarning,
    stacklevel: int = 3,
) -> bool:
    """Emit a diagnostic on both channels, and report whether the printed one fired.

    `warnings.warn` alone does not reach the person reading the executed notebook. Most
    backtest notebooks call `warnings.filterwarnings("ignore")` at import, so a guard
    written on `warnings.warn` reports nothing whether or not it fired - a channel every
    caller disables is not a channel. Printing alone does not reach a library caller or a
    test, which is what `pytest.warns` and a downstream `catch_warnings` are written
    against. So both, every time.

    The printed line is `"  WARN <source>: <message>"`. The two-space indent and the
    `source` prefix are what let a reader scanning a long rendered cell tell an engine
    diagnostic from the notebook's own output.

    `key` deduplicates. It defaults to the message, which is right when the message already
    carries everything that distinguishes one occurrence from another; pass something
    coarser when it carries a value that varies per call and you want one line per
    condition rather than one per call.

    `category` is passed through, because a caller that distinguishes a `RuntimeWarning`
    from a `UserWarning` has a test written against that distinction and a reader has a
    different reason to care. `stacklevel` counts from this function, so a caller that had
    `stacklevel=2` wants 3 here.

    Returns True when the printed line was emitted, False when it was suppressed as a
    repeat. `warnings.warn` fires either way, because `warnings` does its own
    once-per-location bookkeeping and a test may be counting.
    """
    warnings.warn(message, category, stacklevel=stacklevel)
    marker = message if key is None else key
    if marker in _REPORTED:
        return False
    _REPORTED.add(marker)
    print(f"  WARN {source}: {message}")
    return True
