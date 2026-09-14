"""The entity-eligibility rule a persistent-ID panel applies, and who is subject to it.

Here rather than in ``case_studies/utils/latent_factors/panel.py`` beside
:func:`~case_studies.utils.latent_factors.panel.prepare_panel_data`, which is where it
reads most naturally, for one reason: that package's ``__init__`` imports ``torch``
unconditionally and on purpose, so torch's bundled cudart wins symbol resolution before
the ml4t libraries load. Importing anything under it therefore requires torch, and the
coverage guard that reads this rule runs in environments that deliberately have no torch -
CI's ``test-unit`` installs the import surface of the tests it runs and nothing else.

So the definition lives in a module with no such dependency, and ``panel.py`` imports it.
There is still exactly one definition and the builder still owns it; only the file changed.
"""

from __future__ import annotations

import polars as pl

#: Latent-factor models whose panel is built by ``prepare_panel_data`` rather than
#: ``prepare_ragged_panel_data``, and which are therefore subject to the rule below. PCA
#: factorizes a dense ``T x N`` return matrix, so a column that is mostly absent carries no
#: usable covariance with the others; the ragged builder has no such requirement because
#: IPCA, CAE, SAE and SDF all take unbalanced dated cross-sections by construction.
PERSISTENT_PANEL_MODELS: frozenset[str] = frozenset({"pca"})

#: Fraction of the training window's dates an entity must appear on to enter a persistent
#: panel. Not a tuned number and not a quality bar: below one half the entity is absent for
#: most of the window the factors are estimated over.
DEFAULT_MIN_COVERAGE = 0.5


def eligible_persistent_entities(
    keys: pl.DataFrame,
    *,
    entity_col: str,
    date_col: str,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> pl.DataFrame:
    """Entities dense enough for a persistent-ID panel, most complete first.

    ``keys`` is the ``(entity, date)`` rows of the window eligibility is judged over - the
    fold's training window, not the whole dataset, so an entity that lists partway through
    the study is admitted once the rolling window has moved past its start.

    Returned as a frame of ``(entity_col, len)`` rather than a list of names because
    ``prepare_panel_data`` truncates it with ``max_entities`` and needs the ordering to do
    that; callers wanting only the names read the column.

    This is the single definition of the rule. The coverage guard
    (``notebook_contracts.undercovered_prediction_members``) reads it too, so a member is
    charged against the entities its own builder would have admitted rather than against
    the full panel, and the two cannot drift.
    """
    n_dates_total = keys[date_col].n_unique()
    min_dates = max(int(n_dates_total * min_coverage), 10)
    return (
        keys.group_by(entity_col)
        .len()
        .filter(pl.col("len") >= min_dates)
        .sort(["len", entity_col], descending=[True, False])
    )


__all__ = [
    "DEFAULT_MIN_COVERAGE",
    "PERSISTENT_PANEL_MODELS",
    "eligible_persistent_entities",
]
