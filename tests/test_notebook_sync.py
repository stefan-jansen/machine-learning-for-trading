"""Gate: a committed notebook must be its current .py executed in production.

Stamped notebooks carry ``metadata.ml4t_provenance`` recording the git blob of the
paired ``.py`` they were executed from and whether the run used production
parameters. This test fails if any *stamped* notebook is stale (its ``.py`` changed
since execution) or was committed from a TEST-mode run.

Unstamped notebooks are not failed here (adoption is gradual — stamp notebooks as
they are re-run through the canonical path). See
``scripts/notebook_provenance.py`` for the stamp/check tool. To stamp::

    uv run python scripts/notebook_provenance.py stamp <nb.ipynb> --executor <env>
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from notebook_provenance import check_all, production_parameters  # noqa: E402


def test_production_parameters_allow_only_full_run_cache_bypasses():
    assert production_parameters({})
    assert production_parameters({"FORCE_RETRAIN": True})
    assert production_parameters({"FORCE_RETRAIN": "true", "USE_CACHE": "false"})
    assert production_parameters({"FORCE_REBACKTEST": "1"})

    assert not production_parameters({"FORCE_RETRAIN": False})
    assert not production_parameters({"MAX_FOLDS": 1})
    assert not production_parameters({"TRAIN_SAMPLE_FRAC": 0.1})
    assert not production_parameters({"USE_CACHE": True})


def test_stamped_notebooks_are_current_and_production() -> None:
    stale, testmode, _unverified = check_all(strict=False)
    assert not stale and not testmode, (
        "Committed notebooks are out of sync with their source .py:\n"
        + (
            "  STALE (re-run in the canonical env):\n    " + "\n    ".join(stale) + "\n"
            if stale
            else ""
        )
        + (
            "  TEST-MODE (must be a production run):\n    " + "\n    ".join(testmode)
            if testmode
            else ""
        )
    )
