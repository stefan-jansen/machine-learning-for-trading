"""Two notebooks running at once must not write to the same progress log.

`run_notebook` writes cell-level progress to a file that is opened with mode `"w"`,
and it used to be keyed on the notebook's stem alone. `04_model_based_features`,
`12_backtest` and `05_evaluation` are stems shared by all nine case studies and by
chapter notebooks, so two panes running different notebooks truncated each other.

Observed on 2026-08-05 while `cs-sp500_options` was executing its own
`04_model_based_features`: the log held `params={'GARCH_MIN_OBS': 50, 'MAX_SYMBOLS':
5, 'N_RESTARTS': 1}`, which is the **etfs** entry in `tests/overrides.yaml`, written
by a different pane.

The cost is not a lost log. `work/2026-08-05-ci-green/PROTOCOL.md` names this file as
the way to see which cell a hanging notebook is on, "which is faster than waiting for
the traceback" - so under concurrency an agent debugging a hang reads cell progress
belonging to a different case study. Wrong and plausible at the same time is worse
than absent.
"""

from __future__ import annotations

import os
from pathlib import Path

from tests.pm_helpers import REPO_ROOT, _progress_log_path


def test_the_same_stem_in_two_case_studies_gets_two_logs():
    """The collision that was actually observed."""
    etfs = _progress_log_path(REPO_ROOT / "case_studies/etfs/04_model_based_features.py")
    options = _progress_log_path(
        REPO_ROOT / "case_studies/sp500_options/04_model_based_features.py"
    )

    assert etfs != options


def test_two_panes_running_the_same_notebook_get_two_logs():
    """The path alone does not separate them; two panes run the same job all the time."""
    path = REPO_ROOT / "case_studies/etfs/04_model_based_features.py"

    assert str(os.getpid()) in _progress_log_path(path).name


def test_the_log_still_matches_the_glob_the_tooling_uses():
    """`grep -l FAIL /tmp/ml4t-pm-*.log` is in the runbooks and has to keep working."""
    log = _progress_log_path(REPO_ROOT / "case_studies/etfs/04_model_based_features.py")

    assert log.parent == Path("/tmp")
    assert log.match("ml4t-pm-*.log")


def test_the_name_says_which_notebook_it_is():
    """A reader with several logs open has to be able to tell them apart by name."""
    log = _progress_log_path(REPO_ROOT / "case_studies/sp500_options/04_model_based_features.py")

    assert "case_studies_sp500_options_04_model_based_features" in log.name


def test_a_notebook_outside_the_repo_still_gets_a_log(tmp_path: Path):
    """`relative_to` raises for a path outside the repo, and losing the log to that
    would take the diagnostics away exactly when someone is running something unusual.
    """
    log = _progress_log_path(tmp_path / "scratch_notebook.py")

    assert "scratch_notebook" in log.name
    assert log.match("ml4t-pm-*.log")
