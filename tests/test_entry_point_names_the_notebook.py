"""Which notebook registered a training run, on the path where nothing else can say.

`training_runs.entry_point` is the column that answers it, and it was NULL for 785 of the
1,160 rows across the nine production registries on 2026-09-12 - every row written by a
notebook that did not pass `entry_point=` to `open_study`. The value was never derivable
inside the kernel: under papermill the executing file is a temporary `.ipynb`, `__file__` is
absent, and probing a live papermill run for `PAPERMILL_*` in `os.environ`, in `globals()`
and in `dir()` returns three empty collections. The launcher is the only party that knows,
so it says so in `ML4T_ENTRY_POINT` and the study reads it when the caller named nothing.

Every test here asserts on a registered row or on a built request, never on the environment
variable itself: the variable is the mechanism, and a test that checks the mechanism passes
whether or not the column it exists for is filled.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from case_studies.research.configs import model_requests
from case_studies.research.workspace import Study, open_study
from tests.test_research_contract_catalog import _resolved_spec
from tests.test_research_workspace import _seed_release


def _registered_entry_point(study: Study) -> str | None:
    """The column as the registry holds it, for one freshly registered training run."""
    training = study.results.register_training(_resolved_spec())
    db_path = Path(study.root) / "run_log" / "registry.db"
    assert db_path.exists(), f"no registry under {study.root}"
    db = sqlite3.connect(str(db_path))
    try:
        row = db.execute(
            "SELECT entry_point FROM training_runs WHERE training_hash = ?", (training.hash,)
        ).fetchone()
    finally:
        db.close()
    assert row is not None, f"{training.hash} registered no row"
    return row[0]


def _study(tmp_path: Path, **kwargs) -> Study:
    return Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path), **kwargs
    )


class TestTheColumnTheRegistryWrites:
    def test_a_run_records_the_notebook_the_runner_named(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ML4T_ENTRY_POINT", "08_tabular_dl")
        assert _registered_entry_point(_study(tmp_path)) == "08_tabular_dl"

    def test_the_same_run_records_nothing_when_no_one_names_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The state this closes, and the control for the test above.

        Identical call, one difference: nobody said which notebook. Without it the column is
        NULL, which is what the nine production registries are full of. It is asserted rather
        than left implicit so that a change making the entry point mandatory has to come here
        and decide, rather than turning the test above green for a second reason.
        """
        monkeypatch.delenv("ML4T_ENTRY_POINT", raising=False)
        assert _registered_entry_point(_study(tmp_path)) is None

    def test_the_notebook_wins_over_the_runner(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A notebook that names itself is not overruled by the launcher's idea of it.

        This is the case that keeps the runner's value safe to trust: the 20 notebooks that
        already pass `entry_point=` keep recording exactly what they pass, so adopting the
        environment cannot silently rewrite a value that was already right.
        """
        monkeypatch.setenv("ML4T_ENTRY_POINT", "99_wrong")
        study = _study(tmp_path, entry_point="11b_ipca")
        assert _registered_entry_point(study) == "11b_ipca"

    @pytest.mark.parametrize(
        "named",
        [
            "case_studies/etfs/08_tabular_dl.py",
            "/abs/case_studies/etfs/08_tabular_dl.ipynb",
            "08_tabular_dl",
        ],
    )
    def test_a_path_or_a_suffix_records_the_same_stem(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, named: str
    ) -> None:
        """One spelling in the column, whatever the launcher happens to hold.

        `nasdaq100_microstructure` already carries a `14_backtest.py` beside its `06_linear`,
        and a column that sometimes has an extension is a column every query has to strip.
        """
        monkeypatch.setenv("ML4T_ENTRY_POINT", named)
        assert _registered_entry_point(_study(tmp_path)) == "08_tabular_dl"

    def test_an_empty_value_is_absence_not_a_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ML4T_ENTRY_POINT", "")
        assert _registered_entry_point(_study(tmp_path)) is None


class TestTheOtherWaysIn:
    """`open_study` has four branches to a `Study`, and three of them are not `Study.open`."""

    def test_the_isolated_preview_branch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The branch every CI checkout and every clean clone takes.

        It builds a `Study` directly rather than through a constructor, so it is the one place
        a resolver added to the classmethods alone would not reach.
        """
        monkeypatch.setenv("ML4T_ENTRY_POINT", "11e_supervised_autoencoder.py")
        preview = open_study(
            "etfs",
            execution_tier="preview",
            workspace=tmp_path / "isolated",
            release_root=_seed_release(tmp_path),
        )
        assert not (preview.root / "run_log").is_symlink()
        assert preview.execution_tier.value == "preview"
        assert preview.entry_point == "11e_supervised_autoencoder"

    def test_a_read_only_study_over_one_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ML4T_ENTRY_POINT", "19_strategy_analysis")
        release = _seed_release(tmp_path)
        study = Study.at(release / "case_studies" / "etfs")
        assert study.entry_point == "19_strategy_analysis"


class TestTheProvenanceFieldAgrees:
    def test_a_request_takes_the_notebook_from_the_study(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`notebook_path` and `entry_point` answer one question, so they take one answer.

        A notebook that passed both said the same string twice, and the corpus shows what that
        costs: `us_equities_panel` fills `notebook_path` on 96 rows whose column is NULL, while
        `fx_pairs` and `cme_futures` fill neither on 263.
        """
        monkeypatch.setenv("ML4T_ENTRY_POINT", "07_gbm")
        study = _study(tmp_path)
        request = study.model(
            family="gbm",
            label="fwd_ret_21d",
            config_name="lgbm_s",
            overrides={},
            execution_tier="canonical",
            preview_reductions={},
        )
        assert request.notebook == "07_gbm"

    def test_an_explicit_notebook_still_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ML4T_ENTRY_POINT", "07_gbm")
        study = _study(tmp_path)
        request = study.model(
            family="gbm",
            label="fwd_ret_21d",
            config_name="lgbm_s",
            overrides={},
            execution_tier="canonical",
            preview_reductions={},
            notebook="07_gbm_variant",
        )
        assert request.notebook == "07_gbm_variant"


def test_model_requests_carries_it_to_every_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import polars as pl

    monkeypatch.setenv("ML4T_ENTRY_POINT", "06_linear")
    study = _study(tmp_path)
    catalog = pl.DataFrame(
        {
            "family": ["linear", "linear"],
            "label": ["fwd_ret_21d", "fwd_ret_21d"],
            "config_name": ["ridge_s", "lasso_s"],
        }
    )
    requests = model_requests(study, catalog)
    assert [request.notebook for request in requests] == ["06_linear", "06_linear"]


def test_the_runner_puts_the_stem_in_front_of_the_kernel(tmp_path: Path) -> None:
    """The other half, measured through papermill rather than around it.

    Everything above opens a study in this process, where the environment is whatever the test
    set. This runs a real notebook through the real runner and asks the kernel what it received,
    because that is the step the value has to survive: papermill forwards `os.environ` to the
    kernel and nothing else, and the whole mechanism rests on the launcher setting it before it
    calls `pm.execute_notebook`. Without this, deleting that line leaves every other test green.
    """
    from tests.pm_helpers import run_notebook

    probe = tmp_path / "42_probe_entry_point.py"
    probe.write_text(
        "# %%\n"
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        'Path(os.environ["ML4T_PROBE_OUT"]).write_text(str(os.environ.get("ML4T_ENTRY_POINT")))\n'
    )
    seen = tmp_path / "seen.txt"
    result = run_notebook(
        py_path=probe,
        parameters={},
        timeout=120,
        output_dir=tmp_path / "out",
        extra_env={"ML4T_PROBE_OUT": str(seen)},
    )
    assert result["status"] == "ok", result.get("error")
    assert seen.read_text() == "42_probe_entry_point"
