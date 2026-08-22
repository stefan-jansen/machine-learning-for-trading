import os
import re
import sys
import types
from pathlib import Path

import pytest
import yaml

from tests import pm_helpers
from tests.pm_helpers import (
    RECORD_REPLAY,
    RECORD_REWRITE,
    TIER_ON_DEMAND,
    TIER_PER_COMMIT,
    TIER_WEEKLY,
    check_kernel_routing,
    collect_chapter_notebooks,
    current_test_tier,
    get_record_mode,
    get_reruns,
    get_tier,
    injected_parameters,
    missing_required_env,
    research_preview_parameters,
    unusable_parameters,
)

REPO_ROOT = Path(__file__).parent.parent


def test_collect_chapter_notebooks_keeps_real_notebooks_and_skips_helpers() -> None:
    notebooks = {path.as_posix() for path in collect_chapter_notebooks(Path("."), range(1, 28))}

    assert "06_strategy_definition/03_case_study_overview.py" in notebooks
    assert "08_financial_features/case_study_feature_summary.py" in notebooks
    assert "11_ml_pipeline/08_ml_backtest_intro.py" in notebooks
    assert "16_strategy_simulation/01_backtest_first_principles.py" in notebooks
    assert "21_rl_execution_hedging/07_backtest_with_impact.py" in notebooks

    # Helper modules that live beside the notebooks. Each is a file that exists
    # today, so the exclusion is actually exercised: asserting a deleted path is
    # absent proves nothing.
    assert "03_market_microstructure/limit_orderbook.py" not in notebooks
    assert "13_dl_time_series/dl_sequences.py" not in notebooks
    assert "21_rl_execution_hedging/rl_environments.py" not in notebooks
    assert "16_strategy_simulation/_etf_baseline.py" not in notebooks


# ---------------------------------------------------------------------------
# Tier / reruns / record_mode helpers
# ---------------------------------------------------------------------------


def test_get_tier_defaults_to_per_commit() -> None:
    assert get_tier({}) == TIER_PER_COMMIT
    assert get_tier({"tier": None}) == TIER_PER_COMMIT


def test_get_tier_accepts_valid_values() -> None:
    assert get_tier({"tier": "per_commit"}) == TIER_PER_COMMIT
    assert get_tier({"tier": "weekly"}) == TIER_WEEKLY
    assert get_tier({"tier": "on_demand"}) == TIER_ON_DEMAND


def test_get_tier_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="Invalid tier"):
        get_tier({"tier": "nightly"})


def test_current_test_tier_defaults_to_per_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ML4T_TEST_TIER", raising=False)
    assert current_test_tier() == TIER_PER_COMMIT


def test_current_test_tier_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML4T_TEST_TIER", "weekly")
    assert current_test_tier() == TIER_WEEKLY
    monkeypatch.setenv("ML4T_TEST_TIER", "on_demand")
    assert current_test_tier() == TIER_ON_DEMAND


def test_current_test_tier_rejects_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML4T_TEST_TIER", "bogus")
    with pytest.raises(ValueError, match="Invalid ML4T_TEST_TIER"):
        current_test_tier()


def test_get_reruns_default_zero() -> None:
    assert get_reruns({}) == 0


def test_get_reruns_returns_int() -> None:
    assert get_reruns({"reruns": 3}) == 3


def test_get_reruns_rejects_negative_or_nonint() -> None:
    with pytest.raises(ValueError):
        get_reruns({"reruns": -1})
    with pytest.raises(ValueError):
        get_reruns({"reruns": "2"})


def test_get_record_mode_defaults_to_replay() -> None:
    assert get_record_mode({}) == RECORD_REPLAY


def test_get_record_mode_accepts_rewrite() -> None:
    assert get_record_mode({"record_mode": "rewrite"}) == RECORD_REWRITE


def test_get_record_mode_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="Invalid record_mode"):
        get_record_mode({"record_mode": "none"})


# ---------------------------------------------------------------------------
# requires_env — credential gating
# ---------------------------------------------------------------------------


def test_missing_required_env_empty_without_declaration() -> None:
    assert missing_required_env({}) == []


def test_missing_required_env_reports_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ML4T_FAKE_KEY", raising=False)

    assert missing_required_env({"requires_env": "ML4T_FAKE_KEY"}) == ["ML4T_FAKE_KEY"]


def test_missing_required_env_treats_blank_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unset GitHub secret interpolates to the empty string, not to nothing."""
    monkeypatch.setenv("ML4T_FAKE_KEY", "   ")

    assert missing_required_env({"requires_env": "ML4T_FAKE_KEY"}) == ["ML4T_FAKE_KEY"]


def test_missing_required_env_empty_when_all_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML4T_FAKE_KEY", "Jane Doe jane@example.org")
    monkeypatch.setenv("ML4T_OTHER_KEY", "token")

    assert missing_required_env({"requires_env": ["ML4T_FAKE_KEY", "ML4T_OTHER_KEY"]}) == []


def test_missing_required_env_reports_only_the_absent_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ML4T_FAKE_KEY", "present")
    monkeypatch.delenv("ML4T_OTHER_KEY", raising=False)

    assert missing_required_env({"requires_env": ["ML4T_FAKE_KEY", "ML4T_OTHER_KEY"]}) == [
        "ML4T_OTHER_KEY"
    ]


@pytest.fixture(scope="module")
def overrides() -> dict:
    return yaml.safe_load((Path(__file__).parent / "overrides.yaml").read_text())


def test_credential_gated_notebooks_declare_requires_env_not_skip(overrides: dict) -> None:
    """A notebook blocked only on a credential must be gated, never hard-skipped.

    ``skip: true`` is unconditional and is checked after tier routing, so a
    notebook marked that way stays unexecuted even in weekly-external.yml, which
    exists to supply exactly these credentials.

    Selection is by ``requires_env`` rather than by what ``skip_reason`` happens to
    say: an entry carrying both keys is hard-skipped whatever reason it gives, and
    matching on the reason text would let it through under any other wording.
    """
    hard_skipped_despite_a_gate = {
        key
        for key, value in overrides.items()
        if isinstance(value, dict) and value.get("requires_env") and value.get("skip")
    }

    assert hard_skipped_despite_a_gate == set()


def _executable(tmp_path: Path) -> Path:
    interpreter = tmp_path / "python"
    interpreter.write_text("#!/bin/sh\n")
    interpreter.chmod(0o755)
    return interpreter


def test_check_kernel_routing_passes_when_nothing_is_declared() -> None:
    routing = check_kernel_routing({"timeout": 300})

    assert routing.problem is None
    assert routing.python is None
    assert routing.launcher is None


def test_check_kernel_routing_rejects_a_missing_interpreter() -> None:
    routing = check_kernel_routing({"kernel_python": "/opt/nope/bin/python", "docker_env": "py312"})

    assert routing.problem is not None
    assert "/opt/nope/bin/python" in routing.problem
    assert "py312" in routing.problem


def test_check_kernel_routing_names_no_image_when_the_override_declares_none() -> None:
    """`docker_env` is optional, and "Rebuild the None image" is worse than silence."""
    routing = check_kernel_routing({"kernel_python": "/opt/nope/bin/python"})

    assert routing.problem is not None
    assert "None" not in routing.problem


def test_check_kernel_routing_rejects_a_launcher_with_no_interpreter() -> None:
    """run_notebook gates the kernelspec on kernel_python, so a lone launcher is
    dropped in silence and the notebook runs on the pytest interpreter."""
    routing = check_kernel_routing({"kernel_launcher": "envs/py312/bsts_kernel.py"})

    assert routing.problem is not None
    assert "kernel_python" in routing.problem


def test_check_kernel_routing_rejects_a_missing_launcher(tmp_path: Path) -> None:
    """The launcher goes into the kernelspec argv unchecked; a bad path kills the
    kernel at startup with an error that says nothing about the launcher."""
    routing = check_kernel_routing(
        {
            "kernel_python": str(_executable(tmp_path)),
            "kernel_launcher": "envs/py312/does_not_exist.py",
        }
    )

    assert routing.problem is not None
    assert "does_not_exist.py" in routing.problem


def test_check_kernel_routing_rejects_a_directory(tmp_path: Path) -> None:
    """An executable bit is not enough: os.access(X_OK) is true for a searchable
    directory, so /opt/bsts/bin would pass and die at kernel startup.

    A directory is an error in the override, not a stale image, so the message must
    not send the operator off to rebuild one.
    """
    routing = check_kernel_routing({"kernel_python": str(tmp_path), "docker_env": "py312"})

    assert routing.problem is not None
    assert str(tmp_path) in routing.problem
    assert "directory" in routing.problem
    assert "Rebuild" not in routing.problem


def test_check_kernel_routing_rejects_a_file_without_the_executable_bit(tmp_path: Path) -> None:
    """The other half of the interpreter predicate - e.g. a COPY that dropped the
    mode bits leaves a real python that cannot be run."""
    interpreter = tmp_path / "python"
    interpreter.write_text("#!/bin/sh\n")
    interpreter.chmod(0o644)

    routing = check_kernel_routing({"kernel_python": str(interpreter)})

    assert routing.problem is not None
    assert str(interpreter) in routing.problem


def test_check_kernel_routing_returns_what_it_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The call site passes these straight to run_notebook, so returning them
    keeps it from reading the overrides differently from what was checked.

    REPO_ROOT is redirected so this asserts the return contract rather than the
    repo layout - otherwise relocating the real launcher fails the one test whose
    job is to prove the wiring.
    """
    monkeypatch.setattr(pm_helpers, "REPO_ROOT", tmp_path)
    interpreter = _executable(tmp_path)
    launcher = tmp_path / "envs" / "py312" / "kernel.py"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("")

    routing = check_kernel_routing(
        {"kernel_python": str(interpreter), "kernel_launcher": "envs/py312/kernel.py"}
    )

    assert routing.problem is None
    assert routing.python == str(interpreter)
    assert routing.launcher == launcher


def test_every_declared_kernel_launcher_resolves(overrides: dict) -> None:
    """Launcher existence is checkable statically, and nothing else checks it.

    `check_kernel_routing` only runs for a notebook a Docker job selects, and CI
    selects under `-k`, so a moved or mistyped launcher in an unselected entry stays
    invisible. The interpreter cannot be swept this way - its executability depends
    on the image - but the launcher lives in the repo.
    """
    unresolvable = {
        key: value["kernel_launcher"]
        for key, value in overrides.items()
        if isinstance(value, dict)
        and value.get("kernel_launcher")
        and check_kernel_routing(
            {"kernel_python": sys.executable, "kernel_launcher": value["kernel_launcher"]}
        ).problem
    }

    assert unresolvable == {}


def test_no_override_declares_a_launcher_without_an_interpreter(overrides: dict) -> None:
    """Driven through the real function rather than a copy of its predicate, so the
    sweep cannot keep asserting a rule the runtime has stopped applying. Entries
    without `kernel_python` never reach the executability branch, so this stays
    independent of the image."""
    rejected = {
        key
        for key, value in overrides.items()
        if isinstance(value, dict)
        and not value.get("kernel_python")
        and check_kernel_routing({"kernel_launcher": value.get("kernel_launcher")}).problem
    }

    assert rejected == set()


# ---------------------------------------------------------------------------
# Papermill parameter reachability.
#
# Papermill accepts any parameter name and reports nothing when the notebook
# has no use for it, so an override naming a variable the notebook never binds
# runs the notebook at its production defaults while this file records a
# reduction. 292 names across 113 entries had accumulated that way by
# 2026-07-30, including whole chapters' worth of entries for notebooks that no
# longer exist. These two sweeps are what stops it coming back; they ship with
# no allowlist, because a waiver list would restore exactly the silence they
# exist to remove.
#
# They also require the file to stay expanded: the three `&id001`-style anchors
# it used to carry made 18 notebooks share one config block, so pruning any of
# them for its own notebook silently changed the other five.
# ---------------------------------------------------------------------------


def test_every_override_key_names_a_notebook(overrides: dict) -> None:
    """A key that matches no `.py` configures nothing.

    `get_overrides` looks an entry up by the path of the notebook being run, so a
    key nobody matches is never read: its timeout, its `skip` and its parameters
    all go nowhere. Two `data/*/dataset_card` entries meant to hold a download
    notebook out of the per-commit tier sat one directory above the notebook they
    named, which therefore ran every commit.
    """
    orphaned = sorted(key for key in overrides if not (REPO_ROOT / f"{key}.py").exists())

    assert orphaned == []


def test_every_declared_parameter_reaches_its_notebook(overrides: dict) -> None:
    """Every name in a `parameters` block must survive papermill's injection.

    Driven through `unusable_parameters` rather than a copy of its predicate, so
    the sweep cannot go on asserting a rule the helper has stopped applying.
    """
    unreachable = {
        key: unusable_parameters(REPO_ROOT / f"{key}.py", value["parameters"])
        for key, value in overrides.items()
        if isinstance(value, dict)
        and isinstance(value.get("parameters"), dict)
        and unusable_parameters(REPO_ROOT / f"{key}.py", value["parameters"])
    }

    assert unreachable == {}


def _notebook(tmp_path: Path, body: str) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    py = tmp_path / "notebook.py"
    py.write_text(body)
    return py


def test_unusable_parameters_accepts_a_name_bound_in_the_parameters_cell(tmp_path: Path) -> None:
    py = _notebook(
        tmp_path,
        '# %%\nimport os\n\n# %% tags=["parameters"]\nN_EPOCHS = 100\n\n# %%\nprint(N_EPOCHS)\n',
    )

    assert unusable_parameters(py, ["N_EPOCHS"]) == {}


def test_unusable_parameters_accepts_a_name_the_notebook_assigns_above_that_cell(
    tmp_path: Path,
) -> None:
    """Papermill injects after the tagged cell, so it overrides an assignment made
    above one. `case_studies/us_equities_panel/02_labels` sets `START_DATE` in its
    configuration block and reads it below the cell; the override works."""
    py = _notebook(
        tmp_path,
        '# %%\nSTART_DATE = "1990-01-01"\n\n# %% tags=["parameters"]\nSEED = 42\n\n'
        "# %%\nload(start_date=START_DATE)\n",
    )

    assert unusable_parameters(py, ["START_DATE"]) == {}


def test_unusable_parameters_rejects_a_name_the_notebook_never_reads(tmp_path: Path) -> None:
    """`12_gradient_boosting/07_hpo_comparison` declares `N_OPTUNA_TRIALS` in its
    parameters cell and never mentions it again, so the reduction the test states
    has never once applied."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nN_OPTUNA_TRIALS = 50\nSEED = 42\n\n# %%\nprint(SEED)\n',
    )

    assert "never reads it" in unusable_parameters(py, ["N_OPTUNA_TRIALS"])["N_OPTUNA_TRIALS"]


def test_unusable_parameters_rejects_a_name_bound_only_below_the_parameters_cell(
    tmp_path: Path,
) -> None:
    """Papermill's injected cell goes directly after the tagged cell, so a binding
    further down overwrites it — the same silent no-op as never using it at all."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nSEED = 42\n\n# %%\nMAX_SYMBOLS = 0\nprint(MAX_SYMBOLS)\n',
    )

    assert "MAX_SYMBOLS" in unusable_parameters(py, ["MAX_SYMBOLS"])


def test_unusable_parameters_allows_a_rebind_that_reads_the_injected_value_first(
    tmp_path: Path,
) -> None:
    """How the `12_causal_dml` notebooks apply an override: push the injected
    value into the config dict, then read it back under the same name. The value
    survives the rebind, so the entry is valid and must not be swept away."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nCV_FOLDS = 5\n\n# %%\n'
        'cfg["n_folds"] = CV_FOLDS\nCV_FOLDS = cfg.get("n_folds", 5)\n',
    )

    assert unusable_parameters(py, ["CV_FOLDS"]) == {}


def test_unusable_parameters_asks_only_that_one_read_get_the_injected_value(
    tmp_path: Path,
) -> None:
    """The contract is that the value reaches code that uses it, not that it
    survives to the end. Here it reaches line 5; the binding below changes what a
    later read would see, and there is no later read."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nCV_FOLDS = 5\n\n# %%\n'
        'cfg["n_folds"] = CV_FOLDS\nCV_FOLDS = cfg.get("n_folds", 5)\n\n# %%\nCV_FOLDS = 10\n',
    )

    assert unusable_parameters(py, ["CV_FOLDS"]) == {}


def test_unusable_parameters_rejects_a_branch_local_overwrite(tmp_path: Path) -> None:
    """`07_defining_the_learning_task/07_multiple_testing` had this: `n_rad_etf =
    N_RAD_ETF` inside `if min_len > 1:`, read on the next line inside the same
    branch and nowhere else. The injected value can never be the one read, so the
    override sat there stating a 500-simulation run that never happened."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nn_rad_etf = 500\nN_RAD_ETF = 5000\n\n# %%\n'
        "if min_len > 1:\n    n_rad_etf = N_RAD_ETF\n    run(n_simulations=n_rad_etf)\n",
    )

    assert "rebinds the name first" in unusable_parameters(py, ["n_rad_etf"])["n_rad_etf"]


def test_unusable_parameters_allows_a_rebind_on_a_branch_that_does_not_read_it(
    tmp_path: Path,
) -> None:
    """Accepted by design. The parameter reaches the notebook on the path that
    reads it, which is what this function decides; which path a given run takes
    is a question about the notebook."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nMAX_SYMBOLS = 0\n\n# %%\n'
        "if MAX_SYMBOLS:\n    universe = pick(MAX_SYMBOLS)\nelse:\n    MAX_SYMBOLS = 500\n",
    )

    assert unusable_parameters(py, ["MAX_SYMBOLS"]) == {}


def test_unusable_parameters_rejects_a_rebind_that_reads_nothing_first(tmp_path: Path) -> None:
    """The shape ten case study notebooks had: declared in the parameters cell,
    then overwritten a few lines below without the injected value being read."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nSTART_DATE = "2006-01-01"\n\n# %%\n'
        'START_DATE = "2006-01-01"\nprint(START_DATE)\n',
    )

    assert "overwrites the injected value" in unusable_parameters(py, ["START_DATE"])["START_DATE"]


def test_unusable_parameters_allows_a_conditional_derivation(tmp_path: Path) -> None:
    """A notebook that reads the injected value and derives from it under a guard
    is using the parameter, not discarding it. Flagging this would push authors
    toward an allowlist."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nMAX_SYMBOLS = 0\n\n# %%\n'
        "if MAX_SYMBOLS == 0:\n    MAX_SYMBOLS = len(universe)\n",
    )

    assert unusable_parameters(py, ["MAX_SYMBOLS"]) == {}


def test_unusable_parameters_leaves_a_binding_nested_in_control_flow_alone(
    tmp_path: Path,
) -> None:
    """A binding under `if`/`for`/`try` may not run, and deciding whether it does
    needs branch analysis this file has no business carrying. `25_live_trading`
    forces `LIVE_FEED = 0` under a papermill-detection guard; the override still
    reaches every other path."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nLIVE_FEED = 1\n\n# %%\n'
        'if os.environ.get("HEADLESS"):\n    LIVE_FEED = 0\n\n# %%\nprint(LIVE_FEED)\n',
    )

    assert unusable_parameters(py, ["LIVE_FEED"]) == {}


def test_unusable_parameters_reads_both_branch_orders_the_same_way(tmp_path: Path) -> None:
    """Swapping the branches must not swap the verdict. Both orders reach the
    parameter on one path, so both are accepted."""
    read_first = _notebook(
        tmp_path / "a",
        '# %% tags=["parameters"]\nX = 0\n\n# %%\nif cond:\n    use(X)\nelse:\n    X = 1\n',
    )
    bind_first = _notebook(
        tmp_path / "b",
        '# %% tags=["parameters"]\nX = 0\n\n# %%\nif cond:\n    X = 1\nelse:\n    use(X)\n',
    )

    assert unusable_parameters(read_first, ["X"]) == unusable_parameters(bind_first, ["X"]) == {}


def test_unusable_parameters_leaves_a_name_a_function_reads_alone(tmp_path: Path) -> None:
    """A function that reads the parameter may be called before the binding below
    it, so an overwrite cannot be established. Where the analysis cannot be sure
    it says nothing: a missed dead entry costs a stale line, a wrong rejection
    costs a real reduction."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nMAX_SYMBOLS = 0\n\n# %%\n'
        "def report():\n    return MAX_SYMBOLS\n\nreport()\nMAX_SYMBOLS = 500\n",
    )

    assert unusable_parameters(py, ["MAX_SYMBOLS"]) == {}


def test_unusable_parameters_does_not_take_a_function_local_for_a_read(tmp_path: Path) -> None:
    """A name the body also binds is local to it throughout, by Python's scoping
    rule, so reading it there says nothing about the module-level variable."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nMAX_SYMBOLS = 0\n\n# %%\n'
        "def report():\n    MAX_SYMBOLS = 1\n    return MAX_SYMBOLS\n",
    )

    assert "never reads it" in unusable_parameters(py, ["MAX_SYMBOLS"])["MAX_SYMBOLS"]


def test_unusable_parameters_ignores_a_committed_injected_parameters_cell(tmp_path: Path) -> None:
    """`case_studies/etfs/11a_pca` has one: a leftover from a papermill run, which
    the next run replaces. Reading it as notebook code would report the notebook
    overwriting exactly what papermill is about to inject."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nUSE_CACHE = True\n\n'
        '# %% papermill={"duration": 0.1} tags=["injected-parameters"]\n'
        "# Parameters\nUSE_CACHE = False\n\n# %%\nprint(USE_CACHE)\n",
    )

    assert unusable_parameters(py, ["USE_CACHE"]) == {}


def test_unusable_parameters_does_not_take_a_comprehension_target_for_a_rebind(
    tmp_path: Path,
) -> None:
    """A comprehension has its own scope, so its loop variable binds nothing in
    the module. Reading it as a rebinding would redden a valid entry."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nSYMBOLS = ["AAPL"]\n\n# %%\n'
        "picked = [SYMBOLS for SYMBOLS in universe]\nprint(SYMBOLS, picked)\n",
    )

    assert unusable_parameters(py, ["SYMBOLS"]) == {}


def test_unusable_parameters_rejects_a_notebook_with_no_parameters_cell(tmp_path: Path) -> None:
    """Without a tagged cell papermill injects above the imports, so every binding
    the notebook makes wins. Three case study notebooks were in this state."""
    py = _notebook(tmp_path, "# %%\nMAX_FOLDS = 0\nprint(MAX_FOLDS)\n")

    assert "top of the notebook" in unusable_parameters(py, ["MAX_FOLDS"])["MAX_FOLDS"]


def test_unusable_parameters_rejects_a_notebook_that_does_not_exist(tmp_path: Path) -> None:
    problems = unusable_parameters(tmp_path / "gone.py", ["MAX_SYMBOLS"])

    assert "no notebook" in problems["MAX_SYMBOLS"]


def test_unusable_parameters_is_silent_on_an_entry_with_no_parameters(tmp_path: Path) -> None:
    assert unusable_parameters(tmp_path / "gone.py", []) == {}


def test_overrides_shares_no_config_block_between_notebooks() -> None:
    """A YAML anchor makes an edit to one notebook's entry land on every notebook
    that aliases it. The three the file carried were `yaml.dump` artifacts named
    `id001`-`id003`, and they covered 18 case study entries across seven case
    studies, so correcting `TOP_N_PREDICTIONS` for one silently corrected six
    others that read a different set of parameters.
    """
    text = (REPO_ROOT / "tests" / "overrides.yaml").read_text()

    assert re.search(r"^\S.*:\s*[&*]\w+\s*$", text, re.MULTILINE) is None


def test_unusable_parameters_rejects_a_name_an_import_below_the_cell_rebinds(
    tmp_path: Path,
) -> None:
    """`from config import MAX_SYMBOLS` discards the injected value exactly as an
    assignment would, and the read below it reads the imported one."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nMAX_SYMBOLS = 0\n\n# %%\n'
        "from config import MAX_SYMBOLS\n\nprint(MAX_SYMBOLS)\n",
    )

    assert (
        "overwrites the injected value" in unusable_parameters(py, ["MAX_SYMBOLS"])["MAX_SYMBOLS"]
    )


def test_unusable_parameters_does_not_take_a_comprehension_local_for_a_read(
    tmp_path: Path,
) -> None:
    """Inside `[SYMBOLS for SYMBOLS in universe]` the name is the comprehension's
    own, so it shadows the module variable rather than reading it."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nSYMBOLS = ["AAPL"]\n\n# %%\n'
        "picked = [SYMBOLS for SYMBOLS in universe]\nprint(picked)\n",
    )

    assert "never reads it" in unusable_parameters(py, ["SYMBOLS"])["SYMBOLS"]


def test_unusable_parameters_treats_a_class_body_as_running_where_it_is_written(
    tmp_path: Path,
) -> None:
    """Python executes a class body at the `class` statement, not when something
    instantiates it, so a read there cannot protect a value already overwritten
    above it."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nMAX_SYMBOLS = 0\n\n# %%\nMAX_SYMBOLS = 500\n\n'
        "class Universe:\n    size = MAX_SYMBOLS\n",
    )

    assert (
        "overwrites the injected value" in unusable_parameters(py, ["MAX_SYMBOLS"])["MAX_SYMBOLS"]
    )


def test_unusable_parameters_counts_a_class_body_read_as_a_read(tmp_path: Path) -> None:
    """The other half of the same rule: a class body reading the parameter is a
    real use of it, at module level."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nMAX_SYMBOLS = 0\n\n# %%\nclass Universe:\n    size = MAX_SYMBOLS\n',
    )

    assert unusable_parameters(py, ["MAX_SYMBOLS"]) == {}


def test_unusable_parameters_does_not_let_a_nested_local_hide_an_outer_read(
    tmp_path: Path,
) -> None:
    """`MAX_SYMBOLS` is assigned in the inner function and read in the outer one.
    Collecting locals across both scopes would call the outer read local too, and
    reject an override the notebook does use."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nMAX_SYMBOLS = 0\n\n# %%\n'
        "def outer():\n"
        "    def inner():\n"
        "        MAX_SYMBOLS = 1\n"
        "        return MAX_SYMBOLS\n"
        "    return inner(), MAX_SYMBOLS\n",
    )

    assert unusable_parameters(py, ["MAX_SYMBOLS"]) == {}


def test_unusable_parameters_counts_a_comprehension_inside_a_function(tmp_path: Path) -> None:
    """A comprehension has its own scope, but its free variables resolve outward,
    so reading the parameter there is a real use of it. Missing this would delete
    a working reduction."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nMAX_SYMBOLS = 0\n\n# %%\n'
        "def pick(universe):\n    return [s for s in universe[:MAX_SYMBOLS]]\n",
    )

    assert unusable_parameters(py, ["MAX_SYMBOLS"]) == {}


def test_unusable_parameters_does_not_treat_a_try_body_as_certain(tmp_path: Path) -> None:
    """Something above the binding can raise, in which case the handler runs with
    the injected value still in place. The binding is in the `try` body and the
    read is in the handler, so neither is on the other's path."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nMAX_SYMBOLS = 0\n\n# %%\n'
        "try:\n    fetch()\n    MAX_SYMBOLS = 500\nexcept RuntimeError:\n"
        "    fallback(MAX_SYMBOLS)\n",
    )

    assert unusable_parameters(py, ["MAX_SYMBOLS"]) == {}


def test_unusable_parameters_orders_a_multiline_self_rebinding_correctly(tmp_path: Path) -> None:
    """`ast` puts the target of `X = (\\n    X + 1\\n)` on the earlier line, so
    ordering by line number would read this as an overwrite. The right-hand side
    is evaluated first, which is what the event sequence records."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nMAX_SYMBOLS = 0\n\n# %%\n'
        "MAX_SYMBOLS = (\n    MAX_SYMBOLS + 1\n)\nprint(MAX_SYMBOLS)\n",
    )

    assert unusable_parameters(py, ["MAX_SYMBOLS"]) == {}


def test_unusable_parameters_sees_a_class_body_nested_in_a_function(tmp_path: Path) -> None:
    """A class at module level runs where it is written, but one inside a
    function runs when that function is called. Its reads are deferred, and
    missing them would reject a parameter the notebook does use."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nMAX_SYMBOLS = 0\n\n# %%\n'
        "def build():\n    class Config:\n        limit = MAX_SYMBOLS\n\n    return Config\n",
    )

    assert unusable_parameters(py, ["MAX_SYMBOLS"]) == {}


def test_unusable_parameters_keeps_a_nested_class_attribute_class_local(tmp_path: Path) -> None:
    """The name is assigned in the class body before it is read there, so the
    read resolves to the class attribute and says nothing about the parameter."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nMAX_SYMBOLS = 0\n\n# %%\n'
        "def build():\n    class Config:\n        MAX_SYMBOLS = 10\n"
        "        limit = MAX_SYMBOLS\n\n    return Config\n",
    )

    assert "never reads it" in unusable_parameters(py, ["MAX_SYMBOLS"])["MAX_SYMBOLS"]


def test_unusable_parameters_ignores_a_bare_annotation(tmp_path: Path) -> None:
    """`MAX_SYMBOLS: int` annotates without assigning. It binds nothing, so the
    injected value is still what the read below it sees."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nMAX_SYMBOLS = 0\n\n# %%\nMAX_SYMBOLS: int\nprint(MAX_SYMBOLS)\n',
    )

    assert unusable_parameters(py, ["MAX_SYMBOLS"]) == {}


def test_unusable_parameters_orders_a_walrus_value_before_its_target(tmp_path: Path) -> None:
    """`(X := max(1, X))` reads X to compute the value Python then binds. The AST
    lists the target first, so walking children in order would call it an
    overwrite."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nMAX_SYMBOLS = 0\n\n# %%\n'
        "print(MAX_SYMBOLS := max(1, MAX_SYMBOLS))\n",
    )

    assert unusable_parameters(py, ["MAX_SYMBOLS"]) == {}


def test_unusable_parameters_rejects_a_rebind_no_call_can_precede(tmp_path: Path) -> None:
    """`case_studies/sp500_equity_option_analytics/04_model_based_features` had
    this: the parameters cell sets MIN_OBS, a duplicate assignment repeats it
    below the cell, and only functions read it. A function read normally earns
    the benefit of the doubt, but nothing between the cell and the rebinding can
    call one, so the injected value cannot reach any of them."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nMIN_OBS = 252\n\n# %%\nMIN_OBS = 252\n\n# %%\n'
        "def fit(x):\n    return len(x) >= MIN_OBS\n",
    )

    assert "before anything below" in unusable_parameters(py, ["MIN_OBS"])["MIN_OBS"]


def test_unusable_parameters_keeps_a_rebind_a_local_call_precedes(tmp_path: Path) -> None:
    """Same shape, but the notebook calls its own function before the rebinding,
    so that call reads the injected value and the override did land."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nMIN_OBS = 252\n\n# %%\n'
        "def fit(x):\n    return len(x) >= MIN_OBS\n\n# %%\n"
        "first = fit(sample)\nMIN_OBS = 252\n",
    )

    assert unusable_parameters(py, ["MIN_OBS"]) == {}


def test_unusable_parameters_ignores_an_imported_call_before_a_rebind(tmp_path: Path) -> None:
    """An imported helper cannot read this notebook's parameter, so a call to one
    is not what lets a deferred read happen. Counting it would have let the
    sp500 case above pass."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nMIN_OBS = 252\n\n# %%\n'
        'CASE_DIR = get_case_study_dir("x")\nMIN_OBS = 252\n\n# %%\n'
        "def fit(x):\n    return len(x) >= MIN_OBS\n",
    )

    assert "before anything below" in unusable_parameters(py, ["MIN_OBS"])["MIN_OBS"]


def test_unusable_parameters_ignores_a_call_that_cannot_read_the_name(tmp_path: Path) -> None:
    """The call before the rebinding is to a helper that never touches MIN_OBS,
    so it is not evidence that the injected value was read. Counting any local
    call would let this stale override through."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nMIN_OBS = 252\n\n# %%\n'
        "def describe(x):\n    return len(x)\n\n"
        "def fit(x):\n    return len(x) >= MIN_OBS\n\n# %%\n"
        "n = describe(sample)\nMIN_OBS = 252\n\n# %%\nresult = fit(sample)\n",
    )

    assert "before anything below" in unusable_parameters(py, ["MIN_OBS"])["MIN_OBS"]


def test_unusable_parameters_follows_a_call_through_a_local_helper(tmp_path: Path) -> None:
    """`run` does not name MIN_OBS, but it calls `fit`, which does. Calling `run`
    before the rebinding therefore does read the injected value."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nMIN_OBS = 252\n\n# %%\n'
        "def fit(x):\n    return len(x) >= MIN_OBS\n\n"
        "def run(x):\n    return fit(x)\n\n# %%\n"
        "first = run(sample)\nMIN_OBS = 252\n",
    )

    assert unusable_parameters(py, ["MIN_OBS"]) == {}


def test_unusable_parameters_keeps_the_exemption_for_a_lambda_read(tmp_path: Path) -> None:
    """A lambda is not a name a call site can use, so which call triggers its read
    cannot be established. The parameter keeps the benefit of the doubt."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nMIN_OBS = 252\n\n# %%\n'
        "check = lambda x: len(x) >= MIN_OBS\nMIN_OBS = 252\n",
    )

    assert unusable_parameters(py, ["MIN_OBS"]) == {}


def test_unusable_parameters_orders_a_call_in_a_multiline_assignment(tmp_path: Path) -> None:
    """The call that reads MIN_OBS is on a later line than the assignment target
    it feeds, but Python evaluates it first. Ordering by line would place the
    rebinding ahead of the call and reject a parameter that is read."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nMIN_OBS = 252\n\n# %%\n'
        "def fit(x):\n    return len(x) >= MIN_OBS\n\n# %%\n"
        "MIN_OBS = (\n    252 if fit(sample) else 0\n)\n",
    )

    assert unusable_parameters(py, ["MIN_OBS"]) == {}


def test_unusable_parameters_treats_a_class_attribute_as_shadowing(tmp_path: Path) -> None:
    """A class body assigns into its own namespace, so the read after it resolves
    to the attribute. Counting it as a read of the parameter would accept an
    override the notebook cannot use."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nLIMIT = 0\n\n# %%\n'
        "class Config:\n    LIMIT = 500\n    size = LIMIT\n",
    )

    assert "never reads it" in unusable_parameters(py, ["LIMIT"])["LIMIT"]


def test_unusable_parameters_keeps_a_class_read_before_its_attribute(tmp_path: Path) -> None:
    """Read first, assign second: the read still resolves outward to the
    parameter, so the override lands."""
    py = _notebook(
        tmp_path,
        '# %% tags=["parameters"]\nLIMIT = 0\n\n# %%\n'
        "class Config:\n    size = LIMIT\n    LIMIT = 500\n",
    )

    assert unusable_parameters(py, ["LIMIT"]) == {}


class _FakePapermill(types.ModuleType):
    """Stands in for papermill, which `test-unit` deliberately does not install.

    Both execution paths do `import papermill as pm` inside the function, so a module
    placed in sys.modules is what they get. Recording os.environ at that moment is
    exactly what the kernel would inherit.
    """

    class PapermillExecutionError(Exception):
        cell_index = 0
        ename = "x"
        evalue = "x"

    def __init__(self, seen: dict[str, str | None]) -> None:
        super().__init__("papermill")
        self.seen = seen

    def execute_notebook(self, *args, **kwargs) -> None:
        for name in pm_helpers.KERNEL_THREAD_CAPS:
            self.seen[name] = os.environ.get(name)


def _fake_papermill(monkeypatch) -> dict[str, str | None]:
    seen: dict[str, str | None] = {}
    monkeypatch.setitem(sys.modules, "papermill", _FakePapermill(seen))
    for name in pm_helpers.KERNEL_THREAD_CAPS:
        monkeypatch.setenv(name, "24")
    return seen


def test_run_notebook_caps_the_kernel_thread_pools(tmp_path: Path, monkeypatch) -> None:
    """Every pool the kernel can open is pinned before papermill starts it.

    Unpinned, each of scikit-learn, the BLAS and numexpr opens one thread per
    core, several suites run at once, and the OpenMP pools spin rather than block
    while they wait. The wall-clock cost lands inside whichever cell is running
    and is indistinguishable in the log from that cell being slow.
    """
    py = tmp_path / "01_demo.py"
    py.write_text('# %% tags=["parameters"]\nX = 1\n', encoding="utf-8")
    monkeypatch.setattr(pm_helpers, "sync_notebook", lambda p: py.with_suffix(".ipynb"))
    seen = _fake_papermill(monkeypatch)

    pm_helpers.run_notebook(py_path=py, timeout=5)

    assert seen == dict.fromkeys(pm_helpers.KERNEL_THREAD_CAPS, pm_helpers.KERNEL_THREAD_CAP)
    # ...and restored afterwards, so the caps do not leak into the pytest process.
    assert os.environ["OMP_NUM_THREADS"] == "24"


def test_reduced_harness_injects_only_migrated_study_preview_parameters(
    tmp_path: Path,
) -> None:
    migrated = tmp_path / "06_model.py"
    migrated.write_text(
        '# %% tags=["parameters"]\n'
        'EXECUTION_TIER = "canonical"\n'
        'WORKSPACE = "experiments/unrelated"\n'
        "MAX_FOLDS = 8\n"
        "# %%\n"
        "print(EXECUTION_TIER, WORKSPACE, MAX_FOLDS)\n",
        encoding="utf-8",
    )
    legacy = tmp_path / "05_legacy.py"
    legacy.write_text(
        '# %% tags=["parameters"]\nMAX_FOLDS = 8\n# %%\nprint(MAX_FOLDS)\n',
        encoding="utf-8",
    )
    isolated = tmp_path / "isolated"

    migrated_parameters = research_preview_parameters(
        migrated,
        {"MAX_FOLDS": 1, "WORKSPACE": "experiments/unrelated"},
        isolated,
    )
    legacy_parameters = research_preview_parameters(legacy, {"MAX_FOLDS": 1}, isolated)

    assert migrated_parameters == {
        "EXECUTION_TIER": "preview",
        "MAX_FOLDS": 1,
        "WORKSPACE": str(isolated.resolve()),
    }
    assert legacy_parameters == {"MAX_FOLDS": 1}


def test_run_notebook_requires_explicit_research_preview_opt_in(
    tmp_path: Path, monkeypatch
) -> None:
    py = tmp_path / "06_model.py"
    py.write_text(
        '# %% tags=["parameters"]\nEXECUTION_TIER = "canonical"\nWORKSPACE = None\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(pm_helpers, "sync_notebook", lambda path: path.with_suffix(".ipynb"))
    _fake_papermill(monkeypatch)
    calls: list[Path] = []

    def inject(py_path: Path, parameters: dict | None, output_dir: Path | None) -> dict:
        calls.append(py_path)
        return dict(parameters or {})

    monkeypatch.setattr(pm_helpers, "research_preview_parameters", inject)

    pm_helpers.run_notebook(py, output_dir=tmp_path / "canonical")
    assert calls == []

    pm_helpers.run_notebook(
        py,
        output_dir=tmp_path / "preview",
        research_preview=True,
    )
    assert calls == [py]


def test_notebook_worker_caps_the_same_pools(tmp_path: Path, monkeypatch) -> None:
    """The full-execution path builds its own env table, so it can drift from
    run_notebook's. Both read one table, and this is what notices if they stop."""
    from tests import notebook_worker

    py = tmp_path / "01_demo.py"
    py.write_text('# %% tags=["parameters"]\nX = 1\n', encoding="utf-8")
    py.with_suffix(".ipynb").write_text("{}", encoding="utf-8")
    seen = _fake_papermill(monkeypatch)

    notebook_worker._run_full_notebook(
        py_path=py,
        timeout=5,
        output_dir=None,
        data_dir=None,
        extra_env={},
        sync_policy="never",
    )

    assert seen == dict.fromkeys(pm_helpers.KERNEL_THREAD_CAPS, pm_helpers.KERNEL_THREAD_CAP)


def test_injected_parameters_drops_preview_reductions_on_a_canonical_run() -> None:
    """A canonical run must not carry a preview-only parameter.

    ``tests/generate_intermediates.py`` reads the same override entries with
    ``research_preview=False``. The DML request builder rejects a canonical request that
    declares reductions, so injecting them there fails at request construction.
    """
    declared = {"PREVIEW_REDUCTIONS": {"max_samples": 5000}, "MAX_SYMBOLS": 5}
    resolved = injected_parameters(
        Path("case_studies/cme_futures/11_causal_dml.py"),
        declared,
        None,
        research_preview=False,
    )
    assert resolved == {"MAX_SYMBOLS": 5}
    assert declared["PREVIEW_REDUCTIONS"] == {"max_samples": 5000}


def test_injected_parameters_keeps_everything_else_on_a_canonical_run() -> None:
    parameters = {"MAX_SYMBOLS": 5, "TOP_K": 2}
    assert (
        injected_parameters(
            Path("case_studies/cme_futures/13_backtest.py"),
            parameters,
            None,
            research_preview=False,
        )
        == parameters
    )


def test_injected_parameters_keeps_preview_reductions_under_the_preview_tier(
    tmp_path: Path,
) -> None:
    resolved = injected_parameters(
        REPO_ROOT / "case_studies/cme_futures/11_causal_dml.py",
        {"PREVIEW_REDUCTIONS": {"max_samples": 5000, "n_folds": 2}},
        tmp_path,
        research_preview=True,
    )
    assert resolved["PREVIEW_REDUCTIONS"]["max_samples"] == 5000
    assert resolved["EXECUTION_TIER"] == "preview"
