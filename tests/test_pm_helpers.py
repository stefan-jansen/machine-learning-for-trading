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
    missing_required_env,
)


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


def test_credential_gated_notebooks_declare_requires_env_not_skip() -> None:
    """A notebook blocked only on a credential must be gated, never hard-skipped.

    ``skip: true`` is unconditional and is checked after tier routing, so a
    notebook marked that way stays unexecuted even in weekly-external.yml, which
    exists to supply exactly these credentials.

    Selection is by ``requires_env`` rather than by what ``skip_reason`` happens to
    say: an entry carrying both keys is hard-skipped whatever reason it gives, and
    matching on the reason text would let it through under any other wording.
    """
    overrides = yaml.safe_load((Path(__file__).parent / "overrides.yaml").read_text())

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


def test_no_override_declares_a_launcher_without_an_interpreter() -> None:
    """check_kernel_routing only fires for a notebook a Docker job selects, and CI
    runs those under -k. A static sweep catches the misconfiguration in an entry
    nothing currently selects."""
    overrides = yaml.safe_load((Path(__file__).parent / "overrides.yaml").read_text())

    launcher_without_interpreter = {
        key
        for key, value in overrides.items()
        if isinstance(value, dict)
        and value.get("kernel_launcher")
        and not value.get("kernel_python")
    }

    assert launcher_without_interpreter == set()
