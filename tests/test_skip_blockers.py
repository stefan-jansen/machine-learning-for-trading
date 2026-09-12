"""Guard the skips so a reason cannot stay true by never being asked.

``tests/skip_blockers.py`` says why this exists. These tests are the half that makes a
declaration cost something: a condition that has expired is a failure, not a warning, because
it means the notebook could run and the skip is now hiding it.
"""

from __future__ import annotations

from datetime import date

import pytest
import yaml

from tests.pm_helpers import OVERRIDES_PATH
from tests.skip_blockers import (
    DECIDABLE_KINDS,
    UNDECIDABLE_KINDS,
    blocker_unmet_reason,
    declared_kind,
    per_commit_tier,
    skip_declarations,
    unknown_keys,
)

# The escape hatches exist because two conditions genuinely cannot be decided from inside the
# run. The ceiling is what stops them becoming the comfortable answer: a new skip that cannot
# be expressed as a decidable condition has to displace one, or raise this number in a commit
# that says why.
MAX_UNDECIDABLE = 16


def _overrides() -> dict:
    return yaml.safe_load(OVERRIDES_PATH.read_text()) or {}


def _skips() -> dict[str, dict]:
    return skip_declarations(_overrides())


def _decidable_skips() -> dict[str, dict]:
    """The rows whose condition this process can actually evaluate.

    Parametrising the expiry check over every row instead would report the two undecidable
    kinds as skips, and `test-unit-data` treats a skip as a failure precisely because a job
    that runs nothing passes. The undecidable rows are inspected by the two tests below.
    """
    return {
        key: row
        for key, row in _skips().items()
        if declared_kind(row["skip_blocker"]) in DECIDABLE_KINDS
    }


def test_every_skip_declares_a_blocker():
    """A skip with no declared condition is the state this mechanism exists to end."""
    undeclared = sorted(key for key, row in _skips().items() if not row.get("skip_blocker"))
    assert not undeclared, (
        "these rows are skipped with nothing checking the reason: "
        + ", ".join(undeclared)
        + ". Add a skip_blocker naming the condition the reason rests on."
    )


@pytest.mark.parametrize("key", sorted(_skips()))
def test_declaration_is_well_formed(key):
    declaration = _skips()[key]["skip_blocker"]
    assert isinstance(declaration, dict), f"{key}: skip_blocker must be a mapping"
    kind = declared_kind(declaration)
    assert not unknown_keys(declaration, kind), (
        f"{key}: skip_blocker carries keys beside {kind}: {unknown_keys(declaration, kind)}"
    )
    if kind == "fixture_shortfall":
        spec = declaration[kind]
        assert spec.get("note"), f"{key}: fixture_shortfall.note must say what falls short"
        verified = spec.get("verified")
        assert verified, (
            f"{key}: fixture_shortfall.verified must carry the date the reason was last "
            "confirmed by running the notebook"
        )
        if not isinstance(verified, date):
            verified = date.fromisoformat(str(verified))
        assert verified <= date.today(), f"{key}: fixture_shortfall.verified is in the future"


def test_external_blockers_are_off_the_per_commit_tier():
    """Nothing decidable checks an ``external``, so it may not sit in the per-commit suite.

    A per-commit row skipped on an unverifiable reason is exactly the silent exclusion this
    mechanism is about. On a weekly or on-demand tier the dedicated workflow is where the
    service is either present or explicitly absent.
    """
    offenders = sorted(
        key
        for key, row in _skips().items()
        if declared_kind(row["skip_blocker"]) == "external" and per_commit_tier(row)
    )
    assert not offenders, (
        "external blockers must declare a tier other than per_commit: " + ", ".join(offenders)
    )


def test_undecidable_blockers_stay_a_minority():
    kinds = [declared_kind(row["skip_blocker"]) for row in _skips().values()]
    undecidable = [kind for kind in kinds if kind in UNDECIDABLE_KINDS]
    assert len(undecidable) <= MAX_UNDECIDABLE, (
        f"{len(undecidable)} of {len(kinds)} skips rest on a condition nothing can check. "
        "Express the new one as a decidable condition or say in the commit why it cannot be."
    )


@pytest.mark.parametrize("key", sorted(_decidable_skips()))
def test_declaration_has_not_outlived_its_reason(key, populated_data_dir, seeded_output_dir):
    """The condition the skip rests on still holds.

    When the fixture or the registry grows what the reason says is missing, this fails, which
    is the signal to delete the skip or rewrite the reason. The notebook's own test then runs
    with no other edit.

    ``UndecidableHere`` is deliberately not caught. ``populated_data_dir`` has already skipped
    the whole file when there is no fixture, so reaching it means the fixture is present and
    the seeding produced no registry - a silent failure of the thing this check measures
    against, which should be loud rather than another skip.
    """
    declaration = _decidable_skips()[key]["skip_blocker"]
    reason = blocker_unmet_reason(declaration)
    assert reason is not None, (
        f"{key}: the condition this skip rests on no longer holds. Un-skip the notebook and "
        "run it: either the skip goes, or skip_reason and skip_blocker are rewritten to what "
        "blocks it now."
    )


# --- the evaluators can report an expiry, which is what makes the check above worth running --


@pytest.mark.parametrize(
    "declaration,still_blocked",
    [
        ({"absent_fixture_path": "no/such/fixture/file.parquet"}, True),
        ({"absent_fixture_path": "etfs/market/etf_universe.parquet"}, False),
        ({"absent_fixture_path": "equities/market/microstructure/iex/deep/*.pcap.gz"}, True),
        ({"absent_fixture_path": "etfs/market/*.parquet"}, False),
    ],
)
def test_fixture_path_evaluator_reports_both_outcomes(
    declaration, still_blocked, populated_data_dir
):
    """A checker that can only say "still blocked" would pass on every stale declaration."""
    assert (blocker_unmet_reason(declaration) is not None) is still_blocked


@pytest.mark.parametrize(
    "declaration,still_blocked",
    [
        # The fixture registry ships the linear validation population and no other, which makes
        # it the control for both directions without constructing a registry.
        ({"absent_population": {"of": "sp500_options", "names": ["no-such-population-v1"]}}, True),
        (
            {
                "absent_population": {
                    "of": "sp500_options",
                    "names": ["sp500-options-linear-validation-v1"],
                }
            },
            False,
        ),
        (
            {"absent_candidate_set": {"of": "sp500_options", "names": ["no-such-candidate-set"]}},
            True,
        ),
    ],
)
def test_registry_evaluators_report_both_outcomes(
    declaration, still_blocked, populated_data_dir, seeded_output_dir
):
    assert (blocker_unmet_reason(declaration) is not None) is still_blocked


def test_no_canonical_selection_reports_an_expiry_when_the_resolver_returns(
    monkeypatch, populated_data_dir, seeded_output_dir
):
    """The branch the fixture cannot reach today, exercised so it is not dead code.

    Every case study in the fixture registry refuses, so nothing in the declarations above
    ever runs the "condition has expired" path for this kind. A resolver that returns is what
    the fixture will look like once it carries a selectable candidate, and this is the only
    way to see that the evaluator reports it rather than staying quiet.
    """
    from case_studies.utils import strategy_analysis

    monkeypatch.setattr(
        strategy_analysis, "resolve_canonical_rank1_lineage", lambda *a, **k: {"rank1": "present"}
    )
    assert blocker_unmet_reason({"no_canonical_selection": {"of": "etfs"}}) is None


def test_a_declaration_naming_two_kinds_is_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        declared_kind({"external": "x", "absent_fixture_path": "a/b.parquet"})


def test_a_declaration_naming_no_kind_is_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        declared_kind({"reason": "because"})


def test_every_decidable_kind_is_exercised_by_the_declarations():
    """A kind nothing declares is a kind nothing proves works on real data."""
    declared = {declared_kind(row["skip_blocker"]) for row in _skips().values()}
    unused = sorted(set(DECIDABLE_KINDS) - declared)
    assert not unused, (
        "these kinds are implemented but no row declares one, so nothing exercises them "
        f"against the fixture: {', '.join(unused)}"
    )


def _run_docker_runner(monkeypatch, overrides, tmp_path):
    """Drive ``test_docker_notebook`` with one declaration and report what it did.

    Returns ``("skipped", reason)`` or ``("executed", None)``. ``run_notebook`` is replaced by
    a sentinel so a regression reports itself instead of executing a notebook inside the unit
    suite; every decision the runner makes before that point is the real one.
    """
    from tests import test_docker_notebooks as runner

    monkeypatch.setattr(runner, "get_overrides", lambda _: overrides)
    monkeypatch.setattr(runner, "get_tier", lambda _: runner.current_test_tier())
    executed = []
    monkeypatch.setattr(
        runner, "run_notebook", lambda **kwargs: executed.append(kwargs) or {"status": "ok"}
    )
    notebook = tmp_path / "15_causal_estimation" / "05_stand_in.py"
    notebook.parent.mkdir(parents=True, exist_ok=True)
    notebook.write_text("# %%\n")
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    try:
        runner.test_docker_notebook(notebook, tmp_path, tmp_path)
    except pytest.skip.Exception as exc:
        # Skipped derives from BaseException, not Exception. Catching Exception here lets the
        # skip escape and marks THIS test skipped, which reads as a pass in the summary and
        # asserts nothing - the first version of this helper did exactly that.
        return "skipped", str(exc)
    return ("executed", None) if executed else ("returned without executing", None)


def test_the_docker_runner_honours_a_skip_whose_blocker_still_holds(monkeypatch, tmp_path):
    """The defect this replaced: the runner ignored ``skip`` and executed the notebook anyway.

    ``15_causal_estimation/05_momentum_causal_trading`` declared a ``fixture_shortfall`` naming
    the exact ValueError it raises, and the docker runner ran it regardless and collected that
    failure. An image supplies a module; it does not widen the fixture window, so there was
    never a cured condition here to run under.
    """
    outcome, reason = _run_docker_runner(
        monkeypatch,
        {
            "skip": True,
            "skip_reason": "Walk-forward CV needs more bars than the test-data window provides",
            "skip_blocker": {"fixture_shortfall": {"note": "...", "verified": "2026-09-11"}},
        },
        tmp_path,
    )
    assert outcome == "skipped", (
        "the docker runner executed a notebook whose blocker still holds; a skip it ignores "
        "is a declaration that costs nothing and a failure nobody can act on"
    )
    assert "more bars than the test-data window" in reason


def test_the_docker_runner_still_executes_once_the_blocker_expires(monkeypatch, tmp_path):
    """The negative half: honouring a skip must not become skipping unconditionally.

    Without this, deleting the blocker check entirely - or honouring ``skip`` without asking
    whether its condition still holds - passes the test above while silently dropping every
    docker notebook from CI.
    """
    fixture_root = tmp_path / "fixture"
    (fixture_root / "futures").mkdir(parents=True)
    (fixture_root / "futures" / "continuous.parquet").touch()
    monkeypatch.setattr("tests.skip_blockers._fixture_root", lambda: fixture_root)

    outcome, _ = _run_docker_runner(
        monkeypatch,
        {
            "skip": True,
            "skip_reason": "the fixture carries no continuous futures",
            "skip_blocker": {"absent_fixture_path": "futures/continuous.parquet"},
        },
        tmp_path,
    )
    assert outcome == "executed", (
        "the blocker names a file the fixture now carries, so the skip has expired and the "
        "notebook must run again with no edit to overrides.yaml"
    )
