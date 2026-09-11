"""An entry in ``tests/overrides.yaml`` may carry only keys the harness reads.

``invocations_for`` already rejects an unknown key *inside* an invocation. The entry
itself had no such check, so ``env:`` was accepted and reached nothing:
``test_chapter_notebook`` calls ``run_notebook`` without ``extra_env``, and the only
parameters papermill sees come from ``sole_invocation(...).parameters``.

What made it survive is that a reduction arriving nowhere does not present as a broken
reduction. ``12_gradient_boosting/08_shap_analysis`` declared ``MAX_SYMBOLS: 10`` under
``env:`` against a 300 s per-cell timeout and trained on the full universe; the cost read
as the notebook being slow, which is a true statement about a full-universe run and is not
the defect. The same mistake on ``12_gradient_boosting/07_hpo_comparison`` spent a 900 s
timed run before the missing injection was visible.

Two checks, because either alone can be walked around. The first is a list, so it catches
a typo of an existing key and costs an explicit edit to defeat. The second reads the file's
own shape - a mapping of SCREAMING_SNAKE names is a reduction wherever it is written - so a
newly invented key of this class fails without anyone remembering to update the list.
"""

from __future__ import annotations

import pytest
import yaml

from tests.pm_helpers import KNOWN_OVERRIDE_KEYS, OVERRIDES_PATH, invocations_for

SCREAMING_SNAKE = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _entries(raw: dict) -> dict[str, dict]:
    return {key: value for key, value in raw.items() if isinstance(value, dict)}


def unknown_keys(raw: dict) -> dict[str, list[str]]:
    """Every entry's keys that no reader looks at, keyed by notebook."""
    found = {}
    for key, entry in _entries(raw).items():
        unknown = sorted(set(entry) - KNOWN_OVERRIDE_KEYS)
        if unknown:
            found[key] = unknown
    return found


def _looks_like_a_reduction(value: object) -> bool:
    """A mapping whose every name is SCREAMING_SNAKE, i.e. papermill parameter names."""
    if not isinstance(value, dict) or not value:
        return False
    return all(
        isinstance(name, str) and name and set(name) <= set(SCREAMING_SNAKE) for name in value
    )


def reductions_papermill_never_sees(raw: dict) -> dict[str, list[str]]:
    """Parameter names an entry declares somewhere papermill does not read."""
    found = {}
    for key, entry in _entries(raw).items():
        injected = set()
        for run in invocations_for(entry, key=key):
            injected |= set(run.parameters)
        stranded = []
        for name, value in entry.items():
            if name in ("parameters", "invocations") or not _looks_like_a_reduction(value):
                continue
            stranded += [f"{name}.{param}" for param in value if param not in injected]
        if stranded:
            found[key] = sorted(stranded)
    return found


@pytest.fixture(scope="module")
def overrides() -> dict:
    return yaml.safe_load(OVERRIDES_PATH.read_text()) or {}


def test_no_entry_carries_a_key_nothing_reads(overrides) -> None:
    found = unknown_keys(overrides)
    assert not found, (
        f"tests/overrides.yaml entries carry keys no reader looks at: {found}. "
        f"Either the key is a typo for one of {sorted(KNOWN_OVERRIDE_KEYS)}, or it is new "
        f"and needs both a reader and a line in pm_helpers.KNOWN_OVERRIDE_KEYS naming it."
    )


def test_every_declared_reduction_reaches_papermill(overrides) -> None:
    found = reductions_papermill_never_sees(overrides)
    assert not found, (
        f"tests/overrides.yaml declares parameter names papermill is never given: {found}. "
        f"A reduction written under any key but `parameters` (or an invocation's "
        f"`parameters`) leaves the notebook running at production scale."
    )


class TestTheChecksCanFail:
    """The negative half. Without these two the assertions above are unfalsifiable."""

    ENV_SHAPED = {"12_gradient_boosting/08_shap_analysis": {"env": {"MAX_SYMBOLS": 10}}}
    CORRECT = {"12_gradient_boosting/08_shap_analysis": {"parameters": {"MAX_SYMBOLS": 10}}}

    def test_the_key_check_rejects_the_entry_that_prompted_it(self) -> None:
        assert unknown_keys(self.ENV_SHAPED) == {"12_gradient_boosting/08_shap_analysis": ["env"]}

    def test_the_key_check_passes_a_key_the_loader_does_read(self) -> None:
        assert unknown_keys(self.CORRECT) == {}
        assert unknown_keys({"nb": {"timeout": 300, "tier": "weekly", "gpu": True}}) == {}

    def test_the_reduction_check_rejects_the_entry_that_prompted_it(self) -> None:
        assert reductions_papermill_never_sees(self.ENV_SHAPED) == {
            "12_gradient_boosting/08_shap_analysis": ["env.MAX_SYMBOLS"]
        }

    def test_the_reduction_check_passes_the_same_names_written_correctly(self) -> None:
        assert reductions_papermill_never_sees(self.CORRECT) == {}

    def test_the_reduction_check_reads_invocations_too(self) -> None:
        entry = {
            "nb": {
                "elsewhere": {"MAX_SYMBOLS": 10},
                "invocations": [{"id": "a", "parameters": {"MAX_SYMBOLS": 10}}],
            }
        }
        assert reductions_papermill_never_sees(entry) == {}

    def test_a_non_parameter_mapping_is_not_read_as_a_reduction(self) -> None:
        """`skip_reason`-style prose and lowercase mappings are not parameter names."""
        assert reductions_papermill_never_sees({"nb": {"docker_env": {"image": "benchmark"}}}) == {}
        assert unknown_keys({"nb": {"docker_env": "benchmark"}}) == {}
