"""Contracts for the SDF checkpoint schedule and the two numbering schemes around it.

The preset field `checkpoint_epochs` is CONDITIONAL-RELATIVE: the model config validates
it against `n_epochs_cond` alone. The checkpoint labels the run publishes are GLOBAL:
`n_epochs_unc` plus the conditional epoch. Every preset in the corpus was written in the
global scheme, and the canonical path refused it while the preview path passed because
`resolve_checkpoint_epochs` silently dropped the entry that exposed the mismatch.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from case_studies.utils.latent_factors.common import resolve_checkpoint_epochs

REPO = Path(__file__).resolve().parents[1]
SHARED_PRESET = REPO / "case_studies" / "config" / "sdf" / "sdf.yaml"
CASE_STUDIES = (
    "cme_futures",
    "etfs",
    "sp500_equity_option_analytics",
    "us_equities_panel",
    "us_firm_characteristics",
)


def _shared_defaults() -> dict:
    return yaml.safe_load(SHARED_PRESET.read_text())


def _case_study_preset(case_study: str) -> dict:
    document = yaml.safe_load(
        (REPO / "case_studies" / case_study / "config" / "setup.yaml").read_text()
    )
    block = document["modeling"]["latent_factors"]["model_kwargs"]["sdf"]
    return {**_shared_defaults(), **block}


def _presets() -> list[tuple[str, dict]]:
    return [("config/sdf/sdf.yaml", _shared_defaults())] + [
        (case_study, _case_study_preset(case_study)) for case_study in CASE_STUDIES
    ]


def _published_labels(preset: dict) -> list[int]:
    """Reproduce `_resolve_latent_checkpoints` for the SDF branch."""
    unc = int(preset["n_epochs_unc"])
    cond = int(preset["n_epochs_cond"])
    physical = resolve_checkpoint_epochs(
        max(unc, cond),
        checkpoint_interval=None,
        checkpoint_epochs=[int(epoch) for epoch in preset["checkpoint_epochs"]],
    )
    labels = {epoch for epoch in physical if epoch <= unc}
    labels |= {unc + epoch for epoch in physical if epoch <= cond}
    return sorted(labels)


@pytest.mark.parametrize(
    "name,preset", _presets(), ids=lambda value: value if isinstance(value, str) else ""
)
def test_every_sdf_preset_is_within_its_conditional_budget(name: str, preset: dict) -> None:
    """The canonical path hands the raw preset to the model config, which validates it
    against `n_epochs_cond`. A preset above that budget fails production, not review."""
    cond = int(preset["n_epochs_cond"])
    declared = [int(epoch) for epoch in preset["checkpoint_epochs"]]
    assert max(declared) <= cond, f"{name}: {max(declared)} exceeds n_epochs_cond {cond}"


@pytest.mark.parametrize(
    "name,preset", _presets(), ids=lambda value: value if isinstance(value, str) else ""
)
def test_every_sdf_preset_declares_exactly_the_schedule_that_runs(name: str, preset: dict) -> None:
    """The declared list must equal the resolved physical list.

    Asserting on the published labels instead does not work, and the reason is the trap
    worth naming: `include_final` appends `max_epoch`, so `[256,512,768]`,
    `[256,512,768,1024]` and `[256,512,768,1024,1280]` all publish the same labels. A
    truncated preset is invisible in the schedule and still moves the training identity,
    because the declared list is what enters the spec.
    """
    unc = int(preset["n_epochs_unc"])
    cond = int(preset["n_epochs_cond"])
    declared = [int(epoch) for epoch in preset["checkpoint_epochs"]]
    physical = resolve_checkpoint_epochs(
        max(unc, cond), checkpoint_interval=None, checkpoint_epochs=declared
    )
    assert declared == physical, name


@pytest.mark.parametrize(
    "name,preset", _presets(), ids=lambda value: value if isinstance(value, str) else ""
)
def test_every_sdf_preset_publishes_the_full_conditional_horizon(name: str, preset: dict) -> None:
    """The final published label is the end of the conditional stage in global numbering.
    This is the property the renumbering had to preserve."""
    unc = int(preset["n_epochs_unc"])
    cond = int(preset["n_epochs_cond"])
    assert _published_labels(preset)[-1] == unc + cond, name


def test_out_of_range_checkpoint_is_refused_not_dropped() -> None:
    """Filtering silently is what let a globally numbered preset survive review."""
    with pytest.raises(ValueError, match=r"must be within 1\.\.1024"):
        resolve_checkpoint_epochs(
            1024, checkpoint_interval=None, checkpoint_epochs=[256, 512, 768, 1024, 1280]
        )


def test_in_range_checkpoints_resolve_unchanged() -> None:
    assert resolve_checkpoint_epochs(
        1024, checkpoint_interval=None, checkpoint_epochs=[256, 512, 768, 1024]
    ) == [256, 512, 768, 1024]


def test_beta_schedule_is_a_separate_budget_and_is_left_alone() -> None:
    """`beta_checkpoint_epochs` validates against `beta_n_epochs`, not `n_epochs_cond`,
    and is already exactly at its budget. The 256 it shares with `n_epochs_unc` is a
    coincidence of value; renumbering it would break it."""
    for name, preset in _presets():
        beta_total = int(preset["beta_n_epochs"])
        beta_epochs = [int(epoch) for epoch in preset["beta_checkpoint_epochs"]]
        assert max(beta_epochs) == beta_total, name
        assert (
            resolve_checkpoint_epochs(
                beta_total, checkpoint_interval=None, checkpoint_epochs=beta_epochs
            )
            == beta_epochs
        ), name
