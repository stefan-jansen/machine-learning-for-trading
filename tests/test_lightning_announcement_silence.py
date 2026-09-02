"""Lightning's per-trainer banners must not reach the executed notebook.

A checkpointed Darts run builds one Lightning trainer per checkpoint increment, and each one
announces the accelerator, the TPU count, the visible CUDA devices, a logging-service
advertisement and the reason `fit` stopped. At eight folds and a hundred epochs in five-epoch
steps that is a hundred and sixty announcements, and they land in the notebook as output entries
between the results.

`_trainer_kwargs` already turns off the progress bar and the experiment logger. These are
`rank_zero_info` records on Lightning's own loggers, which neither switch reaches, so they
survived both. `tests/test_notebook_output_volume.py` caps the consequence; this pins the cause.

The test drives real `logging` calls rather than reading the level back, because the level that
matters is the effective one - a handler on a parent logger still emits a record the child would
have passed on. Asserting `getLevel()` would pass on a configuration that still prints.
"""

from __future__ import annotations

import logging

import pytest

from case_studies.utils.darts_forecasting import (
    _LIGHTNING_ANNOUNCEMENT_LOGGERS,
    _trainer_kwargs,
    silence_lightning_announcements,
)


@pytest.fixture(autouse=True)
def _restore_levels():
    saved = {name: logging.getLogger(name).level for name in _LIGHTNING_ANNOUNCEMENT_LOGGERS}
    yield
    for name, level in saved.items():
        logging.getLogger(name).setLevel(level)


@pytest.mark.parametrize("name", _LIGHTNING_ANNOUNCEMENT_LOGGERS)
def test_an_announcement_is_dropped(name: str, caplog: pytest.LogCaptureFixture) -> None:
    logging.getLogger(name).setLevel(logging.INFO)
    silence_lightning_announcements()
    with caplog.at_level(logging.DEBUG):
        logging.getLogger(name).info("GPU available: True (cuda), used: True")
    assert not caplog.records


@pytest.mark.parametrize("name", _LIGHTNING_ANNOUNCEMENT_LOGGERS)
def test_a_real_warning_still_gets_through(name: str, caplog: pytest.LogCaptureFixture) -> None:
    """Silencing the banners must not silence Lightning telling us something is wrong."""
    silence_lightning_announcements()
    with caplog.at_level(logging.DEBUG):
        logging.getLogger(name).warning("checkpoint directory is not empty")
    assert [record.getMessage() for record in caplog.records] == [
        "checkpoint directory is not empty"
    ]


def test_building_trainer_kwargs_silences_them() -> None:
    """The suppression travels with the trainer configuration rather than needing a second call.

    Every Darts entry point in this module goes through `_trainer_kwargs`, so a caller cannot
    build a trainer and forget to silence it.
    """
    for name in _LIGHTNING_ANNOUNCEMENT_LOGGERS:
        logging.getLogger(name).setLevel(logging.INFO)

    kwargs = _trainer_kwargs("cpu")

    assert kwargs["enable_model_summary"] is False
    assert all(
        logging.getLogger(name).level == logging.WARNING for name in _LIGHTNING_ANNOUNCEMENT_LOGGERS
    )
