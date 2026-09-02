"""No fitted-feature fold may reach into the holdout, on any configured label.

`04_model_based_features` fits HMM, fractional-differencing and GARCH parameters per fold and
emits their estimates across a window running to `emit_end`. That window is not the primary
label's validation window: it is widened to the latest `val_end` among **every** configured
label, so a model on a shorter-horizon variant finds features for the whole of its own
validation window rather than stopping short of it.

The notebook checks the near side of that - `train_end <= holdout_start`, so no parameter is
estimated from a sealed session, and `sessions_between(val_end, holdout_start) >= horizon`, so
the last validation return resolves before the seal. But the second check runs over the
PRIMARY label's folds only, while the emission window is widened by the variants. A variant
whose validation window ends later than the primary's therefore widens `emit_end` with nothing
comparing the result against the holdout.

Measured on etfs when this was written: `fwd_ret_21d` ends 2023-11-29 and `fwd_ret_5d` ends
2023-12-21, so the variant widens the emission window by 22 days and leaves 11 before the
holdout opens on 2024-01-01. Clear, but only by that margin, and nothing was watching it.

This is deliberately a test and not another notebook assertion. `04_model_based_features`
cannot be re-executed to add one: every training run pins `model_based.parquet` by whole-file
sha256, so re-running it moves every training identity and every backtest underneath. The
property is a fact about the declared configuration, so it can be checked from the
configuration without executing anything.
"""

from __future__ import annotations

import datetime as dt

import pytest
import yaml

from case_studies.utils.cv_window import configured_labels, modeling_fold_boundaries
from utils.paths import REPO_ROOT


def _boundaries(case_study: str, label: str):
    """Fold boundaries, or None when this case study cannot express them as dates.

    An intraday case study carries a time of day on its fold edges, and
    `modeling_fold_boundaries` refuses to truncate one rather than move the fold silently.
    That refusal is correct and it is not what this test is about, so such a label is skipped
    rather than failed - the emission window this guards is a daily-session property.
    """
    try:
        return modeling_fold_boundaries(case_study, label)
    except ValueError:
        return None


def _case_studies() -> list[str]:
    return sorted(p.parts[-3] for p in (REPO_ROOT / "case_studies").glob("*/config/setup.yaml"))


def _holdout_start(case_study: str) -> dt.date | None:
    setup = yaml.safe_load(
        (REPO_ROOT / "case_studies" / case_study / "config" / "setup.yaml").read_text()
    )
    raw = ((setup or {}).get("evaluation") or {}).get("holdout_start")
    if raw is None:
        return None
    return dt.date.fromisoformat(str(raw)[:10])


@pytest.mark.parametrize("case_study", _case_studies())
def test_no_configured_label_validates_into_the_holdout(case_study: str) -> None:
    """Every configured label's folds end before the holdout opens.

    Not just the primary one. The emission window takes the maximum over all of them, so a
    variant reaching past the seal would carry fitted estimates into it - the leak the fold
    contract exists to prevent, arriving through a label the contract does not check.
    """
    holdout_start = _holdout_start(case_study)
    if holdout_start is None:
        pytest.skip(f"{case_study} declares no evaluation.holdout_start")

    checked = 0
    for label in configured_labels(case_study):
        folds = _boundaries(case_study, label)
        if not folds:
            continue
        for split in folds:
            val_end = dt.date.fromisoformat(str(split["val_end"])[:10])
            assert val_end < holdout_start, (
                f"{case_study}/{label} fold {split['fold']} validates to {val_end}, on or after "
                f"the holdout opening {holdout_start}. The fitted-feature emission window is "
                f"the maximum val_end across every configured label, so this reaches into the "
                f"holdout even when the primary label does not."
            )
            checked += 1
    if not checked:
        pytest.skip(f"{case_study} declares no label with resolvable fold boundaries")


@pytest.mark.parametrize("case_study", _case_studies())
def test_the_widening_label_is_reported_not_assumed(case_study: str) -> None:
    """The margin between the widest label and the holdout is positive and knowable.

    A regression here is a configuration change, not a code change - someone adds a label or
    moves a window - so the failure has to name which label consumed the margin rather than
    only that some did.
    """
    holdout_start = _holdout_start(case_study)
    if holdout_start is None:
        pytest.skip(f"{case_study} declares no evaluation.holdout_start")

    widest: tuple[str, dt.date] | None = None
    for label in configured_labels(case_study):
        folds = _boundaries(case_study, label)
        if not folds:
            continue
        end = max(dt.date.fromisoformat(str(s["val_end"])[:10]) for s in folds)
        if widest is None or end > widest[1]:
            widest = (label, end)
    if widest is None:
        pytest.skip(f"{case_study} declares no label with resolvable fold boundaries")

    label, end = widest
    assert end < holdout_start, (
        f"{case_study}: {label} is the widest configured label at {end}, which is not before "
        f"the holdout opening {holdout_start}"
    )
