"""The nasdaq rank-1 rung pin names a label, and that label is the one the case study declares.

The pin restricts to the cost-feasible universe, which spans all four declared labels. Without
the label clause `ORDER BY sharpe DESC LIMIT 1` picks among them, and on 2026-09-14 it would
have picked `fwd_dir_15m` at +2.416 over the primary label's best at +2.300 - moving the rank-1
rung onto a label this case study is not featured on, with nothing in the output saying so.

The clause used to be load-bearing for a narrower reason: the pin also named
`family == "ensemble"`, `14_backtest` registers one ensemble per label, and the three matched
rows were +0.566 (`fwd_ret_15m`), +0.215 (`fwd_ret_5m`) and -0.268 (`fwd_ret_60m`). The family
clause is gone, so the pool the label discriminates over is now the whole cost-feasible sweep
rather than three ensembles, and the margin it decides is 0.116 Sharpe rather than 0.351.

Two things are asserted here, and the second is the one that rots without a test: that the pin
carries a label at all, and that the label it carries is `config/setup.yaml::labels.primary`.
A pin written as a literal beside a declaration it must equal is the shape that drifts when
the declaration moves.
"""

from __future__ import annotations

import polars as pl
import pytest
import yaml

from case_studies.utils.paired_metrics import RUNG_PINS
from utils.paths import get_case_study_dir

PINNED_WITH_A_LABEL = ("nasdaq100_microstructure",)


def declared_primary_label(case_study: str) -> str:
    setup = yaml.safe_load((get_case_study_dir(case_study) / "config" / "setup.yaml").read_text())
    return setup["labels"]["primary"]


@pytest.mark.parametrize("case_study", PINNED_WITH_A_LABEL)
def test_the_pin_names_a_label(case_study):
    pin = RUNG_PINS[case_study]
    assert "label" in pin, (
        f"{case_study}'s rung pin has no label, so one ensemble per label matches it and the "
        "sort decides which anchors the rank-1 rung"
    )


@pytest.mark.parametrize("case_study", PINNED_WITH_A_LABEL)
def test_the_pinned_label_is_the_declared_primary(case_study):
    assert RUNG_PINS[case_study]["label"] == declared_primary_label(case_study)


@pytest.mark.parametrize("case_study", PINNED_WITH_A_LABEL)
def test_the_predicate_and_its_mirror_agree(case_study):
    """The polars predicate and the plain keys beside it must select the same thing.

    The keys exist for the SQL paths that cannot take an expression, so a predicate corrected
    without its mirror leaves two selections under one name. Applied to a frame holding one row
    per label, the predicate must keep exactly the row the mirror names.
    """
    pin = RUNG_PINS[case_study]
    # Families vary across the rows so a predicate that still narrows on one would keep fewer
    # rows than the mirror names, rather than passing on a frame where every family agrees.
    frame = pl.DataFrame(
        {
            "label": ["fwd_ret_5m", "fwd_ret_15m", "fwd_ret_60m", "fwd_dir_15m"],
            "family": ["ensemble", "deep_learning", "gbm", "gbm"],
            "universe_filter": [pin["universe_filter"]] * 4,
            "exit_at_max_days": [None] * 4,
        }
    )
    kept = frame.filter(pin["predicate"])
    assert kept.height == 1, f"the predicate matched {kept.height} of 4 labels, not 1"
    assert kept["label"][0] == pin["label"]
