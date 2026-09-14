"""A second evaluation of a spent holdout window is refused unless it is named.

`holdout_generations_to_retire` has been in the tree since 2026-09-12 and five of the
nine `*_holdout_predictions` notebooks call it. The four that do not are
`cme_futures`, `crypto_perps_funding`, `fx_pairs` and `sp500_equity_option_analytics`
(ml4t/agent-workspace#1174), and `fx_pairs` is where the absence cost something: it
registered a second holdout generation on 2026-09-13 against a window spent on
2026-09-09, and public #1038's audit recorded that "neither guard fires, because both
are scoped per training identity". The conclusion was right and the reason was not -
the helper is scoped per window and would have bucketed each generation as superseding
the other. Nothing fired because the notebook never called it.

The refusal moves here so the nine share one implementation rather than four copies of
it, and it takes an override, because a guard with no way past it gets deleted the
first time someone needs to proceed. The override is per generation and not a boolean:
a boolean is set once and left set, and the guard is then decorative.
"""

from __future__ import annotations

import pytest

from case_studies.utils.strategy_analysis import (
    HoldoutGenerationsToRetire,
    HoldoutWindowSpent,
    refuse_a_second_look,
)

CHECKPOINT = ("epoch", 35)


def _row(prediction_hash: str, config_name: str = "ridge_a1000000.0") -> dict:
    return {
        "prediction_hash": prediction_hash,
        "training_hash": "3cf2fbd5bb3a",
        "config_name": config_name,
    }


def _retire(**buckets) -> HoldoutGenerationsToRetire:
    return HoldoutGenerationsToRetire(
        superseded=tuple(buckets.get("superseded", ())),
        not_out_of_sample=tuple(buckets.get("not_out_of_sample", ())),
        unattributable=tuple(buckets.get("unattributable", ())),
    )


def _call(retire, retiring=()):
    return refuse_a_second_look(
        retire,
        this_configuration="lstm_h64",
        this_training_hash="619480dade83",
        checkpoint=CHECKPOINT,
        retiring=retiring,
    )


def test_an_empty_window_passes() -> None:
    assert _call(_retire()) == ()


def test_a_superseded_generation_refuses_by_default() -> None:
    """The fx_pairs shape: one registered generation, a different configuration arriving."""
    with pytest.raises(HoldoutWindowSpent) as raised:
        _call(_retire(superseded=[_row("9a5dfc90daae")]))

    message = str(raised.value)
    assert "9a5dfc90daae" in message
    assert "ridge_a1000000.0" in message
    assert "lstm_h64" in message, "the refusal must name what would be evaluated, not only what is"
    assert "RETIRE_HOLDOUT_GENERATIONS" in message, "a refusal with no way past it gets deleted"


def test_naming_the_generation_lets_the_run_proceed() -> None:
    rows = [_row("9a5dfc90daae")]
    assert _call(_retire(superseded=rows), retiring=["9a5dfc90daae"]) == tuple(rows)


def test_the_rows_being_retired_come_back_for_the_render() -> None:
    """A second look that proceeded deliberately has to be visible to a reader.

    The registry would otherwise hold two evaluations of the window and the notebook
    would show one, which is the state that makes the out-of-sample claim false rather
    than weak.
    """
    rows = [_row("9a5dfc90daae"), _row("aaaabbbbcccc", "sdf")]
    returned = _call(_retire(superseded=rows), retiring=["9a5dfc90daae", "aaaabbbbcccc"])

    assert [row["prediction_hash"] for row in returned] == ["9a5dfc90daae", "aaaabbbbcccc"]
    assert [row["config_name"] for row in returned] == ["ridge_a1000000.0", "sdf"]


def test_naming_only_one_of_two_still_refuses() -> None:
    with pytest.raises(HoldoutWindowSpent) as raised:
        _call(
            _retire(superseded=[_row("9a5dfc90daae"), _row("aaaabbbbcccc", "sdf")]),
            retiring=["9a5dfc90daae"],
        )

    assert "aaaabbbbcccc" in str(raised.value)
    assert "9a5dfc90daae" not in str(raised.value), "the named one is not what is being refused"


def test_an_override_naming_something_absent_refuses() -> None:
    """A stale override would otherwise sit in the launch line authorizing what arrives next."""
    with pytest.raises(HoldoutWindowSpent) as raised:
        _call(_retire(superseded=[_row("9a5dfc90daae")]), retiring=["deadbeefcafe"])

    assert "deadbeefcafe" in str(raised.value)
    assert "which this window does not carry" in str(raised.value)


def test_an_override_on_an_empty_window_refuses() -> None:
    with pytest.raises(HoldoutWindowSpent):
        _call(_retire(), retiring=["9a5dfc90daae"])


@pytest.mark.parametrize("bucket", ["unattributable", "not_out_of_sample"])
def test_the_other_two_buckets_are_not_overridable(bucket: str) -> None:
    """Naming one would assert a fact the registry does not hold.

    `unattributable` records no CV split, so whether it was refitted for the holdout
    cannot be established either way. `not_out_of_sample` is either a validation-fitted
    model published over the window or a refit filed under its validation identity, and
    the registry cannot tell those apart - an override there would not authorize a second
    look, it would authorize reporting something that may never have been out of sample.
    """
    rows = [_row("9a5dfc90daae")]
    with pytest.raises(HoldoutWindowSpent):
        _call(_retire(**{bucket: rows}), retiring=["9a5dfc90daae"])


def test_an_unattributable_row_is_refused_before_a_superseded_one() -> None:
    """The unshowable bucket is the one to report: it is not a decision anybody can make."""
    with pytest.raises(HoldoutWindowSpent) as raised:
        _call(
            _retire(unattributable=[_row("1111aaaa2222")], superseded=[_row("9a5dfc90daae")]),
            retiring=["9a5dfc90daae"],
        )

    assert "1111aaaa2222" in str(raised.value)
    assert "record no CV split" in str(raised.value)
