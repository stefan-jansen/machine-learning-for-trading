"""Two members fitted on identical files compare as fitted on identical files.

`Result.protocol()` reads `feature_artifacts` out of `computation`, and `CandidateSet.create`
refuses any set whose members differ on a protocol field the caller did not declare comparable.
Six of the seven producers build that field from `mds.input_lineage["artifacts"]`, a mapping of
`{role: {"sha256": <hex>, "size": <int>}}`; the latent adapter builds it from
`case.input_data_spec["files"]`, a list of `{"role": ..., "sha256": "sha256:<hex>"}`.

They are the same statement in different words. Measured on the `etfs` registry 2026-09-07, one
latent-factor run and one linear run at `fwd_ret_21d`: the same three roles - financial, label,
model_based - carrying the same three sha256 values, one rendered with a prefix and no size and
the other with a size and no prefix. A candidate set spanning both families nonetheless refused
on `feature_artifacts`, and the only way past it was to declare that field comparable, which
silences the check for every member it could legitimately compare.

ml4t/agent-workspace#891.
"""

from __future__ import annotations

import pytest

from case_studies.research.comparison import _comparable_protocol_value
from case_studies.research.results import normalized_feature_artifacts

FINANCIAL = "75e4fd36a901a3b74ac85a584051ca903f295c8ba3a363c1e865fa784e83d5e0"
LABEL = "e24ffc0f5d050f6cdb7f748d1a8d070af197d0631fec94c43c0f60dc3bdfdfc6"
MODEL_BASED = "ea4b1074a15d6d5bf4b8b49ff36171d52c9864f946c42737319c96455bd832b4"

# Both taken verbatim from the two etfs runs named in the module docstring.
STANDARD = {
    "financial": {"sha256": FINANCIAL, "size": 146686000},
    "label": {"sha256": LABEL, "size": 3817233},
    "model_based": {"sha256": MODEL_BASED, "size": 3790152},
}
LATENT = [
    {"role": "financial", "sha256": f"sha256:{FINANCIAL}"},
    {"role": "label", "sha256": f"sha256:{LABEL}"},
    {"role": "model_based", "sha256": f"sha256:{MODEL_BASED}"},
]


def test_the_two_producer_shapes_state_the_same_inputs() -> None:
    """The finding, as an assertion: same roles, same content, two renderings."""
    assert normalized_feature_artifacts(STANDARD) == normalized_feature_artifacts(LATENT)
    assert normalized_feature_artifacts(LATENT) == {
        "financial": FINANCIAL,
        "label": LABEL,
        "model_based": MODEL_BASED,
    }


def test_a_different_file_still_differs() -> None:
    """The control. Normalizing the rendering must not normalize away a changed input."""
    moved = [
        {"role": "financial", "sha256": f"sha256:{FINANCIAL}"},
        {"role": "label", "sha256": f"sha256:{LABEL}"},
        {"role": "model_based", "sha256": "sha256:" + "0" * 64},
    ]
    assert normalized_feature_artifacts(STANDARD) != normalized_feature_artifacts(moved)


def test_a_missing_role_still_differs() -> None:
    """A member fitted without the model-based artifact is not the same protocol."""
    assert normalized_feature_artifacts(STANDARD) != normalized_feature_artifacts(LATENT[:2])


def test_an_unrecognized_shape_is_compared_exactly() -> None:
    """An unfamiliar rendering must not collapse to something that equals everything."""
    assert normalized_feature_artifacts("opaque") == "opaque"
    assert normalized_feature_artifacts(None) is None
    assert normalized_feature_artifacts(["a", "b"]) == ["a", "b"]


class TestOnlyFeatureArtifactsIsNormalized:
    """Every other protocol field is compared exactly, as it was."""

    def test_feature_artifacts_goes_through_the_normalizer(self) -> None:
        assert _comparable_protocol_value("feature_artifacts", LATENT) == (
            _comparable_protocol_value("feature_artifacts", STANDARD)
        )

    @pytest.mark.parametrize("field", ["label_artifact", "cv", "split", "execution_tier"])
    def test_other_fields_are_untouched(self, field: str) -> None:
        assert _comparable_protocol_value(field, STANDARD) == STANDARD
        assert _comparable_protocol_value(field, LATENT) == LATENT
        # And two renderings of one thing stay different in a field that is not this one, so
        # nothing here widens a comparison it was not asked to widen.
        assert _comparable_protocol_value(field, STANDARD) != _comparable_protocol_value(
            field, LATENT
        )


# ---------------------------------------------------------------------------
# The contract this buys, end to end
# ---------------------------------------------------------------------------


def test_a_mixed_set_is_created_without_declaring_the_field_comparable(tmp_path) -> None:
    """The consequence #891 names, and the workaround it forced.

    Before this, the only way to build a set spanning both families was to declare
    `feature_artifacts` comparable - which silences the check for every member in the set,
    including the ones whose inputs it could have compared. `cv` must not be declared
    alongside it, because since selection is best validation backtest Sharpe, `cv` is the only
    field checking that two ranked numbers were measured on the same folds.
    """
    import polars as pl

    from case_studies.research import CandidateSet
    from tests.test_research_registry import _predictions, _study, _training_spec

    study = _study(tmp_path)
    frame = _predictions()
    expected = frame.select("symbol", "timestamp", "fold_id")

    members = []
    for artifacts, config in ((STANDARD, "ridge_a"), (LATENT, "ridge_b")):
        training = study.results.register_training(
            _training_spec(feature_artifacts=artifacts, config_name=config)
        )
        members.append(
            study.results.publish_predictions(
                training,
                checkpoint_kind="final",
                checkpoint_value=None,
                split="validation",
                predictions=frame.with_columns(
                    (pl.col("y_score") + len(members) * 0.1).alias("y_score")
                ),
                expected_keys=expected,
            )
        )

    protocols = [member.protocol()["feature_artifacts"] for member in members]
    assert protocols[0] != protocols[1], "the two renderings must still be stored as written"

    candidate_set = CandidateSet.create(study, "mixed-families", members)
    assert set(candidate_set.members) == {member.hash for member in members}
    assert candidate_set.comparison_contract["comparable_fields"] == []

    # Compared normalized, stored raw. The contract's protocol enters `set_hash`, so it keeps
    # the rendering a member actually wrote: rewriting it would re-key all 37 candidate sets
    # across the five case studies that hold one, to change no answer.
    assert candidate_set.comparison_contract["protocol"]["feature_artifacts"] in protocols


def test_a_set_whose_members_read_different_files_is_still_refused(tmp_path) -> None:
    """The control: normalizing the rendering must not wave through a changed input."""
    import polars as pl
    import pytest as _pytest

    from case_studies.research import CandidateSet
    from tests.test_research_registry import _predictions, _study, _training_spec

    study = _study(tmp_path)
    frame = _predictions()
    expected = frame.select("symbol", "timestamp", "fold_id")
    moved = [{**item} for item in LATENT]
    moved[2] = {"role": "model_based", "sha256": "sha256:" + "0" * 64}

    members = []
    for artifacts, config in ((STANDARD, "ridge_a"), (moved, "ridge_b")):
        training = study.results.register_training(
            _training_spec(feature_artifacts=artifacts, config_name=config)
        )
        members.append(
            study.results.publish_predictions(
                training,
                checkpoint_kind="final",
                checkpoint_value=None,
                split="validation",
                predictions=frame.with_columns(
                    (pl.col("y_score") + len(members) * 0.1).alias("y_score")
                ),
                expected_keys=expected,
            )
        )

    with _pytest.raises(ValueError, match="feature_artifacts"):
        CandidateSet.create(study, "mixed-inputs", members)
