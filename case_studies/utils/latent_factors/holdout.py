"""Re-key a locked latent-factor training spec from the validation folds to the holdout fold."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.latent_factors.adapter import (
    _LATENT_MODELS,
    _prepare_expected_keys,
    _resolved_macro_digest,
)

if TYPE_CHECKING:
    from case_studies.research.workspace import Study


def rekey_holdout_spec(
    study: Study,
    spec: dict[str, Any],
    *,
    validation_spec: dict[str, Any],
) -> None:
    """Recompute the fold-derived fields against the holdout fold, in place, before a lock.

    This is the latent-factor half of the hook ``case_studies.research.holdout._rekey_holdout_spec``
    dispatches to, and it exists for the same reason the linear one does: ``spec`` arrives with its
    CV already replaced by the derived holdout fold, while ``expected_prediction_keys`` still
    describes the validation folds. Carrying that forward locks a manifest ``validate_locked_run``
    will reject, and dropping it locks one it will reject differently. Until this existed, every
    latent-factor holdout lock refused with ``NotImplementedError``, which is why
    sp500_equity_option_analytics - whose selected configuration is an ``sae`` - could not close.

    Two fields are re-keyed, and the second only for ``sdf``:

    * ``expected_prediction_keys``, the eligibility manifest, recomputed by running the same
      ``_prepare_expected_keys`` the validation run used against the holdout fold's own rows.
      Recomputing rather than rescaling matters because eligibility is not uniform in time - a
      symbol with too few observations in its ragged window is dropped, and which symbols those
      are differs fold by fold.
    * ``macro_context.resolved_fold_digest``, which ``sdf`` resolves from the macro panel
      restricted to the fold being fitted, so the validation folds' digest describes a different
      panel slice.

    The hook was necessary and not sufficient. Underneath it the deriver and the producer
    disagreed about the holdout's training interval - ``build_holdout_cv`` asked for
    2017-01-05..2020-12-16 while the stage-04 fold covering it begins 2019-01-02, leaving 495
    of 977 training dates covered - and this refused rather than fit on half-null features.
    :func:`holdout_training_floor` below resolves that by bounding the derived window at the
    artifact's own start, so the refusal now fires only for a real gap. ml4t/agent-workspace#977
    has the measurement.

    ``validation_spec`` is unused here. The linear hook needs it to replay a data-derived penalty
    against a recorded fold before trusting the preset; latent-factor models resolve no parameter
    from a fold's own training rows, so there is nothing of that kind to verify. It stays in the
    signature because the dispatch passes it to every family.
    """
    from case_studies.research.contracts import ExecutionTier
    from case_studies.research.models import locked_holdout_split
    from case_studies.utils.latent_factors.case_study import load_case_study_context

    computation = spec["computation"]
    model = computation.get("model")
    model_name = str(model.get("class", "")) if isinstance(model, dict) else ""
    if spec.get("family") != "latent_factors" or model_name not in _LATENT_MODELS:
        raise ValueError(
            "the latent-factor holdout hook was handed a specification it does not fit: "
            f"family={spec.get('family')!r} model={model_name!r}"
        )

    label_ref = study.labels.get(spec["label"], execution_tier=ExecutionTier.CANONICAL)
    case = load_case_study_context(
        study.case_study,
        primary_label=label_ref.name,
        max_symbols=0,
        use_macro=model_name == "sdf",
    )
    split = locked_holdout_split(spec, case.dataset, case.date_col, study.case_study)
    _require_holdout_temporal_features(case, split)
    case.splits = [split]

    expected = _prepare_expected_keys(case, model_name)
    computation["expected_prediction_keys"] = {
        "digest": value_digest(expected, ("symbol", "timestamp", "fold")),
        "n_rows": expected.height,
        "n_folds": expected.get_column("fold").n_unique(),
    }

    macro = computation.get("macro_context")
    if model_name == "sdf" and isinstance(macro, dict):
        macro["resolved_fold_digest"] = _resolved_macro_digest(case)


def _require_holdout_temporal_features(case: Any, split: dict[str, Any]) -> None:
    """Refuse a lock whose holdout fold the model-based feature artifact does not cover.

    The features are joined by ``(entity, date)``, so a holdout fold outside the artifact's rows
    would join from the wrong fold or from nothing, and the holdout would not evaluate the
    configuration selection ranked. The artifact cannot simply be regenerated to include the fold:
    the lock pins it by whole-file sha256, so writing the fold in changes the digest the selection
    was made under. Coverage is the check that can be met.
    """
    from case_studies.research.cv import require_fold_scoped_temporal_holdout_coverage

    if case.temporal_by_fold is None or not case.temporal_keys or not case.temporal_feature_names:
        return
    require_fold_scoped_temporal_holdout_coverage(
        split,
        case.temporal_by_fold,
        source_timeline=case.dataset.get_column(case.date_col),
        declared_folds=case.temporal_artifact_splits,
        date_col=case.date_col,
    )


def holdout_training_floor(study: Study, *, validation_spec: dict[str, Any]) -> Any | None:
    """The earliest date this configuration's fold-scoped features actually reach.

    ``build_holdout_cv`` starts the holdout training window at the earliest validation fold's
    start so the final fit gets the longest history. For a latent-factor configuration that is
    the right rule read against the wrong quantity: the model-based features are fold-scoped, and
    stage 04 emits each fold over a rolling window, so there is no feature history before the
    fold the holdout will be joined against. On sp500_equity_option_analytics the deriver asks
    for 2017-01-05 and the artifact's holdout fold begins 2019-01-02; training the difference
    means fitting 482 of 977 dates on null columns, which is not the configuration selection
    ranked - every validation fold saw a fully populated window.

    So this returns the artifact's own start for the fold the holdout will carry, and
    ``build_holdout_cv`` takes the later of the two. Returning ``None`` leaves the derivation
    untouched, which is the answer whenever the configuration has no fold-scoped features.

    The case-study context is loaded here and again in :func:`rekey_holdout_spec`. That is
    deliberate rather than overlooked: a lock is derived once per case study, and threading a
    cache through two hooks with different dispatch points would buy nothing and couple them.
    """
    from case_studies.utils.latent_factors.case_study import load_case_study_context

    computation = validation_spec.get("computation") or {}
    model = computation.get("model")
    model_name = str(model.get("class", "")) if isinstance(model, dict) else ""
    if validation_spec.get("family") != "latent_factors" or model_name not in _LATENT_MODELS:
        return None

    folds = ((computation.get("cv") or {}).get("folds")) or []
    if not folds:
        return None
    # The fold id build_holdout_cv is about to mint, derived the same way it derives it.
    fold_id = max(int(entry["fold"]) for entry in folds) + 1

    case = load_case_study_context(
        study.case_study,
        primary_label=str(validation_spec.get("label") or ""),
        max_symbols=0,
        use_macro=False,
    )
    if case.temporal_by_fold is None or not case.temporal_keys or not case.temporal_feature_names:
        return None

    import polars as pl

    frame = case.temporal_by_fold
    if isinstance(frame, pl.LazyFrame):
        frame = frame.select(["fold", case.date_col]).collect()
    elif not isinstance(frame, pl.DataFrame):
        frame = pl.from_pandas(frame.loc[:, ["fold", case.date_col]])
    dates = frame.filter(pl.col("fold") == fold_id).get_column(case.date_col)
    # An absent fold is not this hook's refusal to make. `require_fold_scoped_temporal_holdout
    # _coverage` says so with the numbers; returning None here lets it.
    return None if dates.is_empty() else dates.min()
