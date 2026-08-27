from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from case_studies.research.cv import (
    require_fold_scoped_temporal_holdout_coverage,
)
from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.latent_factors.adapter import (
    _LATENT_MODELS,
    _prepare_expected_keys,
    _resolved_macro_digest,
    reconstruct_locked_request,
)

if TYPE_CHECKING:
    from case_studies.research.workspace import Study


def prepare_locked_holdout_spec(
    study: Study,
    spec: dict[str, Any],
    *,
    checkpoint_kind: str,
    checkpoint_value: int | None,
) -> dict[str, Any]:
    """Resolve latent-factor eligibility against the locked holdout fold."""
    from case_studies.research.contracts import ExecutionTier
    from case_studies.research.models import locked_holdout_split
    from case_studies.utils.latent_factors.case_study import load_case_study_context

    prepared = deepcopy(spec)
    computation = prepared.get("computation")
    model = computation.get("model") if isinstance(computation, dict) else None
    model_name = str(model.get("class", "")) if isinstance(model, dict) else ""
    if prepared.get("family") != "latent_factors" or model_name not in _LATENT_MODELS:
        raise ValueError("holdout preparation requires a resolved latent-factor specification")
    study.require_writable()
    study.activate(ExecutionTier.CANONICAL)
    label_ref = study.labels.get(prepared["label"], execution_tier=ExecutionTier.CANONICAL)
    case = load_case_study_context(
        study.case_study,
        primary_label=label_ref.name,
        max_symbols=0,
        use_macro=model_name == "sdf",
    )
    split = locked_holdout_split(prepared, case.dataset, case.date_col, study.case_study)
    if case.temporal_by_fold is not None and case.temporal_keys and case.temporal_feature_names:
        require_fold_scoped_temporal_holdout_coverage(
            split,
            case.temporal_by_fold,
            source_timeline=case.dataset.get_column(case.date_col),
            date_col=case.date_col,
        )
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
    reconstruct_locked_request(
        study,
        prepared,
        checkpoint_kind=checkpoint_kind,
        checkpoint_value=checkpoint_value,
    )
    return prepared
