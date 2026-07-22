"""Tests for case_studies/utils/registry/specs.py.

Hashing determinism is load-bearing: the registry is a content-addressed
store, so any perturbation to hash computation (key ordering, separator
choice, seed handling) silently duplicates runs and corrupts lineage.

These tests pin the exact byte-for-byte hash output so a reformat of
canonical_json or compute_hash cannot change the addresses of existing runs.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from case_studies.utils.registry import registration
from case_studies.utils.registry.specs import (
    DEFAULT_SEED,
    HASH_LENGTH,
    _validate_spec,
    backtest_hash_from_parts,
    build_training_spec,
    canonical_json,
    compute_hash,
    prediction_hash_from_parts,
    training_hash_from_spec,
)

# -----------------------------------------------------------------------------
# canonical_json — deterministic serialization
# -----------------------------------------------------------------------------


def test_canonical_json_sorts_keys() -> None:
    a = canonical_json({"b": 2, "a": 1})
    b = canonical_json({"a": 1, "b": 2})
    assert a == b
    assert a == '{"a":1,"b":2}'


def test_canonical_json_uses_compact_separators() -> None:
    assert canonical_json({"x": 1}) == '{"x":1}'


def test_canonical_json_stringifies_unserializable_via_default() -> None:
    """Path/Enum/datetime-like fields fall through `default=str` so a spec
    never fails serialization."""
    from pathlib import Path

    out = canonical_json({"path": Path("/tmp/x.parquet")})
    assert "/tmp/x.parquet" in out


def test_canonical_json_is_deterministic_across_nested_structures() -> None:
    spec = {
        "outer": {"b": [3, 2, 1], "a": {"z": 9, "y": 8}},
        "flat": 42,
    }
    first = canonical_json(spec)
    second = canonical_json(spec)
    assert first == second
    # Keys are sorted at every level
    assert first.index('"a":') < first.index('"b":')
    assert first.index('"y":') < first.index('"z":')


# -----------------------------------------------------------------------------
# compute_hash — sha256 truncation invariant
# -----------------------------------------------------------------------------


def test_compute_hash_default_length_is_12() -> None:
    h = compute_hash("anything")
    assert len(h) == HASH_LENGTH == 12


def test_compute_hash_is_prefix_of_full_sha256() -> None:
    content = "some_training_content"
    expected_prefix = hashlib.sha256(content.encode()).hexdigest()[:12]
    assert compute_hash(content) == expected_prefix


def test_compute_hash_length_override_respects_arg() -> None:
    assert len(compute_hash("x", length=6)) == 6
    assert len(compute_hash("x", length=64)) == 64


# -----------------------------------------------------------------------------
# training_hash_from_spec
# -----------------------------------------------------------------------------


def _base_spec(**overrides) -> dict:
    spec = {
        "family": "linear",
        "label": "fwd_ret_21d",
        "seed": 42,
        "n_folds": 5,
    }
    spec.update(overrides)
    return spec


def test_training_hash_is_deterministic() -> None:
    spec = _base_spec()
    assert training_hash_from_spec(spec) == training_hash_from_spec(dict(spec))


def test_training_hash_differs_when_seed_changes() -> None:
    assert training_hash_from_spec(_base_spec(seed=1)) != training_hash_from_spec(
        _base_spec(seed=2)
    )


def test_training_hash_differs_when_family_changes() -> None:
    assert training_hash_from_spec(_base_spec(family="gbm")) != training_hash_from_spec(
        _base_spec(family="linear")
    )


def test_training_hash_differs_when_label_changes() -> None:
    assert training_hash_from_spec(_base_spec(label="fwd_ret_5d")) != training_hash_from_spec(
        _base_spec(label="fwd_ret_21d")
    )


def test_training_hash_invariant_under_key_order() -> None:
    """Client code may build the spec dict in arbitrary order; hash must be stable."""
    spec_a = {"family": "gbm", "label": "fwd_ret_21d", "seed": 42}
    spec_b = {"seed": 42, "label": "fwd_ret_21d", "family": "gbm"}
    assert training_hash_from_spec(spec_a) == training_hash_from_spec(spec_b)


# -----------------------------------------------------------------------------
# Spec validation
# -----------------------------------------------------------------------------


def test_validate_spec_missing_seed_injects_default(caplog) -> None:
    """Missing-seed-only case: warn and inject DEFAULT_SEED."""
    with caplog.at_level("WARNING"):
        enriched = _validate_spec({"family": "gbm", "label": "fwd_ret_5d"})

    assert enriched["seed"] == DEFAULT_SEED
    assert "missing 'seed'" in caplog.text


def test_validate_spec_missing_multiple_fields_raises() -> None:
    """Anything beyond a missing seed is a hard error."""
    with pytest.raises(ValueError, match="missing required fields"):
        _validate_spec({"family": "gbm"})  # missing label + seed


def test_validate_spec_does_not_mutate_original() -> None:
    original = {"family": "gbm", "label": "x"}
    enriched = _validate_spec(original)
    assert "seed" not in original  # original untouched
    assert enriched["seed"] == DEFAULT_SEED


# -----------------------------------------------------------------------------
# prediction_hash_from_parts
# -----------------------------------------------------------------------------


def test_prediction_hash_combines_training_hash_checkpoint_split() -> None:
    h = prediction_hash_from_parts("abc123", 100, "val")
    # Reconstruct exact content and compare to the public API
    assert h == compute_hash("abc123|100|val")


def test_prediction_hash_none_checkpoint_becomes_final() -> None:
    h_none = prediction_hash_from_parts("abc", None, "val")
    h_final_str = prediction_hash_from_parts("abc", None, "val")  # Same call
    assert h_none == compute_hash("abc|final|val")
    assert h_none == h_final_str


def test_prediction_hash_distinct_on_split() -> None:
    assert prediction_hash_from_parts("abc", 1, "val") != prediction_hash_from_parts(
        "abc", 1, "test"
    )


def test_prediction_hash_distinct_on_checkpoint() -> None:
    assert prediction_hash_from_parts("abc", 10, "val") != prediction_hash_from_parts(
        "abc", 20, "val"
    )


# -----------------------------------------------------------------------------
# backtest_hash_from_parts
# -----------------------------------------------------------------------------


def test_backtest_hash_combines_prediction_hash_and_strategy_spec() -> None:
    strategy = {"signal": {"method": "equal_weight_top_k", "top_k": 10}}
    h = backtest_hash_from_parts("pred123", strategy)
    assert h == compute_hash(f"pred123|{canonical_json(strategy)}")


def test_backtest_hash_sensitive_to_strategy_change() -> None:
    base = {"top_k": 10}
    variant = {"top_k": 20}
    assert backtest_hash_from_parts("p1", base) != backtest_hash_from_parts("p1", variant)


def test_backtest_hash_invariant_under_strategy_key_order() -> None:
    a = {"signal": {"method": "x", "top_k": 10}, "allocation": {"method": "eq"}}
    b = {"allocation": {"method": "eq"}, "signal": {"top_k": 10, "method": "x"}}
    assert backtest_hash_from_parts("p", a) == backtest_hash_from_parts("p", b)


def test_backtest_hash_excludes_runtime_config_object() -> None:
    planned = {
        "version": 2,
        "strategy": {"signal": {"method": "equal_weight_top_k", "top_k": 10}},
        "backtest_config": {"cash": {"initial": 100_000.0}},
    }
    runtime = {**planned, "_runtime_backtest_config": object()}

    assert backtest_hash_from_parts("pred", runtime) == backtest_hash_from_parts("pred", planned)


def test_backtest_hash_normalizes_default_long_only_direction() -> None:
    implicit = {"strategy": {"signal": {"method": "equal_weight_top_k", "top_k": 5}}}
    explicit = {
        "strategy": {
            "signal": {
                "method": "equal_weight_top_k",
                "top_k": 5,
                "direction": "long_only",
            }
        }
    }
    short = {
        "strategy": {
            "signal": {
                "method": "equal_weight_top_k",
                "top_k": 5,
                "direction": "short_only",
            }
        }
    }
    assert backtest_hash_from_parts("p", implicit) == backtest_hash_from_parts("p", explicit)
    assert backtest_hash_from_parts("p", implicit) != backtest_hash_from_parts("p", short)


# -----------------------------------------------------------------------------
# Regression pin — the exact hash for a canonical spec
# -----------------------------------------------------------------------------


def test_training_hash_regression_pin_for_canonical_spec() -> None:
    """Pin the exact hash of a minimal valid spec. Changing this value
    invalidates every existing registry entry — so any change should be an
    explicit migration, not an accidental refactor."""
    spec = {"family": "linear", "label": "fwd_ret_21d", "seed": 42}
    content = json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str)
    expected = hashlib.sha256(content.encode()).hexdigest()[:12]

    assert training_hash_from_spec(spec) == expected


def test_legacy_input_lineage_remains_part_of_certified_registry_identity() -> None:
    lineage = {"fingerprint": "certified", "artifacts": {"features": {"sha256": "abc"}}}
    spec = build_training_spec(
        "linear",
        "ols",
        "fwd_ret_5d",
        n_folds=2,
        input_lineage=lineage,
    )

    assert spec["input_lineage"] == lineage
    assert training_hash_from_spec(spec) != training_hash_from_spec(
        build_training_spec("linear", "ols", "fwd_ret_5d", n_folds=2)
    )


def test_pca_preset_preserves_shipped_registry_identity() -> None:
    spec = build_training_spec(
        "latent_factors",
        "pca",
        "fwd_ret_5d",
        n_folds=5,
        n_epochs=50,
    )

    assert spec["library"] == "sklearn"
    assert spec["params"] == {"n_factors": 5}
    assert training_hash_from_spec(spec) == "c59b8988c8c7"


def test_epoch_registration_merges_spec_extra_params_on_preset_path(monkeypatch) -> None:
    captured = {}

    def capture_training_run(case_study, *, spec, **kwargs):
        captured["case_study"] = case_study
        captured["spec"] = spec
        return training_hash_from_spec(spec)

    monkeypatch.setattr(registration, "register_training_run", capture_training_run)
    monkeypatch.setattr(registration, "register_prediction_set", lambda *args, **kwargs: "pred")

    result = registration.register_epoch_checkpoint(
        "etfs",
        family="tabular_dl",
        library="tabm",
        config_name="tabm_s",
        label="fwd_ret_21d",
        n_folds=8,
        n_epochs=100,
        best_epoch=50,
        ic_mean=0.01,
        predictions=None,
        spec_extra_params={
            "batch_size": 4096,
            "input_data_spec": {"input_digest": "sha256:input"},
        },
    )

    assert captured["case_study"] == "etfs"
    assert captured["spec"]["params"]["batch_size"] == 4096
    assert captured["spec"]["params"]["input_data_spec"]["input_digest"] == "sha256:input"
    assert result == training_hash_from_spec(captured["spec"])
