from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import polars as pl
import pytest
import yaml

from case_studies.cme_futures import research_workflow
from case_studies.cme_futures.research_workflow import (
    FuturesPricePath,
    _expiry_rules,
    create_label_candidate_sets,
    model_request_catalog,
    product_universe_table,
    publish_product_weights,
    resolved_model_plan,
    run_official_backtest_requests,
    strategy_request_frame,
)
from case_studies.research import (
    DecisionArtifact,
    OfficialPopulation,
    ResolvedModelRequest,
    StateTransitionPolicy,
    Study,
    supersedes_for_run,
)
from case_studies.utils.registry.store import _open_registry
from tests.test_research_contract_catalog import _resolved_spec
from tests.test_research_registry import _training_spec
from utils.paths import REPO_ROOT


@pytest.fixture(autouse=True)
def _restore_output_root():
    yield
    os.environ.pop("ML4T_OUTPUT_DIR", None)
    from case_studies.research import workspace

    workspace._ACTIVE_OUTPUT_ROOT = None
    workspace._clear_root_sensitive_caches()


def _study(tmp_path: Path) -> Study:
    output_root = tmp_path / "workspace"
    root = output_root / "cme_futures"
    root.mkdir(parents=True)
    (root / "config").symlink_to(
        REPO_ROOT / "case_studies" / "cme_futures" / "config", target_is_directory=True
    )
    (output_root / "config").symlink_to(
        REPO_ROOT / "case_studies" / "config", target_is_directory=True
    )
    _open_registry(root).close()
    return Study(
        case_study="cme_futures",
        root=root,
        release_root=tmp_path / "release",
        output_root=output_root,
        read_only=False,
        manifest={"schema_version": 1, "case_study": "cme_futures"},
    )


def _prediction(study: Study):
    training = study.results.register_training(_training_spec(label="fwd_ret_5d"))
    dates = pl.date_range(pl.date(2024, 1, 2), pl.date(2024, 1, 5), eager=True)
    frame = pl.DataFrame(
        {
            "symbol": [product for timestamp in dates for product in ("ES", "NQ")],
            "timestamp": [timestamp for timestamp in dates for _ in range(2)],
            "fold_id": [0] * (2 * len(dates)),
            "y_true": [0.01, -0.01] * len(dates),
            "y_score": [0.02, -0.02] * len(dates),
        }
    )
    return study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold_id"),
    )


def _current_prediction(study: Study, *, alpha: float = 1.0):
    training = study.results.register_training(_resolved_spec(alpha=alpha))
    dates = pl.date_range(pl.date(2024, 1, 2), pl.date(2024, 1, 5), eager=True)
    frame = pl.DataFrame(
        {
            "symbol": [product for timestamp in dates for product in ("ES", "NQ")],
            "timestamp": [timestamp for timestamp in dates for _ in range(2)],
            "fold": [0] * (2 * len(dates)),
            "actual": [0.01, -0.01] * len(dates),
            "prediction": [0.02, -0.02] * len(dates),
        }
    )
    return study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold"),
    )


def _product_prices() -> pl.DataFrame:
    dates = pl.date_range(pl.date(2024, 1, 2), pl.date(2024, 1, 5), eager=True)
    return pl.DataFrame(
        {
            "product": [product for timestamp in dates for product in ("ES", "NQ")],
            "timestamp": [timestamp for timestamp in dates for _ in range(2)],
            "open": [4_800.0, 16_800.0, 4_810.0, 16_820.0] * 2,
            "high": [4_820.0, 16_850.0, 4_830.0, 16_870.0] * 2,
            "low": [4_780.0, 16_750.0, 4_790.0, 16_770.0] * 2,
            "close": [4_810.0, 16_820.0, 4_820.0, 16_840.0] * 2,
            "volume": [1_000] * (2 * len(dates)),
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))


def _returns(result) -> pl.DataFrame:
    path = result.root / "run_log" / "backtest" / result.hash / "daily_returns.parquet"
    return pl.read_parquet(path)


def test_product_decision_matches_existing_futures_engine_path(tmp_path: Path) -> None:
    study = _study(tmp_path)
    prediction = _prediction(study)
    prices = _product_prices()
    signal = {"method": "equal_weight_top_k", "top_k": 1}
    allocation = {"method": "inverse_vol", "vol_window": 2}

    decision = publish_product_weights(
        prediction,
        prices=prices,
        signal=signal,
        allocation=allocation,
    )
    resolved_signal = decision.spec["parameters"]["signal"]
    resolved_allocation = decision.spec["parameters"]["allocation"]
    direct = study.strategy(
        prediction=prediction,
        signal=resolved_signal,
        allocation=resolved_allocation,
    ).run(prices=prices)
    typed = study.strategy(
        prediction=prediction,
        signal=resolved_signal,
        allocation=resolved_allocation,
        decision=decision,
    ).run(prices=prices)

    assert decision.spec["decision_keys"] == ["product", "timestamp"]
    assert decision.load().columns == ["product", "timestamp", "weight", "fold"]
    assert decision.spec["parameters"]["signal"]["long_short"] is True
    assert decision.spec["parameters"]["allocation"]["long_short"] is True
    assert decision.load().filter(pl.col("weight") < 0).height > 0
    assert decision.load().filter(pl.col("weight") > 0).height > 0
    assert typed.spec()["entity_contract"] == {
        "reader_key": "product",
        "engine_key": "symbol",
        "mapping": "one_to_one_at_backtest_boundary",
    }
    assert typed.spec()["decision_artifact"]["decision_keys"] == ["product", "timestamp"]
    assert typed.spec()["decision_artifact"]["parameters"]["contract_position"] == 0
    assert typed.spec()["futures_market"]["roll"]["type"] == "volume"
    assert typed.spec()["futures_market"]["expiry"]["products"] == {
        "ES": {"contract_months": ["H", "M", "U", "Z"], "expiry_rule": "3rd_friday"},
        "NQ": {"contract_months": ["H", "M", "U", "Z"], "expiry_rule": "3rd_friday"},
    }
    assert _returns(typed).equals(_returns(direct))


def test_cme_strategy_rejects_symbol_decisions(tmp_path: Path) -> None:
    study = _study(tmp_path)
    prediction = _prediction(study)
    prices = _product_prices()
    decisions = (
        prediction.load().select("symbol", "timestamp").with_columns(pl.lit(0.5).alias("weight"))
    )
    artifact = DecisionArtifact.publish(
        study,
        kind="target_weights",
        decisions=decisions,
        prediction_hashes=[prediction.hash],
        parameters={"cadence": "7d"},
        state_transition_policy=StateTransitionPolicy(
            fold_boundary="liquidate",
            temporal_gap="continue",
        ),
    )

    with pytest.raises(ValueError, match="canonical product entity key"):
        study.strategy(
            prediction=prediction,
            signal={"method": "equal_weight_top_k", "top_k": 1},
            decision=artifact,
        ).resolve(prices=prices)


def test_cme_strategy_rejects_reader_supplied_symbol_prices(tmp_path: Path) -> None:
    study = _study(tmp_path)
    prediction = _prediction(study)

    with pytest.raises(ValueError, match="canonical product entity key"):
        study.strategy(
            prediction=prediction,
            signal={"method": "equal_weight_top_k", "top_k": 1},
        ).resolve(prices=_product_prices().rename({"product": "symbol"}))


def test_expiry_rules_carry_the_configured_contract_terms() -> None:
    rules = _expiry_rules(["ES", "CL"])

    assert rules.get_column("product").to_list() == ["CL", "ES"]
    assert rules.rows() == [
        ("CL", "3bd_before_25th_prior_month", "F,G,H,J,K,M,N,Q,U,V,X,Z"),
        ("ES", "3rd_friday", "H,M,U,Z"),
    ]


def test_expiry_rules_reject_unknown_and_incomplete_specifications(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="no contract specification"):
        _expiry_rules(["ES", "NOT_A_PRODUCT"])

    market = tmp_path / "data" / "futures" / "market"
    market.mkdir(parents=True)
    (market / "futures_specs.yaml").write_text(
        "products:\n"
        "  ES:\n"
        "    expiry_rule: 3rd_friday\n"
        "    contract_months: []\n"
        "  NQ:\n"
        "    contract_months: [H, M, U, Z]\n"
    )
    monkeypatch.setattr(research_workflow, "REPO_ROOT", tmp_path)

    with pytest.raises(ValueError, match="ES has an incomplete expiry specification"):
        _expiry_rules(["ES"])
    with pytest.raises(ValueError, match="NQ has an incomplete expiry specification"):
        _expiry_rules(["NQ"])


_PRICE_DATES = [datetime(2024, 1, 2), datetime(2024, 1, 3)]


def _engine_price_frame(products: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [product for product in products for _ in _PRICE_DATES],
            "timestamp": [timestamp for _ in products for timestamp in _PRICE_DATES],
            "close": [100.0 for _ in products for _ in _PRICE_DATES],
        }
    )


def _audit_frame(products: list[str], cum_ratio: float = 1.0) -> pl.DataFrame:
    rows = len(products) * len(_PRICE_DATES)
    return pl.DataFrame(
        {
            "product": [product for product in products for _ in _PRICE_DATES],
            "session_date": [timestamp for _ in products for timestamp in _PRICE_DATES],
            "tenor": [0] * rows,
            "adj_open": [100.0 * cum_ratio] * rows,
            "adj_high": [100.0 * cum_ratio] * rows,
            "adj_low": [100.0 * cum_ratio] * rows,
            "adj_close": [100.0 * cum_ratio] * rows,
            "raw_close": [100.0] * rows,
            "cum_ratio": [cum_ratio] * rows,
        }
    )


def _stub_price_loaders(
    monkeypatch: pytest.MonkeyPatch,
    *,
    engine_products: list[str],
    cum_ratio: float = 1.0,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    engine_calls: list[dict[str, object]] = []
    audit_calls: list[dict[str, object]] = []

    def fake_backtest_prices(case_study, label, **kwargs):
        engine_calls.append(kwargs)
        return _engine_price_frame(engine_products)

    def fake_load_cme_futures(**kwargs):
        audit_calls.append(kwargs)
        requested = kwargs.get("products") or engine_products
        return _audit_frame(list(requested), cum_ratio=cum_ratio)

    monkeypatch.setattr(research_workflow, "load_backtest_prices_for", fake_backtest_prices)
    monkeypatch.setattr(research_workflow, "load_cme_futures", fake_load_cme_futures)
    return engine_calls, audit_calls


def test_price_path_audits_exactly_the_products_the_engine_load_sampled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine_calls, audit_calls = _stub_price_loaders(monkeypatch, engine_products=["GC", "NQ"])

    path = research_workflow.load_futures_price_path("fwd_ret_5d", max_products=2)

    assert [call["max_symbols"] for call in engine_calls] == [2]
    assert len(audit_calls) == 1
    assert audit_calls[0]["products"] == ["GC", "NQ"]
    assert not audit_calls[0].get("max_symbols")
    assert path.audit.get_column("product").unique().sort().to_list() == ["GC", "NQ"]
    assert path.expiry_rules.get_column("product").to_list() == ["GC", "NQ"]


@pytest.mark.parametrize("cum_ratio", [float("nan"), float("inf"), 0.0, -1.0])
def test_price_path_rejects_non_finite_or_non_positive_roll_ratios(
    monkeypatch: pytest.MonkeyPatch, cum_ratio: float
) -> None:
    _stub_price_loaders(monkeypatch, engine_products=["ES"], cum_ratio=cum_ratio)

    with pytest.raises(ValueError, match="finite positive"):
        research_workflow.load_futures_price_path("fwd_ret_5d", products=["ES"])


def test_reader_plan_exposes_resolved_population_and_product_contract(tmp_path: Path) -> None:
    expected = pl.DataFrame(
        {
            "symbol": ["ES", "NQ", "ES", "NQ"],
            "timestamp": [
                datetime(2024, 1, 2),
                datetime(2024, 1, 2),
                datetime(2024, 2, 2),
                datetime(2024, 2, 2),
            ],
            "fold": [0, 0, 1, 1],
        }
    )
    spec = _resolved_spec()
    spec["label"] = "fwd_ret_5d"
    spec["execution_tier"] = "preview"
    spec["computation"].update(
        {
            "task": {"type": "regression"},
            "feature_names": ["carry", "momentum"],
            "checkpoint_schedule": [
                {"kind": "final", "value": None},
                {"kind": "epoch", "value": 10},
            ],
        }
    )
    request = ResolvedModelRequest(
        study=_study(tmp_path),
        family="linear",
        spec=spec,
        _context=SimpleNamespace(expected_keys=expected),
    )

    plan = resolved_model_plan([request])
    universe = product_universe_table()

    assert plan.select(
        "family",
        "label",
        "config_name",
        "task",
        "feature_count",
        "eligible_entities",
        "eligible_rows",
        "folds",
        "checkpoints",
        "execution_tier",
        "training_hash",
    ).row(0) == (
        "linear",
        "fwd_ret_5d",
        "ridge",
        "regression",
        2,
        2,
        4,
        2,
        2,
        "preview",
        request.identity,
    )
    configured = yaml.safe_load(
        (REPO_ROOT / "data" / "futures" / "market" / "futures_specs.yaml").read_text()
    )["products"]
    grouped = yaml.safe_load(
        (REPO_ROOT / "case_studies" / "cme_futures" / "config" / "setup.yaml").read_text()
    )["universe"]["product_groups"]
    expected_universe = sorted(
        (sector, product) for sector, products in grouped.items() for product in products
    )

    assert universe.height == 30
    assert universe.height == len(expected_universe)
    assert universe.get_column("product").n_unique() == universe.height
    assert list(zip(universe.get_column("sector"), universe.get_column("product"))) == [
        tuple(row) for row in expected_universe
    ]
    assert universe.get_column("expiry_rule").to_list() == [
        configured[product]["expiry_rule"] for _, product in expected_universe
    ]
    assert universe.get_column("contract_months").to_list() == [
        ",".join(configured[product]["contract_months"]) for _, product in expected_universe
    ]


def test_reader_plan_accepts_flat_family_spec(tmp_path: Path) -> None:
    expected = pl.DataFrame(
        {
            "symbol": ["ES", "NQ"],
            "timestamp": [datetime(2024, 1, 2), datetime(2024, 1, 2)],
            "fold": [0, 0],
        }
    )
    nested = _resolved_spec()
    computation = nested.pop("computation")
    spec = {**nested, **computation}
    spec.update(
        {
            "identity_version": 2,
            "label": "fwd_ret_5d",
            "execution_tier": "preview",
            "checkpoint_schedule": [{"kind": "epoch", "value": 2}],
        }
    )
    request = ResolvedModelRequest(
        study=_study(tmp_path),
        family="deep_learning",
        spec=spec,
        _context=SimpleNamespace(expected_keys=expected),
    )

    plan = resolved_model_plan([request])

    assert plan.item(0, "family") == "deep_learning"
    assert plan.item(0, "checkpoints") == len(spec["checkpoint_schedule"])


def test_model_request_catalog_rejects_each_unknown_requested_config(seeded_output_dir) -> None:
    """`seeded_output_dir` is what puts the training menus under the redirected root.

    `model_request_catalog` reads `config/training/<label>.yaml` through
    `get_case_study_dir`, which resolves under ML4T_OUTPUT_DIR. Without the fixture the
    call raises `ConfigError` for the missing menu before it ever reaches the unknown
    config name, so the test passes only where an earlier test happened to leave the
    variable pointing at a seeded directory.
    """
    with pytest.raises(ValueError, match="missing"):
        model_request_catalog(
            "linear",
            labels=("fwd_ret_5d",),
            config_names=("ols", "missing"),
        )


def test_canonical_decision_rejects_divergent_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from case_studies.cme_futures import research_workflow

    study = _study(tmp_path)
    prediction = _prediction(study)
    prices = _product_prices()
    original = research_workflow.resolve_product_weights
    calls = 0

    def divergent_replay(*args, **kwargs):
        nonlocal calls
        calls += 1
        weights = original(*args, **kwargs)
        if calls == 2:
            weights = weights.with_columns((pl.col("weight") + 0.01).alias("weight"))
        return weights

    monkeypatch.setattr(research_workflow, "resolve_product_weights", divergent_replay)

    with pytest.raises(RuntimeError, match="not deterministic"):
        research_workflow.publish_product_weights(
            prediction,
            prices=prices,
            signal={"method": "equal_weight_top_k", "top_k": 1},
            canonical=True,
        )


def test_visible_requests_snapshot_complete_canonical_backtests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from case_studies.cme_futures import research_workflow

    study = _study(tmp_path)
    prediction = _current_prediction(study)
    second_prediction = _current_prediction(study, alpha=2.0)
    prices = _product_prices()
    path = FuturesPricePath(
        prices=prices,
        audit=pl.DataFrame(
            {
                "product": ["ES", "NQ"],
                "position": [0, 0],
                "timestamp": [prices.item(0, "timestamp"), prices.item(1, "timestamp")],
                "cum_ratio": [1.0, 1.0],
            }
        ),
        roll_transitions=pl.DataFrame(),
        expiry_rules=_expiry_rules(["ES", "NQ"]),
    )
    monkeypatch.setattr(research_workflow, "load_futures_price_path", lambda *args, **kwargs: path)
    original_run_backtests = research_workflow.run_backtests
    backtest_calls = 0

    def run_after_population_snapshot(*args, **kwargs):
        nonlocal backtest_calls
        population = research_workflow.OfficialPopulation.one(
            study,
            name="test-cme-signal",
        )
        assert len(population.members) == 2
        with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
            completed = db.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0]
        assert completed == backtest_calls
        result = original_run_backtests(*args, **kwargs)
        backtest_calls += 1
        return result

    monkeypatch.setattr(research_workflow, "run_backtests", run_after_population_snapshot)
    requests = strategy_request_frame(
        [
            {
                "request_name": "ridge-equal-weight-k1",
                "prediction_hash": prediction.hash,
                "label": "fwd_ret_21d",
                "signal": {"method": "equal_weight_top_k", "top_k": 1},
                "allocation": {"method": "inverse_vol", "vol_window": 2},
                "risk": None,
                "costs": None,
                "chapter": "ch16",
            },
            {
                "request_name": "ridge-alpha2-equal-weight-k1",
                "prediction_hash": second_prediction.hash,
                "label": "fwd_ret_21d",
                "signal": {"method": "equal_weight_top_k", "top_k": 1},
                "allocation": None,
                "risk": None,
                "costs": None,
                "chapter": "ch16",
            },
        ]
    )

    execution = run_official_backtest_requests(
        study,
        requests,
        population_name="test-cme-signal",
    )

    assert execution.catalog_rows.get_column("complete").to_list() == [True, True]
    assert backtest_calls == 2
    assert execution.population.require_complete() == tuple(
        execution.catalog_rows.get_column("backtest_hash")
    )
    assert execution.results[0].spec()["decision_artifact"]["canonical"] is True
    candidate_sets = create_label_candidate_sets(
        study,
        execution,
        stage="signal",
    )
    assert set(candidate_sets) == {"fwd_ret_21d"}
    assert set(candidate_sets["fwd_ret_21d"].members) == set(
        execution.catalog_rows.get_column("backtest_hash")
    )
    shortlist = research_workflow.shortlist_signal_configurations(
        study, label="fwd_ret_21d", limit=1
    )
    assert shortlist[0].hash in set(candidate_sets["fwd_ret_21d"].members)


def test_candidate_set_stage_outside_the_funnel_is_refused() -> None:
    """A stage the funnel does not define never reaches the registry as a new namespace."""
    for stage in research_workflow.CANDIDATE_SET_STAGES:
        assert research_workflow.candidate_set_name(stage, "fwd_ret_21d").startswith("cme_futures-")
    with pytest.raises(ValueError, match="unknown candidate set stage 'cme-signal'"):
        research_workflow.candidate_set_name("cme-signal", "fwd_ret_21d")


_TABULAR_SHAPE = {
    "financial": {"sha256": "aa" * 32, "size": 75849052},
    "label": {"sha256": "bb" * 32, "size": 862351},
    "model_based": {"sha256": "cc" * 32, "size": 26132109},
}
_LATENT_SHAPE = [
    {"role": "financial", "sha256": "sha256:" + "aa" * 32},
    {"role": "label", "sha256": "sha256:" + "bb" * 32},
    {"role": "model_based", "sha256": "sha256:" + "cc" * 32},
    {"role": "setup", "sha256": "sha256:" + "dd" * 32},
]


def _member(name: str, feature_artifacts):
    """A stand-in carrying only what the feature-artifact comparison reads."""
    return SimpleNamespace(
        hash=name,
        protocol=lambda: {"feature_artifacts": feature_artifacts},
    )


def test_the_two_feature_artifact_recording_shapes_normalize_to_one_reading() -> None:
    """The latent adapter writes the same digests differently; the comparison sees through it."""
    tabular = research_workflow._feature_artifact_digests(_member("t", _TABULAR_SHAPE))
    latent = research_workflow._feature_artifact_digests(_member("l", _LATENT_SHAPE))
    assert tabular == latent
    assert tabular == {
        "financial": "aa" * 32,
        "label": "bb" * 32,
        "model_based": "cc" * 32,
    }
    # `setup` is a config digest the latent adapter records and the others do not. It is not a
    # feature input, so it must not enter the comparison and make the two shapes disagree.
    assert "setup" not in latent


def test_a_member_reading_a_different_feature_build_is_refused() -> None:
    """The guard the normalization preserves still fails on a real difference in inputs."""
    stale = {**_TABULAR_SHAPE, "financial": {"sha256": "ee" * 32, "size": 75849052}}
    with pytest.raises(ValueError, match=r"different feature artifacts: \['financial'\]"):
        research_workflow._require_agreeing_feature_artifacts(
            [_member("current", _TABULAR_SHAPE), _member("stale", stale)]
        )
    # Agreement across the two shapes is not an error.
    research_workflow._require_agreeing_feature_artifacts(
        [_member("t", _TABULAR_SHAPE), _member("l", _LATENT_SHAPE)]
    )


def test_a_member_recording_no_digest_for_a_role_is_refused() -> None:
    """A role recorded without a digest cannot pass as agreeing with one that has it."""
    blank = {**_TABULAR_SHAPE, "model_based": {"size": 26132109}}
    with pytest.raises(ValueError, match="records no digest for 'model_based'"):
        research_workflow._require_agreeing_feature_artifacts([_member("blank", blank)])


def test_a_member_reading_an_extra_feature_artifact_is_refused() -> None:
    """An extra feature input is a real difference, unlike the config digest that is dropped."""
    extra = {**_TABULAR_SHAPE, "sentiment": {"sha256": "ff" * 32, "size": 10}}
    with pytest.raises(ValueError, match=r"different feature artifacts: \['sentiment'\]"):
        research_workflow._require_agreeing_feature_artifacts(
            [_member("plain", _TABULAR_SHAPE), _member("extra", extra)]
        )


def test_a_role_recorded_as_a_bare_digest_string_normalizes_too() -> None:
    """Some results record a role as the digest itself rather than a mapping."""
    assert research_workflow._feature_artifact_digests(
        _member("bare", {"financial": "features-a"})
    ) == {"financial": "features-a"}


def _labelled_prediction(study: Study, *, label: str, alpha: float):
    """Publish one validation prediction whose training identity carries the given horizon."""
    spec = _resolved_spec(alpha=alpha)
    spec["label"] = label
    computation = spec.get("computation", spec)
    computation["label_artifact"] = f"{label}-artifact"
    training = study.results.register_training(spec)
    dates = pl.date_range(pl.date(2024, 1, 2), pl.date(2024, 1, 5), eager=True)
    frame = pl.DataFrame(
        {
            "symbol": [product for timestamp in dates for product in ("ES", "NQ")],
            "timestamp": [timestamp for timestamp in dates for _ in range(2)],
            "fold": [0] * (2 * len(dates)),
            "actual": [0.01, -0.01] * len(dates),
            "prediction": [0.02 * alpha, -0.02 * alpha] * len(dates),
        }
    )
    return study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold"),
    )


def _labelled_execution(study: Study, monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """Run one canonical backtest per return horizon and return their hashes by label."""
    prices = _product_prices()
    path = FuturesPricePath(
        prices=prices,
        audit=pl.DataFrame(
            {
                "product": ["ES", "NQ"],
                "position": [0, 0],
                "timestamp": [prices.item(0, "timestamp"), prices.item(1, "timestamp")],
                "cum_ratio": [1.0, 1.0],
            }
        ),
        roll_transitions=pl.DataFrame(),
        expiry_rules=_expiry_rules(["ES", "NQ"]),
    )
    monkeypatch.setattr(research_workflow, "load_futures_price_path", lambda *a, **k: path)
    rows = []
    for index, label in enumerate(research_workflow.ALL_LABELS):
        prediction = _labelled_prediction(study, label=label, alpha=1.0 + index)
        rows.append(
            {
                "request_name": f"{label}-equal-weight-k1",
                "prediction_hash": prediction.hash,
                "label": label,
                "signal": {"method": "equal_weight_top_k", "top_k": 1},
                "allocation": None,
                "risk": None,
                "costs": None,
                "chapter": "ch16",
            }
        )
    execution = run_official_backtest_requests(
        study,
        strategy_request_frame(rows),
        population_name="test-cme-final",
    )
    catalog = execution.catalog_rows
    return {
        label: catalog.filter(pl.col("label") == label).get_column("backtest_hash").to_list()
        for label in research_workflow.ALL_LABELS
    }


def test_final_selection_pool_spans_both_return_horizons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(tmp_path)
    by_label = _labelled_execution(study, monkeypatch)

    def fake_pool(_study, *, label):
        from case_studies.research import CandidateSet, Result

        return CandidateSet.create(
            _study,
            f"test-final-validation-{label}",
            [Result.open(_study, value) for value in by_label[label]],
        )

    monkeypatch.setattr(research_workflow, "final_validation_candidate_set", fake_pool)

    selection = research_workflow.final_selection_candidate_set(study)

    expected = {value for hashes in by_label.values() for value in hashes}
    assert set(selection.members) == expected
    assert len(selection.members) == len(expected) == 2
    assert selection.comparison_contract["comparable_fields"] == sorted(
        research_workflow.HORIZON_DEPENDENT_PROTOCOL_FIELDS
    )

    catalog = research_workflow.selection_catalog(study, selection)
    assert catalog.height == 2
    assert set(catalog.get_column("label")) == set(research_workflow.ALL_LABELS)
    assert catalog.get_column("sharpe").to_list() == sorted(
        catalog.get_column("sharpe").to_list(), reverse=True
    )
    assert selection.best_validation_sharpe().hash == catalog.item(0, "backtest_hash")


def test_selection_catalog_rejects_a_candidate_the_catalog_does_not_describe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from case_studies.research import CandidateSet

    study = _study(tmp_path)
    by_label = _labelled_execution(study, monkeypatch)
    selection = SimpleNamespace(
        members=(*(value for hashes in by_label.values() for value in hashes), "not-a-backtest")
    )

    with pytest.raises(ValueError, match="does not describe every candidate"):
        research_workflow.selection_catalog(study, cast(CandidateSet, selection))


def test_holdout_evidence_is_empty_until_the_lifecycle_is_locked(tmp_path: Path) -> None:
    study = _study(tmp_path)

    assert research_workflow.holdout_evidence(study).is_empty()


def test_holdout_evidence_reports_the_lock_and_its_single_evaluation(tmp_path: Path) -> None:
    study = _study(tmp_path)
    lock_record = {
        "candidate_set_hash": "set-1",
        "label": "fwd_ret_5d",
        "checkpoint_kind": "epoch",
        "checkpoint_value": 20,
        "validation_backtest_hash": "bt-validation",
    }
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        db.execute(
            "INSERT INTO research_locks (lock_hash, lock_json, state, created_at) VALUES (?,?,?,?)",
            ("lock-1", json.dumps(lock_record), "LOCKED", "2026-08-15T00:00:00Z"),
        )

    pending = research_workflow.holdout_evidence(study)
    assert pending.height == 1
    assert pending.item(0, "state") == "LOCKED"
    assert pending.item(0, "label") == "fwd_ret_5d"
    assert pending.item(0, "checkpoint_value") == 20
    assert pending.item(0, "holdout_backtest_hash") is None

    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        db.execute(
            "INSERT INTO holdout_evaluations (lock_hash, holdout_training_hash, "
            "holdout_prediction_hash, holdout_backtest_hash, evaluated_at) VALUES (?,?,?,?,?)",
            ("lock-1", "tr-holdout", "pr-holdout", "bt-holdout", "2026-08-15T01:00:00Z"),
        )
        db.execute(
            "UPDATE research_locks SET state = ? WHERE lock_hash = ?",
            ("HOLDOUT_EVALUATED", "lock-1"),
        )

    evaluated = research_workflow.holdout_evidence(study)
    assert evaluated.height == 1
    assert evaluated.item(0, "state") == "HOLDOUT_EVALUATED"
    assert evaluated.item(0, "holdout_backtest_hash") == "bt-holdout"
    assert evaluated.item(0, "evaluated_at") == "2026-08-15T01:00:00Z"


def test_official_model_catalog_forwards_the_population_it_supersedes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The supersedes value must reach the snapshot, not stop at the boundary.

    It is inside the hashed snapshot, so a run that drops it silently registers a
    different population than the one the operator asked for - and the registry would
    still look consistent afterwards.
    """
    seen: dict[str, object] = {}

    def _create(study, *, name, member_kind, members, supersedes=None):  # noqa: ANN001
        seen.update(name=name, member_kind=member_kind, supersedes=supersedes)
        return SimpleNamespace(require_complete=lambda: None)

    resolved = (
        cast(
            "object",
            SimpleNamespace(spec={"execution_tier": "canonical"}),
        ),
    )
    monkeypatch.setattr(research_workflow, "OfficialPopulation", SimpleNamespace(create=_create))
    monkeypatch.setattr(research_workflow, "expected_prediction_hashes", lambda _requests: ("h1",))
    monkeypatch.setattr(
        research_workflow,
        "run_resolved_model_requests",
        lambda _study, _resolved: SimpleNamespace(
            runs=[SimpleNamespace(predictions=[SimpleNamespace(hash="h1")])]
        ),
    )

    research_workflow.run_official_model_catalog(
        cast("object", SimpleNamespace()),
        pl.DataFrame(),
        population_name="cme_futures-pca-validation-v1",
        resolved_requests=resolved,
        supersedes="2d252634bffb",
    )

    assert seen["supersedes"] == "2d252634bffb"


def test_official_model_catalog_defaults_to_superseding_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A first population must pass None: population.py rejects a supersedes it cannot match."""
    seen: dict[str, object] = {}

    def _create(study, *, name, member_kind, members, supersedes=None):  # noqa: ANN001
        seen["supersedes"] = supersedes
        return SimpleNamespace(require_complete=lambda: None)

    monkeypatch.setattr(research_workflow, "OfficialPopulation", SimpleNamespace(create=_create))
    monkeypatch.setattr(research_workflow, "expected_prediction_hashes", lambda _requests: ("h1",))
    monkeypatch.setattr(
        research_workflow,
        "run_resolved_model_requests",
        lambda _study, _resolved: SimpleNamespace(
            runs=[SimpleNamespace(predictions=[SimpleNamespace(hash="h1")])]
        ),
    )

    research_workflow.run_official_model_catalog(
        cast("object", SimpleNamespace()),
        pl.DataFrame(),
        population_name="cme_futures-pca-validation-v1",
        resolved_requests=(SimpleNamespace(spec={"execution_tier": "canonical"}),),
    )

    assert seen["supersedes"] is None


def _multifold_frames(n_days: int = 60) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Two folds over one product pair, so a fold BOUNDARY exists.

    The single-fold fixture cannot exercise the state-transition policy at all:
    `fold_boundary="liquidate"` only emits a transition where one fold ends and the
    next begins, so with `fold_id` constant the policy is a no-op and the typed and
    direct paths are the same computation. A real run has five folds.
    """
    dates = pl.date_range(
        pl.date(2024, 1, 2), pl.date(2024, 1, 2) + timedelta(days=n_days - 1), eager=True
    )
    products = ("ES", "NQ")
    rows = [(d, p, i) for i, d in enumerate(dates) for p in products]
    base = {"ES": 4_800.0, "NQ": 16_800.0}
    prices = pl.DataFrame(
        {
            "product": [p for _, p, _ in rows],
            "timestamp": [d for d, _, _ in rows],
            "open": [base[p] + i * 10 for _, p, i in rows],
            "high": [base[p] + i * 10 + 20 for _, p, i in rows],
            "low": [base[p] + i * 10 - 20 for _, p, i in rows],
            "close": [base[p] + i * 10 + 10 for _, p, i in rows],
            "volume": [1_000] * len(rows),
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))
    half = len(dates) // 2
    predictions = pl.DataFrame(
        {
            "symbol": [p for _, p, _ in rows],
            "timestamp": [d for d, _, _ in rows],
            "fold_id": [0 if i < half else 1 for _, _, i in rows],
            "y_true": [0.01 if p == "ES" else -0.01 for _, p, _ in rows],
            "y_score": [0.02 if p == "ES" else -0.02 for _, p, _ in rows],
        }
    )
    return predictions, prices


def test_product_decision_equivalence_holds_across_a_fold_boundary(tmp_path: Path) -> None:
    """The typed path must equal the direct path when a fold boundary is crossed.

    Reproduces a real-data failure the single-fold equivalence test cannot reach:
    against the produced canonical predictions the typed path raises

        ValueError: state-transition sequencing requires same-bar engine execution

    because publish_product_weights declares fold_boundary="liquidate" while
    cme_futures executes weekly_friday_close -> monday_open. A liquidation at the
    boundary cannot be honoured under a delayed fill, and the engine refuses rather
    than leaving the position live across a boundary the spec says was flat.
    """
    study = _study(tmp_path)
    frame, prices = _multifold_frames()
    training = study.results.register_training(_training_spec(label="fwd_ret_5d"))
    prediction = study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold_id"),
    )
    assert frame.get_column("fold_id").n_unique() > 1, "fixture must cross a fold boundary"

    signal = {"method": "equal_weight_top_k", "top_k": 1}
    allocation = {"method": "inverse_vol", "vol_window": 2}
    decision = publish_product_weights(
        prediction, prices=prices, signal=signal, allocation=allocation
    )
    # The 7d cadence thins the weight grid, so a short fixture can drop the boundary
    # entirely: at 8 days the weights carry fold 1 only and the policy is a no-op again.
    assert decision.load().get_column("fold").n_unique() > 1, (
        "the WEIGHTS must cross the boundary, not merely the predictions"
    )
    resolved_signal = decision.spec["parameters"]["signal"]
    resolved_allocation = decision.spec["parameters"]["allocation"]

    direct = study.strategy(
        prediction=prediction, signal=resolved_signal, allocation=resolved_allocation
    ).run(prices=prices)
    typed = study.strategy(
        prediction=prediction,
        signal=resolved_signal,
        allocation=resolved_allocation,
        decision=decision,
    ).run(prices=prices)
    assert _returns(typed).equals(_returns(direct))


def test_all_labels_is_read_from_the_sweep_declaration_and_not_restated() -> None:
    """`ALL_LABELS` must be `setup.yaml`'s sweep, in its order, not a copy of it.

    Every CME modelling and strategy notebook reaches its labels through this one
    constant. While it was a literal tuple beside the declaration, nothing compared the
    two: adding a variant to `labels.variants` would have left all twelve notebooks
    fitting the old set and publishing populations one label short, with each notebook's
    own completeness check passing because it is scoped to what that notebook requested.

    The test reads the YAML itself rather than calling the same helper, so reverting the
    constant to a literal fails here as soon as the declaration moves.
    """
    declared = (
        yaml.safe_load(
            (REPO_ROOT / "case_studies" / "cme_futures" / "config" / "setup.yaml").read_text()
        )
        or {}
    ).get("labels") or {}
    primary = str(declared["primary"])
    variants = [str(name) for name in (declared.get("variants") or []) if str(name) != primary]
    assert (primary, *variants) == research_workflow.ALL_LABELS


def test_every_declared_label_has_a_training_menu_for_every_family_requested() -> None:
    """A label in the sweep with no menu entry for a family fits nothing for that family.

    `model_request_catalog` asks `load_configs` per label, so a family the menu omits on
    one label yields a request catalog covering the other, and the population it publishes
    is short by exactly the missing label. The two horizons declare identical family
    menus today; this is what fails if one of them stops.
    """
    menus = {
        label: yaml.safe_load(
            (
                REPO_ROOT / "case_studies" / "cme_futures" / "config" / "training" / f"{label}.yaml"
            ).read_text()
        )
        or {}
        for label in research_workflow.ALL_LABELS
    }
    families = {
        label: {name for name, configs in menu.items() if configs} for label, menu in menus.items()
    }
    first, *rest = research_workflow.ALL_LABELS
    for label in rest:
        assert families[label] == families[first], (
            f"{label} and {first} declare different families: "
            f"{sorted(families[label] ^ families[first])}"
        )


def test_every_declared_model_population_is_published_by_a_notebook() -> None:
    """The contract must name only populations a notebook actually registers.

    `12_model_analysis` and `13_backtest` resolve every entry of
    `MODEL_POPULATION_NAMES` with `OfficialPopulation.one`, which raises on a name that
    does not exist. A name declared here and written nowhere therefore stops those two
    notebooks on a clean registry, and nothing catches it until they run - which is how
    the linear and GBM entries drifted from their producers three times over.

    Reading the literals is the point: the name is the whole contract between the
    producer and the consumer, and it is the literal that has drifted every time.
    """
    notebooks = sorted(
        (Path(__file__).parent.parent / "case_studies" / "cme_futures").glob("[0-9]*.py")
    )
    assert notebooks, "no CME notebooks found"
    published = {
        name
        for notebook in notebooks
        for name in re.findall(
            # `population_name="..."` in a call, and `population_name = POPULATION_NAME or "..."`
            # at module level, which is how 06 and 07 leave the name overridable.
            r'population_name\s*=\s*(?:POPULATION_NAME\s+or\s+)?"([^"]+)"',
            notebook.read_text(),
        )
    }
    declared = set(research_workflow.MODEL_POPULATION_NAMES)
    assert not declared - published, (
        f"MODEL_POPULATION_NAMES declares populations no notebook publishes: "
        f"{sorted(declared - published)}"
    )


def _population(study: Study, name: str, member: str, supersedes: str | None = None):
    return OfficialPopulation.create(
        study, name=name, member_kind="prediction", members=[member], supersedes=supersedes
    )


def test_declared_supersedes_is_dropped_where_the_population_does_not_exist(tmp_path: Path):
    """06_linear and 07_gbm declare a real supersedes hash; a first run must not pass it.

    The hash is a literal in the parameter cell so that running the committed .py
    reproduces the population on record. `run_log/` is not in the repository, so a
    reader's first canonical run finds an empty registry and is the first version of
    its population - and `OfficialPopulation.create` refuses a first version that
    supersedes anything, before any fit happens.
    """
    study = _study(tmp_path)
    declared = "8337482ecb59"
    name = "cme_futures-linear-validation-v1"

    resolved = supersedes_for_run(
        study, population_name=name, declared=declared, execution_tier="canonical"
    )
    assert resolved is None

    # The resolution is what makes the reader's run possible: the declared value is
    # refused outright, and what the notebook actually passes is accepted.
    prediction = _prediction(study)
    with pytest.raises(ValueError, match="first population version cannot supersede"):
        _population(study, name, prediction, supersedes=declared)
    assert _population(study, name, prediction, supersedes=resolved).supersedes is None


def test_declared_supersedes_is_passed_through_once_a_generation_exists(tmp_path: Path):
    """Where the name already has a snapshot, the declared hash must reach the registry."""
    study = _study(tmp_path)
    name = "cme_futures-linear-validation-v1"
    first = _population(study, name, _prediction(study))

    resolved = supersedes_for_run(
        study, population_name=name, declared=first.hash, execution_tier="canonical"
    )
    assert resolved == first.hash

    second = _population(study, name, _current_prediction(study, alpha=2.0), supersedes=resolved)
    assert second.supersedes == first.hash
    assert OfficialPopulation.one(study, name=name).hash == second.hash


def test_a_reader_can_run_the_same_notebook_twice_on_a_clean_clone(tmp_path: Path):
    """The sequence a reader actually performs, which the two tests above miss.

    Each of those seeds the registry to the state it wants and resolves once. A reader
    runs the committed notebook twice against one registry, publishing the same members
    both times, and the second run is where a rule keyed on "does this name have any
    generation" fails: run 1 writes a snapshot whose own supersedes is None, run 2 sees
    a generation and offers the declared hash, and the registry refuses because the hash
    it requires is what run 1 wrote, not what the parameter cell names.

    Reported by the fx_pairs session against the same construction in their notebook.
    """
    study = _study(tmp_path)
    name = "cme_futures-linear-validation-v1"
    declared = "8337482ecb59"
    members = _prediction(study)

    first = _population(
        study,
        name,
        members,
        supersedes=supersedes_for_run(
            study, population_name=name, declared=declared, execution_tier="canonical"
        ),
    )
    assert first.supersedes is None

    # Re-running the notebook unchanged recomputes the same members, so the second run
    # must resolve to the same snapshot rather than trying to supersede one.
    second = _population(
        study,
        name,
        members,
        supersedes=supersedes_for_run(
            study, population_name=name, declared=declared, execution_tier="canonical"
        ),
    )
    assert second.hash == first.hash
    assert OfficialPopulation.one(study, name=name).hash == first.hash


def test_a_disagreeing_supersedes_is_left_for_the_registry_to_refuse(tmp_path: Path):
    """The resolution must not second-guess a wrong hash into a right one.

    A hash that names neither the generation in force nor the one it supersedes describes
    no state this registry is in, so it is withheld. What must not happen is the run
    proceeding: the changed population still has to be refused, by the registry, with the
    message that names the snapshot required.
    """
    study = _study(tmp_path)
    name = "cme_futures-linear-validation-v1"
    first = _population(study, name, _prediction(study))
    stale = "0" * 12

    resolved = supersedes_for_run(
        study, population_name=name, declared=stale, execution_tier="canonical"
    )
    assert resolved is None
    with pytest.raises(ValueError, match=f"must explicitly supersedes {first.hash}"):
        _population(study, name, _current_prediction(study, alpha=2.0), supersedes=resolved)


def test_an_author_declaring_the_generation_in_force_publishes_the_next_one(tmp_path: Path):
    """Superseding is how a refit under a corrected parameter reaches the record.

    `cme_futures-gbm-validation-v1` holds three generations, each written by a run that
    declared the tip it replaced. Withholding the hash wherever the tip does not already
    supersede it would fix a reader's second run by making that publication impossible,
    so the tip's own hash has to resolve as well.
    """
    study = _study(tmp_path)
    name = "cme_futures-gbm-validation-v1"
    first = _population(study, name, _prediction(study))

    resolved = supersedes_for_run(
        study, population_name=name, declared=first.hash, execution_tier="canonical"
    )
    assert resolved == first.hash

    second = _population(study, name, _current_prediction(study, alpha=2.0), supersedes=resolved)
    assert second.supersedes == first.hash
    assert OfficialPopulation.one(study, name=name).hash == second.hash

    # And the run that reproduces that tip declares the same hash, now matched by the
    # other condition, so the committed notebook keeps resolving to the published record.
    assert (
        supersedes_for_run(
            study, population_name=name, declared=first.hash, execution_tier="canonical"
        )
        == first.hash
    )


def test_a_narrowed_run_under_its_own_population_name_drops_the_builtin_hash(tmp_path: Path):
    """The notebooks document publishing a narrowed run under a caller-chosen name.

    That name is its own first generation whatever the built-in default says, so the
    default must not follow it there.
    """
    study = _study(tmp_path)
    _population(study, "cme_futures-linear-validation-v1", _prediction(study))

    assert (
        supersedes_for_run(
            study,
            population_name="my-linear-v1",
            declared="8337482ecb59",
            execution_tier="canonical",
        )
        is None
    )


def test_preview_never_carries_a_supersedes_hash(tmp_path: Path):
    """A preview population is discarded with its workspace and has no lineage to extend."""
    study = _study(tmp_path)
    name = "cme_futures-linear-validation-v1"
    first = _population(study, name, _prediction(study))
    assert (
        supersedes_for_run(
            study, population_name=name, declared=first.hash, execution_tier="preview"
        )
        is None
    )


def test_sdf_checkpoints_chosen_on_validation_never_reach_the_selection_pool() -> None:
    """The four fitted-state aliases are excluded from what 13_backtest can select.

    The stochastic discount factor keeps four checkpoints the library picked by reading the
    validation split: best validation loss and best validation Sharpe, in each of its two
    training phases. Selection downstream ranks on validation Sharpe, so publishing them as
    ordinary members would let a checkpoint win the pool on the same data that chose it.

    The exclusion is asserted against the library's named checkpoints, not against the sign
    of the published value: `_sdf_checkpoint_label` packs both phases onto one integer axis,
    so the signs are an artifact of that packing rather than a contract.
    """
    pytest.importorskip("ml4t.models.stochastic_discount_factor.model")

    from ml4t.models.stochastic_discount_factor.model import (
        VAL_BEST_LOSS_CONDITIONAL,
        VAL_BEST_LOSS_UNCONDITIONAL,
        VAL_BEST_SHARPE_CONDITIONAL,
        VAL_BEST_SHARPE_UNCONDITIONAL,
    )

    from case_studies.utils.latent_factors.cv import _expected_latent_checkpoints
    from case_studies.utils.latent_factors.library_bridge import _sdf_checkpoint_label

    setup = yaml.safe_load(
        (Path(research_workflow.__file__).parent / "config" / "setup.yaml").read_text()
    )
    model_kwargs = setup["modeling"]["latent_factors"]["model_kwargs"]["sdf"]
    n_epochs_unc = int(model_kwargs.get("n_epochs_unc", 256))
    chosen_on_validation = {
        _sdf_checkpoint_label(checkpoint, n_epochs_unc=n_epochs_unc)
        for checkpoint in (
            VAL_BEST_LOSS_UNCONDITIONAL,
            VAL_BEST_SHARPE_UNCONDITIONAL,
            VAL_BEST_LOSS_CONDITIONAL,
            VAL_BEST_SHARPE_CONDITIONAL,
        )
    }
    n_epochs = max(int(value) for value in model_kwargs["checkpoint_epochs"])
    published = set(
        _expected_latent_checkpoints("sdf", n_epochs=n_epochs, model_kwargs=model_kwargs)
    )
    fitted = set(
        _expected_latent_checkpoints(
            "sdf",
            n_epochs=n_epochs,
            model_kwargs=model_kwargs,
            include_internal_aliases=True,
        )
    )

    assert published
    assert not published & chosen_on_validation
    assert fitted - published == chosen_on_validation
