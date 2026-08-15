from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

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
    ResolvedModelRequest,
    StateTransitionPolicy,
    Study,
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


def test_model_request_catalog_rejects_each_unknown_requested_config() -> None:
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
        name_prefix="test-cme-signal",
    )
    assert set(candidate_sets) == {"fwd_ret_21d"}
    assert set(candidate_sets["fwd_ret_21d"].members) == set(
        execution.catalog_rows.get_column("backtest_hash")
    )
