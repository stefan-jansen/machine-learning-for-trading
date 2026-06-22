"""Drift-detector tests for the setup.yaml-driven Ch16-19 sweep loader.

Pins the contract between ``case_studies/{cs}/config/setup.yaml::backtest.sweep``
and the helpers in ``case_studies.utils.sweep_config``:

1. **Loader shape** — ``load_sweep`` and the ``*_for`` / ``get_*`` helpers
   return the expected types and values for migrated case studies.
2. **Registry reconciliation** — the declared sweep covers the rank-1
   ``(method, top_k)`` for every label in ``labels.{primary, variants}``.
3. **Quarantine policy** — V3/V4 deprecated classes (``score_weighted_top_k``
   on the signal stage, ``cross_sectional_percentile``) must not appear in
   the declared sweep.

All 9 case studies have shipped ``backtest.sweep``; ``MIGRATED_CASES`` is
now the full set.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import yaml

from case_studies.utils.sweep_config import (
    get_allocators,
    get_cost_grid_bps,
    get_entry_schemes_for,
    get_portfolio_risk_controls,
    get_position_risk_controls,
    get_top_k_values_for,
    load_sweep,
)
from utils import CASE_STUDIES_DIR

# All case studies ship a ``backtest.sweep`` block in setup.yaml.
MIGRATED_CASES: tuple[str, ...] = (
    "us_firm_characteristics",
    "etfs",
    "fx_pairs",
    "cme_futures",
    "nasdaq100_microstructure",
    "us_equities_panel",
    "sp500_equity_option_analytics",
    "crypto_perps_funding",
    "sp500_options",
)

# Signal-stage methods that must never appear in a declared sweep. These
# correspond to the V3/V4 quarantine list: ``score_weighted_top_k`` is an
# allocator (Ch17), not a signal scheme; ``cross_sectional_percentile``
# was retired during V3 cleanup.
QUARANTINED_SIGNAL_METHODS: frozenset[str] = frozenset(
    {"score_weighted_top_k", "cross_sectional_percentile"}
)


# ---------------------------------------------------------------------------
# Loader contract — us_firm_characteristics (the first migrated case study)
# ---------------------------------------------------------------------------


class TestUsFirmLoader:
    """Pin the loader output for us_firm_characteristics."""

    CS = "us_firm_characteristics"

    def test_load_sweep_returns_expected_keys(self):
        sweep = load_sweep(self.CS)
        assert set(sweep.keys()) >= {
            "top_k_grid",
            "allocators",
            "cost_grid_bps",
            "risk_controls",
        }

    @pytest.mark.parametrize("label", ["fwd_ret_1m", "fwd_ret_1m_win", "fwd_class_1m"])
    def test_top_k_grid_per_label(self, label):
        assert get_top_k_values_for(self.CS, label, n_assets=2500) == [5, 10, 20, 50]

    @pytest.mark.parametrize("label", ["fwd_ret_1m", "fwd_ret_1m_win", "fwd_class_1m"])
    def test_entry_schemes_per_label(self, label):
        schemes = get_entry_schemes_for(self.CS, label, n_assets=2500, long_short=True)
        # Exactly the four equal_weight_top_k schemes declared in setup.yaml.
        assert [s["method"] for s in schemes] == ["equal_weight_top_k"] * 4
        assert [s["top_k"] for s in schemes] == [5, 10, 20, 50]
        assert all(s["long_short"] is True for s in schemes)

    def test_allocators_strip_name(self):
        allocators = get_allocators(self.CS)
        methods = [a["method"] for a in allocators]
        # us_firm is a returns-only firm-characteristics panel with no per-symbol
        # price series, so the moment-based allocators (inverse_vol, risk_parity,
        # hrp, mvo_ledoit_wolf) are intentionally excluded — only the
        # lookback-free allocators are declared (see setup.yaml allocators block).
        assert methods == [
            "equal_weight",
            "score_weighted",
            "conformal_weighted",
        ]
        # No allocator dict should carry the human-readable ``name`` key —
        # the spec hash is computed from the shape that the dispatcher sees.
        assert all("name" not in a for a in allocators)

    def test_cost_grid_bps(self):
        assert get_cost_grid_bps(self.CS) == [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]

    def test_risk_control_counts(self):
        # Position grid: stop_loss (4) + trailing_stop (7) + time_exit (3) = 14
        assert len(get_position_risk_controls(self.CS)) == 14
        # Portfolio-level overlays were removed from all setup.yaml (2026-05-17);
        # only position-level controls are retained.
        assert len(get_portfolio_risk_controls(self.CS)) == 0


# ---------------------------------------------------------------------------
# Quarantine policy — V3/V4 retired classes must not appear in any declared
# sweep
# ---------------------------------------------------------------------------


class TestQuarantinePolicy:
    """Declared sweep must not include V3/V4 retired selection classes."""

    @pytest.mark.parametrize("case_study", MIGRATED_CASES)
    def test_no_score_weighted_top_k_in_signal_sweep(self, case_study):
        sweep = load_sweep(case_study)
        # ``score_weighted_top_k`` should never appear as a top_k_grid /
        # percentile_grid / quantile_grid axis — it belongs in
        # ``allocators`` only.
        # The synthesized entry schemes are the SUT.
        for label in (
            (sweep.get("top_k_grid") or {}).keys()
            | (sweep.get("percentile_grid") or {}).keys()
            | (sweep.get("quantile_grid") or {}).keys()
        ):
            schemes = get_entry_schemes_for(case_study, label, n_assets=10_000, long_short=False)
            methods = {s["method"] for s in schemes}
            assert methods.isdisjoint(QUARANTINED_SIGNAL_METHODS), (
                f"{case_study}/{label}: quarantined signal method appeared: "
                f"{methods & QUARANTINED_SIGNAL_METHODS}"
            )


# ---------------------------------------------------------------------------
# Registry reconciliation — declared sweep covers rank-1 per (CS, label)
# ---------------------------------------------------------------------------


def _registry_path(case_study: str) -> Path:
    return CASE_STUDIES_DIR / case_study / "run_log" / "registry.db"


def _labels_for(case_study: str) -> list[str]:
    setup = yaml.safe_load((CASE_STUDIES_DIR / case_study / "config" / "setup.yaml").read_text())
    labels_block = setup.get("labels") or {}
    primary = labels_block.get("primary")
    variants = labels_block.get("variants") or []
    return [primary, *variants] if primary else list(variants)


class TestRegistryReconciliation:
    """For every label in setup.yaml::labels.{primary,variants}, the rank-1
    signal-stage row in ``backtest_runs`` must use a ``(method, top_k)`` that
    is in the declared sweep.

    Skips a case study if its registry has no signal-stage rows (e.g.,
    immediately after a registry cleanup), since there is nothing to
    reconcile against yet. As each case study completes its Ch16-19 wrap-up,
    its registry rank-1 should reappear and this test should pass.
    """

    @pytest.mark.parametrize("case_study", MIGRATED_CASES)
    def test_rank1_signal_method_in_declared_sweep(self, case_study):
        reg_path = _registry_path(case_study)
        if not reg_path.exists():
            pytest.skip(f"{case_study}: registry.db not present")

        sweep = load_sweep(case_study)
        top_k_by_label = sweep.get("top_k_grid") or {}
        qnt_by_label = sweep.get("quantile_grid") or {}
        pct_by_label = sweep.get("percentile_grid") or {}

        labels = _labels_for(case_study)
        with sqlite3.connect(reg_path) as conn:
            cur = conn.cursor()
            for label in labels:
                row = cur.execute(
                    """
                    SELECT json_extract(r.spec_json, '$.strategy.signal.method'),
                           json_extract(r.spec_json, '$.strategy.signal.top_k'),
                           json_extract(r.spec_json, '$.strategy.signal.n_quantiles'),
                           json_extract(r.spec_json, '$.strategy.signal.max_slots'),
                           bm.sharpe
                    FROM backtest_metrics bm
                    JOIN backtest_runs r ON r.backtest_hash = bm.backtest_hash
                    JOIN prediction_sets p ON p.prediction_hash = r.prediction_hash
                    JOIN training_runs t ON t.training_hash = p.training_hash
                    WHERE t.label = ? AND p.split = 'validation' AND r.stage = 'signal'
                    ORDER BY bm.sharpe DESC LIMIT 1
                    """,
                    (label,),
                ).fetchone()
                if row is None:
                    pytest.skip(f"{case_study}/{label}: no signal-stage rows in registry")

                method, top_k, n_quantiles, max_slots, _sharpe = row
                if method == "equal_weight_top_k":
                    declared_ks = list(top_k_by_label.get(label, []))
                    assert top_k in declared_ks, (
                        f"{case_study}/{label}: registry rank-1 top_k={top_k} "
                        f"not in declared top_k_grid={declared_ks}"
                    )
                elif method == "slot_persistent_signal_exit":
                    # Slot-mechanism signal (nasdaq100 v4 microstructure sweep).
                    # Slots ARE the allocation, so the swept parameter is
                    # ``max_slots`` (declared under the ``signal_nasdaq100``
                    # block), not top_k / n_quantiles.
                    declared_slots = list(
                        (sweep.get("signal_nasdaq100") or {}).get("max_slots", [])
                    )
                    assert max_slots in declared_slots, (
                        f"{case_study}/{label}: registry rank-1 max_slots={max_slots} "
                        f"not in declared signal_nasdaq100.max_slots={declared_slots}"
                    )
                elif method in ("quintile_long_short", "decile_long_short"):
                    declared_qs = list(qnt_by_label.get(label, []))
                    assert n_quantiles in declared_qs, (
                        f"{case_study}/{label}: registry rank-1 n_quantiles="
                        f"{n_quantiles} not in declared quantile_grid={declared_qs}"
                    )
                elif method in QUARANTINED_SIGNAL_METHODS:
                    # The registry still has V3/V4 debris — Ch16-19 sweep
                    # cleanup hasn't run yet for this (case_study, label).
                    # Skip rather than fail; once cleanup runs, the rank-1
                    # method will be canonical and the assertion above takes
                    # over.
                    pytest.skip(
                        f"{case_study}/{label}: rank-1 is {method!r} (V3/V4 "
                        f"quarantine class). Registry cleanup pending; "
                        f"re-rank after task-6/task-7 land."
                    )
                else:
                    pytest.fail(
                        f"{case_study}/{label}: rank-1 method {method!r} is "
                        f"unrecognized by the seam test — extend the test or "
                        f"the loader."
                    )
