from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest
import yaml

from case_studies.research import CausalResult, LabelDefinition, Study
from case_studies.utils import causal
from case_studies.utils.registry.specs import training_hash_from_spec
from tests.test_research_workspace import _seed_release


@pytest.fixture(autouse=True)
def _restore_output_root():
    yield
    os.environ.pop("ML4T_OUTPUT_DIR", None)
    from case_studies.research import workspace

    workspace._ACTIVE_OUTPUT_ROOT = None
    workspace._clear_root_sensitive_caches()


def _causal_fixture(
    tmp_path,
    monkeypatch,
    entity: str = "symbol",
    label_buffer: str = "8H",
    label_horizon: str | None = None,
    treatment_window: int | None = 1,
):
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    setup_path = study.root / "config" / "setup.yaml"
    # `treatment` here is a per-row column carrying its own timestamp, so one bar is
    # the honest window. It is declared rather than omitted because a canonical run now
    # refuses an undeclared window, and every test below that is not about that refusal
    # would otherwise fail on it. Pass treatment_window=None to exercise the refusal.
    causal_block = "causal:\n  treatment: treatment\n  confounders: [confounder]\n"
    if treatment_window is not None:
        causal_block += f"  treatment_window: {treatment_window}\n"
    setup_path.write_text(setup_path.read_text() + causal_block)
    if label_horizon is not None:
        # Merged through a yaml round-trip rather than appended. `labels` already exists in
        # this setup, and a second top-level `labels:` key would replace it outright rather
        # than add to it, taking the buffers with it.
        setup = yaml.safe_load(setup_path.read_text())
        setup.setdefault("labels", {}).setdefault("horizons", {})["fwd_ret_8h"] = label_horizon
        setup_path.write_text(yaml.safe_dump(setup, sort_keys=False))
    rows = []
    for timestamp_index, timestamp in enumerate(
        pl.datetime_range(
            pl.datetime(2024, 1, 1),
            pl.datetime(2024, 1, 9, 16),
            interval="8h",
            eager=True,
            time_zone="UTC",
        )
    ):
        for symbol_index in range(6):
            rows.append(
                {
                    entity: f"S{symbol_index}",
                    "timestamp": timestamp,
                    "feature": float(symbol_index),
                    "treatment": float(symbol_index) + timestamp_index / 100,
                    "confounder": float(timestamp_index % 5),
                    "fwd_ret_8h": (symbol_index - 2.5) / 100 + timestamp_index / 10000,
                }
            )
    frame = pl.DataFrame(rows)
    label = study.labels.publish(
        LabelDefinition("fwd_ret_8h", "regression", label_horizon or label_buffer),
        frame.rename({entity: "symbol"}).select("symbol", "timestamp", "fwd_ret_8h"),
    )
    mds = SimpleNamespace(
        dataset=frame,
        feature_names=["feature", "treatment", "confounder"],
        label_col="fwd_ret_8h",
        label_buffer=label_buffer,
        date_col="timestamp",
        entity_cols=[entity],
        input_lineage={
            "artifacts": {"financial": {"sha256": "features-v1", "size": 1}},
            "fingerprint": "fixture-v1",
        },
    )
    monkeypatch.setattr("utils.modeling.load_modeling_dataset", lambda *args, **kwargs: mds)
    monkeypatch.setattr(
        "utils.modeling.load_configs",
        lambda *args, **kwargs: [
            {
                "config_name": "dml",
                "family": "causal_dml",
                "n_folds": 5,
                "n_placebo": 100,
                "params": {"max_depth": 3, "max_iter": 50},
                "seed": 42,
            }
        ],
    )
    return study, label, frame


def test_causal_request_resolves_timing_gap_and_preview_identity(tmp_path, monkeypatch) -> None:
    study, label, _frame = _causal_fixture(tmp_path, monkeypatch)

    resolved = study.causal(
        method="dml",
        label=label.name,
        execution_tier="preview",
        preview_reductions={
            "max_samples": 240,
            "max_symbols": 6,
            "n_folds": 2,
            "n_placebo": 10,
        },
    ).resolve()
    computation = resolved.spec["computation"]

    assert resolved.spec["identity_version"] == 3
    assert computation["estimand"]["outcome"] == "fwd_ret_8h"
    assert computation["estimand"]["treatment"] == "treatment"
    assert computation["estimand"]["treatment_observed_at"] == "decision_timestamp"
    assert computation["refutation"]["temporal_gap_policy"] == "reset"
    assert computation["refutation"]["observation_cadence"] == "0 days 08:00:00"
    assert computation["analysis_population"]["n_rows"] <= 240
    assert computation["model"]["nuisance_params"]["max_iter"] == 50
    assert computation["preview_reductions"]["n_folds"] == 2


def test_canonical_causal_uses_the_full_declared_population(tmp_path, monkeypatch) -> None:
    study, label, frame = _causal_fixture(tmp_path, monkeypatch)

    canonical = study.causal(method="dml", label=label.name).resolve()
    preview = study.causal(
        method="dml",
        label=label.name,
        execution_tier="preview",
        preview_reductions={
            "max_samples": 60,
            "max_symbols": 6,
            "n_folds": 2,
            "n_placebo": 10,
        },
    ).resolve()

    canonical_population = canonical.spec["computation"]["analysis_population"]
    preview_population = preview.spec["computation"]["analysis_population"]
    # "Full" means the declared panel less the buffer the endpoint cutoff removes, which at
    # an 8H buffer on 8-hour bars is one observation across the fixture's symbols. It does
    # not mean every row: asserting `frame.height` would encode this fixture's labels being
    # non-null through its final bar, which a forward return never is.
    trimmed = frame.get_column("symbol").n_unique()
    assert canonical_population["n_rows"] == frame.height - trimmed
    assert canonical_population["max_samples"] == 0
    assert "preview_reductions" not in canonical.spec["computation"]
    assert preview_population["n_rows"] == 60
    assert preview_population["max_samples"] == 60
    assert canonical.identity != preview.identity


def _set_holdout_start(setup_path, value: str) -> None:
    setup_path.write_text(
        re.sub(
            r"holdout_start: *'?[^'\n]+'?",
            f"holdout_start: '{value}'",
            setup_path.read_text(),
        )
    )


def test_canonical_causal_population_stops_before_the_holdout_endpoint(
    tmp_path, monkeypatch
) -> None:
    """The DML sample must end one outcome horizon before the holdout, and fail closed."""
    study, label, frame = _causal_fixture(tmp_path, monkeypatch)
    setup_path = study.root / "config" / "setup.yaml"

    # The seeded holdout_start (2024-01-11) is past the fixture's last row, so the seal never
    # binds there. Move it inside the fixture's own span to exercise the cutoff itself.
    _set_holdout_start(setup_path, "2024-01-05")
    resolved = study.causal(method="dml", label=label.name).resolve()
    computation = resolved.spec["computation"]

    cutoff = datetime(2024, 1, 5, tzinfo=UTC) - timedelta(hours=8)
    expected = frame.filter(pl.col("timestamp") < cutoff)
    assert computation["estimand"]["holdout_endpoint_cutoff"] == cutoff.isoformat()
    assert computation["analysis_population"]["n_rows"] == expected.height
    # The seal must actually remove rows here, or this test would pass on a no-op filter.
    assert 0 < expected.height < frame.height

    # An empty pre-holdout frame fails closed rather than estimating on whatever is left.
    _set_holdout_start(setup_path, "2023-12-01")
    with pytest.raises(ValueError, match="empty pre-holdout analysis frame"):
        study.causal(method="dml", label=label.name).resolve()


def test_causal_pins_the_thread_pool_and_records_it_in_identity(tmp_path, monkeypatch) -> None:
    """The placebo loop is only bit-reproducible at a fixed thread count.

    `os.environ.setdefault("OMP_NUM_THREADS", "1")` at import does not achieve that: every
    notebook imports sklearn through case_studies.research first, so the pools are already
    built. Measured that way the openmp pool stays at 16. The limit therefore has to be
    applied at run time, and recorded, so two runs at different counts cannot share a hash.
    """
    study, label, _frame = _causal_fixture(tmp_path, monkeypatch)
    resolved = study.causal(method="dml", label=label.name).resolve()

    numerics = resolved.spec["computation"]["numerics"]
    assert numerics["thread_limit"] == causal.DML_THREAD_LIMIT
    assert numerics["deterministic_reduction"] is True

    # The limit must be identity-bearing: a different count is a different result, not the
    # same result computed differently.
    other = deepcopy(resolved.spec)
    other["computation"]["numerics"]["thread_limit"] = causal.DML_THREAD_LIMIT + 1
    assert training_hash_from_spec(other) != training_hash_from_spec(resolved.spec)

    # The resolver's job is to pass the recorded limit down; binding the pool is
    # run_dml_analysis's, so that direct callers get it too - see
    # test_run_dml_analysis_pins_the_pool_for_direct_callers.
    passed: list[int] = []

    def capture(*args, **kwargs):
        passed.append(kwargs["thread_limit"])
        return {
            "dml_result": {"theta": 0.02, "se_hac": 0.01, "n_obs": 120},
            "p_value_hac": 0.04,
            "naive_effect": 0.03,
            "confounding_bias_pct": 50.0,
            "refutation": {"empirical_p": 0.1},
            "started_at": "2026-08-15T00:00:00+00:00",
            "elapsed_s": 1.0,
        }

    monkeypatch.setattr(causal, "run_dml_analysis", capture)
    study.causal(method="dml", label=label.name).run()

    assert passed == [causal.DML_THREAD_LIMIT], (
        f"resolver passed thread_limit={passed}, not [{causal.DML_THREAD_LIMIT}]"
    )


def test_manual_dml_timeseries_pins_the_pool_for_every_caller(monkeypatch) -> None:
    """The pin lives where the nuisance models are fitted, so no caller can bypass it.

    Six case-study DML stages call run_dml_analysis directly, and the chapter-15 notebooks
    15_causal_estimation/03_econml_dml.py and 04_dml_crypto_regime.py call
    manual_dml_timeseries themselves at nine sites, with model_y/model_t unset so they fit the
    default HistGradientBoostingRegressor. All of them import sklearn before this module, so
    the OMP_NUM_THREADS setdefault is inert for every one - while
    cme_futures/12_model_analysis.py:1190 and sp500_options/11_model_analysis.py:992 tell the
    reader the nuisance models are pinned.
    """
    import threadpoolctl

    observed: list[int] = []
    original = causal._walk_forward_indices

    def record(*args, **kwargs):
        observed.extend(
            info["num_threads"]
            for info in threadpoolctl.threadpool_info()
            if info["user_api"] in {"openmp", "blas"}
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(causal, "_walk_forward_indices", record)

    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=(n, 1))
    t = x[:, 0] * 0.5 + rng.normal(scale=0.1, size=n)
    y = t * 0.3 + rng.normal(scale=0.1, size=n)
    causal.manual_dml_timeseries(y, t, x, n_folds=2, embargo=1)

    assert observed, "no thread pool was observed inside manual_dml_timeseries"
    assert set(observed) == {causal.DML_THREAD_LIMIT}, (
        f"manual_dml_timeseries ran with pools at {sorted(set(observed))}, "
        f"not {causal.DML_THREAD_LIMIT}"
    )


def test_run_dml_analysis_pins_the_naive_ols_too(monkeypatch) -> None:
    """The naive-OLS comparison runs after manual_dml_timeseries returns.

    np.linalg.lstsq on a tall design reaches threaded BLAS, so naive_effect - and
    confounding_bias_pct, which is a difference between it and the pinned theta - would vary
    with the ambient pool while the resolved spec records deterministic_reduction: True.
    Observed with manual_dml_timeseries stubbed, so only the outer pin can be in effect.
    """
    import threadpoolctl

    observed: list[int] = []
    n = 200

    def stub(Y, T, X, **kwargs):
        observed.extend(
            info["num_threads"]
            for info in threadpoolctl.threadpool_info()
            if info["user_api"] in {"openmp", "blas"}
        )
        rng = np.random.default_rng(1)
        return {
            "Y_res": rng.normal(size=n),
            "T_res": rng.normal(size=n),
            "theta": 0.02,
            "se_hac": 0.01,
            "n_obs": n,
            "t_stat": 2.0,
            "p_value_hac": 0.04,
            "hac_lags": 1,
            "n_entities": 1,
            "n_periods": n,
            "hac_maxlags": 1,
            "covariance_type": "newey_west",
        }

    monkeypatch.setattr(causal, "manual_dml_timeseries", stub)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 1))
    t = x[:, 0] * 0.5 + rng.normal(scale=0.1, size=n)
    y = t * 0.3 + rng.normal(scale=0.1, size=n)
    causal.run_dml_analysis(
        pd.DataFrame({"y": y, "t": t, "x": x[:, 0]}),
        "t",
        "y",
        ["x"],
        n_folds=2,
        embargo=1,
        n_placebo=0,
    )

    assert observed, "manual_dml_timeseries was not reached"
    assert set(observed) == {causal.DML_THREAD_LIMIT}, (
        f"run_dml_analysis ran with pools at {sorted(set(observed))}, not {causal.DML_THREAD_LIMIT}"
    )


def test_preview_causal_requires_every_reduction(tmp_path, monkeypatch) -> None:
    """A preview that omits max_samples would resolve the full population.

    This path no longer falls back to the shared preset's max_samples, so a partial
    reduction set is an uncapped preview - the opposite of what the tier is for. The
    guard's message already claimed every reduction; now it enforces it.
    """
    study, label, _frame = _causal_fixture(tmp_path, monkeypatch)

    with pytest.raises(
        ValueError, match=r"must declare every reduction; missing \['max_samples'\]"
    ):
        study.causal(
            method="dml",
            label=label.name,
            execution_tier="preview",
            preview_reductions={"max_symbols": 6, "n_folds": 2, "n_placebo": 10},
        ).resolve()


def test_canonical_causal_ignores_a_sample_cap_declared_in_the_preset(
    tmp_path, monkeypatch
) -> None:
    """The shared preset cannot cap a canonical sample.

    config/dml/dml.yaml still declares max_samples for the six case-study DML stages that read
    it as their own default and have not migrated to this path. This resolver must ignore it:
    a canonical run uses the full declared population, and a reduction reaches it only through
    preview_reductions, which a canonical request refuses.
    """
    study, label, frame = _causal_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "utils.modeling.load_configs",
        lambda *args, **kwargs: [
            {
                "config_name": "dml",
                "family": "causal_dml",
                "max_samples": 50_000,
                "n_folds": 5,
                "n_placebo": 100,
                "params": {},
                "seed": 42,
            }
        ],
    )

    resolved = study.causal(method="dml", label=label.name).resolve()
    population = resolved.spec["computation"]["analysis_population"]

    assert population["max_samples"] == 0
    assert population["n_rows"] == frame.height - frame.get_column("symbol").n_unique()

    with pytest.raises(ValueError, match="canonical causal requests cannot declare preview"):
        study.causal(
            method="dml", label=label.name, preview_reductions={"max_samples": 60}
        ).resolve()


def test_causal_run_registers_once_and_reopens_after_restart(tmp_path, monkeypatch) -> None:
    study, label, _frame = _causal_fixture(tmp_path, monkeypatch)

    monkeypatch.setattr(
        causal,
        "run_dml_analysis",
        lambda *args, **kwargs: {
            "dml_result": {"theta": 0.02, "se_hac": 0.01, "n_obs": 120},
            "p_value_hac": 0.04,
            "naive_effect": 0.03,
            "confounding_bias_pct": 50.0,
            "refutation": {"empirical_p": 0.1},
            "started_at": "2026-08-13T00:00:00+00:00",
            "elapsed_s": 1.0,
        },
    )
    request = study.causal(
        method="dml",
        label=label.name,
        execution_tier="preview",
        preview_reductions={
            "max_samples": 240,
            "max_symbols": 6,
            "n_folds": 2,
            "n_placebo": 10,
        },
    )

    first = request.run()
    reopened = CausalResult.open(study, first.hash, include_preview=True)
    second = request.run()

    assert first.complete
    assert reopened.hash == first.hash == second.hash
    assert CausalResult.one(study, label=label.name, execution_tier="preview").hash == first.hash
    assert reopened.metrics["dml_effect"] == 0.02
    assert json.loads(json.dumps(reopened.spec)) == reopened.spec


def test_causal_cache_accepts_provenance_only_drift(tmp_path, monkeypatch) -> None:
    study, label, _frame = _causal_fixture(tmp_path, monkeypatch)
    calls = 0

    def run_analysis(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {
            "dml_result": {"theta": 0.02, "se_hac": 0.01, "n_obs": 120},
            "p_value_hac": 0.04,
            "naive_effect": 0.03,
            "confounding_bias_pct": 50.0,
            "refutation": {"empirical_p": 0.1},
        }

    monkeypatch.setattr(causal, "run_dml_analysis", run_analysis)
    resolved = study.causal(
        method="dml",
        label=label.name,
        execution_tier="preview",
        preview_reductions={
            "max_samples": 240,
            "max_symbols": 6,
            "n_folds": 2,
            "n_placebo": 10,
        },
    ).resolve()
    first = resolved.run()
    provenance_only = deepcopy(resolved.spec)
    provenance_only["provenance"]["baseline_commit"] = "new-provenance-only-commit"

    assert training_hash_from_spec(provenance_only) == first.hash
    second = causal.run_resolved_causal_request(study, provenance_only, resolved._context)

    assert second.hash == first.hash
    assert calls == 1


def test_block_permutation_resets_at_temporal_gap() -> None:
    values = np.arange(8)
    timestamps = np.array(
        [
            "2024-01-01T00:00:00",
            "2024-01-01T08:00:00",
            "2024-01-01T16:00:00",
            "2024-01-02T00:00:00",
            "2024-01-04T00:00:00",
            "2024-01-04T08:00:00",
            "2024-01-04T16:00:00",
            "2024-01-05T00:00:00",
        ],
        dtype="datetime64[s]",
    )

    permuted = causal.block_permute(
        values,
        block_size=2,
        rng=np.random.default_rng(7),
        groups=timestamps,
        expected_step="8h",
    )

    assert set(permuted[:4]) == set(values[:4])
    assert set(permuted[4:]) == set(values[4:])


@pytest.mark.parametrize("entity", ["symbol", "product"])
def test_causal_resolver_accepts_either_canonical_entity_key(tmp_path, monkeypatch, entity) -> None:
    study, label, _frame = _causal_fixture(tmp_path, monkeypatch, entity=entity)

    resolved = study.causal(
        method="dml",
        label=label.name,
        execution_tier="preview",
        preview_reductions={"max_samples": 240, "max_symbols": 6, "n_folds": 2, "n_placebo": 10},
    ).resolve()

    assert resolved.spec["computation"]["analysis_population"]["n_rows"] > 0


@pytest.mark.parametrize(
    ("label_buffer", "expected_block"),
    [("8H", 1), ("24H", 3), ("48H", 6)],
)
def test_the_placebo_block_spans_the_label_horizon(
    tmp_path, monkeypatch, label_buffer, expected_block
) -> None:
    """The resolved spec's block size is the label horizon in bars, not the embargo.

    On 8-hour bars a 24-hour label overlaps three observations, so a block of one
    would leave the placebo free to break exactly the dependence the overlap
    creates. Reverting the resolver to `block_size=embargo` passes only while the
    embargo happens to be derived from the same buffer; this pins the horizon.
    """
    study, label, _frame = _causal_fixture(tmp_path, monkeypatch, label_buffer=label_buffer)

    resolved = study.causal(
        method="dml",
        label=label.name,
        execution_tier="preview",
        preview_reductions={"max_samples": 240, "max_symbols": 6, "n_folds": 2, "n_placebo": 10},
    ).resolve()

    assert resolved.spec["computation"]["refutation"]["block_size"] == expected_block


def test_causal_resolver_rejects_an_unsupported_entity_key(tmp_path, monkeypatch) -> None:
    study, label, _frame = _causal_fixture(tmp_path, monkeypatch, entity="ticker")

    with pytest.raises(ValueError, match="does not support entity key 'ticker'"):
        study.causal(
            method="dml",
            label=label.name,
            execution_tier="preview",
            preview_reductions={
                "max_samples": 240,
                "max_symbols": 6,
                "n_folds": 2,
                "n_placebo": 10,
            },
        ).resolve()


def _session_causal_fixture(tmp_path, monkeypatch):
    """A weekday-only panel, where a calendar buffer and a session buffer disagree."""
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    setup_path = study.root / "config" / "setup.yaml"
    # One bar: `treatment` carries the row's own timestamp. Declared because a canonical
    # run refuses an undeclared window, and this fixture is about the holdout seal.
    setup_path.write_text(
        setup_path.read_text()
        + "causal:\n  treatment: treatment\n  confounders: [confounder]\n  treatment_window: 1\n"
    )
    sessions = [
        timestamp
        for timestamp in pd.date_range("2024-01-01", "2024-03-01", freq="D", tz="UTC")
        if timestamp.weekday() < 5
    ]
    rows = []
    for session_index, timestamp in enumerate(sessions):
        for symbol_index in range(6):
            rows.append(
                {
                    "symbol": f"S{symbol_index}",
                    "timestamp": timestamp.to_pydatetime(),
                    "feature": float(symbol_index),
                    "treatment": float(symbol_index) + session_index / 100,
                    "confounder": float(session_index % 5),
                    "fwd_ret_5d": (symbol_index - 2.5) / 100 + session_index / 10000,
                }
            )
    frame = pl.DataFrame(rows).with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
    label = study.labels.publish(
        LabelDefinition("fwd_ret_5d", "regression", "5D"),
        frame.select("symbol", "timestamp", "fwd_ret_5d"),
    )
    mds = SimpleNamespace(
        dataset=frame,
        feature_names=["feature", "treatment", "confounder"],
        label_col="fwd_ret_5d",
        label_buffer="5D",
        date_col="timestamp",
        entity_cols=["symbol"],
        input_lineage={
            "artifacts": {"financial": {"sha256": "features-v1", "size": 1}},
            "fingerprint": "fixture-v1",
        },
    )
    monkeypatch.setattr("utils.modeling.load_modeling_dataset", lambda *args, **kwargs: mds)
    monkeypatch.setattr(
        "utils.modeling.load_configs",
        lambda *args, **kwargs: [
            {
                "config_name": "dml",
                "family": "causal_dml",
                "n_folds": 5,
                "n_placebo": 100,
                "params": {"max_depth": 3, "max_iter": 50},
                "seed": 42,
            }
        ],
    )
    return study, label, frame, sessions


def test_holdout_seal_counts_sessions_not_calendar_days(tmp_path, monkeypatch) -> None:
    """A 5D buffer means five observations, and on a gapped calendar those differ.

    The panel trades Monday to Friday, so five calendar days back from a Monday holdout
    reaches the Wednesday before, while five observations back reaches the Monday before
    that. The two sessions between them carry outcomes that resolve inside the holdout.
    """
    study, label, frame, sessions = _session_causal_fixture(tmp_path, monkeypatch)
    _set_holdout_start(study.root / "config" / "setup.yaml", "2024-02-19")

    computation = study.causal(method="dml", label=label.name).resolve().spec["computation"]

    holdout = pd.Timestamp("2024-02-19", tz="UTC")
    pre_holdout = [timestamp for timestamp in sessions if timestamp < holdout]
    session_cutoff = pre_holdout[-5]
    calendar_cutoff = holdout - pd.Timedelta("5D")
    # The fixture is only meaningful while the two constructions actually disagree.
    assert session_cutoff < calendar_cutoff

    assert computation["estimand"]["holdout_endpoint_cutoff"] == session_cutoff.isoformat()
    retained = frame.filter(pl.col("timestamp") < session_cutoff)
    assert computation["analysis_population"]["n_rows"] == retained.height
    # Rows the calendar cutoff would have kept, whose own five-session outcome resolves
    # inside the holdout. This is the difference the seal exists to remove.
    leaked = frame.filter(
        (pl.col("timestamp") >= session_cutoff) & (pl.col("timestamp") < calendar_cutoff)
    )
    assert leaked.height > 0
    # Counted from the session list rather than derived from the frame filters above, so it
    # can disagree with them: every session strictly before the cutoff, times every symbol.
    symbols = frame.get_column("symbol").n_unique()
    admissible = len([timestamp for timestamp in sessions if timestamp < session_cutoff])
    assert computation["analysis_population"]["n_rows"] == admissible * symbols


def _patch_modeling_dataset(monkeypatch, frame, buffer: str = "5D") -> None:
    """Re-point the resolver at a modified panel, keeping the fixture's label metadata."""
    mds = SimpleNamespace(
        dataset=frame,
        feature_names=["feature", "treatment", "confounder"],
        label_col="fwd_ret_5d",
        label_buffer=buffer,
        date_col="timestamp",
        entity_cols=["symbol"],
        input_lineage={
            "artifacts": {"financial": {"sha256": "features-v1", "size": 1}},
            "fingerprint": "fixture-v1",
        },
    )
    monkeypatch.setattr("utils.modeling.load_modeling_dataset", lambda *a, **k: mds)


def _retained_rows(frame, cutoffs: dict) -> int:
    """Rows each symbol keeps when sealed against its own cutoff."""
    return sum(
        frame.filter((pl.col("symbol") == symbol) & (pl.col("timestamp") < cutoff)).height
        for symbol, cutoff in cutoffs.items()
    )


def test_holdout_seal_counts_each_entity_own_observations(tmp_path, monkeypatch) -> None:
    """A product missing late sessions is sealed earlier, and only that product is.

    The buffer advances by the entity's own observations. Counting distinct timestamps
    across the panel reaches back five panel sessions, which is fewer than five of a
    sparse product's own, so that product would keep rows resolving inside the holdout.
    """
    study, label, frame, sessions = _session_causal_fixture(tmp_path, monkeypatch)
    holdout = pd.Timestamp("2024-02-19", tz="UTC")
    pre_holdout = [timestamp for timestamp in sessions if timestamp < holdout]
    # S0 stops trading four sessions before the holdout; every other product runs through.
    absent = set(pre_holdout[-4:])
    sparse = frame.filter(~((pl.col("symbol") == "S0") & pl.col("timestamp").is_in(absent)))
    _patch_modeling_dataset(monkeypatch, sparse)
    _set_holdout_start(study.root / "config" / "setup.yaml", "2024-02-19")

    computation = study.causal(method="dml", label=label.name).resolve().spec["computation"]

    sparse_cutoff = [t for t in pre_holdout if t not in absent][-5]
    dense_cutoff = pre_holdout[-5]
    assert sparse_cutoff < dense_cutoff

    cutoffs = {"S0": sparse_cutoff} | {f"S{i}": dense_cutoff for i in range(1, 6)}
    assert computation["analysis_population"]["n_rows"] == _retained_rows(sparse, cutoffs)
    # Sealing every symbol at the earliest cutoff would keep strictly fewer rows, so the
    # count above distinguishes a per-entity seal from a panel-wide collapse of them.
    collapsed = {symbol: sparse_cutoff for symbol in cutoffs}
    assert _retained_rows(sparse, collapsed) < _retained_rows(sparse, cutoffs)
    assert computation["estimand"]["holdout_endpoint_cutoff"] == dense_cutoff.isoformat()


def test_one_entity_exiting_early_does_not_truncate_the_others(tmp_path, monkeypatch) -> None:
    """A product that stops trading long before the holdout seals only itself.

    Collapsing the per-entity cutoffs to their minimum and applying it panel-wide would
    drag every other product back to this one's exit date, which is a much larger loss
    than the leak the seal exists to prevent.
    """
    study, label, frame, sessions = _session_causal_fixture(tmp_path, monkeypatch)
    holdout = pd.Timestamp("2024-02-19", tz="UTC")
    pre_holdout = [timestamp for timestamp in sessions if timestamp < holdout]
    # S0 exits early but still holds well more than the five observations the buffer needs.
    exit_after = pre_holdout[15]
    early = frame.filter(~((pl.col("symbol") == "S0") & (pl.col("timestamp") > exit_after)))
    _patch_modeling_dataset(monkeypatch, early)
    _set_holdout_start(study.root / "config" / "setup.yaml", "2024-02-19")

    computation = study.causal(method="dml", label=label.name).resolve().spec["computation"]

    dense_cutoff = pre_holdout[-5]
    cutoffs = {"S0": pre_holdout[11]} | {f"S{i}": dense_cutoff for i in range(1, 6)}
    assert computation["analysis_population"]["n_rows"] == _retained_rows(early, cutoffs)
    # Sealing every symbol at S0's exit instead keeps strictly fewer rows, which is the
    # truncation this test exists to rule out.
    collapsed = {symbol: pre_holdout[11] for symbol in cutoffs}
    assert _retained_rows(early, collapsed) < _retained_rows(early, cutoffs)


def test_holdout_seal_fails_closed_on_a_panel_shorter_than_its_buffer(
    tmp_path, monkeypatch
) -> None:
    """No entity can absorb the buffer, so the request refuses rather than estimating."""
    study, label, _frame, sessions = _session_causal_fixture(tmp_path, monkeypatch)
    # Three sessions precede this holdout, against a five-observation buffer.
    _set_holdout_start(study.root / "config" / "setup.yaml", "2024-01-04")
    with pytest.raises(ValueError, match="none can absorb the buffer"):
        study.causal(method="dml", label=label.name).resolve()


def test_holdout_seal_holds_a_one_session_horizon_over_a_weekend_boundary(
    tmp_path, monkeypatch
) -> None:
    """A one-session horizon leaks too, whenever a non-session gap precedes the boundary.

    This is a boundary problem rather than a long-horizon one. The holdout opens on a
    Monday, so the two weekend days sit between the last session and the boundary: the
    calendar cutoff lands on the Sunday, a strict `<` still admits the Friday, and that
    Friday's one-session-forward outcome resolves on the first session of the holdout.
    A test that only pins a multi-session buffer passes on a panel that still leaks here.
    """
    study, label, frame, sessions = _session_causal_fixture(tmp_path, monkeypatch)
    _patch_modeling_dataset(monkeypatch, frame, buffer="1D")
    _set_holdout_start(study.root / "config" / "setup.yaml", "2024-02-19")

    computation = study.causal(method="dml", label=label.name).resolve().spec["computation"]

    holdout = pd.Timestamp("2024-02-19", tz="UTC")
    pre_holdout = [timestamp for timestamp in sessions if timestamp < holdout]
    last_session = pre_holdout[-1]
    # The Friday sits strictly before the calendar cutoff, so the old construction kept it.
    assert last_session < holdout - pd.Timedelta("1D")
    # Counting one observation instead seals at that Friday, so the last row retained is
    # the Thursday, whose next session is the Friday and therefore still outside.
    assert computation["estimand"]["holdout_endpoint_cutoff"] == last_session.isoformat()
    symbols = frame.get_column("symbol").n_unique()
    assert computation["analysis_population"]["n_rows"] == (len(pre_holdout) - 1) * symbols


def test_the_bandwidth_takes_the_outcome_horizon_and_the_block_takes_the_buffer(
    tmp_path, monkeypatch
) -> None:
    """The two are different quantities, and only one of them describes outcome overlap.

    The buffer keeps a fold's training rows clear of its validation labels; the
    outcome horizon is how long one outcome stays open, which is what makes
    successive outcomes overlap and therefore what the Newey-West bandwidth has to
    cover. They agree in most case studies here, so a resolver that derives both
    from the buffer passes everywhere except where they differ. On 8-hour bars a
    24-hour buffer over an 8-hour label is three bars against one.
    """
    study, label, _frame = _causal_fixture(
        tmp_path, monkeypatch, label_buffer="24H", label_horizon="8H"
    )

    resolved = study.causal(
        method="dml",
        label=label.name,
        execution_tier="preview",
        preview_reductions={"max_samples": 240, "max_symbols": 6, "n_folds": 2, "n_placebo": 10},
    ).resolve()

    computation = resolved.spec["computation"]
    assert computation["estimand"]["outcome_horizon"] == str(pd.Timedelta("8h"))
    assert computation["refutation"]["block_size"] == 3
    assert resolved._context.horizon == 1


def test_a_horizon_longer_than_its_buffer_is_refused(tmp_path, monkeypatch) -> None:
    """The placebo block and the per-entity seal assume the buffer is the longer one.

    Both values are hand-authored in setup.yaml and nothing pairs them, so the
    ordering that used to hold structurally now holds by configuration. Reversed,
    the placebo block would be shorter than the dependence it holds fixed and the
    seal would leave outcomes reaching into the holdout, neither with a symptom.
    """
    study, label, _frame = _causal_fixture(
        tmp_path, monkeypatch, label_buffer="8H", label_horizon="24H"
    )

    # Matched on the claim rather than on the phrasing of the comparison. The two spans are
    # now compared as observation counts rather than as durations, because a calendar month
    # is not a fixed span and `pd.Timedelta` refuses it, so a regex tied to how the numbers
    # are worded fails on a message that says the same thing.
    with pytest.raises(ValueError, match="cannot be shorter than the outcome it is holding"):
        study.causal(
            method="dml",
            label=label.name,
            execution_tier="preview",
            preview_reductions={
                "max_samples": 240,
                "max_symbols": 6,
                "n_folds": 2,
                "n_placebo": 10,
            },
        ).resolve()


def test_a_canonical_run_refuses_an_undeclared_treatment_window(tmp_path, monkeypatch) -> None:
    """A block that cannot be shown to span the treatment must not reach the registry.

    Preview warns and proceeds - that tier exists to run reduced and be thrown away. Canonical
    refuses, because the alternative is a registered refutation whose p-value reads stronger
    than it is and that nothing downstream can distinguish from a correctly sized one.
    """
    study, label, _frame = _causal_fixture(tmp_path, monkeypatch, treatment_window=None)

    with pytest.raises(ValueError, match="no construction window is declared"):
        study.causal(method="dml", label=label.name, execution_tier="canonical").resolve()


def test_preview_warns_on_an_undeclared_window_rather_than_failing(tmp_path, monkeypatch) -> None:
    """Failing preview would block CI on every case study that had not declared one yet."""
    study, label, _frame = _causal_fixture(tmp_path, monkeypatch, treatment_window=None)

    with pytest.warns(UserWarning, match="no construction window is declared"):
        study.causal(
            method="dml",
            label=label.name,
            execution_tier="preview",
            preview_reductions={
                "max_samples": 240,
                "max_symbols": 6,
                "n_folds": 2,
                "n_placebo": 10,
            },
        ).resolve()


def test_the_block_spans_the_treatment_when_it_outlasts_the_buffer(tmp_path, monkeypatch) -> None:
    """The case the defect produced: a wide treatment permuted in blocks sized by the label.

    ETFs permuted a six-month momentum column in blocks of 21 sessions this way. The spec must
    also record which of the two scales bound it, so a reader can tell the two apart.
    """
    study, label, _frame = _causal_fixture(tmp_path, monkeypatch, treatment_window=21)

    resolved = study.causal(method="dml", label=label.name, execution_tier="canonical").resolve()

    refutation = resolved.spec["computation"]["refutation"]
    assert refutation["block_size"] == 21
    assert refutation["block_size_basis"] == "treatment_window"
    assert refutation["treatment_window_steps"] == 21
    assert refutation["label_buffer_steps"] < 21


def test_the_outcome_horizon_is_registered_and_is_not_the_buffer(tmp_path, monkeypatch) -> None:
    """The bandwidth the second stage is HAC-corrected at has to be readable from the spec.

    ``run_dml_analysis`` is already given the outcome horizon in observation periods, and until
    now the resolver computed it and threw it away. A notebook comparing bandwidth against block
    size then had nothing registered to read: crypto's ``12_model_analysis`` printed a value it
    derived from a key that had never existed under any version of the resolver.

    The assertion that carries this is the second one. ``label_buffer_steps`` is the CV gap and
    may be deliberately longer than the outcome it seals, so a horizon read off the buffer is
    wrong in exactly the case where the two differ - which is the fixture here.
    """
    # A buffer three cadences long over an eight-hour panel, sealing a one-cadence outcome.
    # That gap is the case the two quantities are told apart in: a horizon read off the
    # buffer would report 3 where the label resolves in 1.
    study, label, _frame = _causal_fixture(
        tmp_path, monkeypatch, treatment_window=21, label_buffer="24H", label_horizon="8H"
    )

    refutation = (
        study.causal(method="dml", label=label.name, execution_tier="canonical")
        .resolve()
        .spec["computation"]["refutation"]
    )

    assert refutation["label_buffer_steps"] == 3
    assert refutation["label_horizon_steps"] == 1


def test_naming_the_notebook_records_provenance_without_moving_the_identity(
    tmp_path, monkeypatch
) -> None:
    """`notebook=` answers which notebook wrote a row, and must not reprice the analysis.

    `entry_point` in a causal spec names the module, `case_studies.utils.causal`, which every
    `*_causal_dml` notebook shares - so it cannot say which one ran. `notebook_path` can, and
    it is in `_V2_PROVENANCE_FIELDS`, so recording it moves no hash and forces no refit. That
    is the whole reason this is safe to add to notebooks whose rows already exist, and it is
    worth pinning rather than asserting: `computation` is hashed whole, so a request field
    that leaked into it would reprice every causal row in the corpus.
    """
    study, label, _frame = _causal_fixture(tmp_path, monkeypatch)

    unnamed = study.causal(method="dml", label=label.name).resolve()
    named = study.causal(method="dml", label=label.name, notebook="11_causal_dml").resolve()

    from case_studies.utils.registry.specs import training_hash_from_spec

    assert named.spec["computation"] == unnamed.spec["computation"]
    assert training_hash_from_spec(named.spec) == training_hash_from_spec(unnamed.spec)

    assert named.spec["provenance"]["notebook_path"] == "11_causal_dml"
    assert "notebook_path" not in unnamed.spec["provenance"]
    assert named.spec["provenance"]["entry_point"] == "case_studies.utils.causal"
