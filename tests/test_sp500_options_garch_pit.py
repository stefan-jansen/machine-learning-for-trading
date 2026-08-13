"""Regression tests for the S&P 500 options causal GJR-GARCH filter."""

from __future__ import annotations

import ast
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import yaml
from ml4t.diagnostic.evaluation.stats import benjamini_hochberg_fdr

from case_studies.sp500_options._underlying_returns import reconcile_underlying_log_returns

NOTEBOOK_SOURCE = Path("case_studies/sp500_options/04_model_based_features.py")
GARCH_FIT_FUNCTIONS = {
    "causal_gjr_garch_filter",
    "fit_garch_with_retry",
    "summarize_garch_fit",
    "training_garch_filter_state",
    "failed_garch_diagnostic",
    "fit_gjr_garch_symbol",
}


def _load_causal_filter():
    tree = ast.parse(NOTEBOOK_SOURCE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "causal_gjr_garch_filter"
    )
    namespace = {"np": np, "pd": pd}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK_SOURCE), "exec"),
        namespace,
    )
    return namespace["causal_gjr_garch_filter"]


def _load_sv_calibrator(fake_fit):
    tree = ast.parse(NOTEBOOK_SOURCE.read_text())
    function_names = {
        "stable_segment_seed",
        "sv_diagnostics_pass",
        "run_sv_calibration_attempt",
        "accepted_sv_calibration",
        "sv_training_window",
        "sv_calibration_diagnostic",
        "calibrate_sigma_eta",
    }
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    namespace = {
        "np": np,
        "pd": pd,
        "SEED": 42,
        "SV_RETRY_DRAWS": 4_000,
        "SV_RETRY_TUNE": 4_000,
        "fit_sv_calibration_symbol": fake_fit,
        "hashlib": __import__("hashlib"),
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK_SOURCE), "exec"),
        namespace,
    )
    return namespace["calibrate_sigma_eta"]


def _load_sv_pool_selector():
    tree = ast.parse(NOTEBOOK_SOURCE.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {"option_coverage_in_window", "return_coverage_in_window", "select_sv_pool"}
    ]
    namespace = {"pd": pd, "pl": pl, "date": date}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK_SOURCE), "exec"),
        namespace,
    )
    return namespace["select_sv_pool"]


def _load_temporal_ic_summarizer():
    tree = ast.parse(NOTEBOOK_SOURCE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "summarize_temporal_ic"
    )
    namespace = {"pl": pl, "benjamini_hochberg_fdr": benjamini_hochberg_fdr}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK_SOURCE), "exec"),
        namespace,
    )
    return namespace["summarize_temporal_ic"]


def _load_seed_and_particle_filter():
    tree = ast.parse(NOTEBOOK_SOURCE.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"stable_segment_seed", "particle_filter_sv"}
    ]
    namespace = {
        "PERIODS_PER_YEAR": 252,
        "hashlib": __import__("hashlib"),
        "np": np,
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK_SOURCE), "exec"),
        namespace,
    )
    return namespace["stable_segment_seed"], namespace["particle_filter_sv"]


def _load_public_schema_validator():
    tree = ast.parse(NOTEBOOK_SOURCE.read_text())
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "MODEL_BASED_SCHEMA"
                for target in node.targets
            )
        )
        or (isinstance(node, ast.FunctionDef) and node.name == "validate_public_temporal_schema")
    ]
    namespace = {"pl": pl}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK_SOURCE), "exec"),
        namespace,
    )
    return namespace["validate_public_temporal_schema"]


def _load_label_endpoint_sealer():
    tree = ast.parse(NOTEBOOK_SOURCE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "seal_incremental_label_endpoints"
    )
    namespace = {"pl": pl, "date": date}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK_SOURCE), "exec"),
        namespace,
    )
    return namespace["seal_incremental_label_endpoints"]


@pytest.mark.parametrize(
    ("symbol", "return_std", "fit_scale", "seed"),
    [
        ("LOW_VOL", 0.0001, 1000.0, 1),
        ("BA", 1.0, 1.0, 2),
        ("HIGH_VOL", 100.0, 0.1, 3),
    ],
)
def test_causal_garch_prefix_is_invariant_to_same_time_and_future_returns(
    symbol: str,
    return_std: float,
    fit_scale: float,
    seed: int,
) -> None:
    causal_filter = _load_causal_filter()
    assert symbol
    rng = np.random.default_rng(seed)
    returns = pd.Series(rng.normal(0, return_std, 400))
    scaled_variance = float(np.var(returns.to_numpy() * fit_scale))
    params = pd.Series(
        {
            "mu": 0.0,
            "omega": max(scaled_variance * 0.05, 1e-8),
            "alpha[1]": 0.05,
            "gamma[1]": 0.05,
            "beta[1]": 0.85,
        }
    )
    bounds = (max(scaled_variance / 1e8, 1e-12), max(scaled_variance * 1e8, 1.0))
    event = 250
    future_event = 325

    base = causal_filter(returns, params, fit_scale, scaled_variance, bounds).to_numpy()
    expected_initial_variance = (
        params["omega"]
        + (params["alpha[1]"] + 0.5 * params["gamma[1]"] + params["beta[1]"]) * scaled_variance
    )
    assert base[0] == pytest.approx(np.sqrt(expected_initial_variance) / fit_scale)
    same_time = returns.copy()
    same_time.iloc[event] += 50 * return_std
    same_path = causal_filter(same_time, params, fit_scale, scaled_variance, bounds).to_numpy()
    future = returns.copy()
    future.iloc[future_event] -= 50 * return_std
    future_path = causal_filter(future, params, fit_scale, scaled_variance, bounds).to_numpy()

    np.testing.assert_array_equal(base[: event + 1], same_path[: event + 1])
    assert abs(base[event + 1] - same_path[event + 1]) > 0
    np.testing.assert_array_equal(base[: future_event + 1], future_path[: future_event + 1])


def test_notebook_does_not_use_arch_fixed_result_for_filtering() -> None:
    source = NOTEBOOK_SOURCE.read_text()
    assert ".fix(" not in source
    assert "causal_gjr_garch_filter(" in source


def test_notebook_uses_identity_safe_segment_returns() -> None:
    source = NOTEBOOK_SOURCE.read_text()
    assert "reconcile_underlying_log_returns(underlying)" in source
    assert "segment_returns: dict[tuple[str, int], pd.Series]" in source
    assert "for (symbol, sec_id), ret_series in segment_returns.items()" in source
    assert "validate_segment_feature_panel(garch_df" in source
    assert "validate_segment_feature_panel(sv_df" in source


def test_segment_scale_and_prior_identity_do_not_change_filtered_state() -> None:
    causal_filter = _load_causal_filter()
    start = date(2020, 1, 1)
    prices = pl.DataFrame(
        {
            "timestamp": [start + timedelta(days=i) for i in range(10)],
            "symbol": ["AAPL"] * 10,
            "sec_id": [1] * 5 + [2] * 5,
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 130.0, 131.0, 129.0, 132.0, 133.0],
            "adj_factor": [1.0] * 10,
        }
    )
    perturbed = prices.with_columns(
        pl.when(pl.col("sec_id") == 1)
        .then(pl.col("close") * 17)
        .when(pl.col("sec_id") == 2)
        .then(pl.col("close") * 11)
        .alias("close")
    )
    baseline = reconcile_underlying_log_returns(prices)
    changed = reconcile_underlying_log_returns(perturbed)
    segment = baseline.filter(pl.col("sec_id") == 2)["clean_log_return"].drop_nulls().to_numpy()
    changed_segment = (
        changed.filter(pl.col("sec_id") == 2)["clean_log_return"].drop_nulls().to_numpy()
    )
    np.testing.assert_array_equal(segment, changed_segment)

    segment_returns = pd.Series(segment * 100)
    changed_returns = pd.Series(changed_segment * 100)
    params = pd.Series(
        {
            "mu": 0.0,
            "omega": 0.05,
            "alpha[1]": 0.05,
            "gamma[1]": 0.05,
            "beta[1]": 0.85,
        }
    )
    baseline_state = causal_filter(segment_returns, params, 1.0, 1.0, (1e-12, 1e12))
    changed_state = causal_filter(changed_returns, params, 1.0, 1.0, (1e-12, 1e12))
    np.testing.assert_array_equal(baseline_state, changed_state)


def test_sv_pool_uses_training_option_coverage_not_generic_return_history() -> None:
    select_pool = _load_sv_pool_selector()
    training_dates = [timestamp.date() for timestamp in pd.date_range("2020-01-01", periods=300)]
    post_train_dates = [timestamp.date() for timestamp in pd.date_range("2020-10-27", periods=200)]
    security_ids = {"A": 1, "B": 2, "C": 3}

    identity_rows = []
    return_rows = []
    for symbol, sec_id in security_ids.items():
        for timestamp in training_dates + post_train_dates:
            identity_rows.append({"timestamp": timestamp, "symbol": symbol, "sec_id": sec_id})
            return_rows.append({"timestamp": timestamp, "symbol": symbol, "sec_id": sec_id})

    option_rows = []
    for symbol, coverage in {"A": 150, "B": 250, "C": 50}.items():
        option_rows.extend(
            {"timestamp": timestamp, "symbol": symbol} for timestamp in training_dates[:coverage]
        )
    option_rows.extend({"timestamp": timestamp, "symbol": "C"} for timestamp in post_train_dates)

    option_panel = pl.DataFrame(option_rows)
    identity_panel = pl.DataFrame(identity_rows)
    return_panel = pl.DataFrame(return_rows)
    train_start = training_dates[0]
    train_end = training_dates[-1]

    selected = select_pool(
        option_panel,
        identity_panel,
        return_panel,
        train_start,
        train_end,
        2,
    )
    selected_without_future = select_pool(
        option_panel.filter(pl.col("timestamp") <= train_end),
        identity_panel,
        return_panel,
        train_start,
        train_end,
        2,
    )

    assert selected == [("B", 2), ("A", 1)]
    assert selected == selected_without_future


def test_temporal_ic_screen_controls_false_discovery_rate() -> None:
    summarize = _load_temporal_ic_summarizer()
    temporal_ic = {
        "strong": {
            "mean_ic": 0.03,
            "naive_se": 0.008,
            "hac_se": 0.009,
            "t_stat": 3.2,
            "p_value": 0.001,
        },
        "nominal_only": {
            "mean_ic": 0.02,
            "naive_se": 0.008,
            "hac_se": 0.010,
            "t_stat": 2.1,
            "p_value": 0.04,
        },
        "null": {
            "mean_ic": -0.01,
            "naive_se": 0.009,
            "hac_se": 0.010,
            "t_stat": -1.0,
            "p_value": 0.20,
        },
    }

    summary, n_discoveries = summarize(temporal_ic)
    by_feature = {row["feature"]: row for row in summary.iter_rows(named=True)}

    assert n_discoveries == 1
    assert by_feature["strong"]["fdr_pval"] == pytest.approx(0.003)
    assert by_feature["strong"]["significant_fdr05"] is True
    assert by_feature["nominal_only"]["hac_pval"] < 0.05
    assert by_feature["nominal_only"]["fdr_pval"] == pytest.approx(0.06)
    assert by_feature["nominal_only"]["significant_fdr05"] is False


def test_temporal_hac_uses_hold_to_expiry_trading_horizon() -> None:
    tree = ast.parse(NOTEBOOK_SOURCE.read_text())
    source = NOTEBOOK_SOURCE.read_text()
    hac_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "compute_ic_hac_stats"
    ]

    assert 'LABEL_HORIZON_TRADING_DAYS = int(_setup["features"]["hold_sessions"])' in source
    setup = yaml.safe_load(Path("case_studies/sp500_options/config/setup.yaml").read_text())
    assert setup["features"]["hold_sessions"] == 21
    assert len(hac_calls) == 1
    horizon = next(
        keyword.value for keyword in hac_calls[0].keywords if keyword.arg == "label_horizon"
    )
    assert isinstance(horizon, ast.Name)
    assert horizon.id == "LABEL_HORIZON_TRADING_DAYS"


def test_future_only_segment_cannot_change_existing_particle_path() -> None:
    stable_seed, particle_filter = _load_seed_and_particle_filter()
    validation_end = pd.Timestamp("2020-12-31")
    existing_returns = pd.Series(
        np.linspace(-2.0, 2.0, 300),
        index=pd.date_range("2020-01-01", periods=300, freq="D"),
    )
    future_returns = pd.Series(
        np.linspace(-1.0, 1.0, 100),
        index=pd.date_range("2022-01-01", periods=100, freq="D"),
    )

    def filter_paths(segments):
        paths = {}
        for (symbol, sec_id), returns in segments.items():
            available = returns[returns.index <= validation_end]
            if available.empty:
                continue
            seed = stable_seed(42, "cv-fold:0", symbol, sec_id)
            paths[(symbol, sec_id)] = particle_filter(
                available.to_numpy(), 0.2, n_particles=200, seed=seed
            )
        return paths

    baseline = filter_paths({("ZZZ", 9): existing_returns})
    with_future_first = filter_paths(
        {("AAA_FUTURE", 1): future_returns, ("ZZZ", 9): existing_returns}
    )

    np.testing.assert_array_equal(baseline[("ZZZ", 9)], with_future_first[("ZZZ", 9)])


def test_public_temporal_schema_matches_frozen_contract() -> None:
    validate = _load_public_schema_validator()
    valid = pl.DataFrame(
        {
            "timestamp": [date(2020, 1, 2)],
            "symbol": ["AAPL"],
            "garch_cond_vol": [0.2],
            "fold": [0],
            "sv_vol": [0.21],
            "garch_vrp": [0.05],
            "sv_vrp": [0.04],
        },
        schema_overrides={"fold": pl.Int32},
    )
    validate(valid)

    with pytest.raises(RuntimeError, match="schema"):
        validate(valid.with_columns(pl.col("fold").cast(pl.Int64)))


def test_incremental_screen_purges_labels_settling_in_holdout() -> None:
    seal = _load_label_endpoint_sealer()
    frame = pl.DataFrame(
        {
            "timestamp": [date(2020, 11, 30), date(2020, 12, 15), date(2020, 12, 31)],
            "symbol": ["A", "B", "C"],
            "dte_calendar": [31, 35, 1],
            "ret_to_expiry": [0.1, 0.2, 0.3],
        },
        schema_overrides={"dte_calendar": pl.Int32},
    )

    retained, purged_rows, max_endpoint = seal(frame, date(2021, 1, 1))

    assert retained["symbol"].to_list() == ["A"]
    assert purged_rows == 2
    assert max_endpoint == date(2020, 12, 31)


@pytest.mark.parametrize(
    ("symbol", "close_before", "factor_before", "close_after", "factor_after"),
    [
        ("AAPL", 499.23, 8.100549, 129.04, 32.402197),
        ("FAST", 64.28, 2.668525, 31.34, 5.337051),
        ("NVDA", 751.19, 1.632315, 186.12, 6.529261),
    ],
)
def test_known_split_jumps_disappear_from_adjusted_returns(
    symbol: str,
    close_before: float,
    factor_before: float,
    close_after: float,
    factor_after: float,
) -> None:
    assert symbol
    raw_return = np.log(close_after / close_before)
    adjusted_return = np.log((close_after * factor_after) / (close_before * factor_before))
    assert abs(raw_return) > 0.6
    assert abs(adjusted_return) < 0.1


def test_nonconverged_garch_retry_is_excluded() -> None:
    tree = ast.parse(NOTEBOOK_SOURCE.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in GARCH_FIT_FUNCTIONS
    ]

    class FakeResult:
        convergence_flag = 1
        params = pd.Series(
            {
                "mu": 0.0,
                "omega": 0.05,
                "alpha[1]": 0.05,
                "gamma[1]": 0.05,
                "beta[1]": 0.85,
            }
        )
        loglikelihood = -100.0

    class FakeModel:
        def __init__(self) -> None:
            self.fit_calls: list[dict] = []

        def fit(self, **kwargs):
            self.fit_calls.append(kwargs)
            return FakeResult()

    fake_model = FakeModel()
    namespace = {
        "np": np,
        "pd": pd,
        "date": date,
        "arch_model": lambda *args, **kwargs: fake_model,
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK_SOURCE), "exec"),
        namespace,
    )
    returns = pd.Series(
        np.linspace(-1, 1, 400),
        index=pd.date_range("2018-01-01", periods=400, freq="D"),
    )

    feature, diagnostics = namespace["fit_gjr_garch_symbol"](
        returns,
        date(2018, 1, 1),
        date(2018, 12, 31),
        date(2019, 2, 4),
    )

    assert feature is None
    assert diagnostics is not None
    assert diagnostics["converged"] is False
    assert diagnostics["retried"] is True
    assert len(fake_model.fit_calls) == 2
    assert fake_model.fit_calls[1]["options"]["maxiter"] == 2_000


def test_errored_garch_attempt_returns_failed_diagnostic() -> None:
    tree = ast.parse(NOTEBOOK_SOURCE.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in GARCH_FIT_FUNCTIONS
    ]

    def fail_model(*args, **kwargs):
        raise ValueError("synthetic fit failure")

    namespace = {"np": np, "pd": pd, "date": date, "arch_model": fail_model}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK_SOURCE), "exec"),
        namespace,
    )
    returns = pd.Series(
        np.linspace(-1, 1, 400),
        index=pd.date_range("2018-01-01", periods=400, freq="D"),
    )

    feature, diagnostic = namespace["fit_gjr_garch_symbol"](
        returns,
        date(2018, 1, 1),
        date(2018, 12, 31),
        date(2019, 2, 4),
    )

    assert feature is None
    assert diagnostic is not None
    assert diagnostic["converged"] is False
    assert diagnostic["n_fit"] == 365
    assert diagnostic["error_type"] == "ValueError"
    assert diagnostic["error_message"] == "synthetic fit failure"


def test_sv_retry_uses_only_passing_posterior() -> None:
    calls = []
    failed = {
        "rhat": 1.02,
        "ess_bulk": 800.0,
        "ess_tail": 1_000.0,
        "divergences": 0,
        "max_treedepth_hits": 0,
    }
    passed = failed | {"rhat": 1.001}

    def fake_fit(returns, draws, tune, chains, seed):
        calls.append((len(returns), draws, tune, chains, seed))
        if len(calls) == 1:
            return np.array([1.0, 1.0]), failed
        return np.array([2.0, 2.0]), passed

    calibrate = _load_sv_calibrator(fake_fit)
    series = pd.Series(
        np.linspace(-1, 1, 300),
        index=pd.date_range("2018-01-01", periods=300, freq="D"),
    )
    pooled, diagnostics = calibrate(
        [("BA", 1)],
        {("BA", 1): series},
        date(2018, 1, 1),
        date(2018, 10, 27),
        n_draws=2_000,
        n_tune=2_000,
        n_chains=4,
    )

    assert pooled == 2.0
    assert diagnostics[0]["retried"] is True
    assert diagnostics[0]["sigma_eta"] == 2.0
    assert len(diagnostics[0]["attempts"]) == 2
    assert [call[1:3] for call in calls] == [(2_000, 2_000), (4_000, 4_000)]


def test_sv_pool_weights_segments_equally_after_longer_retry() -> None:
    calls = []
    failed = {
        "rhat": 1.02,
        "ess_bulk": 800.0,
        "ess_tail": 1_000.0,
        "divergences": 0,
        "max_treedepth_hits": 0,
    }
    passed = failed | {"rhat": 1.001}

    def fake_fit(returns, draws, tune, chains, seed):
        calls.append((draws, tune, seed))
        if len(calls) == 1:
            return np.zeros(2_000), failed
        if len(calls) == 2:
            return np.ones(4_000), passed
        return np.full(2_000, 3.0), passed

    calibrate = _load_sv_calibrator(fake_fit)
    series = pd.Series(
        np.linspace(-1, 1, 300),
        index=pd.date_range("2018-01-01", periods=300, freq="D"),
    )
    pooled, diagnostics = calibrate(
        [("A", 1), ("B", 2)],
        {("A", 1): series, ("B", 2): series},
        date(2018, 1, 1),
        date(2018, 10, 27),
        n_draws=2_000,
        n_tune=2_000,
        n_chains=4,
    )

    assert pooled == pytest.approx(2.0)
    assert [diagnostic["sigma_eta"] for diagnostic in diagnostics] == [1.0, 3.0]
    assert diagnostics[0]["retried"] is True
    assert diagnostics[1]["retried"] is False


def test_sv_retry_fails_closed_when_second_attempt_misses_gate() -> None:
    failed = {
        "rhat": 1.02,
        "ess_bulk": 800.0,
        "ess_tail": 1_000.0,
        "divergences": 0,
        "max_treedepth_hits": 0,
    }

    def fake_fit(returns, draws, tune, chains, seed):
        return np.array([1.0, 1.0]), failed

    calibrate = _load_sv_calibrator(fake_fit)
    series = pd.Series(
        np.linspace(-1, 1, 300),
        index=pd.date_range("2018-01-01", periods=300, freq="D"),
    )

    with pytest.raises(RuntimeError, match="failed after retry"):
        calibrate(
            [("BA", 1)],
            {("BA", 1): series},
            date(2018, 1, 1),
            date(2018, 10, 27),
            n_draws=2_000,
            n_tune=2_000,
            n_chains=4,
        )
