"""Regression tests for S&P 500 options financial features."""

from __future__ import annotations

import ast
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from case_studies.sp500_options._underlying_returns import reconcile_underlying_log_returns
from case_studies.utils.feature_engineering import EPS, trailing_return, trailing_volatility

NOTEBOOK = Path("case_studies/sp500_options/03_financial_features.py")
SEGMENT = ["symbol", "sec_id"]

# The notebook binds these from config/setup.yaml. The tests exercise its functions in isolation,
# so they supply the same values here; test_notebook_binds_every_window_from_the_configuration
# is what keeps the two in step.
WINDOWS = {
    "underlying_return": [1, 5, 10, 21],
    "realized_volatility": [5, 10, 21, 42, 63],
    "volume_zscore": 20,
    "instrument_return": [1, 5],
    "instrument_cost_momentum": 5,
    "vrp": [5, 10, 21, 42, 63],
    "vrp_reference": 21,
    "vrp_zscore": 252,
    "vrp_momentum": [5, 10],
    "iv_zscore": [63, 252],
    "iv_momentum": [5, 10, 21],
}
MIN_OBS = 0.8


def _load_notebook_functions(*wanted_names: str) -> dict[str, object]:
    """Execute the named notebook functions in a namespace carrying their dependencies."""
    tree = ast.parse(NOTEBOOK.read_text())
    wanted = set(wanted_names)
    definitions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    missing = wanted - {node.name for node in definitions}
    if missing:
        raise AssertionError(f"the notebook no longer defines {sorted(missing)}")
    module = ast.Module(body=definitions, type_ignores=[])
    namespace = {
        "np": np,
        "pl": pl,
        "EPS": EPS,
        "SEGMENT": SEGMENT,
        "WINDOWS": WINDOWS,
        "MIN_OBS": MIN_OBS,
        "PERIODS_PER_YEAR": 252,
        "TARGET_DTE": 30,
        "LEVEL": {"vega_floor": 0.001, "realized_volatility_floor": 0.01},
        "reconcile_underlying_log_returns": reconcile_underlying_log_returns,
        "trailing_return": trailing_return,
        "trailing_volatility": trailing_volatility,
    }
    exec(compile(module, NOTEBOOK, "exec"), namespace)
    return {name: namespace[name] for name in wanted_names}


def _identity_panel() -> pl.DataFrame:
    start = date(2020, 1, 1)
    closes = [100.0, 101.0, 102.0, 25.75, 25.50, 25.80, 26.0]
    closes += [130.0, 131.0, 132.0, 133.0, 134.0, 135.0, 136.0]
    factors = [1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0] + [1.0] * 7
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(days=i) for i in range(14)],
            "symbol": ["AAPL"] * 14,
            "sec_id": [1] * 7 + [2] * 7,
            "close": closes,
            "adj_factor": factors,
            "volume": [1_000_000 + i for i in range(14)],
        }
    )


def test_underlying_horizons_use_adjusted_returns_and_restart_at_the_identity() -> None:
    compute = _load_notebook_functions("underlying_features")["underlying_features"]
    prices = _identity_panel()
    features = compute(prices)
    expected = reconcile_underlying_log_returns(prices).with_columns(
        pl.col("clean_log_return")
        .rolling_sum(5, min_samples=5)
        .over(SEGMENT)
        .exp()
        .sub(1)
        .alias("expected_ret_5d")
    )
    comparison = features.join(
        expected.select("timestamp", "symbol", "expected_ret_5d"), on=["timestamp", "symbol"]
    )
    positions = features.with_columns(
        pl.col("timestamp").cum_count().over(SEGMENT).alias("position")
    )

    assert np.allclose(
        comparison["ret_5d"].drop_nulls(), comparison["expected_ret_5d"].drop_nulls()
    )
    # A split changes the quote but not the return, so the 4x factor leaves no jump behind.
    assert features["ret_1d"].drop_nulls().abs().max() < 0.05
    # The first observation of a new security is not a return, and each horizon warms up again.
    assert features.filter(pl.col("identity_boundary")).select("ret_1d", "ret_5d", "rv_5d").null_count().row(0) == (1, 1, 1)  # fmt: skip
    assert positions.filter(pl.col("position") <= 5)["ret_5d"].null_count() == 10


def _session_panel(quoted_positions: set[int], length: int = 400) -> pl.DataFrame:
    """One security on a dense session grid, quoted only on *quoted_positions*."""
    start = date(2018, 1, 1)
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(days=i) for i in range(length)],
            "symbol": ["SAME"] * length,
            "sec_id": [1] * length,
            "instrument_id": [
                "straddle_30d_atm" if i in quoted_positions else None for i in range(length)
            ],
            "instr_mid": [10.0 + i / 100 if i in quoted_positions else None for i in range(length)],
            "instr_pct_of_S": [
                0.04 + i / 100_000 if i in quoted_positions else None for i in range(length)
            ],
            "instr_theta": [-0.2 if i in quoted_positions else None for i in range(length)],
            "instr_vega": [0.5 if i in quoted_positions else None for i in range(length)],
            "instr_dte": [30.0 if i in quoted_positions else None for i in range(length)],
            "instr_delta": [0.01 if i in quoted_positions else None for i in range(length)],
            "iv_atm": [0.20 + i / 10_000 if i in quoted_positions else None for i in range(length)],
            "call_iv": [
                0.21 + i / 10_000 if i in quoted_positions else None for i in range(length)
            ],
            "put_iv": [0.19 + i / 10_000 if i in quoted_positions else None for i in range(length)],
            **{
                f"rv_{w}d": [0.17 + w / 1_000 + i / 20_000 for i in range(length)]
                for w in WINDOWS["realized_volatility"]
            },
        }
    )


def _stateful(panel: pl.DataFrame) -> pl.DataFrame:
    functions = _load_notebook_functions(
        "instrument_features",
        "premium_features",
        "dynamics_features",
        "session_zscore",
        "min_observations",
    )
    result = functions["instrument_features"](panel)
    result = functions["premium_features"](result)
    return functions["dynamics_features"](result)


def test_a_shift_spans_sessions_and_not_quotes() -> None:
    """The defect this stage was rebuilt for: a gap in the quotes must null the window."""
    # Quoted in six-session bursts every nineteen sessions, which is how a real symbol on the
    # monthly expiry cycle appears. No session has a quote exactly ten sessions earlier.
    burst = {i for i in range(400) if i % 19 < 6}
    result = _stateful(_session_panel(burst))

    # No session in a six-long burst has a quote ten sessions back, in it or in the one before.
    assert result["iv_mom_10d"].null_count() == len(result)
    # The five-session shift reaches the start of its own burst, and only that: one value per
    # complete burst, against the 395 a fully quoted grid would carry.
    complete_bursts = sum(1 for start in range(0, 400, 19) if start + 5 < 400)
    assert result["iv_mom_5d"].drop_nulls().len() == complete_bursts == 21
    assert result["instr_ret_1d"].drop_nulls().len() > 0

    dense = _stateful(_session_panel(set(range(400))))
    assert dense["iv_mom_10d"].drop_nulls().len() == 400 - 10
    assert dense["iv_mom_5d"].drop_nulls().len() == 400 - 5


def test_the_zscore_needs_the_configured_share_of_the_window_quoted() -> None:
    dense = _stateful(_session_panel(set(range(400))))
    positions = dense.with_columns(pl.col("timestamp").cum_count().over(SEGMENT).alias("position"))
    required = round(WINDOWS["iv_zscore"][0] * MIN_OBS)

    # Even on a fully quoted grid the rule, not the window, sets the first session that can
    # carry a value: 50 of the 63, so the value appears at position 50 and not at 63.
    assert positions.filter(pl.col("position") < required)["iv_atm_z_63"].null_count() == required - 1  # fmt: skip
    assert positions.filter(pl.col("position") == required)["iv_atm_z_63"].null_count() == 0

    # Drop every fourth session: 75% quoted is below the configured 80% rule, so a 63-session
    # window never holds enough and the z-score is null throughout.
    thin = _stateful(_session_panel({i for i in range(400) if i % 4}))
    assert thin["iv_atm_z_63"].null_count() == len(thin)
    assert required == 50


def test_a_prior_security_cannot_change_the_next_security_features() -> None:
    panel = pl.concat(
        [
            _session_panel(set(range(300)), length=300),
            _session_panel(set(range(300)), length=300).with_columns(
                pl.col("sec_id") + 1,
                pl.col("timestamp") + timedelta(days=300),
            ),
        ]
    )
    stateful = ["instr_ret_1d", "instr_ret_5d", "vrp_zscore_252", "iv_atm_z_63", "iv_mom_5d"]
    baseline = _stateful(panel)
    perturbed = _stateful(
        panel.with_columns(
            pl.when(pl.col("sec_id") == 1)
            .then(pl.col("iv_atm") + 2)
            .otherwise(pl.col("iv_atm"))
            .alias("iv_atm"),
            pl.when(pl.col("sec_id") == 1)
            .then(pl.col("instr_mid") * 100)
            .otherwise(pl.col("instr_mid"))
            .alias("instr_mid"),
        )
    )

    assert (
        baseline.filter(pl.col("sec_id") == 2)
        .select(stateful)
        .equals(perturbed.filter(pl.col("sec_id") == 2).select(stateful))
    )


def test_appending_a_later_security_cannot_change_the_existing_prefix() -> None:
    first = _session_panel(set(range(300)), length=300)
    later = first.with_columns(pl.col("sec_id") + 1, pl.col("timestamp") + timedelta(days=300))
    stateful = ["instr_ret_5d", "vrp_zscore_252", "iv_atm_z_63", "iv_mom_5d"]

    baseline = _stateful(first)
    extended = _stateful(pl.concat([first, later]))

    assert baseline.select("timestamp", *stateful).equals(
        extended.filter(pl.col("sec_id") == 1).select("timestamp", *stateful)
    )


def test_security_identity_join_is_unique_complete_and_fail_loud() -> None:
    attach = _load_notebook_functions("attach_security_identity")["attach_security_identity"]
    straddles = pl.DataFrame(
        {
            "timestamp": [date(2020, 1, 1), date(2020, 1, 2)],
            "symbol": ["SAME", "SAME"],
            "instr_mid": [1.0, 2.0],
        }
    )
    identity = pl.DataFrame(
        {
            "timestamp": [date(2020, 1, 1), date(2020, 1, 2)],
            "symbol": ["SAME", "SAME"],
            "sec_id": [1, 2],
        }
    )

    assert attach(straddles, identity)["sec_id"].to_list() == [1, 2]
    with pytest.raises(ValueError, match="not unique"):
        attach(straddles, pl.concat([identity, identity.head(1)]))
    with pytest.raises(ValueError, match="lack contemporaneous"):
        attach(straddles, identity.head(1))


def test_notebook_binds_every_window_from_the_configuration() -> None:
    """No window, threshold or ranked column may be retyped in the notebook."""
    source = NOTEBOOK.read_text()
    body = source.split('# %% tags=["parameters"]', 1)[1]
    # Only the code: prose may name a window when it is explaining one.
    code = "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("#"))

    assert 'WINDOWS = FEATURES["windows"]' in source
    assert 'MIN_OBS = FEATURES["min_observations_fraction"]' in source
    assert 'NULL_POLICY = list(FEATURES["null_policy"])' in source
    # The window lengths themselves must not appear as literals anywhere in the code.
    for literal in ("252,", "rolling_mean(63", "shift(21)", "shift(10)", "/ 30.0"):
        assert literal not in code, f"{literal!r} is retyped rather than bound from setup.yaml"


def test_notebook_does_not_run_the_evaluation_screen_this_stage_does_not_own() -> None:
    """The IC / HAC / FDR screen belongs to 05_evaluation, which runs it fold-aware."""
    source = NOTEBOOK.read_text()

    for forbidden in ("benjamini_hochberg_fdr", "compute_ic_hac_stats", "spearmanr", "ic_mean"):
        assert forbidden not in source


def test_notebook_code_cells_respect_publication_line_limit() -> None:
    source = NOTEBOOK.read_text()
    code_cells = [
        cell for cell in source.split("# %%")[1:] if not cell.lstrip().startswith("[markdown]")
    ]
    oversized = [
        len(cell.rstrip().splitlines())
        for cell in code_cells
        if len(cell.rstrip().splitlines()) > 40
    ]

    assert oversized == []
