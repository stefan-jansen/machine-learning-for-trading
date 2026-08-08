"""Regression tests for the price bases used by the S&P 500 options labels."""

from __future__ import annotations

import ast
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from case_studies.sp500_options._underlying_returns import reconcile_underlying_log_returns

NOTEBOOK = Path(__file__).parents[1] / "case_studies" / "sp500_options" / "02_labels.py"


def _assignment_nodes(*targets: str) -> list[ast.stmt]:
    tree = ast.parse(NOTEBOOK.read_text())
    selected = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if names.intersection(targets):
            selected.append(node)
    return selected


def _run_assignments(targets: tuple[str, ...], namespace: dict[str, object]) -> None:
    module = ast.Module(body=_assignment_nodes(*targets), type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, NOTEBOOK, "exec"), namespace)


def test_vrp_rv_respects_splits_security_boundaries_and_segment_scale() -> None:
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 10), eager=True)
    index = np.arange(len(dates))
    first_segment = index < 35
    adjusted = np.where(
        first_segment,
        np.linspace(100.0, 103.4, len(dates)),
        np.linspace(200.0, 203.4, len(dates)),
    )
    split = index >= 20
    raw_close = np.where(first_segment & split, adjusted / 2, adjusted)
    adj_factor = np.where(first_segment & split, 2.0, 1.0)
    bars = pl.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["SPLT"] * len(dates),
            "sec_id": np.where(first_segment, 1, 2),
            "close": raw_close,
            "adj_factor": adj_factor,
        }
    )
    straddles = pl.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["SPLT"] * len(dates),
            "iv_atm": [0.25] * len(dates),
        }
    )
    labels = pl.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["SPLT"] * len(dates),
            "ret_to_expiry": np.linspace(-0.5, 0.5, len(dates)),
        }
    )

    def run_baseline(prices: pl.DataFrame) -> dict[str, object]:
        namespace = {
            "np": np,
            "pl": pl,
            "reconcile_underlying_log_returns": reconcile_underlying_log_returns,
            "load_sp500_daily_bars": lambda: prices,
            "load_sp500_options_straddles": lambda: straddles,
            "PRIMARY_LABEL": "ret_to_expiry",
            "RV_WINDOW": 21,
            "SESSIONS_PER_YEAR": 252,
            "dev": {"ret_to_expiry": labels},
        }
        _run_assignments(
            ("straddles", "underlying", "dense", "RV_COL", "annualised_rv", "realised", "baseline"),
            namespace,
        )
        return namespace

    baseline = run_baseline(bars)
    scaled = run_baseline(
        bars.with_columns(
            pl.when(pl.col("sec_id") == 2)
            .then(pl.col("close") * 11)
            .otherwise(pl.col("close"))
            .alias("close")
        )
    )
    baseline_rv = baseline["realised"]
    boundaries = baseline_rv.filter(pl.col("identity_boundary"))
    baseline_vrp = baseline["baseline"].sort("timestamp")
    scaled_vrp = scaled["baseline"].sort("timestamp")

    assert boundaries.height == 1
    assert boundaries["clean_log_return"].null_count() == 1
    assert baseline_vrp.select(["timestamp", "symbol"]).equals(
        scaled_vrp.select(["timestamp", "symbol"])
    )
    np.testing.assert_allclose(baseline_vrp["rv_21d"], scaled_vrp["rv_21d"], atol=1e-12)
    assert baseline_vrp["rv_21d"].max() < 0.01


def test_expiry_intrinsic_value_keeps_historical_close_basis() -> None:
    """Settlement reads the unadjusted close, on the same basis as the listed strike.

    `adj_factor` is 4 here, so a label built from the split-adjusted close would price the
    intrinsic value at a quarter of the strike's basis and report a profit where there is
    a loss. Running the notebook's own `panel` assignments rather than a transcription of
    them is what makes the check bind to the shipped formula.
    """
    expiration = date(2020, 8, 31)
    signal = date(2020, 8, 3)
    bars = pl.DataFrame(
        {
            "timestamp": [expiration],
            "symbol": ["SPLT"],
            "close": [52.0],
            "adj_factor": [4.0],
        }
    )
    contract_returns = pl.DataFrame(
        {
            "feature_date": [signal],
            "expiration": [expiration],
            "symbol": ["SPLT"],
            "strike": [50.0],
            "entry_straddle_mid": [10.0],
        }
    )
    calendar = pl.DataFrame({"timestamp": [signal, expiration], "_bar": [0, 21]})
    namespace = {
        "pl": pl,
        "contract_returns": contract_returns,
        "calendar": calendar,
        "underlying": bars,
        "hedge_path": None,
        "accrued_hedge_pnl": lambda _: pl.DataFrame(
            {"timestamp": [signal], "symbol": ["SPLT"], "hedge_pnl_5d": [0.0]}
        ),
        "PRIMARY_LABEL": "ret_to_expiry",
    }

    _run_assignments(("panel", "settlement", "expiry_bar"), namespace)
    result = namespace["panel"]

    assert result["ret_to_expiry"].item() == 0.8
    assert result["dte_calendar"].item() == 28
    # Entry is one session after the signal, so 21 sessions to expiry is 20 of exposure.
    assert result["window"].item() == 20


def test_rv_window_is_counted_on_the_market_calendar_not_on_quoted_rows() -> None:
    """A session the market was open for and the stock missed must null the window.

    Counting the window on the stock's own rows closes over the absence, so a 21-session
    volatility spans 21 rows that cover more than 21 sessions and nothing says so. The
    two symbols are what makes the check bind: `FULL` trades every session and keeps its
    value on the session `GAPS` misses, so a fix that simply nulled more would fail here.
    """
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 20), eager=True)
    n, hole = len(dates), 30
    frames = []
    for symbol in ("FULL", "GAPS"):
        frame = pl.DataFrame(
            {
                "timestamp": dates,
                "symbol": [symbol] * n,
                "sec_id": [1] * n,
                "close": np.linspace(100.0, 110.0, n),
                "adj_factor": [1.0] * n,
            }
        )
        frames.append(
            frame.filter(pl.col("timestamp") != dates[hole]) if symbol == "GAPS" else frame
        )
    bars = pl.concat(frames)

    namespace = {
        "np": np,
        "pl": pl,
        "reconcile_underlying_log_returns": reconcile_underlying_log_returns,
        "load_sp500_daily_bars": lambda: bars,
        "RV_WINDOW": 21,
        "SESSIONS_PER_YEAR": 252,
    }
    _run_assignments(("underlying", "dense", "RV_COL", "annualised_rv", "realised"), namespace)

    assert namespace["dense"].height == 2 * n, "the absent session is not reindexed back in"
    realised = namespace["realised"].sort(["symbol", "timestamp"])
    rv = {
        symbol: group["rv_21d"].to_list()
        for (symbol,), group in realised.group_by(["symbol"], maintain_order=True)
    }

    # The absent session leaves no return on itself and none on the session after it, and
    # the window is 21 returns wide, so every window from the absence to 21 sessions past
    # the session after it is short of an observation and yields nothing.
    assert rv["FULL"][21] is not None and rv["FULL"][hole] is not None
    assert rv["GAPS"][hole - 1] is not None
    assert all(value is None for value in rv["GAPS"][hole : hole + 22])
    assert rv["GAPS"][hole + 22] is not None


def test_notebook_rv_rolls_within_full_security_identity() -> None:
    assignments = _assignment_nodes("annualised_rv", "realised")

    assert len(assignments) == 2
    expression = "".join(ast.dump(node) for node in assignments)
    assert "reconcile_underlying_log_returns" in expression
    assert "clean_log_return" in expression
    assert "symbol" in expression
    assert "sec_id" in expression


def test_baseline_ic_uses_sorted_horizon_aware_hac() -> None:
    tree = ast.parse(NOTEBOOK.read_text())
    hac_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "compute_ic_hac_stats"
    ]

    assert len(hac_calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in hac_calls[0].keywords}
    # The bandwidth is a name bound from the observed windows, not a literal: this label
    # resolves at expiration, so its exposure varies from trade to trade and a constant
    # here would drift from the data the first time the contract calendar moves.
    horizon = keywords["label_horizon"]
    assert isinstance(horizon, ast.Name)
    assert horizon.id == "LONGEST_WINDOW"
    window_bindings = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "LONGEST_WINDOW" for t in node.targets)
    ]
    assert len(window_bindings) == 1
    assert "max" in ast.dump(window_bindings[0])

    sorted_ic_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ic" for target in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "sort"
    ]
    assert len(sorted_ic_assignments) == 1
    sort_call = sorted_ic_assignments[0].value
    assert isinstance(sort_call.args[0], ast.Constant)
    assert sort_call.args[0].value == "timestamp"
