"""A price panel narrower than the predictions is reported, not acted on.

ml4t/agent-workspace#911. The vectorized path computes `gross_ret = weight *
y_true` from the predictions frame and reads `prices` only for the rebalance
calendar - the same set of decision dates whichever symbols are in the panel. So
a preview taken at a reduced `MAX_SYMBOLS` was the production sweep at the
production cost per backtest: measured on us_firm_characteristics/11_backtest,
32 backtests at 300 symbols and at 3,708 agreed in every column to six decimals.

Two ways of acting on it inside the runner are wrong, and each was tried.
Narrowing the predictions to the panel unasked makes the traded universe decide
the portfolio without entering the backtest identity - the caller hashes its
specification before this module sees the run, so a reduced preview would be
served the full-universe result. Refusing the run stops a preview that is
legitimately configured this way: the CI `us_firm_characteristics` fixture holds
a 5-symbol panel against 20-symbol predictions, and refusing took four notebooks
down.

What closes it is the caller declaring the universe it trades, before it hashes.
`traded_universe_declaration` builds that declaration from the panel and
`build_backtest_spec(traded_universe=...)` puts it in `strategy.signal`, which is
hashed whole; `apply_traded_universe` reads it back in the runner, checks the
panel against it, and narrows the predictions to it. A caller that declares
nothing gets the old behaviour in every respect, which is what leaves every
registered backtest at the identity it was written under - the first two groups
of tests below hold both halves of that.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

SPEC = {
    "version": 2,
    "strategy": {
        "signal": {"method": "equal_weight_top_k", "top_k": 2, "long_short": False},
        "rebalance": {"mode": "vectorized", "cadence": "daily", "step": 1},
    },
    "backtest_config": {
        "cash": {"initial": 100_000.0},
        "commission": {"model": "percentage", "rate": 0.0},
        "slippage": {"model": "percentage", "rate": 0.0},
        "account": {"allow_short_selling": False},
    },
}


def _predictions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1)] * 4,
            "symbol": ["A", "B", "C", "D"],
            "y_score": [0.9, 0.8, 0.7, 0.6],
            "y_true": [0.1, 0.2, 0.3, 0.4],
        }
    )


def _prices(symbols: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1)] * len(symbols),
            "symbol": symbols,
            "close": [100.0] * len(symbols),
        }
    )


def _run(monkeypatch, prices, **kwargs) -> dict:
    """Call run_backtest and capture what the vectorized path was handed."""
    import case_studies.utils.backtest_runner as br
    import case_studies.utils.conformal as conformal

    captured: dict = {}

    monkeypatch.setattr(br, "get_backtest_config", lambda _: object())
    monkeypatch.setattr(br, "ensure_backtest_spec", lambda *args, **kw: args[2])
    monkeypatch.setattr(conformal, "ensure_conformal_calibration_identity", lambda spec: spec)
    monkeypatch.setattr(br, "substitute_continuous_return_for_classification", lambda p, *_: p)

    def fake_vectorized(**kw):
        captured.update(kw)
        return {
            "daily_returns": pl.DataFrame(
                {"timestamp": [datetime(2024, 1, 1)], "daily_return": [0.0]}
            ),
            "metrics": {"sharpe": 0.0},
        }

    monkeypatch.setattr(br, "_run_vectorized", fake_vectorized)

    br.run_backtest(
        "us_firm_characteristics",
        "pred1",
        SPEC,
        prices=prices,
        predictions=_predictions(),
        register=False,
        **kwargs,
    )
    return captured


def test_a_panel_that_cannot_price_the_predictions_says_so(monkeypatch) -> None:
    """The defect: the panel was reduced and the sweep ran over all four names."""
    with pytest.warns(UserWarning) as record:
        captured = _run(monkeypatch, _prices(["A", "C"]))
    message = str(record[0].message)
    assert "2 of which it cannot price" in message
    # Names the knob that does reduce this stage, so the reader is not left to guess.
    assert "TOP_N_PREDICTIONS" in message
    # And says it without changing the run: the identity the caller already hashed
    # describes a sweep over all four names, and that is what it gets.
    assert captured["predictions"]["symbol"].to_list() == ["A", "B", "C", "D"]


def test_a_covering_panel_is_silent(monkeypatch) -> None:
    """The control: nothing is reported when the panel carries every predicted name."""
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", UserWarning)
        captured = _run(monkeypatch, _prices(["A", "B", "C", "D", "E"]))
    assert captured["predictions"]["symbol"].to_list() == ["A", "B", "C", "D"]
    assert sorted(captured["weights"]["symbol"].to_list()) == ["A", "B"]


def test_a_precomputed_allocation_is_reported_on_too(monkeypatch) -> None:
    """The Ch19 risk sweep reads y_true from the same predictions frame."""
    weights = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1)] * 4,
            "symbol": ["A", "B", "C", "D"],
            "weight": [0.25, 0.25, 0.25, 0.25],
        }
    )
    with pytest.warns(UserWarning, match="cannot price"):
        _run(monkeypatch, _prices(["A", "C"]), precomputed_weights=weights)


def test_an_empty_panel_says_nothing() -> None:
    """No panel is not a reduction to zero; the engine's own guards cover it."""
    import warnings as _warnings

    from case_studies.utils.backtest_runner import warn_if_the_panel_does_not_bound_the_universe

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", UserWarning)
        warn_if_the_panel_does_not_bound_the_universe(
            _predictions(), pl.DataFrame(), case_study="demo", label="fwd_ret_1m"
        )


def test_the_htm_option_path_keeps_its_own_universe(monkeypatch) -> None:
    """sp500_options/ret_to_expiry indexes prices and predictions differently."""
    import case_studies.utils.backtest_runner as br
    import case_studies.utils.conformal as conformal

    captured: dict = {}
    monkeypatch.setattr(br, "get_backtest_config", lambda _: object())
    monkeypatch.setattr(br, "ensure_backtest_spec", lambda *args, **kw: args[2])
    monkeypatch.setattr(conformal, "ensure_conformal_calibration_identity", lambda spec: spec)
    monkeypatch.setattr(br, "substitute_continuous_return_for_classification", lambda p, *_: p)

    def fake_htm(**kw):
        captured.update(kw)
        return {
            "daily_returns": pl.DataFrame(
                {"timestamp": [datetime(2024, 1, 1)], "daily_return": [0.0]}
            ),
            "metrics": {"sharpe": 0.0},
        }

    monkeypatch.setattr(br, "_run_htm_daily_mtm", fake_htm)
    monkeypatch.setattr(br, "declared_rebalance_step", lambda *_: None)
    br.run_backtest(
        "sp500_options",
        "pred1",
        SPEC,
        prices=_prices(["X"]),
        predictions=_predictions(),
        label="ret_to_expiry",
        register=False,
    )
    assert captured["predictions"]["symbol"].to_list() == ["A", "B", "C", "D"]


def test_the_notebooks_blanket_warning_filter_does_not_hide_it(capsys) -> None:
    """Eleven backtest notebooks ignore warnings at import; the reader still sees this.

    `us_firm_characteristics/11_backtest.py:62` and ten siblings call
    `warnings.filterwarnings("ignore")` before the first backtest runs, so a
    diagnostic that only warns reaches nobody who reads the executed notebook.
    This asserts the message survives that filter, which is the condition the
    notebook actually runs under - `pytest.warns` in the tests above installs its
    own filter and cannot see the difference.
    """
    import warnings as _warnings

    from case_studies.utils.backtest_runner import warn_if_the_panel_does_not_bound_the_universe

    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore")
        warn_if_the_panel_does_not_bound_the_universe(
            _predictions(), _prices(["A", "B"]), case_study="filtered", label="fwd_ret_1m"
        )
    assert "cannot price" in capsys.readouterr().out


def test_a_sweep_over_one_prediction_set_says_it_once(capsys) -> None:
    """A twelve-scheme sweep calls `run_backtest` twelve times over one panel.

    Each call sees the same panel and the same predictions, so twelve copies of
    one diagnostic would bury the cell output it is printed into. The case study
    name here is distinct from every other test in this file, so what is counted
    is this loop and not a report some earlier test already made.
    """
    import warnings as _warnings

    from case_studies.utils.backtest_runner import warn_if_the_panel_does_not_bound_the_universe

    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore")
        for _ in range(12):
            warn_if_the_panel_does_not_bound_the_universe(
                _predictions(), _prices(["A", "B"]), case_study="swept", label="fwd_ret_1m"
            )
    assert capsys.readouterr().out.count("cannot price") == 1


# ---------------------------------------------------------------------------
# The declaration: what a reduced run says about itself, and what that changes.
# ---------------------------------------------------------------------------


def _declaration(symbols: list[str]) -> dict:
    from case_studies.utils.backtest_presets import traded_universe_declaration

    return traded_universe_declaration(_prices(symbols))


def _declared_spec(symbols: list[str]) -> dict:
    from copy import deepcopy

    spec = deepcopy(SPEC)
    spec["strategy"]["signal"]["traded_universe"] = _declaration(symbols)
    return spec


def test_a_declared_universe_narrows_the_run_to_the_panel(monkeypatch) -> None:
    """The reduction reduces: two names priced, two names traded, not four."""
    import warnings as _warnings
    from copy import deepcopy

    import case_studies.utils.backtest_runner as br

    spec = _declared_spec(["A", "C"])
    captured: dict = {}
    monkeypatch.setattr(br, "get_backtest_config", lambda _: object())
    monkeypatch.setattr(br, "ensure_backtest_spec", lambda *args, **kw: args[2])
    monkeypatch.setattr(br, "substitute_continuous_return_for_classification", lambda p, *_: p)
    import case_studies.utils.conformal as conformal

    monkeypatch.setattr(conformal, "ensure_conformal_calibration_identity", lambda s: s)

    def fake_vectorized(**kw):
        captured.update(kw)
        return {
            "daily_returns": pl.DataFrame(
                {"timestamp": [datetime(2024, 1, 1)], "daily_return": [0.0]}
            ),
            "metrics": {"sharpe": 0.0},
        }

    monkeypatch.setattr(br, "_run_vectorized", fake_vectorized)
    with _warnings.catch_warnings():
        # And the warning goes quiet, because the panel now does bound the run. Leaving it
        # would tell the reader a reduced run is inert at exactly the point it stopped being.
        _warnings.simplefilter("error", UserWarning)
        br.run_backtest(
            "us_firm_characteristics",
            "pred1",
            deepcopy(spec),
            prices=_prices(["A", "C"]),
            predictions=_predictions(),
            register=False,
        )
    assert captured["predictions"]["symbol"].to_list() == ["A", "C"]


def test_a_reduced_run_does_not_hash_like_the_full_run() -> None:
    """The identity half. Without this the reduced run is served the full run's result."""
    from case_studies.utils.registry.specs import backtest_hash_from_parts

    full = backtest_hash_from_parts("pred1", SPEC)
    reduced = backtest_hash_from_parts("pred1", _declared_spec(["A", "C"]))
    assert full != reduced


def test_two_universes_of_the_same_size_do_not_hash_alike() -> None:
    """A symbol count is not a universe: {A, C} and {A, D} are different portfolios."""
    from case_studies.utils.registry.specs import backtest_hash_from_parts

    assert _declaration(["A", "C"])["n_symbols"] == _declaration(["A", "D"])["n_symbols"] == 2
    assert backtest_hash_from_parts("pred1", _declared_spec(["A", "C"])) != (
        backtest_hash_from_parts("pred1", _declared_spec(["A", "D"]))
    )


def test_a_full_run_hashes_exactly_as_it_did_before() -> None:
    """The compatibility half, and the reason the key is emitted only on request.

    `build_backtest_spec(traded_universe=None)` must produce the spec it produced before the
    parameter existed, or all 21,117 registered backtests re-key and every sweep recomputes
    from scratch.
    """
    from copy import deepcopy

    import case_studies.utils.backtest_loaders as bl
    from case_studies.utils.backtest_presets import build_backtest_spec
    from case_studies.utils.registry.specs import backtest_hash_from_parts

    config = bl.get_backtest_config("us_firm_characteristics")
    prices = _prices(["A", "B", "C", "D"])
    kwargs = dict(
        prices=prices,
        prediction_hash="pred1",
        initial_cash=100_000.0,
        signal={"method": "equal_weight_top_k", "top_k": 2, "long_short": False},
        label=config.primary_label,
    )
    without = build_backtest_spec("us_firm_characteristics", config, **deepcopy(kwargs))
    explicit_none = build_backtest_spec(
        "us_firm_characteristics", config, traded_universe=None, **deepcopy(kwargs)
    )
    declared = build_backtest_spec(
        "us_firm_characteristics",
        config,
        traded_universe=_declaration(["A", "C"]),
        **deepcopy(kwargs),
    )
    assert "traded_universe" not in without["strategy"]["signal"]
    assert backtest_hash_from_parts("pred1", without) == backtest_hash_from_parts(
        "pred1", explicit_none
    )
    assert backtest_hash_from_parts("pred1", without) != backtest_hash_from_parts("pred1", declared)


def test_a_panel_that_is_not_the_declared_universe_stops_the_run() -> None:
    """The declaration is checked against the panel, not trusted.

    Registering a portfolio under an identity that describes a different one is the failure
    the key exists to prevent, so a spec and a panel that disagree is a refusal.
    """
    from case_studies.utils.backtest_runner import apply_traded_universe

    signal = {"traded_universe": _declaration(["A", "C"])}
    with pytest.raises(ValueError, match="not the universe this spec declares"):
        apply_traded_universe(
            _predictions(), _prices(["A", "D"]), signal, case_study="us_firm_characteristics"
        )


def test_no_declaration_leaves_the_predictions_alone() -> None:
    from case_studies.utils.backtest_runner import apply_traded_universe

    for signal in ({}, None, {"method": "equal_weight_top_k"}):
        out = apply_traded_universe(
            _predictions(), _prices(["A", "C"]), signal, case_study="us_firm_characteristics"
        )
        assert out["symbol"].to_list() == ["A", "B", "C", "D"]


def test_every_notebook_that_reduces_its_panel_declares_what_it_trades() -> None:
    """The fleet-wide statement, so a new backtest notebook cannot reintroduce this.

    A notebook that passes `MAX_SYMBOLS` to a price loader and then hashes a specification
    has to say which universe that specification is for. Reading the source rather than
    keeping a list is what stops this going stale silently.
    """
    import re
    from pathlib import Path

    from utils.paths import REPO_ROOT

    missing = []
    for path in sorted((Path(REPO_ROOT) / "case_studies").glob("*/[0-9]*.py")):
        text = path.read_text()
        if "max_symbols=MAX_SYMBOLS" not in text or "build_backtest_spec(" not in text:
            continue
        n_calls = len(re.findall(r"build_backtest_spec\(", text))
        n_declared = len(re.findall(r"traded_universe=", text))
        if n_declared < n_calls:
            missing.append(f"{path.parent.name}/{path.stem}: {n_declared} of {n_calls} calls")
    assert not missing, (
        "backtest specs built from a reducible panel with no universe declared: " + str(missing)
    )


def test_the_reduced_and_the_full_run_now_disagree(monkeypatch) -> None:
    """The isolating measurement this issue was left open for.

    Its recorded reproduction backtested predictions that were themselves produced at the
    reduced width, so it could not separate `MAX_SYMBOLS` reducing the backtest from the
    predictions already being narrow. Here one prediction set over four names is backtested
    twice through the real vectorized path: once against the full panel and once against a
    two-name panel the spec declares. `top_k=2` picks A and B from the full cross-section
    and C and D from the declared one, so the two runs hold different portfolios and report
    different returns - which is what `MAX_SYMBOLS` was supposed to do and did not.
    """
    from copy import deepcopy

    import case_studies.utils.backtest_runner as br
    import case_studies.utils.conformal as conformal

    monkeypatch.setattr(br, "get_backtest_config", lambda _: object())
    monkeypatch.setattr(br, "ensure_backtest_spec", lambda *args, **kw: args[2])
    monkeypatch.setattr(conformal, "ensure_conformal_calibration_identity", lambda s: s)
    monkeypatch.setattr(br, "substitute_continuous_return_for_classification", lambda p, *_: p)

    def run(spec, panel):
        return br.run_backtest(
            "us_firm_characteristics",
            "pred1",
            deepcopy(spec),
            prices=_prices(panel),
            predictions=_predictions(),
            label="fwd_ret_21d",
            register=False,
        )

    full = run(SPEC, ["A", "B", "C", "D"])
    reduced = run(_declared_spec(["C", "D"]), ["C", "D"])

    full_ret = full.daily_returns["daily_return"].to_list()
    reduced_ret = reduced.daily_returns["daily_return"].to_list()
    # y_true is 0.1/0.2 for A/B against 0.3/0.4 for C/D, so the two portfolios cannot agree.
    assert full_ret != reduced_ret, (full_ret, reduced_ret)
    assert sorted(full.weights["symbol"].to_list()) == ["A", "B"]
    assert sorted(reduced.weights["symbol"].to_list()) == ["C", "D"]
    # `register=False` returns no hash, so the identity half is stated on the specs the two
    # runs were built from - the same pair the caller hashes before deciding what to skip.
    from case_studies.utils.registry.specs import backtest_hash_from_parts

    assert backtest_hash_from_parts("pred1", SPEC) != backtest_hash_from_parts(
        "pred1", _declared_spec(["C", "D"])
    )


# ---------------------------------------------------------------------------
# What a distinct identity lets in: the two now coexist, so nothing downstream
# may rank one against the other.
# ---------------------------------------------------------------------------


def test_precomputed_and_ordinary_execution_agree_on_a_reduced_book(monkeypatch) -> None:
    """The Ch19 risk sweep brings its own weights and skips weight construction.

    Both paths have to narrow at the same point, and that point is before ranking. Narrowing
    the finished weights instead is a different operation and a worse one: `top_k=2` over the
    full cross-section picks A and B at half each, and dropping B from that leaves a book half
    in cash - an overlay holding a different portfolio from its own parent, under one
    identity. Narrowing first picks A and C at half each, which is what ordinary execution
    does.
    """
    from copy import deepcopy

    import case_studies.utils.backtest_runner as br
    import case_studies.utils.conformal as conformal

    spec = _declared_spec(["A", "C"])
    prices = _prices(["A", "C"])

    precomputed = br.precompute_weights(
        _predictions(), deepcopy(spec), prices, case_study="us_firm_characteristics"
    )

    captured: dict = {}
    monkeypatch.setattr(br, "get_backtest_config", lambda _: object())
    monkeypatch.setattr(br, "ensure_backtest_spec", lambda *args, **kw: args[2])
    monkeypatch.setattr(conformal, "ensure_conformal_calibration_identity", lambda s: s)
    monkeypatch.setattr(br, "substitute_continuous_return_for_classification", lambda p, *_: p)

    def fake_vectorized(**kw):
        captured.update(kw)
        return {
            "daily_returns": pl.DataFrame(
                {"timestamp": [datetime(2024, 1, 1)], "daily_return": [0.0]}
            ),
            "metrics": {"sharpe": 0.0},
        }

    monkeypatch.setattr(br, "_run_vectorized", fake_vectorized)
    br.run_backtest(
        "us_firm_characteristics",
        "pred1",
        deepcopy(spec),
        prices=prices,
        predictions=_predictions(),
        register=False,
    )
    ordinary = captured["weights"]

    key = ["timestamp", "symbol"]
    assert sorted(precomputed["symbol"].to_list()) == ["A", "C"]
    assert (
        precomputed.sort(key)
        .select(key + ["weight"])
        .equals(ordinary.sort(key).select(key + ["weight"]))
    ), (precomputed.sort(key).to_dicts(), ordinary.sort(key).to_dicts())
    # Fully invested, which is the half that catches narrowing after allocation.
    assert precomputed["weight"].sum() == pytest.approx(ordinary["weight"].sum())


def test_a_reduced_run_is_refused_on_the_canonical_tier() -> None:
    """Coexistence is made impossible rather than filtered for.

    A reduced run now has a backtest identity of its own, so where it used to be skipped as
    already-done it would register a row beside the full one. Nothing downstream distinguishes
    them: `resolve_best_predictions` takes MAX(sharpe) over every backtest of a prediction and
    `resolve_best_backtest_runs` the top Sharpe at a stage, and a Sharpe earned over a handful
    of names would advance a configuration ahead of one earned over the whole panel.

    Filtering at those ten call sites cannot have one right default - excluding reduced rows is
    correct for the canonical registry and empties a preview workspace, where every row is
    reduced. So the two are never allowed into one registry instead: a reduced run is a preview
    run, and every notebook that hashes a specification off a reducible panel says so. The
    refusal is read out of the source by `canonically_refused_parameters`, which is what makes
    the canonical fixture path drop the name rather than raise on it.
    """
    import re
    from pathlib import Path

    from tests.pm_helpers import canonically_refused_parameters
    from utils.paths import REPO_ROOT

    missing = []
    for path in sorted((Path(REPO_ROOT) / "case_studies").glob("*/[0-9]*.py")):
        text = path.read_text()
        if "max_symbols=MAX_SYMBOLS" not in text or "build_backtest_spec(" not in text:
            continue
        if "MAX_SYMBOLS" not in canonically_refused_parameters(path):
            missing.append(f"{path.parent.name}/{path.stem}")
        n_calls = len(re.findall(r"build_backtest_spec\(", text))
        if len(re.findall(r"traded_universe=", text)) < n_calls:
            missing.append(f"{path.parent.name}/{path.stem} (undeclared universe)")
    assert not missing, (
        "these hash a specification off a reducible panel without refusing the reduction "
        f"canonically, so a reduced row could reach the canonical registry: {missing}"
    )
