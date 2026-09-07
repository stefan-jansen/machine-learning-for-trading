"""Central sweep configuration for Ch16-20 parametric backtesting.

The Ch16-19 sweep grid (entry schemes, allocators, cost grid, risk controls)
is declared per-case-study under ``backtest.sweep`` in each case study's
``config/setup.yaml``. ``load_sweep(case_study)`` and the ``*_for`` / ``get_*``
helpers read that block and synthesize dispatcher-shaped config dicts. No
implicit fallback — the block is required, and ``KeyError`` is raised when
missing.

Usage:
    from case_studies.utils.sweep_config import (
        get_entry_schemes_for, get_top_k_values_for,
        get_allocators, get_cost_grid_bps,
        get_position_risk_controls, get_portfolio_risk_controls,
    )

    schemes = get_entry_schemes_for(
        "us_firm_characteristics", label="fwd_ret_1m",
        n_assets=2500, long_short=True,
    )
"""

from __future__ import annotations

import polars as pl
import yaml

# ---------------------------------------------------------------------------
# MAE/MFE-calibrated risk controls (Ch19)
# ---------------------------------------------------------------------------


def calibrate_trailing_stops(
    prices: pl.DataFrame,
    horizons: list[int] | None = None,
    percentiles: tuple[float, ...] = (10, 25),
    asset_col: str = "symbol",
    time_col: str = "timestamp",
    price_col: str = "close",
) -> list[dict]:
    """Compute case-study-specific trailing stop thresholds from MAE percentiles.

    Uses the ml4t-diagnostic excursion module to analyze how far prices
    typically draw down over various holding horizons, then converts
    MAE percentiles into trailing stop thresholds.

    Args:
        prices: Long-format DataFrame with [timestamp, symbol, close] (at minimum).
        horizons: Bar horizons to analyze. Default uses [10, 20, 40] for daily data.
        percentiles: MAE percentiles to convert to stops. Default (10, 25) gives
            tight and moderate thresholds.
        asset_col: Name of the asset identifier column.
        time_col: Name of the timestamp column.
        price_col: Name of the close price column.

    Returns:
        List of risk control dicts in the same shape as the declared
        ``backtest.sweep.risk_controls.position`` entries in setup.yaml,
        with names like ``trailing_mae_p10_h20`` (10th percentile MAE at
        20-bar horizon).
    """
    import numpy as np
    from ml4t.diagnostic.evaluation.excursion import analyze_excursions

    if horizons is None:
        horizons = [10, 20, 40]

    # Aggregate across all assets for a universe-level excursion profile
    # Key: (horizon, percentile) → list of per-asset MAE values
    symbols = prices[asset_col].unique().to_list()
    mae_by_hp: dict[tuple[int, float], list[float]] = {
        (h, p): [] for h in horizons for p in percentiles
    }

    for sym in symbols:
        sym_prices = (
            prices.filter(pl.col(asset_col) == sym)
            .sort(time_col)
            .get_column(price_col)
            .drop_nulls()
        )
        if len(sym_prices) < max(horizons) + 10:
            continue

        result = analyze_excursions(
            sym_prices,
            horizons=horizons,
            percentiles=list(percentiles),
        )

        for h in horizons:
            if h not in result.statistics:
                continue
            stats = result.statistics[h]
            for p in percentiles:
                val = stats.mae_percentiles.get(p)
                if val is not None and not np.isnan(val):
                    mae_by_hp[(h, p)].append(abs(val))

    # Build calibrated trailing stop configs from aggregated MAE percentiles
    controls: list[dict] = []
    seen_thresholds: set[float] = set()

    for h in horizons:
        for p in percentiles:
            vals = mae_by_hp[(h, p)]
            if not vals:
                continue
            # Median across all assets gives a robust universe-level threshold
            threshold = float(np.median(vals))
            threshold = round(threshold, 3)
            if threshold <= 0 or threshold in seen_thresholds:
                continue
            seen_thresholds.add(threshold)
            pct_str = f"{threshold * 100:.1f}".replace(".", "p")
            controls.append(
                {
                    "name": f"trailing_mae_p{int(p)}_h{h}_{pct_str}pct",
                    "type": "trailing_stop",
                    "threshold": threshold,
                    "calibration": {"horizon": h, "percentile": p, "source": "mae"},
                }
            )

    # Sort by threshold for readable sweep ordering
    controls.sort(key=lambda c: c["threshold"])
    return controls


# ---------------------------------------------------------------------------
# setup.yaml-driven sweep loader (Ch16-19)
# ---------------------------------------------------------------------------


def load_sweep(case_study: str) -> dict:
    """Return the ``backtest.sweep`` block from a case study's setup.yaml.

    The block declares the Ch16-19 sweep grid (signal selection, allocators,
    cost grid, risk controls) per-case-study. Raises ``KeyError`` if missing —
    there is no fallback to module-level constants. See
    ``case_studies/us_firm_characteristics/config/setup.yaml`` for the schema.
    """
    setup_path = _setup_path(case_study)
    setup = yaml.safe_load(setup_path.read_text())
    sweep = (setup.get("backtest") or {}).get("sweep")
    if sweep is None:
        raise KeyError(
            f"backtest.sweep missing from case_studies/{case_study}/config/"
            "setup.yaml — the Ch16-19 sweep grid must be declared explicitly. "
            "See case_studies/us_firm_characteristics/config/setup.yaml for "
            "the schema."
        )
    return sweep


def _setup_path(case_study: str):
    # Resolve through get_case_study_dir so setup.yaml is read from the same place
    # the model stages read it: the source tree in production, or ML4T_OUTPUT_DIR
    # when set (tests, and the create_experiment workflow where the reader edits
    # backtest.sweep/costs/execution in their isolated experiment copy).
    from utils.paths import get_case_study_dir

    # create=False: reading config must never materialize the case-study dir. A
    # side-effect mkdir under ML4T_OUTPUT_DIR would make create_experiment think
    # the experiment already exists.
    return get_case_study_dir(case_study, create=False) / "config" / "setup.yaml"


def _load_setup(case_study: str) -> dict:
    return yaml.safe_load(_setup_path(case_study).read_text())


def get_declared_long_short(case_study: str) -> bool:
    """Whether a case study's declared selection mode is cross-sectional long-short.

    Delegates to ``get_backtest_config``, which reads it off
    ``mapping.position_state_space``. That is the selection mode, not
    ``account.allow_short_selling``: the account flag is an execution permission, and
    the two come apart. ``sp500_options`` sets the permission because it sells
    straddles, but its state space is ``short_straddle_hedged`` - a one-sided short
    that builds no cross-sectional short sleeve, and its backtest passes
    ``long_short: False`` accordingly. Sizing its concentration grid off the account
    flag would halve a ceiling that nothing halves at run time.
    """
    from case_studies.utils.backtest_loaders import get_backtest_config

    return bool(get_backtest_config(case_study).long_short)


def get_execution_defaults(case_study: str) -> dict:
    """Return the ``execution:`` block from setup.yaml.

    Single source of truth for engine-level defaults — ``initial_cash``,
    ``share_type``, ``allocator_lookback``. Raises ``KeyError`` if missing;
    the Ch16-19 notebooks must not declare local INITIAL_CASH constants.
    """
    setup = _load_setup(case_study)
    block = setup.get("execution")
    if block is None:
        raise KeyError(
            f"execution: block missing from case_studies/{case_study}/config/"
            "setup.yaml — must declare initial_cash, share_type, allocator_lookback."
        )
    return block


def get_allocator_lookback(case_study: str) -> int:
    """Return the CS-level lookback (bars-of-underlying) for moment-based allocators.

    Read from ``setup.yaml::execution.allocator_lookback``. Applied uniformly
    to inverse_vol, risk_parity, hrp, mvo_ledoit_wolf so allocators compete
    on method, not on window choice. Bars count on the price DataFrame the
    allocator consumes (typically daily even when rebalance is monthly).
    """
    exec_block = get_execution_defaults(case_study)
    lb = exec_block.get("allocator_lookback")
    if lb is None:
        raise KeyError(
            f"execution.allocator_lookback missing from case_studies/{case_study}/"
            "config/setup.yaml — required for moment-based allocators."
        )
    return int(lb)


_STAGE_DEFAULTS = {
    "signal": 0,  # 0 = all predictions (equal-weight baseline)
    "allocation": 10,  # top-10 model configs by baseline Sharpe
    "cost_sensitivity": 1,  # top-1 of {signal+allocation} per label
    "risk_overlay": 1,  # top-1 of {signal+allocation} per label
}

_DEFAULT_CHECKPOINTS_PER_CONFIG = 1


def get_top_n_predictions(case_study: str, stage: str) -> int:
    """Return the top-N predictions to feed into ``stage`` from the upstream stage.

    Reads ``backtest.sweep.top_n_predictions[stage]`` with safe fallbacks.
    ``stage`` is one of ``signal | allocation | cost_sensitivity | risk_overlay``.
    Unknown stage names raise ``ValueError`` rather than ``KeyError`` so the stack
    trace clearly distinguishes a typo'd lookup from a missing key in the YAML.

    For ``allocation`` the unit counted is a *model config* — one ``(family,
    config_name)`` pair — not a prediction. See ``get_checkpoints_per_config``
    for how many checkpoints each advancing config contributes.
    """
    if stage not in _STAGE_DEFAULTS:
        raise ValueError(f"unknown stage {stage!r}; expected one of {sorted(_STAGE_DEFAULTS)}")
    block = load_sweep(case_study).get("top_n_predictions") or {}
    return int(block.get(stage, _STAGE_DEFAULTS[stage]))


def get_checkpoints_per_config(case_study: str) -> int:
    """Return how many checkpoints each advancing model config contributes.

    Reads ``backtest.sweep.checkpoints_per_config``. The default of 1 means a
    config enters the allocation sweep through its single best checkpoint, so
    a long checkpoint sweep of one model cannot crowd out other model families.
    """
    value = load_sweep(case_study).get("checkpoints_per_config")
    if value is None:
        return _DEFAULT_CHECKPOINTS_PER_CONFIG
    return int(value)


def get_expensive_allocators_skip(case_study: str) -> bool:
    """Return whether MVO/HRP should be skipped at Ch17 (intraday escape hatch)."""
    return bool(load_sweep(case_study).get("expensive_allocators_skip", False))


def get_cost_grid_half_spread_usd(case_study: str) -> list[float]:
    """Return the half-spread (USD per share) grid for per-share cost-regime sweep.

    Used by Ch18 cost sensitivity for CSes whose declared cost model is
    ``per_share_plus_spread`` (etfs, nasdaq100_microstructure). Returns an
    empty list when the key is absent — the bps grid is the only cost
    dimension for that CS.
    """
    block = load_sweep(case_study).get("cost_grid_half_spread_usd")
    if block is None:
        return []
    return [float(v) for v in block]


def get_per_share_commission(case_study: str, default: float = 0.0035) -> float:
    """Return the per-share commission (USD/share) from ``costs.per_share``.

    Single source of truth for the per-share companion cost regime in Ch18.
    For CSes whose headline cost model is bps (e.g. sp500_equity_option_analytics)
    the ``costs.per_share`` key may be absent; ``default`` (IBKR Pro Tiered top
    tier, $0.0035/share) is returned in that case. Replaces ad-hoc
    ``open(setup.yaml)["costs"]["per_share"]`` reads in cost notebooks.
    """
    costs = _load_setup(case_study).get("costs") or {}
    return float(costs.get("per_share", default))


def get_cadence_sweep(case_study: str) -> list[str]:
    """Return the alternative-cadence list for the Ch18 cadence × cost heatmap.

    Read from ``backtest.sweep.cadence_sweep`` in setup.yaml. Used by CSes
    that explore rebalance-cadence sensitivity (nasdaq100_microstructure).
    Returns an empty list when the key is absent — the CS does not run a
    cadence sweep. Tokens are the engine's cadence vocabulary (e.g.
    ``15_minute``, ``1_hour``).
    """
    block = load_sweep(case_study).get("cadence_sweep")
    if block is None:
        return []
    return [str(c) for c in block]


# --- Calendar-aware allocator lookback resolution ---------------------------

# Calendar tokens we recognize in setup.yaml allocator entries. The numeric
# multiplier is read from the token prefix (e.g., ``3M`` → 3 × month).
_CALENDAR_UNITS_PER_YEAR = {
    "Y": 1,
    "M": 12,
    "W": 52,
    "D": None,  # special-cased to periods_per_year (handles intraday cadences)
}


def _resolve_calendar_lookback(value, periods_per_year: float) -> int:
    """Translate a ``"3M"``/``"6M"``/``"1Y"`` token into bars-of-underlying.

    ``periods_per_year`` is interpreted as the **price-data bars per year**
    (not the Sharpe-annualization factor — for most CSes the two coincide
    because both equal 252 when underlying prices are daily). The allocator
    consumes the raw ``prices`` DataFrame whose row cadence is the price
    cadence, not the rebalance cadence; rolling-vol windows are measured in
    those rows.

    Cases where the two definitions diverge:
      - crypto_perps_funding: ``evaluation.periods_per_year=365`` is the
        daily-equivalent annualization factor, but the underlying data is
        8-hourly (1095 bars/yr). Override per-CS via a future
        ``evaluation.bars_per_year`` field; until then prefer literal bars
        in setup.yaml for non-daily-underlying CSes.
      - nasdaq100_microstructure: same situation (intraday underlying).

    Integer values pass through unchanged (legacy ``vol_window: 63``).
    """
    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str):
        raise TypeError(
            f"lookback must be int or calendar string, got {type(value).__name__}: {value!r}"
        )
    token = value.strip().upper()
    if not token:
        raise ValueError(f"empty lookback string: {value!r}")
    unit = token[-1]
    if unit not in _CALENDAR_UNITS_PER_YEAR:
        raise ValueError(
            f"unsupported lookback unit {unit!r} in {value!r}; expected one of Y/M/W/D"
        )
    try:
        n = int(token[:-1])
    except ValueError as exc:
        raise ValueError(f"cannot parse lookback prefix in {value!r}") from exc
    units_per_year = _CALENDAR_UNITS_PER_YEAR[unit]
    if units_per_year is None:
        # D → use periods_per_year directly (assumes ppy counts daily bars;
        # intraday CSes that nonetheless evaluate at daily MTM keep ppy=252)
        bars_per_unit = periods_per_year / 252.0
        return max(1, int(round(n * bars_per_unit)))
    bars_per_unit = periods_per_year / units_per_year
    return max(1, int(round(n * bars_per_unit)))


def _periods_per_year_for(case_study: str) -> float:
    setup = _load_setup(case_study)
    ppy = (setup.get("evaluation") or {}).get("periods_per_year")
    if ppy is None:
        raise KeyError(
            f"evaluation.periods_per_year missing from case_studies/{case_study}/"
            "config/setup.yaml — required for calendar-aware allocator lookbacks."
        )
    return float(ppy)


# How many registered prediction sets the width resolver reads before it decides.
# One is enough where they agree and cannot detect it where they do not.
_RANKED_WIDTH_SAMPLE = 8


def ranked_cross_section_width(
    case_study: str, label: str, *, split: str = "validation"
) -> int | None:
    """Distinct symbols in the predictions an entry-scheme sweep will rank.

    ``None`` when nothing is registered yet for ``(label, split)``, when the
    artifacts behind the registered sets are missing, or when the sampled sets do
    not agree on a width. A case study whose model stages have not run has no
    ranked cross-section, and inventing one would refuse a sweep that has not
    been set up wrong.

    The registry is sampled rather than asked about the population a particular
    caller is sweeping, and those are different questions: `cme_futures/13_backtest`
    selects official or preview candidates, and a width taken from an unrelated
    set could remove a concentration that is feasible for the ones actually being
    swept. So the resolver only speaks when the sample is unanimous - up to eight
    registered sets, all reporting the same width - and otherwise returns ``None``
    and leaves the caller's own number standing. A caller that knows its
    population passes ``ranked_width`` and skips this entirely.

    Unanimity is the common case, not a hope: measured 2026-09-07 over six
    registered sets per label in etfs, us_firm_characteristics,
    sp500_equity_option_analytics and us_equities_panel, every label had exactly
    one width - 99, 3,708, 549 and 3,147 respectively.

    Distinct symbols over the window, which is the same count the callers take
    off the price panel, so the comparison in ``get_entry_schemes_for`` is between
    two measurements of the same kind. The number that literally bounds a ranking
    is the per-date cross-section, which is smaller - 88 against 99 for etfs,
    2,032 against 3,708 for us_firm_characteristics - and moving the rule onto it
    would change which concentrations are feasible today. That is a separate
    decision about the rule; this function is about which set the rule reads.
    """
    import sqlite3
    import warnings

    from case_studies.utils.registry.store import _case_dir, _registry_db_path, _run_log_dir

    case_dir = _case_dir(case_study)
    db_path = _registry_db_path(case_dir)
    if not db_path.is_file():
        return None
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
            rows = db.execute(
                """
                SELECT p.prediction_hash
                FROM prediction_sets p
                JOIN training_runs t ON t.training_hash = p.training_hash
                WHERE t.label = ? AND p.split = ?
                LIMIT ?
                """,
                (label, split, _RANKED_WIDTH_SAMPLE),
            ).fetchall()
    except sqlite3.Error:
        return None

    widths: set[int] = set()
    for (prediction_hash,) in rows:
        artifact = _run_log_dir(case_dir) / "predictions" / prediction_hash / "predictions.parquet"
        if not artifact.is_file():
            continue
        try:
            widths.add(
                int(pl.scan_parquet(artifact).select(pl.col("symbol").n_unique()).collect().item())
            )
        except (pl.exceptions.PolarsError, OSError, ValueError):
            return None
    if not widths:
        return None
    if len(widths) > 1:
        warnings.warn(
            f"{case_study}/{label}: registered prediction sets disagree on the ranked "
            f"cross-section ({sorted(widths)}), so the entry-scheme feasibility rule falls "
            "back to the caller's price-panel width. Pass ranked_width= with the width of "
            "the population actually being swept.",
            stacklevel=2,
        )
        return None
    return widths.pop()


def get_entry_schemes_for(
    case_study: str,
    label: str,
    n_assets: int,
    long_short: bool,
    *,
    ranked_width: int | None = None,
) -> list[dict]:
    """Synthesize Ch16 entry schemes for ``(case_study, label)`` from setup.yaml.

    Reads ``backtest.sweep.{top_k_grid, percentile_grid, quantile_grid}``
    keyed by label, produces one scheme dict per (axis × value), filtered for
    feasibility against ``n_assets``. Output dicts match the shape consumed
    by ``Ch16 backtest notebooks`` (one scheme per (axis, value)).

    Quantile schemes (``quintile_long_short`` / ``decile_long_short``) carry
    ``long_short=True`` regardless of the ``long_short`` argument — they are
    inherently long-short by construction. ``long_short`` controls only the
    sign of top-k / percentile schemes.

    When the case study declares a ``backtest.sweep.signal_nasdaq100`` block
    (the nasdaq100 v4 slot-mechanism sweep), schemes from that block are
    appended — see ``get_signal_nasdaq100_schemes_for`` for the cross-product.

    ``n_assets`` is the width of the price panel. What bounds a ranked selection
    is the prediction cross-section, and the two are not the same number: every
    caller measures ``n_assets`` off the panel it loaded, while the ranking runs
    over whatever the fitting stages scored. ``ranked_width`` is that second
    number, resolved from the registry when the caller does not supply it, and
    the feasibility rule below applies to the narrower of the two.

    They agree on `main` today, which is why nothing is currently wrong and why
    the check could not fire on the condition it exists to catch
    (ml4t/agent-workspace#1003). Any change that narrows the fitting stages
    without narrowing the panel - a new preview tier, a per-notebook
    ``max_symbols``, a family that scores a subset - reproduces
    ml4t/agent-workspace#989: twelve backtests with a Sharpe, zero trades, and
    four notebooks failing several stages downstream of the cause.
    """
    sweep = load_sweep(case_study)
    schemes: list[dict] = []
    if ranked_width is None:
        ranked_width = ranked_cross_section_width(case_study, label)
    tradeable = n_assets if ranked_width is None else min(n_assets, ranked_width)

    top_k_by_label = sweep.get("top_k_grid") or {}
    pct_by_label = sweep.get("percentile_grid") or {}
    qnt_by_label = sweep.get("quantile_grid") or {}

    # Strict label gate. If any of the three grids is declared in the YAML
    # but the label appears in none of them, raise — silently returning an
    # empty scheme list lets a typo'd LABEL papermill parameter register
    # zero backtests with no warning (Ch16/13 loops over schemes have no
    # else-clause). The legacy "no backtest.sweep block at all" path is
    # preserved: when all three grids are empty/absent, we fall through and
    # return [] (callers that explicitly opt into the legacy
    # ``get_entry_schemes(...)`` helper keep working).
    any_grid_declared = (
        "top_k_grid" in sweep or "percentile_grid" in sweep or "quantile_grid" in sweep
    )
    label_known = label in top_k_by_label or label in pct_by_label or label in qnt_by_label
    if any_grid_declared and not label_known:
        raise KeyError(
            f"label {label!r} not declared in any of backtest.sweep.{{top_k_grid, "
            f"percentile_grid, quantile_grid}} for case_studies/{case_study}/"
            f"config/setup.yaml; known labels: top_k={sorted(top_k_by_label)}, "
            f"pct={sorted(pct_by_label)}, qnt={sorted(qnt_by_label)}"
        )

    declared_top_k = [int(k) for k in top_k_by_label.get(label, [])]
    top_k_schemes: list[dict] = []
    for k in declared_top_k:
        # k == tradeable holds the whole cross-section = equal-weight benchmark, not
        # a prediction-based portfolio; exclude it to match get_top_k_values_for.
        if k >= tradeable:
            continue
        # Long and short selections must be disjoint. Keep the declared grid and
        # exclude only members that the available cross-section cannot realize.
        if long_short and 2 * k > tradeable:
            continue
        top_k_schemes.append(
            {
                "name": f"ew_top{k}",
                "method": "equal_weight_top_k",
                "top_k": k,
                "long_short": long_short,
            }
        )
    # Raised here rather than left to the caller's empty-list guard, because only
    # this scope holds both numbers. A caller that reports `n_assets` back is
    # reporting the number the check did not use, which is how the original
    # failure presented. The panel-too-narrow case (`tradeable == n_assets`) is
    # left to that guard, which names the right number for it.
    if declared_top_k and not top_k_schemes and tradeable < n_assets:
        raise ValueError(
            f"no concentration declared for {case_study}/{label} can be realized: "
            f"backtest.sweep.top_k_grid declares {sorted(declared_top_k)}, and the "
            f"predictions this sweep ranks carry {tradeable} symbols against a price "
            f"panel of {n_assets}. Every k is at or above the ranked cross-section, so "
            "each would hold all of it rather than select from it, and every backtest "
            "in the sweep would record num_trades = 0 while still reporting a Sharpe. "
            "Narrow the price panel and the fitting stages together, or declare a "
            "smaller concentration."
        )
    schemes.extend(top_k_schemes)

    for p in pct_by_label.get(label, []):
        p = float(p)
        schemes.append(
            {
                "name": f"cs_pct{int(p)}",
                "method": "cross_sectional_percentile",
                "percentile": p,
                "long_short": long_short,
            }
        )

    for n_q in qnt_by_label.get(label, []):
        n_q = int(n_q)
        if n_q == 5:
            name, method = "quintile_ls", "quintile_long_short"
        elif n_q == 10:
            name, method = "decile_ls", "decile_long_short"
        else:
            raise ValueError(
                f"quantile_grid only supports n_quantiles ∈ {{5, 10}}; "
                f"got {n_q} for label {label!r} in case_studies/{case_study}/"
                f"config/setup.yaml. The backtest dispatcher has no method for "
                f"q{n_q}_long_short."
            )
        if n_assets < 2 * n_q:
            continue
        schemes.append(
            {
                "name": name,
                "method": method,
                "n_quantiles": n_q,
                "long_short": True,
            }
        )

    # nasdaq100 v4 slot mechanism — appended when the block is present.
    if "signal_nasdaq100" in sweep:
        schemes.extend(get_signal_nasdaq100_schemes_for(case_study, label, n_assets))

    return schemes


def get_signal_nasdaq100_schemes_for(
    case_study: str,
    label: str,
    n_assets: int,
) -> list[dict]:
    """Expand the ``backtest.sweep.signal_nasdaq100`` block into entry schemes.

    Block shape (all keys required unless noted)::

        signal_nasdaq100:
          selection_method: [slot_persistent_signal_exit, eq_w_topk]
          long_q: [0.90, 0.95, 0.99]      # slot only — entry quantile
          direction: [long_only, long_short]
          max_slots: [5, 10, 20]           # slot only — concurrent holdings
          hold_bars: [8, 16, 32]           # slot only — max-hold backstop
          exit_signal_q: [null, 0.30, ...] # slot only — stay threshold, null disables
          pred_freshness_max_min: 14       # slot only — backward-asof tolerance
          bars_per_day_grid: [14]          # slot only — execution cadence
          top_k_grid: [5, 10, 20]          # eq_w_topk only — top-k holdings
          lookback_days: 21                # slot only — rolling window depth

    Slot mechanism is single-direction (the slot book cannot be both long
    and short simultaneously); slot × long_short combinations are dropped.
    eq_w_topk supports both directions via the canonical long_short axis.

    Schemes carry a ``name`` derived from the cross-product coordinates so
    the registry rows are distinguishable. ``selection_method_config`` keys
    are flattened into the scheme dict so the existing 14_backtest.py loop
    passes them through to ``run_backtest`` unchanged.
    """
    sweep = load_sweep(case_study)
    block = sweep.get("signal_nasdaq100")
    if block is None:
        msg = (
            f"backtest.sweep.signal_nasdaq100 missing from "
            f"case_studies/{case_study}/config/setup.yaml"
        )
        raise KeyError(msg)

    methods = list(block.get("selection_method", []))
    if not methods:
        msg = (
            f"signal_nasdaq100.selection_method must list at least one method "
            f"for case_studies/{case_study}"
        )
        raise ValueError(msg)
    long_qs = [float(v) for v in block.get("long_q", [])]
    directions = list(block.get("direction", []))
    max_slots_grid = [int(v) for v in block.get("max_slots", [])]
    hold_bars_grid = [int(v) for v in block.get("hold_bars", [])]
    # Tolerate a scalar/`null` exit_signal_q (e.g. ``exit_signal_q: null``)
    # instead of raising an opaque ``list(None)`` TypeError far from the config.
    raw_exit = block.get("exit_signal_q", [None])
    exit_qs = list(raw_exit) if isinstance(raw_exit, list) else [raw_exit]
    pred_freshness = block.get("pred_freshness_max_min")
    bpd_grid = [int(v) for v in block.get("bars_per_day_grid", [])]
    top_k_grid = [int(v) for v in block.get("top_k_grid", [])]
    lookback_days = int(block.get("lookback_days", 21))

    # Fail loudly on unknown directions and on missing/typo'd required keys: a
    # YAML typo (e.g. ``max_slot:`` or ``bars_per_day:``) would otherwise leave
    # the corresponding grid empty, silently collapse the cross-product to zero
    # schemes, and register zero backtests — the failure mode the strict-label
    # gate exists to prevent.
    _allowed_dirs = {"long_only", "short_only", "long_short"}
    bad_dirs = [d for d in directions if d not in _allowed_dirs]
    if bad_dirs:
        msg = (
            f"signal_nasdaq100.direction has unknown value(s) {bad_dirs} for "
            f"case_studies/{case_study}; allowed: {sorted(_allowed_dirs)}"
        )
        raise ValueError(msg)
    _required: dict[str, list] = {"direction": directions}
    if "slot_persistent_signal_exit" in methods:
        _required.update(
            long_q=long_qs,
            max_slots=max_slots_grid,
            hold_bars=hold_bars_grid,
            bars_per_day_grid=bpd_grid,
        )
    if "eq_w_topk" in methods:
        _required["top_k_grid"] = top_k_grid
    missing = sorted(k for k, v in _required.items() if not v)
    if missing:
        msg = (
            f"signal_nasdaq100 is missing/empty required key(s) {missing} for "
            f"case_studies/{case_study} given selection_method={methods}; "
            f"check for a YAML typo in the sweep block."
        )
        raise ValueError(msg)

    schemes: list[dict] = []
    for method in methods:
        n_before = len(schemes)
        if method == "slot_persistent_signal_exit":
            for long_q in long_qs:
                for direction in directions:
                    # slot books are single-direction by construction:
                    # long_only/short_only are supported, long_short is dropped.
                    if direction not in ("long_only", "short_only"):
                        continue
                    for max_slots in max_slots_grid:
                        if max_slots >= n_assets:
                            continue
                        for hold_bars in hold_bars_grid:
                            for exit_q in exit_qs:
                                exit_q_norm = None if exit_q is None else float(exit_q)
                                if exit_q_norm is not None and exit_q_norm >= long_q:
                                    continue
                                for bpd in bpd_grid:
                                    eq_tag = (
                                        "noexit"
                                        if exit_q_norm is None
                                        else f"q{int(exit_q_norm * 100):02d}"
                                    )
                                    name = (
                                        f"slot_{direction[0]}_lq{int(long_q * 100):02d}"
                                        f"_s{max_slots}_h{hold_bars}_{eq_tag}_b{bpd}"
                                    )
                                    schemes.append(
                                        {
                                            "name": name,
                                            "method": "slot_persistent_signal_exit",
                                            "long_q": long_q,
                                            "lookback_days": lookback_days,
                                            "bars_per_day": bpd,
                                            "max_slots": max_slots,
                                            "hold_bars": hold_bars,
                                            "exit_signal_q": exit_q_norm,
                                            "pred_freshness_max_min": pred_freshness,
                                            "direction": direction,
                                            "long_short": False,
                                        }
                                    )
        elif method == "eq_w_topk":
            for direction in directions:
                ls = direction == "long_short"
                for top_k in top_k_grid:
                    # Same ceiling as `get_top_k_values_for`, and it has to be
                    # applied here too: this branch sets `long_short` off its own
                    # `direction` axis, so a `k` in `(n_assets // 2, n_assets)`
                    # would register as `ewtopk_ls_k<k>` and run at the clamped
                    # half. `direction` is per-scheme, so use it rather than the
                    # case study's declared mode.
                    if top_k >= n_assets or (ls and 2 * top_k > n_assets):
                        continue
                    dtag = "ls" if ls else direction[0]
                    schemes.append(
                        {
                            "name": f"ewtopk_{dtag}_k{top_k}",
                            "method": "equal_weight_top_k",
                            "top_k": top_k,
                            "long_short": ls,
                            "direction": "long_only" if ls else direction,
                        }
                    )
        else:
            msg = (
                f"signal_nasdaq100.selection_method has unknown method "
                f"{method!r} for case_studies/{case_study}; supported: "
                f"slot_persistent_signal_exit, eq_w_topk"
            )
            raise ValueError(msg)
        # A requested method that expands to zero schemes is the silent-zero
        # failure the validation exists to catch (e.g. a slot-only block with
        # direction=[long_short], which slot drops, or every grid value filtered
        # out by max_slots/top_k >= n_assets). Fail loudly instead.
        if len(schemes) == n_before:
            msg = (
                f"signal_nasdaq100 method {method!r} produced zero schemes for "
                f"case_studies/{case_study} (n_assets={n_assets}); every grid "
                f"combination was filtered out — check direction/max_slots/top_k."
            )
            raise ValueError(msg)
    return schemes


def get_top_k_values_for(
    case_study: str,
    label: str,
    n_assets: int,
    *,
    long_short: bool | None = None,
    ranked_width: int | None = None,
) -> list[int]:
    """Return the top-K grid for ``(case_study, label)`` used by Ch17.

    Filters out k >= n_assets (holding everything is the equal-weight
    benchmark, not a prediction-based portfolio). Raises ``KeyError`` if
    ``backtest.sweep.top_k_grid[label]`` is not declared, and ``ValueError``
    when the filter empties the grid.

    A long-short strategy takes ``k`` names on each side, so its ceiling is half
    the universe, not all of it. ``signals.py`` clamps ``eff_k`` to
    ``n_assets // 2`` when ``long_short`` is set, and ``top_k`` is identity-bearing
    through ``plan_backtests(signal=...)`` - so a ``k`` in the half-open interval
    ``(n_assets // 2, n_assets)`` passes this filter, registers under the declared
    ``k``, and runs at the clamped one. Two such values register distinct identities
    over a byte-identical weight series. ``get_entry_schemes_for`` has excluded that
    interval since it took a ``long_short`` argument; this function is the other half
    of the same grid and has to agree with it.

    ``long_short`` defaults to the case study's declared selection mode - the same
    ``mapping.position_state_space`` that ``get_backtest_config`` reads, which is what
    every caller already passes to ``get_entry_schemes_for`` as ``bt_config.long_short``.
    Not ``account.allow_short_selling``: that is an execution permission, and
    ``sp500_options`` holds it while selecting long-only. Pass it explicitly to override.

    An empty grid is never a legitimate result: the caller multiplies it into
    a sweep size, so zero concentrations means zero backtests, and the sweep
    loop then completes without registering anything while still reporting
    itself done. The downstream risk-overlay notebook is the only thing that
    notices, and it blames the operator for not having run this stage. Raise
    here instead, where the cause - a universe cap smaller than the smallest
    declared k - is still visible.

    ``ranked_width`` is the prediction cross-section, resolved from the registry
    when not supplied, and the filter runs against the narrower of it and
    ``n_assets`` for the reason given under ``get_entry_schemes_for``
    (ml4t/agent-workspace#1003). The two functions are halves of one grid and
    have to filter it the same way or a sweep and its plumbing test disagree
    about which concentrations exist.
    """
    sweep = load_sweep(case_study)
    grid = (sweep.get("top_k_grid") or {}).get(label)
    if grid is None:
        raise KeyError(
            f"backtest.sweep.top_k_grid[{label!r}] not declared in "
            f"case_studies/{case_study}/config/setup.yaml"
        )
    if long_short is None:
        long_short = get_declared_long_short(case_study)
    if ranked_width is None:
        ranked_width = ranked_cross_section_width(case_study, label)
    tradeable = n_assets if ranked_width is None else min(n_assets, ranked_width)
    ceiling = tradeable // 2 if long_short else tradeable
    values = [
        int(k) for k in grid if int(k) < tradeable and not (long_short and 2 * int(k) > tradeable)
    ]
    if not values:
        smallest = min(int(k) for k in grid)
        # What the universe has to hold, which is not the same number in the two
        # cases. Long-short takes `k` names on each side, so it needs `2k`, and
        # saying "raise MAX_SYMBOLS above k" there sends the reader to a cap that
        # cannot fix it - or worse, to a cap the fixture cannot reach at all.
        needed = (
            f"at least {2 * smallest} names for the smallest declared k of {smallest} "
            f"({smallest} long and {smallest} short)"
            if long_short
            else f"more than {smallest} names for the smallest declared k"
        )
        side = f"long-short ceiling ({ceiling})" if long_short else "universe"
        raise ValueError(
            f"top_k_grid[{label!r}] = {list(grid)} is empty after filtering against "
            f"{tradeable} tradeable names for case_studies/{case_study} (price panel "
            f"{n_assets}, ranked cross-section {ranked_width}): every declared k holds "
            f"the whole {side}. A cross-sectional sweep here needs {needed}. Raise the "
            f"universe cap (MAX_SYMBOLS) if the panel has them, widen the panel if it "
            f"does not, or declare a smaller k."
        )
    return values


_MOMENT_ALLOCATORS = {"inverse_vol", "risk_parity", "hrp", "mvo_ledoit_wolf", "mvo"}
_LOOKBACK_KEYS = ("vol_window", "lookback")


def get_allocators(case_study: str) -> list[dict]:
    """Return the Ch17 allocator configs (lookback-injected from setup.yaml).

    Each dict matches the shape consumed by
    ``case_studies.utils.backtest_runner._apply_allocation``:
    ``{"method": str, ...kwargs}``. Common kwargs: ``vol_window``,
    ``lookback``, ``max_weight``.

    Moment-based allocators (inverse_vol, risk_parity, hrp, mvo_ledoit_wolf)
    receive ``vol_window``/``lookback`` from the CS-level
    ``execution.allocator_lookback`` — a single window keeps allocators
    comparable. Calendar-string fallback (``"3M"``/``"6M"``) is still
    supported on individual entries when an override is needed, but the
    standard path is the CS-level lookback.

    The ``name`` key in setup.yaml is human-readable metadata only; it is
    stripped here so the dispatcher sees a stable spec shape and the
    allocation-stage registry hash is reproducible.
    """
    raw = load_sweep(case_study).get("allocators") or []
    cs_lookback = None
    if any(a.get("method") in _MOMENT_ALLOCATORS for a in raw):
        cs_lookback = get_allocator_lookback(case_study)
    ppy = _periods_per_year_for(case_study) if _needs_calendar_resolve(raw) else None
    resolved = []
    for entry in raw:
        out = {k: v for k, v in entry.items() if k != "name"}
        # CS-level lookback injection for moment-based allocators that don't
        # carry an explicit per-entry override.
        if out.get("method") in _MOMENT_ALLOCATORS and cs_lookback is not None:
            if out["method"] in {"mvo", "mvo_ledoit_wolf"}:
                out.setdefault("lookback", cs_lookback)
            else:
                out.setdefault("vol_window", cs_lookback)
        # Calendar-string overrides (``"3M"`` etc.) resolved against ppy.
        for key in _LOOKBACK_KEYS:
            if key in out and isinstance(out[key], str):
                if ppy is None:
                    raise RuntimeError("calendar lookback resolution requires periods_per_year")
                out[key] = _resolve_calendar_lookback(out[key], ppy)
        resolved.append(out)
    return resolved


def _needs_calendar_resolve(allocators: list[dict]) -> bool:
    return any(isinstance(a.get(k), str) for a in allocators for k in _LOOKBACK_KEYS)


def get_allocator_label(alloc: dict) -> str:
    """Return a human-readable label for an allocator dict (``alloc['method']``)."""
    return str(alloc.get("method", "unknown"))


def get_cost_grid_bps(case_study: str) -> list[float]:
    """Return the Ch18 cost-sweep grid (bps; commission + slippage combined)."""
    return [float(c) for c in (load_sweep(case_study).get("cost_grid_bps") or [])]


def get_htm_cost_cascade(case_study: str) -> dict:
    """Return the Ch18 HTM cost-cascade block (sp500_options only).

    The cascade dispatches the O'Donovan & Yu (2025) hold-to-expiry cost
    analysis: entry-only half-spread fractions, optionally restricted to a
    liquid-universe subset (rung-3). Raises ``KeyError`` if the block is
    missing — case studies that use the standard bps regime should call
    :func:`get_cost_grid_bps` instead.
    """
    block = load_sweep(case_study).get("htm_cost_cascade")
    if block is None:
        raise KeyError(
            f"backtest.sweep.htm_cost_cascade missing from "
            f"case_studies/{case_study}/config/setup.yaml — only the "
            f"HTM-cascade case studies (sp500_options) declare this block."
        )
    return block


def get_universe_filters_for(case_study: str) -> list[str | None]:
    """Return the list of ``strategy.signal.universe_filter`` axis values to sweep.

    ``None`` represents the full universe (no filter applied, equivalent to
    ``apply_universe_filter`` returning predictions unchanged). Other values
    are passed through to ``apply_universe_filter`` in ``backtest_runner.py``
    where they drive a spec-declared universe restriction at the
    rebalance-date grain (currently only ``"liquid"`` is supported, for the
    sp500_options bottom-quantile half-spread subset).

    Sourced from ``backtest.sweep.universe_filter`` in ``setup.yaml``: a
    single scalar value pinning the canonical sweep to one universe. For
    sp500_options this is ``"liquid"`` (the only economic universe for the
    HTM straddle strategy after costs); absent everywhere else, which
    yields ``[None]``. ``"full"`` and ``"none"`` are normalized to ``None``
    so pre-universe-axis registry rows (which carry no
    ``signal.universe_filter`` in their spec) remain hash-stable.

    Note: ``backtest.sweep.htm_cost_cascade.universes`` is a separate
    Ch18-only block consumed directly by ``15_costs.py`` via
    ``get_htm_cost_cascade``; it is the cost-comparison axis (full vs
    liquid) and does NOT participate in the canonical rank-1 sweep.
    """
    sweep = load_sweep(case_study)
    uf = sweep.get("universe_filter")
    if uf is not None:
        return [(None if str(uf).lower() in ("full", "none") else str(uf))]
    return [None]


def get_position_risk_controls(case_study: str) -> list[dict]:
    """Return the Ch19 position-level risk controls (engine case studies only)."""
    risk = load_sweep(case_study).get("risk_controls") or {}
    return list(risk.get("position") or [])


def get_portfolio_risk_controls(case_study: str) -> list[dict]:
    """Return the Ch19 portfolio-level risk controls (all case studies)."""
    risk = load_sweep(case_study).get("risk_controls") or {}
    return list(risk.get("portfolio") or [])
