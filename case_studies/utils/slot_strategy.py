"""Persistent-slot signal-exit strategy for intraday execution.

The slot mechanism is the operationally-defensible variant of the canonical
``eq_w_topk`` selection used in chapters 11-19. It is designed for
microstructure case studies (nasdaq100) where:

  - predictions arrive at one cadence (e.g. 1-min) but execution rebalances
    on a coarser cadence (e.g. 15-min)
  - per-symbol score distributions are heterogeneous (a 0.6 score for AAPL
    is not equivalent to a 0.6 score for AMZN), so entry uses a per-symbol
    rolling quantile rather than cross-sectional rank
  - the chapter narrative needs a *signal-based exit* — close positions
    when the score crosses back below a stay-threshold — which neither
    ``eq_w_topk`` nor ``risk_controls.position`` (Ch19) expresses

Mechanism:
  1. Align predictions to price-grid timestamps via backward asof (freshest
     prediction within ``pred_freshness_max_min``).
  2. Compute per-symbol rolling entry threshold at quantile ``long_q``
     (delegates to ``signals.per_symbol_rolling_percentile_signal``).
  3. Optionally compute per-symbol rolling stay threshold at quantile
     ``exit_signal_q`` < ``long_q``.
  4. Walk bars in time order maintaining ``open_slots: dict[sym -> entry_bar]``.
     At each bar:
       (a) close slots whose age >= ``hold_bars`` (max-hold backstop)
       (b) close slots whose current score < stay_threshold (signal-exit)
       (c) close slots hitting ``take_profit`` / ``stop_loss`` vs entry price
       (d) open new slots from the top ``max_slots - len(open_slots)`` entries
           sorted by score descending
       (e) emit ``weight_per_slot`` for every currently-held (ts, sym)

  Take-profit and stop-loss are per-slot exit legs evaluated against each
  slot's entry price. They are an intrinsic property of the slot mechanism's
  event-driven holding period (entry -> exit on the FIRST trigger), distinct
  from Ch19 ``risk_controls.position`` overlays which act on a continuously
  rebalanced weight series. The sandbox finding (nasdaq100 v4) is that
  signal-exit OR take-profit each help out-of-sample but stacking them does
  not, so callers sweep them as mutually exclusive exit variants.

The output schema ``[timestamp, symbol, weight]`` is what
``backtest_runner._run_engine`` consumes as ``weights``.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Literal

import polars as pl

from case_studies.utils.signals import per_symbol_rolling_percentile_signal


def _run_slot_simulation(
    signals_by_ts: dict[datetime, list[tuple[str, float]]],
    all_bars_sorted: list[datetime],
    max_slots: int,
    weight_per_slot: float,
    hold_bars: int,
    *,
    score_by_ts_sym: Mapping[tuple[datetime, str], float] | None,
    stay_threshold_by_ts_sym: Mapping[tuple[datetime, str], float] | None,
    price_by_ts_sym: Mapping[tuple[datetime, str], float] | None = None,
    take_profit: float | None = None,
    stop_loss: float | None = None,
) -> tuple[pl.DataFrame, dict]:
    """Pure-mechanism slot simulator with optional signal-exit and TP/SL.

    Walks ``all_bars_sorted`` in order. Returns long-only weights frame and
    a stats dict with per-exit-cause counts. Exit priority per bar:
    max-hold, then signal-exit, then take-profit, then stop-loss. TP/SL
    compare the current bar's price to the slot's entry price and require
    ``price_by_ts_sym``; absent a current/entry price the TP/SL legs are
    skipped (the slot still honours max-hold/signal-exit).
    """
    if max_slots <= 0:
        raise ValueError(f"max_slots must be positive, got {max_slots}")
    if hold_bars <= 0:
        raise ValueError(f"hold_bars must be positive, got {hold_bars}")
    if not (0 < weight_per_slot <= 1.0):
        raise ValueError(f"weight_per_slot must be in (0, 1], got {weight_per_slot}")
    if stop_loss is not None and stop_loss < 0:
        raise ValueError(f"stop_loss must be positive (sign applied internally), got {stop_loss}")
    if take_profit is not None and take_profit <= 0:
        raise ValueError(f"take_profit must be positive, got {take_profit}")

    score_lookup = score_by_ts_sym or {}
    stay_lookup = stay_threshold_by_ts_sym or {}
    price_lookup = price_by_ts_sym or {}
    use_tp_sl = take_profit is not None or stop_loss is not None

    open_slots: dict[str, dict] = {}  # sym -> {"entry_i", "entry_px"}
    rows: list[dict] = []
    n_entries = 0
    n_exits_maxhold = 0
    n_exits_signal = 0
    n_exits_tp = 0
    n_exits_sl = 0

    for i, ts in enumerate(all_bars_sorted):
        # 1. Expire slots — max-hold, then signal-exit, then TP, then SL
        to_close: list[tuple[str, str]] = []
        for sym, slot in open_slots.items():
            if i - slot["entry_i"] >= hold_bars:
                to_close.append((sym, "maxhold"))
                continue
            key = (ts, sym)
            current_score = score_lookup.get(key)
            stay_thresh = stay_lookup.get(key)
            if (
                current_score is not None
                and stay_thresh is not None
                and current_score < stay_thresh
            ):
                to_close.append((sym, "signal"))
                continue
            if use_tp_sl and slot["entry_px"] is not None:
                current_px = price_lookup.get(key)
                if current_px is not None and slot["entry_px"] > 0:
                    ret = current_px / slot["entry_px"] - 1.0
                    if take_profit is not None and ret >= take_profit:
                        to_close.append((sym, "tp"))
                        continue
                    if stop_loss is not None and ret <= -stop_loss:
                        to_close.append((sym, "sl"))
                        continue

        for sym, cause in to_close:
            del open_slots[sym]
            if cause == "maxhold":
                n_exits_maxhold += 1
            elif cause == "signal":
                n_exits_signal += 1
            elif cause == "tp":
                n_exits_tp += 1
            else:
                n_exits_sl += 1

        # 2. New entries — sorted by score desc, capacity-limited
        candidates = signals_by_ts.get(ts, [])
        if candidates:
            fresh = [(s, sc) for s, sc in candidates if s not in open_slots]
            fresh.sort(key=lambda x: -x[1])
            capacity = max_slots - len(open_slots)
            for sym, _score in fresh[:capacity]:
                entry_px = price_lookup.get((ts, sym)) if use_tp_sl else None
                open_slots[sym] = {"entry_i": i, "entry_px": entry_px}
                n_entries += 1

        # 3. Emit weights for currently-held symbols
        for sym in open_slots:
            rows.append({"timestamp": ts, "symbol": sym, "weight": weight_per_slot})

    stats = {
        "n_entries": n_entries,
        "n_exits_maxhold": n_exits_maxhold,
        "n_exits_signal": n_exits_signal,
        "n_exits_tp": n_exits_tp,
        "n_exits_sl": n_exits_sl,
        "n_exits_total": n_exits_maxhold + n_exits_signal + n_exits_tp + n_exits_sl,
        "max_slots": max_slots,
        "hold_bars": hold_bars,
        "n_bars": len(all_bars_sorted),
    }
    if not rows:
        empty = pl.DataFrame(
            schema={"timestamp": pl.Datetime("us"), "symbol": pl.String, "weight": pl.Float64}
        )
        return empty, stats
    out = pl.DataFrame(rows).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    return out, stats


def _align_predictions_to_bars(
    predictions: pl.DataFrame,
    bar_grid: pl.DataFrame,
    *,
    pred_freshness_max_min: int | None,
    score_col: str,
    time_col: str,
    asset_col: str,
) -> pl.DataFrame:
    """Backward-asof align predictions to a (symbol, timestamp) bar grid.

    ``bar_grid`` carries the rebalance schedule. For each (sym, bar_ts) row,
    pull the freshest prediction with timestamp <= bar_ts and stale by at
    most ``pred_freshness_max_min`` minutes. Predictions older than the
    tolerance are dropped, leaving rows with null ``y_score`` which are
    then filtered out.

    When ``pred_freshness_max_min`` is None, the asof tolerance is
    unbounded (typical when predictions and prices share the same cadence).
    """
    bars = bar_grid.select([asset_col, time_col]).sort([asset_col, time_col])
    preds = predictions.select([asset_col, time_col, score_col]).sort([asset_col, time_col])
    tol = f"{pred_freshness_max_min}m" if pred_freshness_max_min is not None else None
    aligned = bars.join_asof(
        preds,
        on=time_col,
        by=asset_col,
        strategy="backward",
        tolerance=tol,
    )
    return aligned.filter(pl.col(score_col).is_not_null()).sort([asset_col, time_col])


def _signals_to_lookup(
    signals_df: pl.DataFrame,
    *,
    score_col: str,
    time_col: str,
    asset_col: str,
) -> dict[datetime, list[tuple[str, float]]]:
    """Convert ``per_symbol_rolling_percentile_signal`` output (signal==1 rows)
    to ``dict[ts -> list[(sym, score)]]`` for the slot simulator.
    """
    fired = signals_df.filter(pl.col("signal") == 1).select([time_col, asset_col, score_col])
    out: dict[datetime, list[tuple[str, float]]] = {}
    for row in fired.iter_rows(named=True):
        ts = row[time_col]
        if ts not in out:
            out[ts] = []
        out[ts].append((row[asset_col], float(row[score_col])))
    return out


def build_persistent_slot_weights_hybrid(
    predictions: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    long_q: float,
    lookback_days: int,
    bars_per_day: int,
    max_slots: int,
    hold_bars: int,
    weight_per_slot: float | None = None,
    exit_signal_q: float | None = None,
    take_profit: float | None = None,
    stop_loss: float | None = None,
    pred_freshness_max_min: int | None = None,
    direction: Literal["long_only", "short_only"] = "long_only",
    score_col: str = "y_score",
    time_col: str = "timestamp",
    asset_col: str = "symbol",
    price_col: str = "close",
) -> tuple[pl.DataFrame, dict]:
    """Library entry point for the persistent-slot signal-exit selection method.

    Pipeline:
      1. Align ``predictions`` to ``prices`` grid via backward-asof.
      2. Compute per-symbol rolling entry threshold at ``long_q``; entry signal
         where y_score > threshold.
      3. If ``exit_signal_q`` is set (< ``long_q``), compute per-symbol rolling
         stay threshold at that quantile.
      4. Run slot simulation with ``max_slots`` capacity and ``hold_bars`` cap,
         plus optional ``take_profit`` / ``stop_loss`` per-slot exit legs
         (evaluated on the bar-close return vs the slot's entry price).
      5. Apply ``direction`` sign — ``short_only`` flips the weight sign.

    ``take_profit`` / ``stop_loss`` are decimals (0.005 = 0.5%). They free the
    slot on trigger so the weight series stops emitting the symbol until a fresh
    entry signal — the slot-native re-entry semantics that engine-level
    ``risk_controls.position`` rules cannot express against a dense target
    series. The nasdaq100 v4 sandbox finding is that signal-exit OR take-profit
    each help out-of-sample but stacking them does not, so callers pass at most
    one of ``exit_signal_q`` / ``take_profit`` per configuration.

    Returns ``(weights_df, stats_dict)``. ``weights_df`` has schema
    ``[timestamp, symbol, weight]`` matching ``_run_engine`` input.

    Note: long_short is not supported — slot books are inherently
    single-direction (a symbol cannot occupy a long and short slot
    simultaneously). The cross-asset long-short story belongs to
    ``eq_w_topk`` / ``quintile_long_short``.
    """
    if direction not in ("long_only", "short_only"):
        raise ValueError(
            f"slot direction must be 'long_only' or 'short_only', got {direction!r}; "
            "long_short is not supported for the slot mechanism"
        )
    if exit_signal_q is not None and exit_signal_q >= long_q:
        raise ValueError(
            f"exit_signal_q ({exit_signal_q}) must be < long_q ({long_q}) "
            "so the stay threshold sits below the entry threshold"
        )
    if weight_per_slot is None:
        weight_per_slot = 1.0 / max_slots

    aligned = _align_predictions_to_bars(
        predictions,
        prices,
        pred_freshness_max_min=pred_freshness_max_min,
        score_col=score_col,
        time_col=time_col,
        asset_col=asset_col,
    )

    sig_df = per_symbol_rolling_percentile_signal(
        aligned,
        long_q=long_q,
        lookback_days=lookback_days,
        bars_per_day=bars_per_day,
        score_col=score_col,
        time_col=time_col,
        asset_col=asset_col,
        signal_type="long_only",
        stay_q=exit_signal_q,
    )

    signals_by_ts = _signals_to_lookup(
        sig_df,
        score_col=score_col,
        time_col=time_col,
        asset_col=asset_col,
    )

    if exit_signal_q is not None:
        stay_lookup = {
            (r[time_col], r[asset_col]): float(r["stay_thresh"])
            for r in sig_df.filter(pl.col("stay_thresh").is_not_null())
            .select([time_col, asset_col, "stay_thresh"])
            .iter_rows(named=True)
        }
        score_lookup = {
            (r[time_col], r[asset_col]): float(r[score_col])
            for r in aligned.select([time_col, asset_col, score_col]).iter_rows(named=True)
        }
    else:
        stay_lookup = None
        score_lookup = None

    if take_profit is not None or stop_loss is not None:
        price_lookup = {
            (r[time_col], r[asset_col]): float(r[price_col])
            for r in prices.select([time_col, asset_col, price_col])
            .filter(pl.col(price_col).is_not_null())
            .iter_rows(named=True)
        }
    else:
        price_lookup = None

    schedule = sorted(aligned[time_col].unique().to_list())

    weights, stats = _run_slot_simulation(
        signals_by_ts=signals_by_ts,
        all_bars_sorted=schedule,
        max_slots=max_slots,
        weight_per_slot=weight_per_slot,
        hold_bars=hold_bars,
        score_by_ts_sym=score_lookup,
        stay_threshold_by_ts_sym=stay_lookup,
        price_by_ts_sym=price_lookup,
        take_profit=take_profit,
        stop_loss=stop_loss,
    )

    if direction == "short_only" and not weights.is_empty():
        weights = weights.with_columns((-pl.col("weight")).alias("weight"))

    stats["direction"] = direction
    stats["long_q"] = long_q
    stats["exit_signal_q"] = exit_signal_q
    stats["take_profit"] = take_profit
    stats["stop_loss"] = stop_loss
    return weights, stats
