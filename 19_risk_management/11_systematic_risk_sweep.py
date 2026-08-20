# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Systematic Risk Parameter Sweep
# **Chapter 19: Risk Management**
#
# ## Purpose
# Calibrate position-level exit rules on a chronological calibration window, then compare the
# selected rules once on a later evaluation window. One-dimensional sweeps show sensitivity to
# each rule width, joint grids expose interactions, and MAE/MFE percentiles provide a second
# calibration method.
#
# ## Learning Objectives
# After completing this notebook, you will be able to:
# - separate exit-rule calibration from chronological evaluation;
# - run one-dimensional and joint StopLoss, TakeProfit, and TrailingStop sweeps;
# - derive stop and target priors from closed-trade MAE/MFE paths;
# - compare preselected configurations with CAGR-based Calmar ratios.
#
# ## Book reference
# Section 19.7, "Adaptive Risk Controls Without Leakage," and Figures 19.6-19.7.
#
# ## Prerequisites
# Complete [`02_exit_strategies`](02_exit_strategies.ipynb),
# [`03_position_sizing_mae_mfe`](03_position_sizing_mae_mfe.ipynb), and
# [`10_ml4t_backtest_risk_demo`](10_ml4t_backtest_risk_demo.ipynb) first.

# %%
"""Calibrate and evaluate systematic position-exit rules without sample reuse."""

from datetime import date

import numpy as np
import plotly.graph_objects as go
import polars as pl
from IPython.display import Markdown, display
from ml4t.backtest import BacktestConfig, DataFeed, Engine, ExecutionMode, Strategy
from ml4t.backtest.analytics.trades import MAEMFEAnalyzer
from ml4t.backtest.execution.rebalancer import RebalanceConfig, TargetWeightExecutor
from ml4t.backtest.risk.position import RuleChain, StopLoss, TakeProfit, TrailingStop
from plotly.subplots import make_subplots

from data import load_etfs
from utils.paths import get_output_dir
from utils.style import COLORS, ml4t_diverging, show_plotly_with_alt

# %% tags=["parameters"]
START_DATE = "2019-01-02"
END_DATE = "2023-12-31"
CALIBRATION_END = "2021-12-31"
N_BARS = 1260
LOOKBACK_BARS = 60
CADENCE_BARS = 21
INITIAL_CASH = 100_000

# %%
OUTPUT_DIR = get_output_dir(19, "risk_sweep")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SYMBOLS = ["SPY", "QQQ", "IWM", "XLF", "EEM"]
calibration_end = date.fromisoformat(CALIBRATION_END)

# %% [markdown]
# ## 1. Point-in-Time ETF Panel and Split
#
# The fixed five-ETF panel is an illustrative, present-day universe rather than a
# point-in-time membership reconstruction. It therefore supports a rule-calibration lesson, not
# an unbiased claim about historical ETF selection. Momentum uses each close at time *t*, and the
# engine executes resulting targets on the next bar.

# %%
prices_df = load_etfs(
    symbols=SYMBOLS,
    start_date=START_DATE,
    end_date=END_DATE,
).sort(["symbol", "timestamp"])
prices_df = (
    prices_df.with_columns(_bar=pl.col("timestamp").cum_count().over("symbol"))
    .filter(pl.col("_bar") <= N_BARS)
    .drop("_bar")
)

# %% [markdown]
# Before any backtest, fail closed on schema, duplicate keys, null OHLC data, and incomplete daily
# snapshots. This makes a missing asset an input error instead of an exception to suppress inside
# the executor.

# %%
required_columns = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
assert required_columns <= set(prices_df.columns)
assert prices_df["timestamp"].dtype == pl.Date
assert set(prices_df["symbol"].unique()) == set(SYMBOLS)
assert prices_df.select("symbol", "timestamp").is_duplicated().sum() == 0
assert prices_df.select(sorted(required_columns)).null_count().row(0) == (0,) * len(
    required_columns
)
panel_counts = prices_df.group_by("timestamp").agg(pl.col("symbol").n_unique().alias("n_symbols"))
assert panel_counts.filter(pl.col("n_symbols") != len(SYMBOLS)).is_empty()

all_dates = prices_df["timestamp"].unique().sort().to_list()
calibration_dates = [ts for ts in all_dates if ts <= calibration_end]
evaluation_dates = [ts for ts in all_dates if ts > calibration_end]
assert len(calibration_dates) > LOOKBACK_BARS
assert len(evaluation_dates) > LOOKBACK_BARS

# %% tags=["results"]
calibration_prices = prices_df.filter(pl.col("timestamp") <= calibration_end)
evaluation_prices = prices_df.filter(pl.col("timestamp") > calibration_end)
calibration_schedule = calibration_dates[LOOKBACK_BARS::CADENCE_BARS]
evaluation_schedule = evaluation_dates[::CADENCE_BARS]

display(
    Markdown(
        f"Loaded **{prices_df.height:,} complete bars** for **{len(SYMBOLS)} ETFs**. "
        f"Calibration ends on **{calibration_dates[-1]}**; the untouched evaluation window runs "
        f"from **{evaluation_dates[0]}** through **{evaluation_dates[-1]}**. Rebalances are "
        f"scheduled every **{CADENCE_BARS} trading bars**."
    )
)

# %% [markdown]
# The weight builder computes trailing returns independently within each symbol. Passing an
# explicit schedule prevents the strategy from mixing trading-bar schedules with calendar-day
# comparisons.


# %%
def build_momentum_weights(
    prices: pl.DataFrame,
    schedule: list[date],
    lookback: int,
    top_n: int = 3,
) -> dict[date, dict[str, float]]:
    """Return top-N equal weights for explicit decision timestamps."""
    momentum = (
        prices.sort(["symbol", "timestamp"])
        .with_columns(
            momentum=(pl.col("close") / pl.col("close").shift(lookback) - 1).over("symbol")
        )
        .select("timestamp", "symbol", "momentum")
        .drop_nulls()
    )
    weights: dict[date, dict[str, float]] = {}
    for timestamp in schedule:
        snapshot = momentum.filter(pl.col("timestamp") == timestamp).sort(
            "momentum", descending=True
        )
        assert snapshot.height == len(SYMBOLS)
        selected = snapshot.head(top_n)["symbol"].to_list()
        weights[timestamp] = {symbol: 1.0 / len(selected) for symbol in selected}
    return weights


# %%
calibration_weights = build_momentum_weights(
    calibration_prices,
    calibration_schedule,
    LOOKBACK_BARS,
)
evaluation_weights = build_momentum_weights(
    prices_df,
    evaluation_schedule,
    LOOKBACK_BARS,
)
assert list(calibration_weights) == calibration_schedule
assert list(evaluation_weights) == evaluation_schedule

# %% [markdown]
# ## 2. Backtest Contract
#
# A target is submitted only on its scheduled trading bar. `NEXT_BAR` execution means the signal
# formed from the current close cannot fill on that same close. The complete-panel assertion above
# makes missing-symbol recovery unnecessary.


# %%
class SweepStrategy(Strategy):
    """Apply a fixed rule chain and rebalance on explicit decision timestamps."""

    def __init__(self, weights: dict[date, dict[str, float]], rules=None):
        self.rules = rules
        self.weights = weights
        self.executor = TargetWeightExecutor(
            config=RebalanceConfig(
                min_trade_value=100.0,
                min_weight_change=0.005,
                allow_fractional=True,
            )
        )

    def on_start(self, broker):
        if self.rules is not None:
            broker.set_position_rules(self.rules)

    def on_data(self, timestamp, data, context, broker):
        del context
        if timestamp not in self.weights:
            return
        assert set(data) == set(SYMBOLS)
        self.executor.execute(self.weights[timestamp], data, broker)


# %% [markdown]
# The runner delegates annualization to `ml4t.backtest`: Calmar is the library's CAGR divided by
# absolute maximum drawdown.
#
# Trade statistics are a separate matter. A backtest that ends while positions are open produces
# trades in three states - closed, partly unwound, and still open - and the last two carry a PnL
# that is a mark-to-market rather than a result. The library's `num_trades` and `win_rate` count
# all three, which inflates the win rate with positions that have not finished losing. Everything
# trade-level below is therefore computed over the closed trades only, which is also the population
# the MAE/MFE analysis reads.


# %%
def run_risk_sweep(
    prices: pl.DataFrame,
    weights: dict[date, dict[str, float]],
    rules=None,
    initial_cash: float = 100_000,
) -> dict:
    """Run one exit-rule configuration and return closed-trade metrics."""
    config = BacktestConfig(
        initial_cash=initial_cash,
        commission_rate=0.001,
        slippage_rate=0.001,
        execution_mode=ExecutionMode.NEXT_BAR,
    )
    result = Engine(
        feed=DataFeed(prices_df=prices),
        strategy=SweepStrategy(weights, rules=rules),
        config=config,
    ).run()
    closed_trades = [trade for trade in result.trades if trade.status == "closed"]
    closed_wins = sum(1 for trade in closed_trades if trade.pnl > 0)
    return {
        # Equity-curve metrics come from the library and are unaffected by trade bookkeeping.
        "sharpe": float(result.metrics["sharpe"]),
        "max_dd": float(result.equity.max_dd),
        "calmar": float(result.metrics["calmar"]),
        "cagr": float(result.metrics["cagr"]),
        "total_return": float(result.metrics["total_return"]),
        # Trade-level statistics are computed over the closed trades only, matching the population
        # the MAE/MFE analysis reads. The library's own num_trades and win_rate count every trade
        # including the ones still open at the final bar, whose PnL is a mark rather than a result.
        "n_trades_all": int(result.metrics["num_trades"]),
        "n_trades": len(closed_trades),
        "win_rate": closed_wins / len(closed_trades) if closed_trades else float("nan"),
        "trades": closed_trades,
    }


# %% tags=["results"]
baseline_calibration = run_risk_sweep(
    calibration_prices,
    calibration_weights,
    initial_cash=INITIAL_CASH,
)
if not baseline_calibration["trades"]:
    raise ValueError("The calibration baseline closed no trades, so no trade statistics exist")

display(
    Markdown(
        f"The calibration baseline has Sharpe **{baseline_calibration['sharpe']:.2f}**, "
        f"CAGR **{baseline_calibration['cagr']:.1%}**, maximum drawdown "
        f"**{baseline_calibration['max_dd']:.1%}**, and Calmar "
        f"**{baseline_calibration['calmar']:.2f}**. Of the "
        f"**{baseline_calibration['n_trades_all']}** positions the backtest opened, "
        f"**{baseline_calibration['n_trades']}** closed before the window ended; the rest were "
        "still open or only partly unwound, and every trade statistic below counts the closed "
        "ones only."
    )
)

# %% [markdown]
# ## 3. Calibration Sweeps
#
# Every candidate below is fit and ranked on the calibration window only. The evaluation window is
# not consulted until one candidate from each calibration method has been frozen.

# %% [markdown]
# This helper runs a one-dimensional rule family while retaining the raw metrics needed for the
# diagnostic and publication-data outputs.


# %%
def run_1d_sweep(
    rule_class,
    levels: list[float],
    name: str,
) -> list[dict]:
    """Evaluate one rule family on the calibration window."""
    results = []
    for level in levels:
        metrics = run_risk_sweep(
            calibration_prices,
            calibration_weights,
            rules=RuleChain([rule_class(pct=level)]),
            initial_cash=INITIAL_CASH,
        )
        results.append({**metrics, "rule": name, "param": level})
    return results


# %%
sl_levels = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
tp_levels = [0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
trail_levels = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]

sl_results = run_1d_sweep(StopLoss, sl_levels, "StopLoss")
tp_results = run_1d_sweep(TakeProfit, tp_levels, "TakeProfit")
trail_results = run_1d_sweep(TrailingStop, trail_levels, "TrailingStop")

# %% [markdown]
# The solid line reports calibration Sharpe; the dotted line reports closed-trade count. Their
# different axes make turnover changes visible without treating trade count as a performance
# metric.


# %%
def add_sweep_panel(
    fig,
    row: int,
    sweep: list[dict],
    name: str,
    color: str,
) -> None:
    """Add one calibrated sweep panel with a secondary trade-count axis."""
    fig.add_trace(
        go.Scatter(
            x=[row["param"] for row in sweep],
            y=[row["sharpe"] for row in sweep],
            mode="lines+markers",
            line={"color": color},
            name=f"{name} Sharpe",
        ),
        row=row,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=[row["param"] for row in sweep],
            y=[row["n_trades"] for row in sweep],
            mode="lines+markers",
            line={"color": COLORS["neutral"], "dash": "dot"},
            marker={"symbol": "square"},
            name=f"{name} closed trades",
        ),
        row=row,
        col=1,
        secondary_y=True,
    )
    fig.add_hline(
        y=baseline_calibration["sharpe"],
        line={"color": COLORS["slate"], "dash": "dash"},
        row=row,
        col=1,
        secondary_y=False,
    )


# %%
fig = make_subplots(
    rows=3,
    cols=1,
    specs=[[{"secondary_y": True}] for _ in range(3)],
    subplot_titles=["StopLoss", "TakeProfit", "TrailingStop"],
    vertical_spacing=0.09,
)
for row, (sweep, name, color) in enumerate(
    [
        (sl_results, "StopLoss", COLORS["blue"]),
        (tp_results, "TakeProfit", COLORS["positive"]),
        (trail_results, "TrailingStop", COLORS["amber"]),
    ],
    start=1,
):
    add_sweep_panel(fig, row, sweep, name, color)
    fig.update_xaxes(
        title_text="Rule width" if row == 3 else None, tickformat=".0%", row=row, col=1
    )
    fig.update_yaxes(title_text="Sharpe ratio", secondary_y=False, row=row, col=1)
    fig.update_yaxes(title_text="Closed trades", secondary_y=True, row=row, col=1)
fig.update_layout(
    height=900,
    margin={"l": 75, "r": 85, "t": 105, "b": 70},
    title=(
        "Calibration Sharpe Depends on Exit Width"
        "<br><sup>2019-2021 calibration only; dashed line is the no-rule baseline</sup>"
    ),
    showlegend=False,
)
show_plotly_with_alt(
    fig,
    "Sharpe and trade count against threshold for each of the three one-dimensional sweeps, showing how each rule's parameter trades performance against how often it fires.",
)

# %%
sweep_rows = [
    {
        "rule": row["rule"],
        "param": float(row["param"]),
        "sharpe": float(row["sharpe"]),
        "n_trades": int(row["n_trades"]),
        "sample": "calibration",
    }
    for sweep in (sl_results, tp_results, trail_results)
    for row in sweep
]
pl.DataFrame(sweep_rows).write_parquet(OUTPUT_DIR / "risk_rule_1d_sweeps.parquet")

# %% [markdown]
# Joint grids ask whether one rule's useful width depends on the other. The selection score is
# calibration Calmar, but the later evaluation reports all candidates under one untouched window.


# %%
def run_rule_grid(
    left_levels: list[float],
    right_levels: list[float],
    right_rule,
) -> tuple[np.ndarray, np.ndarray]:
    """Return calibration Sharpe and Calmar matrices for a two-rule grid."""
    sharpe = np.zeros((len(left_levels), len(right_levels)))
    calmar = np.zeros_like(sharpe)
    for row, stop in enumerate(left_levels):
        for column, right in enumerate(right_levels):
            metrics = run_risk_sweep(
                calibration_prices,
                calibration_weights,
                rules=RuleChain([StopLoss(pct=stop), right_rule(pct=right)]),
                initial_cash=INITIAL_CASH,
            )
            sharpe[row, column] = metrics["sharpe"]
            calmar[row, column] = metrics["calmar"]
    return sharpe, calmar


# %%
sl_grid = [0.03, 0.05, 0.08, 0.10, 0.15]
tp_grid = [0.05, 0.10, 0.15, 0.20, 0.30]
trail_grid = [0.02, 0.04, 0.06, 0.08, 0.10]

sl_tp_sharpe, sl_tp_calmar = run_rule_grid(sl_grid, tp_grid, TakeProfit)
sl_trail_sharpe, sl_trail_calmar = run_rule_grid(sl_grid, trail_grid, TrailingStop)
assert np.isfinite(sl_tp_sharpe).all() and np.isfinite(sl_tp_calmar).all()
assert np.isfinite(sl_trail_sharpe).all() and np.isfinite(sl_trail_calmar).all()

sl_tp_sharpe_delta = sl_tp_sharpe - baseline_calibration["sharpe"]
sl_trail_sharpe_delta = sl_trail_sharpe - baseline_calibration["sharpe"]
sl_tp_calmar_delta = sl_tp_calmar - baseline_calibration["calmar"]
sl_trail_calmar_delta = sl_trail_calmar - baseline_calibration["calmar"]
sharpe_limit = float(
    np.max(np.abs(np.concatenate([sl_tp_sharpe_delta.ravel(), sl_trail_sharpe_delta.ravel()])))
)
calmar_limit = float(
    np.max(np.abs(np.concatenate([sl_tp_calmar_delta.ravel(), sl_trail_calmar_delta.ravel()])))
)

# %% [markdown]
# Each metric uses one pooled, symmetric scale across both rule pairs. Colors therefore encode the
# change from the same no-rule baseline instead of re-normalizing every panel independently.


# %%
def add_surface(
    fig,
    data: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    row: int,
    column: int,
    limit: float,
    colorbar_title: str,
) -> None:
    """Add one baseline-relative heatmap with an explicit pooled range."""
    fig.add_trace(
        go.Heatmap(
            z=data,
            x=x_labels,
            y=y_labels,
            zmin=-limit,
            zmax=limit,
            zmid=0,
            colorscale=ml4t_diverging(),
            text=np.round(data, 2),
            texttemplate="%{text:+.2f}",
            showscale=row == 1,
            colorbar={
                "title": colorbar_title,
                "len": 0.8,
                "y": 0.5,
                "x": 1.03 if column == 1 else 1.17,
                "thickness": 12,
            },
        ),
        row=row,
        col=column,
    )


# %% [markdown]
# Explicit categorical ticks prevent Plotly's static renderer from dropping percentage signs.


# %%
def apply_surface_axes(
    fig,
    stop_labels: list[str],
    take_profit_labels: list[str],
    trailing_labels: list[str],
) -> None:
    """Apply explicit percentage categories and units to every grid axis."""
    panels = [(1, 1, take_profit_labels), (1, 2, take_profit_labels)]
    panels += [(2, 1, trailing_labels), (2, 2, trailing_labels)]
    for row, column, labels in panels:
        fig.update_xaxes(
            type="category",
            tickmode="array",
            tickvals=labels,
            ticktext=labels,
            title_text="Second rule width (%)" if row == 2 else None,
            row=row,
            col=column,
        )
        fig.update_yaxes(
            type="category",
            tickmode="array",
            tickvals=stop_labels,
            ticktext=stop_labels,
            title_text="StopLoss width (%)" if column == 1 else None,
            row=row,
            col=column,
        )


# %%
sl_labels = [f"{value:.0%}" for value in sl_grid]
tp_labels = [f"{value:.0%}" for value in tp_grid]
trail_labels = [f"{value:.0%}" for value in trail_grid]
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=[
        "SL x TP: Sharpe change",
        "SL x TP: Calmar change",
        "SL x Trail: Sharpe change",
        "SL x Trail: Calmar change",
    ],
    vertical_spacing=0.16,
    horizontal_spacing=0.14,
)
add_surface(fig, sl_tp_sharpe_delta, tp_labels, sl_labels, 1, 1, sharpe_limit, "Delta Sharpe")
add_surface(fig, sl_tp_calmar_delta, tp_labels, sl_labels, 1, 2, calmar_limit, "Delta Calmar")
add_surface(fig, sl_trail_sharpe_delta, trail_labels, sl_labels, 2, 1, sharpe_limit, "")
add_surface(fig, sl_trail_calmar_delta, trail_labels, sl_labels, 2, 2, calmar_limit, "")
apply_surface_axes(fig, sl_labels, tp_labels, trail_labels)
fig.update_layout(
    width=980,
    height=720,
    margin={"l": 80, "r": 190, "t": 100, "b": 75},
    title=(
        "Joint Exit Rules Move Calibration Performance Around the Baseline"
        "<br><sup>Pooled ranges make colors comparable within each metric</sup>"
    ),
)
show_plotly_with_alt(
    fig,
    "A heatmap of Calmar over the stop-loss by take-profit grid, with a broad region of similar values rather than a sharp optimum.",
)

# %% tags=["results"]
best_sl_tp_index = np.unravel_index(np.nanargmax(sl_tp_calmar), sl_tp_calmar.shape)
best_sl_trail_index = np.unravel_index(np.nanargmax(sl_trail_calmar), sl_trail_calmar.shape)
best_sl_tp = (sl_grid[best_sl_tp_index[0]], tp_grid[best_sl_tp_index[1]])
best_sl_trail = (sl_grid[best_sl_trail_index[0]], trail_grid[best_sl_trail_index[1]])

display(
    Markdown(
        f"Calibration selects **SL {best_sl_tp[0]:.0%} / TP {best_sl_tp[1]:.0%}** "
        f"at Calmar **{sl_tp_calmar[best_sl_tp_index]:.2f}**, and "
        f"**SL {best_sl_trail[0]:.0%} / Trail {best_sl_trail[1]:.0%}** at Calmar "
        f"**{sl_trail_calmar[best_sl_trail_index]:.2f}**. These are frozen before the "
        "evaluation window is opened."
    )
)

# %%
grid_rows = [
    {
        "stop_loss": float(stop),
        "take_profit": float(target),
        "sharpe": float(sl_tp_sharpe[row, column]),
        "calmar": float(sl_tp_calmar[row, column]),
        "sample": "calibration",
    }
    for row, stop in enumerate(sl_grid)
    for column, target in enumerate(tp_grid)
]
pl.DataFrame(grid_rows).write_parquet(OUTPUT_DIR / "stoploss_takeprofit_grid.parquet")

# %% [markdown]
# ## 4. MAE/MFE Calibration on Closed Trades
#
# MAE/MFE percentiles are another fitted rule. They use only closed calibration trades, so neither
# an end-of-window mark nor any evaluation path can influence the selected stop and target.

# %% tags=["results"]
calibration_trades = baseline_calibration["trades"]
mae_mfe = MAEMFEAnalyzer(calibration_trades)
suggested_sl = mae_mfe.suggest_stop_loss(percentile=90)
suggested_tp = mae_mfe.suggest_take_profit(percentile=75)
calibrated_sl_pct = abs(suggested_sl)
calibrated_tp_pct = abs(suggested_tp)

display(
    Markdown(
        f"Across **{mae_mfe.num_trades} closed calibration trades**, mean MAE is "
        f"**{mae_mfe.mae_mean:.1%}** and mean MFE is **{mae_mfe.mfe_mean:.1%}**. The frozen "
        f"percentile prior is **SL {calibrated_sl_pct:.1%} / TP {calibrated_tp_pct:.1%}**. "
        "The small trade sample makes this a deliberately low-confidence prior."
    )
)

# %%
distribution = mae_mfe.distribution_data()
realized_returns = [trade.pnl_percent for trade in calibration_trades]
color_limit = max(abs(min(realized_returns)), abs(max(realized_returns)))
fig = go.Figure(
    go.Scatter(
        x=distribution["mae"],
        y=distribution["mfe"],
        mode="markers",
        marker={
            "size": 8,
            "color": realized_returns,
            "colorscale": ml4t_diverging(),
            "cmin": -color_limit,
            "cmax": color_limit,
            "colorbar": {"title": "Realized return", "tickformat": ".0%"},
            "line": {"width": 0.5, "color": COLORS["neutral"]},
        },
        text=[f"Realized return: {value:.1%}" for value in realized_returns],
        hovertemplate="MAE: %{x:.1%}<br>MFE: %{y:.1%}<br>%{text}<extra></extra>",
    )
)
fig.add_vline(x=suggested_sl, line_dash="dash", line_color=COLORS["negative"])
fig.add_hline(y=suggested_tp, line_dash="dash", line_color=COLORS["positive"])
fig.update_layout(
    height=480,
    title=(
        "Closed Calibration Trades Define the MAE/MFE Prior"
        "<br><sup>Dashed lines are the 90th-percentile stop and 75th-percentile target</sup>"
    ),
    xaxis={"title": "Maximum adverse excursion", "tickformat": ".0%"},
    yaxis={"title": "Maximum favorable excursion", "tickformat": ".0%"},
)
show_plotly_with_alt(
    fig,
    "A heatmap of Calmar over the stop-loss by trailing-stop grid on the same colour scale as the previous one.",
)

# %% [markdown]
# ## 5. One-Time Chronological Evaluation
#
# Only the no-rule baseline and the three configurations frozen above enter the 2022-2023 window.
# This comparison estimates one realized-window outcome; it does not provide multiple-testing-
# adjusted inference or establish that any rule is universally optimal.

# %%
evaluation_configs = {
    "No rules": None,
    f"Grid SL/TP {best_sl_tp[0]:.0%}/{best_sl_tp[1]:.0%}": RuleChain(
        [StopLoss(pct=best_sl_tp[0]), TakeProfit(pct=best_sl_tp[1])]
    ),
    f"Grid SL/Trail {best_sl_trail[0]:.0%}/{best_sl_trail[1]:.0%}": RuleChain(
        [StopLoss(pct=best_sl_trail[0]), TrailingStop(pct=best_sl_trail[1])]
    ),
    f"MAE/MFE {calibrated_sl_pct:.0%}/{calibrated_tp_pct:.0%}": RuleChain(
        [StopLoss(pct=calibrated_sl_pct), TakeProfit(pct=calibrated_tp_pct)]
    ),
}
evaluation_results = {
    name: run_risk_sweep(
        evaluation_prices,
        evaluation_weights,
        rules=rules,
        initial_cash=INITIAL_CASH,
    )
    for name, rules in evaluation_configs.items()
}

# %%
comparison = pl.DataFrame(
    [
        {
            "configuration": name,
            "sharpe": metrics["sharpe"],
            "cagr": metrics["cagr"],
            "max_dd": metrics["max_dd"],
            "calmar": metrics["calmar"],
            "closed_trades": metrics["n_trades"],
            "win_rate": metrics["win_rate"],
        }
        for name, metrics in evaluation_results.items()
    ]
).sort("calmar", descending=True)
winner = comparison.row(0, named=True)
baseline_evaluation = evaluation_results["No rules"]
comparison

# %% [markdown]
# The trade columns count closed trades and the win rate is computed over those same trades, so
# both describe the positions that actually finished. Read them alongside the equity-curve columns
# rather than instead of them: a configuration can close few trades and still shape the curve,
# because the rules act on positions that remain open as well.

# %%
plot_rows = comparison.sort("calmar", descending=True)
plot_labels = [name.replace(" ", "<br>", 1) for name in plot_rows["configuration"]]
fig = go.Figure(
    go.Bar(
        x=plot_labels,
        y=plot_rows["calmar"],
        marker_color=[
            COLORS["amber"] if name == winner["configuration"] else COLORS["blue"]
            for name in plot_rows["configuration"]
        ],
        text=[f"{value:.2f}" for value in plot_rows["calmar"]],
        textposition="outside",
        cliponaxis=False,
        customdata=np.column_stack([plot_rows["sharpe"], plot_rows["cagr"], plot_rows["max_dd"]]),
        hovertemplate=(
            "Calmar: %{y:.2f}<br>Sharpe: %{customdata[0]:.2f}<br>"
            "CAGR: %{customdata[1]:.1%}<br>Max drawdown: %{customdata[2]:.1%}<extra></extra>"
        ),
    )
)
fig.update_layout(
    width=850,
    height=480,
    margin={"l": 75, "r": 45, "t": 100, "b": 90},
    title=(
        f"{winner['configuration']} Leads the One-Time Evaluation"
        "<br><sup>2022-2023; all rule choices frozen on 2019-2021 calibration data</sup>"
    ),
    xaxis={"title": None, "automargin": True},
    yaxis={"title": "Calmar ratio (CAGR / |maximum drawdown|)", "zeroline": True},
    showlegend=False,
)
show_plotly_with_alt(
    fig,
    "Bars of evaluation-window Calmar per configuration, sorted, with the highest-ranked bar highlighted. Several bars sit below zero.",
)

# %%
pl.DataFrame(
    [
        {
            "sample": "calibration",
            "sharpe": baseline_calibration["sharpe"],
            "n_trades": baseline_calibration["n_trades"],
        },
        {
            "sample": "evaluation",
            "sharpe": baseline_evaluation["sharpe"],
            "n_trades": baseline_evaluation["n_trades"],
        },
    ]
).write_parquet(OUTPUT_DIR / "risk_rule_baseline.parquet")

# %% [markdown]
# ## Key Takeaways

# %% tags=["results"]
display(
    Markdown(
        f"""
1. **Calibrate before evaluating.** The grids and MAE/MFE percentiles use only 2019-2021 data;
   the 2022-2023 window is opened once after all three rule configurations are frozen.
2. **Use the metric you name.** The evaluation ranks configurations by CAGR-based Calmar. In this
   run, **{winner["configuration"]}** ranks first at **{winner["calmar"]:.2f}**, while the no-rule
   baseline records **{baseline_evaluation["calmar"]:.2f}**.
3. **Treat small-sample excursion percentiles as priors.** The MAE/MFE levels come from only
   **{mae_mfe.num_trades} closed calibration trades**, so they seed a search rather than settle it.
4. **Keep the operational contract explicit.** Signals use completed closes, orders execute on the
   next bar, schedules count trading bars, and incomplete panels fail before the backtest.

This capstone closes the chapter's progression from risk measurement to calibrated controls.
"""
    )
)
