"""Shared allocation functions for Ch17 portfolio construction.

Each function takes (predictions, prices_df, top_k, ...) and returns
pl.DataFrame with columns [time_col, symbol, weight].

Predictions must have columns: [time_col, symbol, y_score].
Prices must have columns: [time_col, symbol, close] or [time_col, symbol, ret].
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _select_top_bottom(
    predictions: pl.DataFrame,
    top_k: int,
    long_short: bool,
    time_col: str = "timestamp",
    score_col: str = "y_score",
) -> pl.DataFrame:
    """Rank cross-sectionally and select top-K (and bottom-K if long_short)."""
    ranked = predictions.with_columns(
        cs_rank=pl.col(score_col).rank(method="ordinal", descending=True).over(time_col),
        n_assets=pl.col(score_col).count().over(time_col),
    )
    if long_short:
        selected = ranked.filter(
            (pl.col("cs_rank") <= top_k) | (pl.col("cs_rank") > pl.col("n_assets") - top_k)
        ).with_columns(
            side=pl.when(pl.col("cs_rank") <= top_k).then(pl.lit("long")).otherwise(pl.lit("short"))
        )
    else:
        selected = ranked.filter(pl.col("cs_rank") <= top_k).with_columns(side=pl.lit("long"))
    return selected


def _filter_prices_to_prediction_assets(
    prices_df: pl.DataFrame,
    predictions: pl.DataFrame,
    asset_col: str = "symbol",
) -> pl.DataFrame:
    """Pre-filter prices to only assets in predictions (performance optimization)."""
    pred_assets = predictions[asset_col].unique()
    return prices_df.filter(pl.col(asset_col).is_in(pred_assets))


def _returns_from_prices(
    prices_df: pl.DataFrame,
    time_col: str = "timestamp",
    asset_col: str = "symbol",
) -> pl.DataFrame:
    """Extract returns from prices: use 'ret' if available, else pct_change('close')."""
    if "ret" in prices_df.columns:
        return prices_df.select([time_col, asset_col, "ret"])
    return (
        prices_df.sort(time_col, asset_col)
        .with_columns(ret=pl.col("close").pct_change().over(asset_col))
        .select([time_col, asset_col, "ret"])
    )


def _compute_rolling_vol(
    prices_df: pl.DataFrame,
    vol_window: int = 63,
    time_col: str = "timestamp",
    asset_col: str = "symbol",
    target_dtype: pl.DataType | None = None,
) -> pl.DataFrame:
    """Compute rolling volatility from daily returns."""
    returns = _returns_from_prices(prices_df, time_col, asset_col)
    result = returns.with_columns(vol=pl.col("ret").rolling_std(vol_window).over(asset_col)).select(
        [time_col, asset_col, "vol"]
    )
    if target_dtype is not None and result[time_col].dtype != target_dtype:
        result = result.cast({time_col: target_dtype})
    return result


def _normalize_within_sides(
    selected: pl.DataFrame,
    time_col: str = "timestamp",
) -> pl.DataFrame:
    """Normalize inverse-vol weights within long/short sides separately."""
    selected = selected.with_columns(inv_vol=1.0 / pl.col("vol").clip(lower_bound=1e-6))

    long_w = selected.filter(pl.col("side") == "long").with_columns(
        weight=pl.col("inv_vol") / pl.col("inv_vol").sum().over(time_col)
    )
    short_w = selected.filter(pl.col("side") == "short").with_columns(
        weight=-pl.col("inv_vol") / pl.col("inv_vol").sum().over(time_col)
    )
    parts = [long_w]
    if short_w.height > 0:
        parts.append(short_w)
    return pl.concat(parts, how="diagonal_relaxed")


def _cap_weights(
    df: pl.DataFrame,
    max_weight: float,
    time_col: str = "timestamp",
) -> pl.DataFrame:
    """Cap per-asset weight and redistribute excess proportionally.

    Iterates until no weight exceeds max_weight (handles cascading overflow).
    Operates on long side only; short side is handled symmetrically.
    """
    if max_weight >= 1.0:
        return df

    for _ in range(20):  # convergence guard
        over = df.filter(pl.col("weight").abs() > max_weight + 1e-9)
        if over.is_empty():
            break
        df = df.with_columns(
            clipped=pl.col("weight").clip(-max_weight, max_weight),
        )
        # Redistribute excess within each timestamp
        excess_per_ts = df.group_by(time_col).agg(
            excess=(pl.col("weight") - pl.col("clipped")).sum()
        )
        n_uncapped = (
            df.filter(pl.col("weight").abs() <= max_weight).group_by(time_col).agg(n_free=pl.len())
        )
        adj = excess_per_ts.join(n_uncapped, on=time_col, how="left").with_columns(
            bump=(pl.col("excess") / pl.col("n_free").fill_null(1)).fill_null(0.0)
        )
        df = df.join(adj.select([time_col, "bump"]), on=time_col, how="left")
        df = df.with_columns(
            weight=pl.when(pl.col("weight").abs() <= max_weight)
            .then(pl.col("clipped") + pl.col("bump"))
            .otherwise(pl.col("clipped"))
        ).drop(["clipped", "bump"])

    return df


def _cluster_var(cov: np.ndarray, indices: list[int]) -> float:
    """Cluster variance: inverse-vol portfolio variance within cluster."""
    sub_cov = cov[np.ix_(indices, indices)]
    diag = np.clip(np.diag(sub_cov), 1e-10, None)
    inv_vol = 1.0 / np.sqrt(diag)
    w = inv_vol / inv_vol.sum()
    return float(w @ sub_cov @ w)


def _hrp_weights(cov_matrix: np.ndarray, corr_matrix: np.ndarray) -> np.ndarray:
    """HRP weights via Lopez de Prado (2016): cluster, quasi-diag, bisect."""
    n = cov_matrix.shape[0]
    if n <= 1:
        return np.ones(n)

    # Correlation distance
    dist = np.sqrt(0.5 * (1 - corr_matrix))
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)

    try:
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method="single")
    except Exception:
        return np.ones(n) / n

    # Quasi-diagonalize
    sort_ix = leaves_list(link).tolist()

    # Recursive bisection
    weights = np.ones(n)
    cluster_items = [sort_ix]

    while cluster_items:
        new_clusters = []
        for items in cluster_items:
            if len(items) <= 1:
                continue
            mid = len(items) // 2
            left, right = items[:mid], items[mid:]

            left_var = _cluster_var(cov_matrix, left)
            right_var = _cluster_var(cov_matrix, right)
            alpha = 1 - left_var / (left_var + right_var) if (left_var + right_var) > 0 else 0.5

            weights[left] *= alpha
            weights[right] *= 1 - alpha

            if len(left) > 1:
                new_clusters.append(left)
            if len(right) > 1:
                new_clusters.append(right)
        cluster_items = new_clusters

    weights /= weights.sum()
    return weights


# ---------------------------------------------------------------------------
# Public allocation functions
# ---------------------------------------------------------------------------


def compute_conformal_weights(
    predictions: pl.DataFrame,
    conformal_widths: pl.DataFrame,
    top_k: int,
    long_short: bool = False,
    *,
    floor_quantile: float = 0.01,
    time_col: str = "timestamp",
) -> pl.DataFrame:
    """Conformal inverse-width position sizing.

    Selects top-K by ``y_score`` and weights each selected asset by 1/Δ_i,
    normalized within each side (long/short) so the leg sums to ±1. Widths
    come from ``case_studies.utils.conformal.compute_conformal_widths`` and
    are joined on (timestamp, symbol). Assets without a calibrated width at
    that timestamp are dropped from the leg (the inverse-width sum
    renormalizes accordingly).

    A small floor at ``floor_quantile`` of the in-sample width distribution
    prevents 1/Δ blow-up when residuals happen to be identical.
    """
    selected = _select_top_bottom(predictions, top_k, long_short, time_col)

    widths = conformal_widths.select(time_col, "symbol", "width")
    # Harmonize join dtypes to predictions/weights.
    if widths[time_col].dtype != selected[time_col].dtype:
        widths = widths.cast({time_col: selected[time_col].dtype})
    if widths["symbol"].dtype != selected["symbol"].dtype:
        widths = widths.cast({"symbol": selected["symbol"].dtype})

    selected = selected.join(widths, on=[time_col, "symbol"], how="inner")
    if selected.is_empty():
        raise ValueError(
            "conformal_weighted: empty join between selected top-K predictions "
            "and conformal_widths. Likely cause: widths not computed for this "
            "prediction_hash, or fold_id range mismatch. Run "
            "compute_conformal_widths() before backtest."
        )

    floor = float(selected["width"].quantile(floor_quantile))
    floor = max(floor, 1e-12)
    selected = selected.with_columns(inv_w=1.0 / pl.max_horizontal(pl.col("width"), pl.lit(floor)))

    long_w = selected.filter(pl.col("side") == "long").with_columns(
        weight=pl.col("inv_w") / pl.col("inv_w").sum().over(time_col)
    )
    parts = [long_w]
    if long_short:
        short_w = selected.filter(pl.col("side") == "short").with_columns(
            weight=-pl.col("inv_w") / pl.col("inv_w").sum().over(time_col)
        )
        if short_w.height > 0:
            parts.append(short_w)

    result = pl.concat(parts, how="diagonal_relaxed")
    return result.select([time_col, "symbol", "weight"]).filter(pl.col("weight") != 0.0)


def compute_inverse_vol_weights(
    predictions: pl.DataFrame,
    prices_df: pl.DataFrame,
    top_k: int,
    vol_window: int = 63,
    long_short: bool = False,
    time_col: str = "timestamp",
) -> pl.DataFrame:
    """Inverse-volatility weighting: select top-K by score, weight by 1/vol.

    Normalizes weights within each side (long/short) separately.
    """
    selected = _select_top_bottom(predictions, top_k, long_short, time_col)
    _prices = _filter_prices_to_prediction_assets(prices_df, predictions)
    vol = _compute_rolling_vol(
        _prices, vol_window, time_col, target_dtype=predictions[time_col].dtype
    )

    selected = selected.join(vol, on=[time_col, "symbol"], how="left").with_columns(
        pl.col("vol").fill_null(pl.col("vol").median())
    )

    result = _normalize_within_sides(selected, time_col)
    return result.select([time_col, "symbol", "weight"]).filter(pl.col("weight") != 0.0)


def compute_risk_parity_weights(
    predictions: pl.DataFrame,
    prices_df: pl.DataFrame,
    top_k: int,
    vol_window: int = 63,
    long_short: bool = False,
    time_col: str = "timestamp",
) -> pl.DataFrame:
    """Simplified risk-parity (approximate ERC) using vol^1.5 exponent.

    Uses inverse-vol^1.5 as a proxy for equal risk contribution --- accounts
    for the empirical relationship between volatility and correlation.
    """
    selected = _select_top_bottom(predictions, top_k, long_short, time_col)
    _prices = _filter_prices_to_prediction_assets(prices_df, predictions)
    vol = _compute_rolling_vol(
        _prices, vol_window, time_col, target_dtype=predictions[time_col].dtype
    )

    selected = selected.join(vol, on=[time_col, "symbol"], how="left").with_columns(
        pl.col("vol").fill_null(pl.col("vol").median())
    )

    # Risk-parity approximation: w_i proportional to 1 / vol_i^1.5
    selected = selected.with_columns(inv_vol=1.0 / (pl.col("vol").clip(lower_bound=1e-6) ** 1.5))

    long_w = selected.filter(pl.col("side") == "long").with_columns(
        weight=pl.col("inv_vol") / pl.col("inv_vol").sum().over(time_col)
    )
    parts = [long_w]
    if long_short:
        short_w = selected.filter(pl.col("side") == "short").with_columns(
            weight=-pl.col("inv_vol") / pl.col("inv_vol").sum().over(time_col)
        )
        if short_w.height > 0:
            parts.append(short_w)

    result = pl.concat(parts, how="diagonal_relaxed")
    return result.select([time_col, "symbol", "weight"]).filter(pl.col("weight") != 0.0)


def compute_mvo_weights(
    predictions: pl.DataFrame,
    prices_df: pl.DataFrame,
    top_k: int,
    lookback: int = 126,
    max_weight: float = 0.15,
    long_short: bool = False,
    time_col: str = "timestamp",
) -> pl.DataFrame:
    """MVO with Ledoit-Wolf shrinkage and position cap.

    At each rebalance date:
    1. Select top-K assets by score
    2. Estimate covariance via LedoitWolf
    3. Use ML z-scores as expected returns
    4. Solve constrained QP: max Sharpe s.t. position caps
    """
    from scipy.optimize import minimize
    from sklearn.covariance import LedoitWolf

    selected = _select_top_bottom(predictions, top_k, long_short, time_col)

    # Pre-filter prices to prediction assets (performance: avoids pct_change on full universe)
    _prices = _filter_prices_to_prediction_assets(prices_df, predictions)

    # Cast prices time column to match predictions dtype
    if _prices[time_col].dtype != predictions[time_col].dtype:
        _prices = _prices.cast({time_col: predictions[time_col].dtype})

    rets = _returns_from_prices(_prices, time_col)

    all_timestamps = selected[time_col].unique().sort().to_list()
    rows = []

    for ts in all_timestamps:
        ts_selected = selected.filter(pl.col(time_col) == ts)
        assets = ts_selected["symbol"].to_list()
        scores = ts_selected.select(["symbol", "y_score"])
        side_map = dict(
            zip(ts_selected["symbol"].to_list(), ts_selected["side"].to_list(), strict=False)
        )
        if len(assets) < 3:
            continue

        recent = rets.filter((pl.col(time_col) <= ts) & pl.col("symbol").is_in(assets)).sort(
            time_col
        )
        recent_dates = recent[time_col].unique().sort()
        if len(recent_dates) > lookback:
            recent = recent.filter(pl.col(time_col).is_in(recent_dates.tail(lookback)))
        window_rets = (
            recent.pivot(on="symbol", index=time_col, values="ret").sort(time_col).drop(time_col)
        )

        if window_rets.height < lookback // 2:
            if long_short:
                long_assets = [a for a in assets if side_map.get(a) == "long"]
                short_assets = [a for a in assets if side_map.get(a) == "short"]
                if long_assets:
                    lw = 1.0 / len(long_assets)
                    for a in long_assets:
                        rows.append({time_col: ts, "symbol": a, "weight": lw})
                if short_assets:
                    sw = -1.0 / len(short_assets)
                    for a in short_assets:
                        rows.append({time_col: ts, "symbol": a, "weight": sw})
            else:
                w = 1.0 / len(assets)
                for a in assets:
                    rows.append({time_col: ts, "symbol": a, "weight": w})
            continue

        ret_matrix = window_rets.to_numpy()
        valid_mask = ~np.all(np.isnan(ret_matrix), axis=0)
        valid_assets = [a for a, v in zip(window_rets.columns, valid_mask, strict=False) if v]
        ret_matrix = ret_matrix[:, valid_mask]
        ret_matrix = ret_matrix[~np.any(np.isnan(ret_matrix), axis=1)]

        min_obs = max(top_k, lookback // 2)
        if ret_matrix.shape[0] < min_obs or ret_matrix.shape[1] < 3:
            if long_short:
                long_assets = [a for a in assets if side_map.get(a) == "long"]
                short_assets = [a for a in assets if side_map.get(a) == "short"]
                if long_assets:
                    lw = 1.0 / len(long_assets)
                    for a in long_assets:
                        rows.append({time_col: ts, "symbol": a, "weight": lw})
                if short_assets:
                    sw = -1.0 / len(short_assets)
                    for a in short_assets:
                        rows.append({time_col: ts, "symbol": a, "weight": sw})
            else:
                w = 1.0 / len(assets)
                for a in assets:
                    rows.append({time_col: ts, "symbol": a, "weight": w})
            continue

        cov = LedoitWolf().fit(ret_matrix).covariance_
        score_map = dict(zip(scores["symbol"].to_list(), scores["y_score"].to_list(), strict=False))
        mu = np.array([score_map.get(a, 0.0) for a in valid_assets])
        mu_std = mu.std()
        if mu_std > 0:
            mu = (mu - mu.mean()) / mu_std

        n = len(valid_assets)

        def neg_sharpe(w, mu=mu, cov=cov):
            port_ret = w @ mu
            port_vol = np.sqrt(w @ cov @ w)
            return -port_ret / max(port_vol, 1e-8)

        if long_short:
            bounds = [(-max_weight, max_weight)] * n
            constraints = [{"type": "eq", "fun": lambda w: np.sum(w)}]  # dollar neutral
            w0 = mu / max(np.abs(mu).sum(), 1e-8)
        else:
            bounds = [(0.0, max_weight)] * n
            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
            w0 = np.ones(n) / n

        result = minimize(
            neg_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10},
        )

        w_opt = result.x if result.success else w0
        if long_short:
            w_sum = np.abs(w_opt).sum()
            if w_sum > 0:
                w_opt = w_opt / w_sum
        else:
            w_opt = np.maximum(w_opt, 0)
            w_opt /= w_opt.sum()

        for a, w in zip(valid_assets, w_opt, strict=False):
            if abs(w) > 1e-6:
                rows.append({time_col: ts, "symbol": a, "weight": float(w)})

    if not rows:
        return pl.DataFrame(
            schema={
                time_col: predictions[time_col].dtype,
                "symbol": pl.String,
                "weight": pl.Float64,
            }
        )

    return pl.DataFrame(rows).sort(time_col, "symbol")


def compute_hrp_weights(
    predictions: pl.DataFrame,
    prices_df: pl.DataFrame,
    top_k: int,
    vol_window: int = 63,
    long_short: bool = False,
    min_coverage: float = 0.5,
    time_col: str = "timestamp",
) -> pl.DataFrame:
    """HRP allocation (Lopez de Prado, 2016).

    Applies HRP separately to long and short legs using a rolling
    correlation window. Falls back to equal-weight if insufficient history.

    Bug fix vs original: drops assets with <min_coverage observations in the
    rolling window instead of fill_null(0.0), which corrupted the covariance
    matrix for sparse panels.
    """
    selected = _select_top_bottom(predictions, top_k, long_short, time_col)

    # Pre-filter prices to prediction assets (performance: avoids pct_change on full universe)
    _prices = _filter_prices_to_prediction_assets(prices_df, predictions)

    # Cast prices time column to match predictions dtype
    if _prices[time_col].dtype != predictions[time_col].dtype:
        _prices = _prices.cast({time_col: predictions[time_col].dtype})

    returns = _returns_from_prices(_prices, time_col)

    timestamps = sorted(selected[time_col].unique().to_list())
    all_weights: list[dict] = []

    for ts in timestamps:
        for side_label, sign in [("long", 1.0), ("short", -1.0)]:
            if side_label == "short" and not long_short:
                continue
            side_assets = selected.filter(
                (pl.col(time_col) == ts) & (pl.col("side") == side_label)
            )["symbol"].to_list()
            if not side_assets:
                continue

            # Get recent returns for these assets
            recent = returns.filter(
                (pl.col(time_col) <= ts) & (pl.col("symbol").is_in(side_assets))
            ).sort(time_col)

            recent_dates = recent[time_col].unique().sort()
            if len(recent_dates) > vol_window:
                recent = recent.filter(pl.col(time_col).is_in(recent_dates.tail(vol_window)))

            # Pivot to wide format — drop assets with insufficient coverage
            pivot = recent.pivot(on="symbol", index=time_col, values="ret").drop(time_col)

            if pivot.shape[0] < 20 or pivot.shape[1] < 2:
                w = 1.0 / len(side_assets)
                for a in side_assets:
                    all_weights.append({time_col: ts, "symbol": a, "weight": sign * w})
                continue

            # Drop columns (assets) with <50% non-null coverage in the window
            min_obs = int(pivot.shape[0] * min_coverage)
            valid_cols = [c for c in pivot.columns if pivot[c].drop_nulls().len() >= min_obs]

            if len(valid_cols) < 2:
                w = 1.0 / len(side_assets)
                for a in side_assets:
                    all_weights.append({time_col: ts, "symbol": a, "weight": sign * w})
                continue

            # Use only valid columns, drop rows with any remaining NaN
            ret_matrix = pivot.select(valid_cols).drop_nulls().to_numpy()

            if ret_matrix.shape[0] < 20 or ret_matrix.shape[1] < 2:
                w = 1.0 / len(side_assets)
                for a in side_assets:
                    all_weights.append({time_col: ts, "symbol": a, "weight": sign * w})
                continue

            cov = np.cov(ret_matrix.T)
            std = np.sqrt(np.clip(np.diag(cov), 1e-16, None))
            corr = cov / np.outer(std, std)
            corr = np.clip(corr, -1, 1)

            hrp_w = _hrp_weights(cov, corr)

            # Assign HRP weights to valid assets; equal-weight the rest
            hrp_asset_map = dict(zip(valid_cols, hrp_w, strict=False))
            remaining = [a for a in side_assets if a not in hrp_asset_map]

            # Rescale: HRP assets get their share, remaining get residual
            if remaining:
                hrp_total = sum(hrp_asset_map.values())
                remain_w = (1.0 - hrp_total) / len(remaining) if hrp_total < 1.0 else 0.0
                for a in remaining:
                    all_weights.append({time_col: ts, "symbol": a, "weight": sign * remain_w})

            for a, w in hrp_asset_map.items():
                all_weights.append({time_col: ts, "symbol": a, "weight": sign * w})

    if not all_weights:
        return pl.DataFrame(
            schema={time_col: pl.Datetime, "symbol": pl.String, "weight": pl.Float64}
        )

    return pl.DataFrame(all_weights).sort(time_col, "symbol")
