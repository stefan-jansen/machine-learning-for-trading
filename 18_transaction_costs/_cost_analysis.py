"""Shared transaction cost analysis for Ch18 notebooks and case study costs.py.

Provides:
- corwin_schultz_spread(): High-low spread estimator (Corwin & Schultz 2012)
- roll_spread(): Serial covariance spread estimator (Roll 1984)
- compute_adv(): Rolling average daily volume
- compute_adv_usd(): Rolling average daily dollar volume
- calibrate_sqrt_impact(): Fit η in Impact = σ·η·√(Q/V)
- estimate_kyle_lambda(): Linear price-impact coefficient ΔP = λQ
- estimate_capacity(): Max AUM given impact coefficient and alpha
- breakeven_alpha(): Required gross alpha given turnover and costs
- get_fee_schedule(): Codified exchange fee schedules

All functions operate on Polars Series/DataFrames.
"""

from __future__ import annotations

import numpy as np
import polars as pl

# =============================================================================
# SPREAD ESTIMATION
# =============================================================================


def corwin_schultz_spread(
    high: pl.Series | pl.Expr,
    low: pl.Series | pl.Expr,
    window: int = 1,
) -> pl.Series | pl.Expr:
    """Corwin-Schultz (2012) high-low spread estimator.

    Estimates the bid-ask spread from daily high and low prices using the
    insight that daily highs (lows) are predominantly at ask (bid) prices.

    The two-period estimator uses:
        β = E[ln(H/L)²]  over consecutive single periods
        γ = ln(H₂/L₂)²   where H₂, L₂ are 2-period high/low
        α = (√2β - √β) / (3 - 2√2) - √(γ / (3 - 2√2))

    Spread S = 2(eᵅ - 1) / (1 + eᵅ)

    Args:
        high: High prices (Series or Expr)
        low: Low prices (Series or Expr)
        window: Rolling window for averaging β (default 1 = raw estimator)

    Returns:
        Estimated spread as fraction (not bps). Negative values clamped to 0.
    """
    ln_hl = (high / low).log()
    ln_hl_sq = ln_hl**2

    # β: average of sum of consecutive single-period squared log ranges
    beta = ln_hl_sq + ln_hl_sq.shift(1)

    # γ: squared log range over 2-period high/low
    high_2 = high.rolling_max(2)
    low_2 = low.rolling_min(2)
    gamma = (high_2 / low_2).log() ** 2

    if window > 1:
        beta = beta.rolling_mean(window)
        gamma = gamma.rolling_mean(window)

    # α coefficient
    denom = 3 - 2 * np.sqrt(2)  # ≈ 0.1716

    alpha = (((2 * beta).sqrt() - beta.sqrt()) / denom) - (gamma / denom).sqrt()

    # Spread = 2(eᵅ - 1) / (1 + eᵅ)
    exp_alpha = alpha.exp()
    spread = 2 * (exp_alpha - 1) / (1 + exp_alpha)

    # Clamp negatives to zero
    return spread.clip(lower_bound=0)


def roll_spread(close: pl.Series | pl.Expr, window: int = 20) -> pl.Series | pl.Expr:
    """Roll (1984) serial covariance spread estimator.

    If the bid-ask bounce is the dominant source of serial correlation in
    returns, then: Spread = 2√(-Cov(Δpₜ, Δpₜ₋₁))

    Only defined when autocovariance is negative (efficient market condition).

    Args:
        close: Closing prices
        window: Rolling window for covariance estimation

    Returns:
        Estimated spread as fraction. Returns 0 where cov > 0.
    """
    ret = close.pct_change()
    ret_lag = ret.shift(1)

    # Rolling covariance: Cov(rₜ, rₜ₋₁)
    # Using: Cov(X,Y) = E[XY] - E[X]E[Y]
    cov = (ret * ret_lag).rolling_mean(window) - ret.rolling_mean(window) * ret_lag.rolling_mean(
        window
    )

    # Spread = 2 * sqrt(-cov) where cov < 0, else 0
    neg_cov = (-cov).clip(lower_bound=0)
    return 2 * neg_cov.sqrt()


# =============================================================================
# VOLUME & IMPACT
# =============================================================================


def compute_adv(volume: pl.Series | pl.Expr, window: int = 20) -> pl.Series | pl.Expr:
    """Rolling average daily volume (shares/contracts)."""
    return volume.rolling_mean(window)


def compute_adv_usd(
    volume: pl.Series | pl.Expr,
    close: pl.Series | pl.Expr,
    window: int = 20,
) -> pl.Series | pl.Expr:
    """Rolling average daily dollar volume."""
    return (volume * close).rolling_mean(window)


def calibrate_sqrt_impact(
    returns: np.ndarray,
    volume: np.ndarray,
    sigma: np.ndarray,
    adv: np.ndarray,
    *,
    min_adv: float = 1e3,
) -> dict:
    """Calibrate η in the square-root impact model: |r| = σ · η · √(V/ADV).

    Uses OLS regression of |r|/σ on √(V/ADV) to estimate η.

    Args:
        returns: Daily returns
        volume: Daily volume
        sigma: Rolling volatility (same frequency as returns)
        adv: Average daily volume
        min_adv: Minimum ADV filter to avoid division by near-zero

    Returns:
        dict with keys: eta, r_squared, std_err, n_obs
    """
    from sklearn.linear_model import LinearRegression

    # Filter valid observations
    mask = (
        np.isfinite(returns)
        & np.isfinite(volume)
        & np.isfinite(sigma)
        & np.isfinite(adv)
        & (sigma > 0)
        & (adv > min_adv)
    )
    r = np.abs(returns[mask])
    s = sigma[mask]
    v = volume[mask]
    a = adv[mask]

    if len(r) < 30:
        return {"eta": np.nan, "r_squared": np.nan, "std_err": np.nan, "n_obs": len(r)}

    # y = |r| / σ,  x = √(V / ADV)
    y = r / s
    x = np.sqrt(v / a).reshape(-1, 1)

    reg = LinearRegression(fit_intercept=False)
    reg.fit(x, y)

    y_pred = reg.predict(x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    n = len(y)
    std_err = np.sqrt(ss_res / (n - 1)) / np.sqrt(np.sum(x**2)) if n > 1 else np.nan

    return {
        "eta": float(reg.coef_[0]),
        "r_squared": float(r_squared),
        "std_err": float(std_err),
        "n_obs": n,
    }


def estimate_kyle_lambda(
    price_changes: np.ndarray,
    signed_volume: np.ndarray,
) -> dict:
    """Estimate Kyle's lambda: ΔP = λ · Q + ε.

    Uses HuberRegressor for robustness to outliers.

    Args:
        price_changes: Price changes (ΔP)
        signed_volume: Signed order flow (Q, positive = buy-initiated)

    Returns:
        dict with keys: lambda_, r_squared, std_err, n_obs
    """
    from sklearn.linear_model import HuberRegressor

    mask = np.isfinite(price_changes) & np.isfinite(signed_volume) & (signed_volume != 0)
    dp = price_changes[mask]
    sv = signed_volume[mask].reshape(-1, 1)

    if len(dp) < 30:
        return {"lambda_": np.nan, "r_squared": np.nan, "std_err": np.nan, "n_obs": len(dp)}

    reg = HuberRegressor(fit_intercept=True)
    reg.fit(sv, dp)

    y_pred = reg.predict(sv)
    ss_res = np.sum((dp - y_pred) ** 2)
    ss_tot = np.sum((dp - dp.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    n = len(dp)
    std_err = (
        np.sqrt(ss_res / (n - 2)) / np.sqrt(np.sum((sv - sv.mean()) ** 2)) if n > 2 else np.nan
    )

    return {
        "lambda_": float(reg.coef_[0]),
        "r_squared": float(r_squared),
        "std_err": float(std_err),
        "n_obs": n,
    }


# =============================================================================
# CAPACITY & BREAKEVEN
# =============================================================================


def estimate_capacity(
    adv_usd: float,
    impact_coeff: float,
    gross_alpha_bps: float,
    turnover: float = 1.0,
    max_participation: float = 0.01,
) -> dict:
    """Estimate strategy capacity (maximum AUM).

    A strategy's capacity is limited by market impact eating into gross alpha.
    At max AUM, net alpha ≈ 0.

    Uses: Impact_bps ≈ impact_coeff * 10_000 * √(trade_$ / ADV_$)
    where trade_$ = AUM * turnover * max_participation

    Args:
        adv_usd: Average daily dollar volume of the universe
        impact_coeff: Calibrated η from sqrt impact model
        gross_alpha_bps: Expected gross alpha in bps per rebalance
        turnover: One-way turnover per rebalance (fraction)
        max_participation: Maximum volume participation rate

    Returns:
        dict with max_aum_usd, breakeven_participation, impact_at_max_bps
    """
    if impact_coeff <= 0 or gross_alpha_bps <= 0 or adv_usd <= 0:
        return {"max_aum_usd": 0.0, "breakeven_participation": 0.0, "impact_at_max_bps": 0.0}

    # Solve: gross_alpha_bps = impact_coeff * 10_000 * sqrt(participation)
    # => participation = (gross_alpha_bps / (impact_coeff * 10_000))²
    breakeven_participation = (gross_alpha_bps / (impact_coeff * 10_000)) ** 2
    breakeven_participation = min(breakeven_participation, max_participation)

    # AUM = participation * ADV / turnover
    max_aum = breakeven_participation * adv_usd / max(turnover, 1e-6)

    impact_at_max = impact_coeff * 10_000 * np.sqrt(breakeven_participation)

    return {
        "max_aum_usd": float(max_aum),
        "breakeven_participation": float(breakeven_participation),
        "impact_at_max_bps": float(impact_at_max),
    }


def breakeven_alpha(turnover: float, cost_bps: float) -> float:
    """Required gross alpha (as decimal) to break even after costs.

    Args:
        turnover: Annual one-way turnover (e.g., 12 for monthly rebalance)
        cost_bps: Round-trip cost in basis points

    Returns:
        Required annual gross alpha as decimal (e.g., 0.01 = 1%)
    """
    return turnover * cost_bps / 10_000


# =============================================================================
# FEE SCHEDULES
# =============================================================================

FEE_SCHEDULES = {
    "us_equities": {
        "name": "US Equities (IB Pro)",
        "commission_per_share": 0.005,
        "min_commission": 1.00,
        "sec_fee_per_million": 27.80,
        "finra_taf_per_share": 0.000166,
        "exchange_rebate_per_share": -0.002,  # Maker rebate
        "exchange_fee_per_share": 0.003,  # Taker fee
        "notes": "IB Pro tiered pricing. SEC/FINRA fees on sells only.",
    },
    "etfs": {
        "name": "ETFs (IB Pro)",
        "commission_per_share": 0.005,
        "min_commission": 1.00,
        "sec_fee_per_million": 27.80,
        "notes": "Same as equities. Commission-free at some brokers.",
    },
    "crypto_perps": {
        "name": "Crypto Perpetuals (Binance)",
        "taker_bps": 4.0,
        "maker_bps": 2.0,
        "funding_rate_note": "8h funding rate (not a trading cost)",
        "notes": "Binance USDT-M futures. VIP tiers reduce fees.",
    },
    "cme_futures": {
        "name": "CME Futures",
        "commission_per_contract": 2.00,
        "exchange_fee_per_contract": 1.50,
        "nfa_fee_per_contract": 0.02,
        "clearing_fee_per_contract": 0.10,
        "notes": "CME Group all-in costs. Varies by product.",
    },
    "fx_spot": {
        "name": "FX Spot (OANDA-style)",
        "spread_bps_major": 1.5,
        "spread_bps_cross": 4.0,
        "commission_bps": 0.0,
        "swap_points_note": "Overnight roll cost varies by pair and direction",
        "notes": "Spread-only pricing. No separate commission.",
    },
    "sp500_options": {
        "name": "US Equity Options (IB Pro)",
        "commission_per_contract": 0.65,
        "min_commission": 1.00,
        "exchange_fee_per_contract": 0.30,
        "occ_fee_per_contract": 0.055,
        "notes": "Options Clearing Corporation + exchange fees.",
    },
}


def get_fee_schedule(asset_class: str) -> dict:
    """Get codified fee schedule for an asset class.

    Args:
        asset_class: One of 'us_equities', 'etfs', 'crypto_perps',
            'cme_futures', 'fx_spot', 'sp500_options'

    Returns:
        Dict with fee components and notes
    """
    if asset_class not in FEE_SCHEDULES:
        available = ", ".join(sorted(FEE_SCHEDULES.keys()))
        raise ValueError(f"Unknown asset class '{asset_class}'. Available: {available}")
    return FEE_SCHEDULES[asset_class]


def standardized_cost_per_100k(asset_class: str, price: float = 50.0) -> dict:
    """Estimate total cost for a $100K trade by asset class.

    Useful for cross-asset comparison. Returns cost breakdown in bps.

    Args:
        asset_class: Fee schedule key
        price: Representative price per unit (for per-share fees)

    Returns:
        dict with commission_bps, exchange_bps, total_bps
    """
    trade_usd = 100_000
    fees = get_fee_schedule(asset_class)

    if asset_class in ("us_equities", "etfs"):
        shares = trade_usd / price
        commission = max(shares * fees["commission_per_share"], fees["min_commission"])
        exchange = shares * fees.get("exchange_fee_per_share", 0.003)
        sec = (trade_usd / 1e6) * fees.get("sec_fee_per_million", 27.80)
        total = commission + exchange + sec
    elif asset_class == "crypto_perps":
        total = trade_usd * fees["taker_bps"] / 10_000
        commission = total
        exchange = 0
    elif asset_class == "cme_futures":
        # Representative CME contract: median notional ~$75K across product mix
        # (e.g., E-mini ES ~$260K, corn ~$23K, crude ~$70K, gold ~$230K)
        contracts = trade_usd / 75_000
        per_contract = (
            fees["commission_per_contract"]
            + fees["exchange_fee_per_contract"]
            + fees["nfa_fee_per_contract"]
            + fees["clearing_fee_per_contract"]
        )
        total = contracts * per_contract
        commission = contracts * fees["commission_per_contract"]
        exchange = total - commission
    elif asset_class == "fx_spot":
        total = trade_usd * fees["spread_bps_major"] / 10_000
        commission = 0
        exchange = total
    elif asset_class == "sp500_options":
        # Assume ATM option at $5 premium, 100 shares per contract
        contracts = trade_usd / 500  # $5 * 100
        per_contract = (
            fees["commission_per_contract"]
            + fees["exchange_fee_per_contract"]
            + fees["occ_fee_per_contract"]
        )
        total = contracts * per_contract
        commission = contracts * fees["commission_per_contract"]
        exchange = total - commission
    else:
        return {"commission_bps": 0, "exchange_bps": 0, "total_bps": 0}

    total_bps = total / trade_usd * 10_000
    commission_bps = commission / trade_usd * 10_000
    exchange_bps = (total - commission) / trade_usd * 10_000

    return {
        "commission_bps": round(float(commission_bps), 2),
        "exchange_bps": round(float(exchange_bps), 2),
        "total_bps": round(float(total_bps), 2),
    }
