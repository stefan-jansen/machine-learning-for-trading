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
# # Slow Features and Context: Fundamentals, Macro, Calendar
#
# **Chapter 8: Feature Engineering**
# **Section Reference**: 8.4 - Contextual and Slow-Moving Features
# **Docker image**: `ml4t`
#
# ## Purpose
#
# This notebook covers **slow-moving features** that provide context for faster signals:
#
# 1. **Fundamentals**: Value, quality, growth factors from financial statements
# 2. **Macro**: Economic indicators, yield curves, credit spreads, risk regimes
# 3. **Calendar**: Cyclical encodings for seasonal patterns
#
# ## Key Principle
#
# Slow features update infrequently (quarterly, monthly, or by schedule) but
# condition daily decisions. The binding constraint is **data integrity** —
# ensuring each observation reflects only what was knowable at decision time.
#
# ## Data Policy
#
# All examples use **real data** (SEC XBRL, FRED macro).
#
# ## References
#
# - Fama and French (1992, 1993): Value, size, profitability factors
# - Cochrane (2011): "Presidential Address: Discount Rates" — factor zoo
# - Harvey, Liu, and Zhu (2016): "...and the Cross-Section of Expected Returns"
#
# ## Case Study Mapping
#
# | Case Study | Relevant Features |
# |------------|-------------------|
# | ETFs (`etfs`) | Calendar encodings, macro regimes |
# | US Firm Characteristics (`us_firm_characteristics`) | All fundamental factors |
# | S&P 500 Equity+Options (`sp500_equity_option_analytics`) | Macro + VIX regime |

# %%
"""Slow Features and Context: Fundamentals, Macro, Calendar — contextual features that condition faster signals."""

from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta

import polars as pl

warnings.filterwarnings("ignore")

from data import load_macro as _load_macro_canonical
from data import load_sec_xbrl_fundamentals

# %% tags=["parameters"]
SEED = 42
CALENDAR_START_DATE = "2015-01-01"

# %% [markdown]
# ---
#
# # Part 1: Fundamental Factors
#
# Fundamental factors update quarterly but inform daily trading decisions.
#
# **Key challenges**:
# - Point-in-time accuracy (use announcement date, not period end)
# - Forward-filling to daily frequency
# - Factor staleness between announcements

# %% [markdown]
# ## 1.1 Load Fundamental Data
#
# ### Scope: scaffolding for the construction mechanics, not a real-data value pipeline
#
# `load_fundamentals()` reads SEC XBRL filings. XBRL publishes accounting numbers
# (book equity, earnings, operating cash flow, capex) but does **not** publish
# market capitalization — that comes from market prices on the announcement
# date. To keep the value-factor cells below executable on the XBRL output
# alone, this notebook approximates `market_cap = 2 × book_value`. This is a
# **scaffolding** value: it lets the downstream `compute_value_factors()` cell
# show the mechanics of book-to-market, earnings yield, and cash-flow yield,
# but the resulting numbers are **not** the real-data factor values.
#
# The lookahead-safe, real-data version (XBRL fundamentals joined to daily
# adjusted prices on the announcement date, with point-in-time discipline) is
# demonstrated in the `us_firm_characteristics` case study and uses the
# `load_firm_characteristics()` loader from `data/equities/loader.py`. See
# Chapter 11's case study pipeline (`case_studies/us_firm_characteristics/`)
# and the Chen-Pelger-Zhu (2020) panel for the production version.

# %%
# Denominator safety constant (used by all factor computations)
EPSILON = 1e-10


# %%
# Map lowercase us-gaap concepts to the shorter names used downstream.
# The XBRL loader exposes one column per us-gaap concept in lowercase.
_XBRL_RENAMES = {
    "stockholdersequity": "book_value",
    "netincomeloss": "earnings",
    "revenues": "revenue",
    "netcashprovidedbyusedinoperatingactivities": "operating_cf",
    "longtermdebt": "total_debt",
    "paymentstoacquirepropertyplantandequipment": "capex",
}


def load_fundamentals() -> pl.DataFrame:
    """Load SEC XBRL fundamentals and normalize to factor-friendly names.

    Note: `market_cap` remains a SCAFFOLDING approximation (2× book value)
    because XBRL does not publish market capitalization. Production systems
    should join with actual price data on the announcement date.
    """
    df = load_sec_xbrl_fundamentals().rename(_XBRL_RENAMES)

    # `assets` preserves its lowercase concept name; alias for downstream code.
    df = df.with_columns(
        [
            # Market cap approximation — SCAFFOLDING only (XBRL has no market cap)
            (pl.col("book_value") * 2.0).alias("market_cap"),
            pl.col("assets").alias("total_assets"),
        ]
    )

    # Accruals (earnings - operating CF)
    if "operating_cf" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("operating_cf").is_not_null())
            .then(pl.col("earnings") - pl.col("operating_cf"))
            .otherwise(0.0)
            .alias("accruals")
        )

    return df


fundamentals = load_fundamentals()
print(f"Fundamental data: {len(fundamentals):,} rows, {fundamentals['symbol'].n_unique()} symbols")
fundamentals.head(5)

# %% [markdown]
# ## 1.2 Value Factors
#
# Value factors identify stocks trading at discounts relative to fundamentals.
#
# > **Reminder**: every factor below has `market_cap` in the denominator and
# > `market_cap` is the `2 × book_value` scaffolding from §1.1. The cell
# > demonstrates the *construction* of book-to-market, earnings yield, and
# > cash-flow yield; the *values* are not the real-data factor values. See
# > the `us_firm_characteristics` case study for the production version.


# %%
def compute_value_factors(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute value factors with denominator clipping for safety.
    """
    return df.with_columns(
        [
            # Book-to-Market
            (pl.col("book_value") / pl.col("market_cap").clip(EPSILON, None)).alias(
                "book_to_market"
            ),
            # Earnings yield
            (pl.col("earnings") / pl.col("market_cap").clip(EPSILON, None)).alias("earnings_yield"),
            # Cash flow yield
            (pl.col("operating_cf") / pl.col("market_cap").clip(EPSILON, None)).alias("cf_yield"),
            # FCF yield
            (
                (pl.col("operating_cf") - pl.col("capex"))
                / pl.col("market_cap").clip(EPSILON, None)
            ).alias("fcf_yield"),
        ]
    )


value_df = compute_value_factors(fundamentals)
print("Value factors computed:")
value_df.select(["symbol", "fiscal_quarter_end", "book_to_market", "earnings_yield"]).tail(10)

# %% [markdown]
# **Interpretation**: A book-to-market ratio of 0.5 means the stock trades at
# 2x its book value — the market assigns significant intangible/growth premium.
# Earnings yield is the inverse of the P/E ratio, making higher values more
# "value-oriented."

# %% [markdown]
# ## 1.3 Quality Factors
#
# Quality factors identify financially healthy companies.


# %%
def compute_quality_factors(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute quality factors with denominator safety.
    """
    return df.with_columns(
        [
            # ROE
            (pl.col("earnings") / pl.col("book_value").clip(EPSILON, None)).alias("roe"),
            # ROA
            (pl.col("earnings") / pl.col("total_assets").clip(EPSILON, None)).alias("roa"),
            # Accruals ratio (lower = better quality)
            (pl.col("accruals") / pl.col("total_assets").clip(EPSILON, None)).alias(
                "accruals_ratio"
            ),
            # Leverage
            (pl.col("total_debt") / pl.col("total_assets").clip(EPSILON, None)).alias(
                "debt_to_assets"
            ),
        ]
    )


quality_df = compute_quality_factors(value_df)
print("Quality factors computed:")
quality_df.select(["symbol", "fiscal_quarter_end", "roe", "roa", "accruals_ratio"]).tail(10)

# %% [markdown]
# ## 1.4 Daily Alignment with Correct ASOF Join
#
# **Critical**: Both DataFrames must be sorted by the join keys.
#
# ```python
# # WRONG: Only sorting by date
# daily_df.join_asof(fundamental_df.sort("timestamp"), on="timestamp")
#
# # CORRECT: Sort both by [symbol, date]
# daily_df.sort(["symbol", "timestamp"]).join_asof(
#     fundamental_df.sort(["symbol", "announcement_date"]),
#     left_on="timestamp",
#     right_on="announcement_date",
#     by="symbol",
# )
# ```


# %%
def align_factors_to_daily(
    factor_df: pl.DataFrame,
    daily_dates: pl.DataFrame,
    announcement_col: str = "announcement_date",
) -> pl.DataFrame:
    """
    Align quarterly factors to daily frequency using ASOF join.

    CRITICAL: Both frames must be sorted by join keys.
    """
    # Ensure sorting on both frames (REQUIRED for join_asof)
    factor_sorted = factor_df.sort(["symbol", announcement_col])
    daily_sorted = daily_dates.sort(["symbol", "timestamp"])

    # ASOF join: each day gets most recent announced values
    aligned = daily_sorted.join_asof(
        factor_sorted,
        left_on="timestamp",
        right_on=announcement_col,
        by="symbol",
        strategy="backward",
    )

    return aligned


# Create daily dates for alignment demo
symbols = quality_df["symbol"].unique().to_list()
daily_dates = (
    pl.DataFrame(
        {"timestamp": pl.date_range(date(2024, 1, 1), date(2024, 12, 31), "1d", eager=True)}
    )
    .filter(pl.col("timestamp").dt.weekday() < 5)  # Business days
    .join(pl.DataFrame({"symbol": symbols}), how="cross")
)

aligned = align_factors_to_daily(
    quality_df.select(
        ["symbol", "announcement_date", "fiscal_quarter_end", "roe", "book_to_market"]
    ),
    daily_dates,
)

print(f"Daily aligned: {len(aligned):,} rows")
aligned.filter(pl.col("symbol") == symbols[0]).head(10)

# %% [markdown]
# ### 1.5 Fake Sample Size Warning
#
# Forward-filling quarterly data to daily frequency inflates the apparent
# sample size. Each unique fundamental observation appears ~63 times (one
# quarter of trading days), but carries the same information.

# %%
# Count unique fundamental observations vs total daily rows
if len(aligned) > 0:
    n_daily = len(aligned)
    # Approximate unique observations: distinct (symbol, fiscal_quarter_end) pairs
    n_unique = (
        aligned.drop_nulls(["fiscal_quarter_end"])
        .select(["symbol", "fiscal_quarter_end"])
        .unique()
        .shape[0]
    )
    inflation_ratio = n_daily / max(n_unique, 1)

    print(f"Daily rows:           {n_daily:,}")
    print(f"Unique observations:  {n_unique:,}")
    print(f"Inflation ratio:      {inflation_ratio:.0f}x")
    print(
        "\nEach fundamental observation is repeated ~63 times via forward-fill."
        "\nThis inflates t-statistics if not accounted for."
        "\nSee Section 7.2 on uniqueness weighting for the correction."
    )

# %% [markdown]
# ---
#
# # Part 2: Macro Features
#
# Macro data comes at mixed frequencies (daily, weekly, monthly, quarterly).
#
# **Key considerations**:
# - **Publication lag**: Monthly data has 2-4 week delay
# - **Revisions**: Initial estimates are often revised
# - **Forward-fill carefully**: Limit to avoid stale data

# %% [markdown]
# ## 2.1 Load Macro Data
#
# > **Publication Lag Warning**: Macro data has significant publication delays.
# > Conservative approach: Lag monthly data by 30+ days.


# %%
macro = _load_macro_canonical()
print(f"Macro data: {len(macro):,} rows")
print(f"Columns: {[c for c in macro.columns if c != 'timestamp'][:10]}")

# %% [markdown]
# ## 2.2 Trend Features with Publication Lag
#
# > **Conservative Lagging**: For monthly data, add 30-day lag to ensure
# > the data was actually available at the trading date.


# %%
def create_macro_trend_features(
    df: pl.DataFrame,
    cols: list[str],
    windows: list[int] = [21, 63, 252],
    conservative_lag: int = 0,  # Days to lag for publication delay
) -> pl.DataFrame:
    """
    Create trend features from macro data.

    Args:
        df: Macro data
        cols: Columns to process
        windows: Rolling window sizes
        conservative_lag: Days to lag for publication delay safety
    """
    # Apply conservative lag if specified
    if conservative_lag > 0:
        lag_exprs = [
            pl.col(c).shift(conservative_lag).alias(f"{c}_lagged") for c in cols if c in df.columns
        ]
        df = df.with_columns(lag_exprs)
        cols = [f"{c}_lagged" for c in cols if c in df.columns]

    feature_exprs = []
    for col in cols:
        if col not in df.columns:
            continue

        for w in windows:
            # Z-score
            feature_exprs.append(
                (
                    (pl.col(col) - pl.col(col).rolling_mean(w))
                    / pl.col(col).rolling_std(w).clip(EPSILON, None)
                ).alias(f"{col}_zscore_{w}d")
            )
            # Rate of change
            feature_exprs.append(pl.col(col).pct_change(w).alias(f"{col}_roc_{w}d"))

    return df.with_columns(feature_exprs)


# Apply to VIX (daily, no lag needed)
daily_cols = ["vixcls", "dgs10", "t10y2y"]
macro_features = create_macro_trend_features(
    macro,
    [c for c in daily_cols if c in macro.columns],
    windows=[21, 63],
)

print(f"Macro features: {len(macro_features.columns)} columns")

# %% [markdown]
# **Interpretation**: Z-scored macro data measures whether the current indicator
# level is unusual relative to its recent history. A VIX z-score of +2 means
# fear is elevated relative to the last 21 or 63 days — this conditions how
# momentum and carry signals perform.

# %% [markdown]
# ## 2.3 Monthly Features with Correct Forward-Fill
#
# **Fix**: Use forward-filled version for YoY/3m changes, not raw monthly.


# %%
def create_monthly_features(
    df: pl.DataFrame,
    monthly_cols: list[str],
    conservative_lag: int = 30,  # Monthly data publication delay
) -> pl.DataFrame:
    """
    Create features from monthly macro data.

    Uses forward-filled version for change calculations.
    Applies conservative lag for publication delay.
    """
    feature_exprs = []

    for col in monthly_cols:
        if col not in df.columns:
            continue

        # Forward-fill with limit (avoid very stale data)
        ffill_col = f"{col}_ffill"
        df = df.with_columns(
            pl.col(col).shift(conservative_lag).forward_fill(limit=45).alias(ffill_col)
        )

        # YoY change (using forward-filled, lagged version)
        feature_exprs.append(pl.col(ffill_col).pct_change(252).alias(f"{col}_yoy"))
        # 3-month change
        feature_exprs.append(pl.col(ffill_col).pct_change(63).alias(f"{col}_3m_chg"))

    if feature_exprs:
        df = df.with_columns(feature_exprs)

    return df


# Example: unemployment (monthly)
if "unrate" in macro.columns:
    macro_features = create_monthly_features(macro_features, ["unrate"], conservative_lag=30)

# %% [markdown]
# ## 2.4 Relative Value Features
#
# **Naming fix**: Rolling median ≠ percentile rank. Be precise.


# %%
def create_relative_value_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create relative value features with correct naming.

    Note: rolling_median is NOT a percentile rank - it's the median value.
    Percentile rank would be: rank(current) / count (0-100 scale).
    """
    feature_exprs = []

    # Credit spread (if available)
    if "bamlc0a0cm" in df.columns:
        feature_exprs.append(pl.col("bamlc0a0cm").alias("credit_spread"))

    # Term spread (if available)
    if "t10y2y" in df.columns:
        feature_exprs.append(pl.col("t10y2y").alias("term_spread"))

    if feature_exprs:
        df = df.with_columns(feature_exprs)

    # Rolling MEDIAN (not percentile - be precise about naming)
    median_cols = ["vixcls", "credit_spread", "term_spread"]
    median_exprs = [
        pl.col(c).rolling_median(252).alias(f"{c}_rolling_median_252d")
        for c in median_cols
        if c in df.columns
    ]

    if median_exprs:
        df = df.with_columns(median_exprs)

    return df


macro_features = create_relative_value_features(macro_features)

# %% [markdown]
# ## 2.4b Yield-Curve Slope Feature
#
# The yield-curve slope (10Y-2Y spread) is loaded as `t10y2y`, but the text
# specifies additional processing: a 5-day EMA for smoothing and a 250-day
# z-score for regime-relative positioning.

# %%
# Yield-curve slope: EMA smoothing + rolling z-score
macro_features = macro_features.with_columns(
    pl.col("t10y2y").ewm_mean(span=5, ignore_nulls=True).alias("yc_slope_ema5"),
).with_columns(
    [
        (
            (pl.col("yc_slope_ema5") - pl.col("yc_slope_ema5").rolling_mean(250))
            / pl.col("yc_slope_ema5").rolling_std(250).clip(EPSILON, None)
        ).alias("yc_slope_zscore_250d"),
    ]
)
print("Yield-curve slope feature:")
macro_features.select(["timestamp", "t10y2y", "yc_slope_ema5", "yc_slope_zscore_250d"]).tail(5)

# %% [markdown]
# **Interpretation**: The z-score centers the slope relative to its recent history.
# Values above +2 indicate an unusually steep curve (risk-on, growth expectations);
# below -2 indicates inversion (recession signal). The EMA removes daily noise
# without introducing significant lag.

# %% [markdown]
# ## 2.5 Risk Regime Features


# %%
def create_risk_regime_features(df: pl.DataFrame) -> pl.DataFrame:
    """Create risk regime indicators."""
    feature_exprs = []

    # VIX regime (thresholds: <15 low, 15-25 normal, >25 high)
    if "vixcls" in df.columns:
        feature_exprs.append(
            pl.when(pl.col("vixcls") < 15)
            .then(0)
            .when(pl.col("vixcls") < 25)
            .then(1)
            .otherwise(2)
            .alias("vix_regime")
        )
        # VIX ratio to 252-day max
        feature_exprs.append(
            (pl.col("vixcls") / pl.col("vixcls").rolling_max(252).clip(EPSILON, None)).alias(
                "vix_relative_to_max"
            )
        )

    # Credit regime
    if "credit_spread" in df.columns:
        feature_exprs.append(
            pl.when(pl.col("credit_spread") < 1.0)
            .then(0)
            .when(pl.col("credit_spread") < 2.0)
            .then(1)
            .otherwise(2)
            .alias("credit_regime")
        )

    return df.with_columns(feature_exprs) if feature_exprs else df


macro_features = create_risk_regime_features(macro_features)
print("Risk regime features:")
macro_features.select([c for c in macro_features.columns if "regime" in c or "relative" in c]).tail(
    5
)

# %% [markdown]
# ---
#
# # Part 3: Calendar and Seasonal Encodings
#
# Calendar features encode **predictable clocks**: sessions, day-of-week,
# month-of-year, and scheduled events. The key principle is to encode
# **phase and proximity**, not outcomes.

# %% [markdown]
# ## 3.1 Cyclical Encoding
#
# Encoding month as an integer (1-12) implies an ordinal relationship
# (December > January). Cyclical sin/cos encoding removes this artifact:
#
# $$x_{\sin} = \sin\left(\frac{2\pi \cdot m}{12}\right), \quad x_{\cos} = \cos\left(\frac{2\pi \cdot m}{12}\right)$$

# %%
from ml4t.engineer.features.ml import cyclical_encode

from data import load_etfs

etfs = load_etfs()
spy = etfs.filter(pl.col("symbol") == "SPY").sort("timestamp")
calendar_start_dt = datetime.fromisoformat(CALENDAR_START_DATE)
spy = spy.filter(pl.col("timestamp") >= calendar_start_dt)

# Cyclical encoding for month
month_encoded = cyclical_encode(pl.col("timestamp").dt.month(), period=12, name_prefix="month")
cal_df = spy.with_columns(**month_encoded)

# Day-of-week encoding (Monday=1, Friday=5)
dow_encoded = cyclical_encode(pl.col("timestamp").dt.weekday(), period=5, name_prefix="dow")
cal_df = cal_df.with_columns(**dow_encoded)

print("Calendar encodings (last 10 rows):")
cal_df.select(["timestamp", "month_sin", "month_cos", "dow_sin", "dow_cos"]).tail(10)

# %% [markdown]
# **Usage**: Calendar features are primarily **state variables** for conditioning.
# For example, momentum signals may behave differently in January (tax-loss selling
# reversal) versus other months. Time-to-event encodings (e.g., days to next
# earnings, days to FOMC) follow the same pattern.
#
# **Note**: Volatility state features (vol ratio, percentile, decile) and
# price-derived regime indicators (variance ratio, fractal efficiency) are
# covered in `01_price_volume_features` since they derive from price data.
# Signal × state interactions and feasibility overlays are in
# `06_robustness_sensitivity`.

# %% [markdown]
# ## 3.2 Time-to-Event Encoding
#
# Time-to-event measures proximity to a known future event (earnings, FOMC,
# rebalance). The text specifies:
#
# $$d_{t,a} = \min(T_{\text{next}} - t, \; H_{\max})$$
#
# where $T_{\text{next}}$ is the next event date and $H_{\max}$ caps the
# feature to avoid extreme values far from events.

# %%
# Synthetic earnings calendar for demonstration
# Real systems would load from SEC EDGAR filing dates
earnings_dates = []
for symbol in ["AAPL", "MSFT", "GOOGL"]:
    # Quarterly earnings approximately 45 days after quarter end
    for q_end in [
        date(2023, 3, 31),
        date(2023, 6, 30),
        date(2023, 9, 30),
        date(2023, 12, 31),
        date(2024, 3, 31),
        date(2024, 6, 30),
        date(2024, 9, 30),
        date(2024, 12, 31),
    ]:
        ann_date = q_end + timedelta(days=45)
        earnings_dates.append({"symbol": symbol, "earnings_date": ann_date})

earnings_cal = pl.DataFrame(earnings_dates).sort(["symbol", "earnings_date"])

# %%
# Create daily dates and compute time-to-event features
daily = (
    pl.DataFrame(
        {"timestamp": pl.date_range(date(2023, 1, 1), date(2024, 12, 31), "1d", eager=True)}
    )
    .filter(pl.col("timestamp").dt.weekday() < 5)
    .join(pl.DataFrame({"symbol": ["AAPL", "MSFT", "GOOGL"]}), how="cross")
    .sort(["symbol", "timestamp"])
)

# Rolling forward join: for each date, find next earnings date
H_MAX = 63  # Cap at 63 trading days

daily_with_events = daily.join_asof(
    earnings_cal.sort(["symbol", "earnings_date"]),
    left_on="timestamp",
    right_on="earnings_date",
    by="symbol",
    strategy="forward",
).with_columns(
    [
        (pl.col("earnings_date") - pl.col("timestamp"))
        .dt.total_days()
        .clip(0, H_MAX)
        .alias("days_to_earnings"),
    ]
)

# %%
# Bin into pre/post windows
daily_with_events = daily_with_events.with_columns(
    pl.when(pl.col("days_to_earnings") <= 2)
    .then(pl.lit("pre_2d"))
    .when(pl.col("days_to_earnings") <= 5)
    .then(pl.lit("pre_5d"))
    .when(pl.col("days_to_earnings") > H_MAX - 1)
    .then(pl.lit("far"))
    .otherwise(pl.lit("normal"))
    .alias("event_proximity")
)

print("Time-to-event features:")
# Show a sample around an earnings date
sample_symbol = "AAPL"
print(
    daily_with_events.filter(
        (pl.col("symbol") == sample_symbol)
        & (pl.col("timestamp").is_between(date(2023, 4, 25), date(2023, 5, 20)))
    ).select(["timestamp", "symbol", "days_to_earnings", "event_proximity"])
)

# %% [markdown]
# **Interpretation**: Time-to-event serves as a **state variable** — a label
# that partitions trading days into discrete proximity windows
# (pre-2d, pre-5d, normal, far). These windows feed downstream signal × state
# interactions (see `06_robustness_sensitivity` for the IC-conditioning
# pattern); this notebook covers only the encoding step.

# %% [markdown]
# ## Summary
#
# ### Fundamentals
# - **Value**: Book-to-market, earnings yield, CF yield
# - **Quality**: ROE, ROA, accruals ratio
# - **Alignment**: ASOF join with both frames sorted by `[symbol, date]`
# - **Scaffolding**: Market cap approximation is for teaching only
#
# ### Macro
# - **Publication lag**: Add 30-day lag for monthly data
# - **Forward-fill**: Use filled version for YoY/3m changes
# - **Naming**: Rolling median $\neq$ percentile rank (be precise)
# - **Risk regimes**: VIX thresholds, credit regime from spread levels
#
# ### Calendar
# - **Cyclical encoding**: sin/cos for month, day-of-week, time-to-event
# - **Phase, not outcome**: Encode timing, not post-event realized moves
#
# ### Key Patterns
#
# | Feature Type | Update Freq | Alignment | Use Case |
# |--------------|-------------|-----------|----------|
# | Fundamentals | Quarterly | ASOF by announcement | Factor signals |
# | Macro | Daily/Monthly | Forward-fill + lag | Context, regime |
# | Calendar | Deterministic | Direct encoding | Seasonality |
#
# ### Next Notebooks
#
# - `05_feature_selection` — Feature selection and deduplication (§8.6)
# - `06_robustness_sensitivity` — Regime conditioning, interactions (§8.6)
