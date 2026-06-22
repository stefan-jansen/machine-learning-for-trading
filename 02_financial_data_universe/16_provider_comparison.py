# ---
# jupyter:
#   jupytext:
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
# # Provider Comparison: Multi-Source Data Acquisition
#
# **Docker image**: `ml4t`
#
# ## Purpose
# Demonstrate the `ml4t.data.providers` unified `fetch_ohlcv()` interface
# across YahooFinance, WikiPrices and FRED, then build a fallback strategy
# and a price-level provider-comparison report. The notebook seeds the
# ETF rotation universe used by the ETFs case study downstream.
#
# ## Learning Objectives
# - Fetch OHLCV from multiple providers with one call signature.
# - Build a priority-ordered fallback fetcher for production resilience.
# - Quantify provider disagreement (row counts, price-difference stats)
#   so multi-source stitching is auditable.
# - Use FRED (vintage-aware) to pull macro series for risk overlays.
#
# ## Book reference
# Chapter 2, §2.3 (multi-source stitching). The ETF universe materialised
# here feeds the `case_studies/etfs/` pipeline.
#
# ## Prerequisites
# - Network access for live YahooFinance + FRED calls (the notebook is a
#   provider-acquisition demo — these are the only places in the
#   publication pass where live API calls are intentional).
# - WikiPrices parquet under `ML4T_DATA_PATH/equities/market/us_equities/us_equities.parquet`
#   for the historical leg.
# - `FRED_API_KEY` set (free signup) for §8.

# %%
"""Provider Comparison — Multi-source data acquisition with ml4t-data providers."""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from ml4t.data.providers import WikiPricesProvider, YahooFinanceProvider
from ml4t.data.providers.fred import FREDProvider

HAS_FRED = bool(os.getenv("FRED_API_KEY"))
if not HAS_FRED:
    raise RuntimeError(
        "FRED_API_KEY not set. Get a free key at "
        "https://fred.stlouisfed.org/docs/api/api_key.html and export it."
    )

# Reproducibility: a fixed as-of date keeps outputs stable between book
# editions. Bump when the book is revised.
AS_OF_DATE = "2025-01-15"


# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI

# %% [markdown]
# ---
#
# ## Section 1: Understanding the Provider Architecture
#
# ml4t-data provides a **unified interface** for fetching data from multiple sources. All providers inherit from `BaseProvider` and implement the same `fetch_ohlcv()` method.
#
# ### Key Benefits:
# 1. **Consistency**: Same API regardless of data source
# 2. **Validation**: Automatic OHLC invariant checks
# 3. **Polars Output**: 10-100x faster than pandas alternatives
# 4. **Rate Limiting**: Built-in throttling to avoid API bans
# 5. **Circuit Breaker**: Automatic failure detection and recovery

# %%
# The ml4t-data provider landscape
# Note: This notebook demonstrates Yahoo, WikiPrices, and FRED. Other providers
# are available but require API keys and are covered in asset-class-specific notebooks.
providers_info = pl.DataFrame(
    {
        "provider": [
            "YahooFinanceProvider",
            "WikiPricesProvider",
            "FREDProvider",
            "EODHDProvider",
            "BinanceAPIProvider",
        ],
        "asset_class": [
            "US Equities, ETFs",
            "US Equities (Historical)",
            "Economic Indicators",
            "Global Equities",
            "Crypto",
        ],
        "api_key_required": [
            "No",
            "No (local file)",
            "Yes (free)",
            "Yes (free tier)",
            "No",
        ],
        "date_range": ["~30 years", "1962-2018", "50+ years", "~20 years", "~5 years"],
        "demonstrated_in": [
            "this notebook",
            "this notebook",
            "this notebook",
            "10_crypto_perps_eda",
            "10_crypto_perps_eda",
        ],
    }
)
providers_info

# %% [markdown]
# ---
#
# ## Section 2: Fetching Data with Yahoo Finance
#
# Yahoo Finance is the easiest starting point - no API key required. Let's fetch our ETF universe for the momentum strategy.

# %%
# Define our ETF universe for the rotation strategy
ETF_UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD"]

# Date range: 5 years of daily data (relative to AS_OF_DATE for reproducibility)
end_date = AS_OF_DATE
start_date = (datetime.strptime(AS_OF_DATE, "%Y-%m-%d") - timedelta(days=5 * 365)).strftime(
    "%Y-%m-%d"
)

print(f"Fetching data from {start_date} to {end_date}")
print(f"ETF Universe: {ETF_UNIVERSE}")

# %%
# Create Yahoo Finance provider
yahoo = YahooFinanceProvider()

# Fetch SPY as our primary example
spy_data = yahoo.fetch_ohlcv("SPY", start_date, end_date)

print(f"Fetched {len(spy_data)} rows of SPY data")
print(f"Date range: {spy_data['timestamp'].min()} to {spy_data['timestamp'].max()}")
print("\nSample data:")
print(spy_data.head())

# %%
# Examine the schema - ml4t-data provides consistent column names
print("Schema (consistent across all providers):")
for name, dtype in spy_data.schema.items():
    print(f"  {name}: {dtype}")

# %%
# Fetch the full ETF universe — fail loudly on missing tickers
etf_data = {symbol: yahoo.fetch_ohlcv(symbol, start_date, end_date) for symbol in ETF_UNIVERSE}
print(
    f"Fetched {len(etf_data)}/{len(ETF_UNIVERSE)} ETFs · "
    f"{sum(len(d) for d in etf_data.values()):,} total daily rows"
)

# %%
# Combine into a single DataFrame with symbol column
combined_dfs = []
for symbol, df in etf_data.items():
    combined_dfs.append(df.with_columns(pl.lit(symbol).alias("symbol")))

etf_universe_df = pl.concat(combined_dfs).sort(["symbol", "timestamp"])

(
    etf_universe_df.group_by("symbol")
    .agg(
        pl.col("timestamp").min().alias("start"),
        pl.col("timestamp").max().alias("end"),
        pl.len().alias("rows"),
        pl.col("close").last().alias("last_close"),
    )
    .sort("symbol")
)

# %% [markdown]
# ---
#
# ## Section 3: Historical Data with WikiPrices
#
# For long-term backtests (30+ years), Yahoo Finance has limitations. WikiPrices provides
# long-horizon U.S. equity history (including many delisted names) from roughly 1962 to 2018.
#
# **Key Advantage**: Includes delisted companies, but you still need explicit survivorship
# checks and delisting return handling (see `08_survivorship_bias_detection`).

# %%
from utils import ML4T_DATA_PATH

WIKI_PATHS = [
    # Primary: canonical layout under ML4T_DATA_PATH
    ML4T_DATA_PATH / "equities" / "market" / "us_equities" / "us_equities.parquet",
    # Docker mount (same nested layout)
    Path("/data/equities/market/us_equities/us_equities.parquet"),
    # Test fixture (CI subset)
    Path("/app/tests/fixtures/data/equities/market/us_equities/us_equities.parquet"),
    Path("tests/fixtures/data/equities/market/us_equities/us_equities.parquet"),
]

wiki = None
wiki_path_used = None
wiki_load_errors = []

# %% [markdown]
# ### WikiPrices Schema Adapter
#
# When WikiPrices data has been canonicalized to `symbol`/`timestamp` columns,
# this adapter provides the same `fetch_ohlcv()` interface as the library provider.


# %%
class CanonicalWikiPricesAdapter:
    """Adapter for canonicalized local Wiki Prices parquet."""

    def __init__(self, parquet_path: Path):
        self.parquet_path = Path(parquet_path)

    def fetch_ohlcv(self, symbol: str, start: str, end: str) -> pl.DataFrame:
        start_date = datetime.strptime(start, "%Y-%m-%d").date()
        end_date = datetime.strptime(end, "%Y-%m-%d").date()
        return (
            pl.scan_parquet(self.parquet_path)
            .filter(
                (pl.col("symbol") == symbol)
                & pl.col("timestamp").is_between(start_date, end_date, closed="both")
            )
            .select(
                [
                    pl.col("timestamp").cast(pl.Datetime("us")).alias("timestamp"),
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "adj_close",
                    "adj_open",
                    "adj_high",
                    "adj_low",
                    "adj_volume",
                    "ex_dividend",
                    "split_ratio",
                ]
            )
            .sort("timestamp")
            .collect()
        )

    def list_available_symbols(self) -> list[str]:
        return (
            pl.scan_parquet(self.parquet_path)
            .select(pl.col("symbol").unique().sort())
            .collect()
            .to_series()
            .to_list()
        )

    def close(self) -> None:
        """Mirror provider lifecycle API."""
        return None


# %% [markdown]
# ### Load WikiPrices Provider
#
# Try multiple paths to find WikiPrices data (local install, Docker, CI fixtures).

# %%
for wiki_path in WIKI_PATHS:
    if wiki_path.exists():
        print(f"  Found: {wiki_path}")
        try:
            wiki = WikiPricesProvider(parquet_path=wiki_path)
            wiki_path_used = wiki_path
            break
        except Exception as e:
            # File exists but failed to load - this is a real error, not silent skip
            wiki_load_errors.append((wiki_path, str(e)))
            print(f"  ERROR loading {wiki_path}: {e}")

# If we found files but couldn't load any of them, that's a bug - fail loudly
if wiki is None and wiki_load_errors:
    error_msg = "WikiPrices files found but failed to load:\n"
    for path, err in wiki_load_errors:
        error_msg += f"  {path}: {err}\n"
    raise RuntimeError(error_msg)

# %%
if wiki is None:
    raise RuntimeError(
        f"WikiPrices parquet not found in any of: {[str(p) for p in WIKI_PATHS]}. "
        "Materialise it via WikiPricesProvider.download(api_key=<nasdaq>) first."
    )

print(f"WikiPrices loaded from: {wiki_path_used}")

# Fetch long-term AAPL history; the canonical local schema may need the
# adapter wrapper if the file uses asset/date instead of symbol/timestamp.
try:
    aapl_historical = wiki.fetch_ohlcv("AAPL", "1990-01-01", "2018-03-27")
except Exception as e:
    if 'unable to find column "ticker"' not in str(e):
        raise
    print("Detected canonical schema; switching to CanonicalWikiPricesAdapter.")
    wiki = CanonicalWikiPricesAdapter(wiki_path_used)
    aapl_historical = wiki.fetch_ohlcv("AAPL", "1990-01-01", "2018-03-27")

print(
    f"AAPL data: {len(aapl_historical)} rows · "
    f"{aapl_historical['timestamp'].min()} → {aapl_historical['timestamp'].max()}"
)
aapl_historical.head()

# %%
# WikiPrices includes delisted companies — critical for avoiding survivorship bias.
available_symbols = wiki.list_available_symbols()
print(f"WikiPrices contains {len(available_symbols):,} symbols; sample: {available_symbols[:10]}")

# %% [markdown]
# ---
#
# ## Section 4: Multi-Provider Fallback Strategy
#
# For production systems, we need a robust strategy that handles:
# 1. API failures
# 2. Rate limits
# 3. Data gaps
# 4. Historical coverage
#
# The **fallback pattern** tries providers in order until one succeeds.


# %%
@dataclass
class FetchResult:
    """Result of a multi-provider fetch attempt."""

    success: bool
    data: pl.DataFrame | None
    provider_used: str | None
    providers_tried: list[str]
    error_messages: dict[str, str]


# %% [markdown]
# ### Fallback Fetch
#
# Try multiple providers in priority order; return the first successful result.


# %%
def fetch_with_fallback(
    symbol: str,
    start: str,
    end: str,
    providers: list,
    provider_names: list[str],
) -> FetchResult:
    """
    Fetch data trying multiple providers in order.

    Parameters
    ----------
    symbol : str
        Ticker symbol
    start, end : str
        Date range (YYYY-MM-DD)
    providers : list
        Provider instances to try
    provider_names : list[str]
        Names for logging

    Returns
    -------
    FetchResult
        Result with data and metadata
    """
    errors = {}
    tried = []

    for provider, name in zip(providers, provider_names, strict=False):
        tried.append(name)
        try:
            data = provider.fetch_ohlcv(symbol, start, end)
            if len(data) > 0:
                return FetchResult(
                    success=True,
                    data=data,
                    provider_used=name,
                    providers_tried=tried,
                    error_messages=errors,
                )
            else:
                errors[name] = "Empty result"
        except Exception as e:
            errors[name] = str(e)

    return FetchResult(
        success=False,
        data=None,
        provider_used=None,
        providers_tried=tried,
        error_messages=errors,
    )


# %%
# Try Yahoo first, then WikiPrices as fallback
providers = [yahoo, wiki]
provider_names = ["Yahoo Finance", "WikiPrices"]

result = fetch_with_fallback("AAPL", "2020-01-01", "2020-12-31", providers, provider_names)
if not result.success:
    raise RuntimeError(f"All providers failed: {result.error_messages}")
print(
    f"Success via {result.provider_used}: {len(result.data):,} rows; tried {result.providers_tried}"
)


# %% [markdown]
# ---
#
# ## Section 5: Combining Historical and Recent Data
#
# For 30+ year backtests, we combine:
# 1. **WikiPrices** (1962-2018) - includes delisted names
# 2. **Yahoo Finance** (2018-present) - current data
#
# This gives us the best of both worlds.


# %%
def build_complete_history(
    symbol: str,
    yahoo_provider,
    wiki_provider=None,
    start_date: str = "1990-01-01",
    end_date: str = None,
) -> pl.DataFrame:
    """Build complete OHLCV history combining WikiPrices (pre-2018) and Yahoo Finance (2018+)."""
    if end_date is None:
        end_date = AS_OF_DATE

    parts = []

    # WikiPrices cutoff
    wiki_end = "2018-03-27"

    # Part 1: Historical data from WikiPrices
    if wiki_provider and start_date < wiki_end:
        try:
            historical = wiki_provider.fetch_ohlcv(symbol, start_date, wiki_end)
            if len(historical) > 0:
                parts.append(historical)
                print(f"  WikiPrices: {len(historical)} rows ({start_date} to {wiki_end})")
        except Exception as e:
            print(f"  WikiPrices: {e}")

    # Part 2: Recent data from Yahoo Finance
    yahoo_start = "2018-03-28" if start_date < wiki_end else start_date
    try:
        recent = yahoo_provider.fetch_ohlcv(symbol, yahoo_start, end_date)
        if len(recent) > 0:
            parts.append(recent)
            print(f"  Yahoo Finance: {len(recent)} rows ({yahoo_start} to {end_date})")
    except Exception as e:
        print(f"  Yahoo Finance: {e}")

    if not parts:
        raise ValueError(f"No data found for {symbol}")

    # Standardize columns and types before concatenation
    # Both providers should have: timestamp, open, high, low, close, volume
    standard_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    standardized = []
    for df in parts:
        # Select only standard columns (ensure all have same schema)
        df_std = df.select(standard_cols)
        # Normalize types: timestamp to us precision, numerics to Float64
        df_std = df_std.with_columns(
            pl.col("timestamp").cast(pl.Datetime("us")),
            pl.col("open", "high", "low", "close").cast(pl.Float64),
            pl.col("volume").cast(
                pl.Float64
            ),  # Volume can be Int64 or Float64 depending on provider
        )
        standardized.append(df_std)

    # Combine and deduplicate
    combined = pl.concat(standardized).sort("timestamp").unique("timestamp")
    print(f"  Combined: {len(combined)} rows total")

    return combined


# %%
# Build complete AAPL history (1990-now)
aapl_complete = build_complete_history("AAPL", yahoo, wiki, "1990-01-01")
print(
    f"Combined: {len(aapl_complete):,} rows · "
    f"{aapl_complete['timestamp'].min()} → {aapl_complete['timestamp'].max()}"
)


# %% [markdown]
# ---
#
# ## Section 6: Provider Data Comparison
#
# Yahoo Finance and WikiPrices both expose a close price, but they apply different
# adjustment conventions and cover different date ranges, so the two series can
# diverge sharply. This section demonstrates how to:
#
# 1. Align data from multiple providers by date
# 2. Quantify differences (row counts, price discrepancies)
# 3. Identify systematic vs random variations
# 4. Diagnose the causes (splits, dividends, timing)
#
# **Why this matters**: Multi-source strategies must understand when providers disagree.
# Small discrepancies compound over backtests; large ones signal data errors.


# %%
def compare_providers_detailed(
    symbol: str,
    start: str,
    end: str,
    provider_a,
    provider_b,
    name_a: str = "Provider A",
    name_b: str = "Provider B",
) -> dict:
    """
    Detailed comparison of two data providers for a symbol.

    Returns dict with:
    - summary: high-level stats
    - aligned: row-by-row comparison DataFrame
    - discrepancies: rows where providers disagree significantly
    """
    try:
        data_a = provider_a.fetch_ohlcv(symbol, start, end)
        data_b = provider_b.fetch_ohlcv(symbol, start, end)
    except Exception as e:
        return {"error": str(e), "summary": None, "aligned": None, "discrepancies": None}

    # Normalize timestamps to date for joining (providers may have different time components)
    data_a = data_a.with_columns(pl.col("timestamp").dt.date().alias("timestamp"))
    data_b = data_b.with_columns(pl.col("timestamp").dt.date().alias("timestamp"))

    # Inner join on date to get overlapping rows
    aligned = data_a.select(["timestamp", "open", "high", "low", "close", "volume"]).join(
        data_b.select(["timestamp", "open", "high", "low", "close", "volume"]),
        on="timestamp",
        suffix="_b",
    )

    # Calculate differences
    aligned = aligned.with_columns(
        [
            ((pl.col("close") - pl.col("close_b")) / pl.col("close_b") * 100).alias(
                "close_diff_pct"
            ),
            ((pl.col("volume") - pl.col("volume_b")) / pl.col("volume_b") * 100).alias(
                "volume_diff_pct"
            ),
        ]
    )

    # Identify significant discrepancies (>0.1% price difference)
    discrepancies = aligned.filter(pl.col("close_diff_pct").abs() > 0.1)

    # Summary statistics
    summary = {
        "symbol": symbol,
        "period": f"{start} to {end}",
        f"{name_a}_rows": len(data_a),
        f"{name_b}_rows": len(data_b),
        "aligned_rows": len(aligned),
        "missing_in_a": len(data_b) - len(aligned),
        "missing_in_b": len(data_a) - len(aligned),
        "mean_close_diff_pct": aligned["close_diff_pct"].mean() if len(aligned) > 0 else None,
        "max_close_diff_pct": aligned["close_diff_pct"].abs().max() if len(aligned) > 0 else None,
        "discrepancy_days": len(discrepancies),
        "exact_matches": len(aligned.filter(pl.col("close_diff_pct").abs() < 0.01)),
    }

    return {"summary": summary, "aligned": aligned, "discrepancies": discrepancies}


# %%
comparison_result = compare_providers_detailed(
    "AAPL", "2017-01-01", "2017-12-31", yahoo, wiki, "Yahoo", "WikiPrices"
)
if comparison_result.get("error"):
    raise RuntimeError(f"Comparison failed: {comparison_result['error']}")

summary = comparison_result["summary"]
summary_df = pl.DataFrame([summary])
summary_df

# %%
# Show top discrepancies (if any)
disc = comparison_result["discrepancies"]
if len(disc) > 0:
    disc.select(["timestamp", "close", "close_b", "close_diff_pct"]).head(5)
else:
    print("No significant Yahoo↔WikiPrices price discrepancies for AAPL 2017.")


# %% [markdown]
# ### What Real Provider Discrepancies Look Like
#
# When comparing Yahoo Finance to WikiPrices on historical data, common findings include:
#
# | Pattern | Typical Magnitude | Cause |
# |---------|-------------------|-------|
# | **Exact match** | <0.01% | Same underlying source |
# | **Small drift** | 0.01-0.1% | Rounding, adjustment timing |
# | **Step change** | 1-10% | Different split adjustment date |
# | **Systematic offset** | Consistent % | Different dividend treatment |
#
# **Key insight**: Raw quotes share the same exchange source, but *adjustments*
# diverge — and that divergence can dominate the comparison. The AAPL 2017 run
# above is a case in point: all 249 overlapping days differ by ~77% (zero exact
# matches) because Yahoo's close retroactively reflects Apple's 2020 4:1 split,
# while WikiPrices ends in 2018 and never applied it. Always align the adjustment
# basis (and watch the coverage window) before diffing two providers.

# %% [markdown]
# ---
#
# ## Section 7: ETF Universe Performance
#
# We visualize the 7 core ETFs defined at the start of this notebook. These represent
# distinct asset classes for the rotation strategy:
#
# | Symbol | Asset Class | Role in Portfolio |
# |--------|-------------|-------------------|
# | SPY | US Large Cap | Risk-on equity |
# | QQQ | US Tech | High-beta growth |
# | IWM | US Small Cap | Cyclical exposure |
# | EFA | Intl Developed | Geographic diversification |
# | EEM | Emerging Markets | Growth/risk allocation |
# | TLT | Long Treasury | Flight-to-quality |
# | GLD | Gold | Inflation/crisis hedge |
#
# A rotation strategy switches between these based on momentum signals (Chapter 7).

# %%
# Calculate normalized prices (start = 100)
fig = go.Figure()

for symbol, df in etf_data.items():
    # Normalize to 100 at start
    normalized = (df["close"] / df["close"][0]) * 100

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"].to_list(),
            y=normalized.to_list(),
            name=symbol,
            mode="lines",
        )
    )

fig.update_layout(
    title=f"Core ETF Universe Performance ({len(etf_data)} ETFs, Normalized to 100)",
    xaxis_title="Date",
    yaxis_title="Normalized Price",
    height=500,
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)

# Add horizontal line at 100
fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)

fig.show()

# %%
# Calculate performance statistics
performance_stats = []

for symbol, df in etf_data.items():
    returns = df["close"].pct_change().drop_nulls()

    total_return = (df["close"][-1] / df["close"][0] - 1) * 100
    annual_return = ((1 + total_return / 100) ** (252 / len(df)) - 1) * 100
    volatility = returns.std() * np.sqrt(252) * 100
    sharpe = (annual_return - 2) / volatility if volatility > 0 else 0  # Assume 2% risk-free

    performance_stats.append(
        {
            "symbol": symbol,
            "total_return_pct": total_return,
            "annual_return_pct": annual_return,
            "volatility_pct": volatility,
            "sharpe_ratio": sharpe,
            "trading_days": len(df),
        }
    )

performance_df = pl.DataFrame(performance_stats).sort("annual_return_pct", descending=True)
performance_df

# %% [markdown]
# ---
#
# ## Section 8: Economic Data with FREDProvider
#
# For macro regime detection and risk management, FRED (Federal Reserve Economic Data)
# provides 800,000+ economic time series. Key series for trading include:
#
# | Series | Description | Frequency |
# |--------|-------------|-----------|
# | VIXCLS | VIX Volatility Index | Daily |
# | DGS10 | 10-Year Treasury Yield | Daily |
# | T10Y2Y | Yield Curve Slope | Daily |
# | UNRATE | Unemployment Rate | Monthly |
# | ICSA | Initial Jobless Claims | Weekly |
#
# **Get free API key**: https://fred.stlouisfed.org/docs/api/api_key.html

# %%
# Pull VIX and 10Y Treasury from FRED for the same window
fred = FREDProvider()
vix = fred.fetch_ohlcv("VIXCLS", "2023-01-01", "2024-01-01")
treasury_10y = fred.fetch_ohlcv("DGS10", "2023-01-01", "2024-01-01")
fred.close()

vix_mean = float(vix["close"].mean())
vix_max = float(vix["close"].max())
last_yield = float(treasury_10y.filter(pl.col("close").is_not_null())["close"][-1])
print(
    f"VIX 2023: {len(vix):,} obs · mean {vix_mean:.1f}, max {vix_max:.1f} · "
    f"DGS10 last yield {last_yield:.2f}%"
)
vix.head()

# %% [markdown]
# ## Key Takeaways
#
# Multi-source data acquisition profile for the 7-ETF rotation universe.
#
# ### Quantitative Findings
# - **ETF universe** (5 years, 1,256 trading days/symbol): QQQ
#   +138.5 % total return / +19.1 % annualised / Sharpe 0.66 leads;
#   TLT −28.6 % / −6.5 % / Sharpe −0.47 trails. EEM essentially flat
#   (+0.85 % total, Sharpe −0.08). The 5-year window covers a
#   bond-bear / equity-bull regime — useful for the rotation case study
#   to learn that "diversification" requires non-equity diversifiers
#   beyond duration alone.
# - **WikiPrices coverage**: ~3,200 historical symbols (1962-2018)
#   including delisted names — required for any backtest that wants to
#   avoid survivorship bias on legacy data.
# - **Yahoo↔WikiPrices comparison (AAPL 2017)**: zero exact matches — all
#   249 overlapping days differ by ~77% because Yahoo's auto-adjusted close
#   reflects Apple's 2020 4:1 split while WikiPrices ends in 2018 and predates
#   it. The helper surfaces these price-difference statistics so adjustment-basis
#   mismatches are caught row-by-row rather than glossed.
# - **AAPL stitched history**: WikiPrices + Yahoo joins to ~8,800 daily
#   rows (1990-now) without duplication after the 2018-03-27 cutoff.
# - **FRED**: VIXCLS and DGS10 fetched in the same call signature as
#   the equity providers — the unified API generalises cleanly to
#   macro overlays.
#
# ### Implications for Practitioners
# - **One signature, many sources**: The unified `fetch_ohlcv()` makes
#   provider swaps cheap. Treat the provider as configuration, not as
#   bespoke per-source code.
# - **Fallback over fail-fast** for production data acquisition: prefer
#   the next provider over a missing day, but log the source used so
#   the audit trail is intact.
# - **Comparison is the contract**: never trust two providers blindly —
#   run a row-by-row diff on overlapping windows and treat any
#   adjustment-method mismatch as a bug, not a feature.
#
# **Next**: `17_complete_pipeline` consumes this universe end-to-end
# (ingestion → quality gate → storage → query); `18_data_management`
# adds the `DataManager`/`Universe`/`HiveStorage` layer for production.

# %%
# Close provider sessions
yahoo.close()
wiki.close()
print(f"   - Total rows: {len(etf_universe_df):,}")
