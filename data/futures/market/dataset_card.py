# %% [markdown]
# # CME Futures Dataset
#
# Continuous futures contracts from CME Group via Databento.
#
# | Property | Value |
# |----------|-------|
# | **Provider** | Databento |
# | **Asset Class** | Futures (Equity, Rates, Energy, Metals, FX, Ags) |
# | **Frequency** | Hourly, Daily (derived) |
# | **Products** | 30 core + 6 extension |
# | **Coverage** | 2011-2025 |
# | **Size** | ~500 MB |
# | **API Key** | `DATABENTO_API_KEY` (**PAID**) |
# | **Loader** | `load_cme_futures()` |
#
# **WARNING**: Databento is a paid data provider. Always estimate costs before downloading.

# %%
"""CME Futures - download, explore, and update workflow."""

import json
import os
from pathlib import Path

import polars as pl
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# %% [markdown]
# ## 1. Configuration
#
# The futures universe is defined in `config.yaml`. Includes 30 core
# products across equity indices, treasuries, energy, metals, currencies, and
# agriculture.

# %%
# Load and display configuration
config_path = Path("config.yaml")
config = yaml.safe_load(config_path.read_text())

print("=== CME Futures Configuration ===")
print(f"Dataset: {config['dataset']}")
print(f"Schema: {config['schema']}")
print(f"Roll type: {config['roll_type']} (volume-based)")
print(f"Tenors: {config['tenors']} (front, second, third month)")
print(f"Date range: {config['default_start']} to {config['default_end']}")
print("\nProduct categories:")

# Count by category
categories = {}
for product, info in config["products"].items():
    cat = info.get("category", "unknown")
    categories[cat] = categories.get(cat, 0) + 1

for cat, count in sorted(categories.items()):
    print(f"  {cat}: {count} products")

print(f"\nTotal core products: {len(config['products'])}")
print(f"Extension products: {len(config.get('extension_products', {}))}")

# %% [markdown]
# ## 2. API Key Setup
#
# **Databento is a paid data provider.** New accounts receive $125 free credit.
#
# ### Getting a Databento API Key
#
# 1. Sign up at [Databento](https://databento.com/signup) ($125 free credit)
# 2. Navigate to **API Keys** in your dashboard
# 3. Create a new API key
# 4. Add to your `.env` file in the repository root:
#
# ```bash
# DATABENTO_API_KEY=db-your-api-key-here
# ```
#
# ### Cost Reference
#
# | Data Type | Cost Estimate |
# |-----------|---------------|
# | Hourly OHLCV | ~$0.50-1.00 per product per year |
# | Daily OHLCV | ~$0.05-0.10 per product per year |
# | Full 30 products x 15 years | ~$75-100 |
#
# **ALWAYS run cost estimation before downloading!**

# %%
# Verify API key is configured
api_key = os.getenv("DATABENTO_API_KEY")
if api_key:
    # Show partial key for verification
    print(f"DATABENTO_API_KEY: {api_key[:8]}... (configured)")
else:
    print("WARNING: DATABENTO_API_KEY not set in environment")
    print("Sign up at: https://databento.com/signup ($125 free credit)")
    print("Add to .env file: DATABENTO_API_KEY=db-your-key-here")

# %% [markdown]
# ## 3. Download Data
#
# **IMPORTANT**: Always run cost estimation before downloading!
#
# The download:
# - Uses Hive partitioning by product/year for efficient updates
# - Downloads full date range per product in one API call (cost efficient)
# - Stores V0, V1, V2 tenors (front, second, third month) stacked


# %%
def estimate_futures_cost(products: list[str] | None = None) -> float:
    """Estimate download cost from Databento.

    Args:
        products: Specific products to estimate (default: all from config)

    Returns:
        Estimated cost in USD
    """
    import databento as db

    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise ValueError("DATABENTO_API_KEY not set. See API Key Setup section.")

    # Load config
    config = yaml.safe_load(config_path.read_text())

    if products is None:
        products = list(config["products"].keys())

    client = db.Historical()
    total_cost = 0.0

    print("=== Cost Estimation ===")
    print(f"Products: {len(products)}")
    print(f"Tenors: {config['tenors']}")
    print(f"Date range: {config['default_start']} to {config['default_end']}")
    print()

    for product in products:
        product_info = config["products"].get(product, {})
        start = product_info.get("start", config["default_start"])

        # Build symbols for continuous contracts
        symbols = [f"{product}.{config['roll_type']}.{pos}" for pos in config["tenors"]]

        try:
            cost = client.metadata.get_cost(
                dataset=config["dataset"],
                symbols=symbols,
                schema=config["schema"],
                start=start,
                end=config["default_end"],
                stype_in="continuous",
            )
            total_cost += cost
            print(f"  {product}: ${cost:.2f}")
        except Exception as e:
            print(f"  {product}: ERROR - {e}")

    print(f"\n{'=' * 40}")
    print(f"TOTAL ESTIMATED COST: ${total_cost:.2f}")
    print(f"{'=' * 40}")

    return total_cost


def download_futures_data(
    products: list[str] | None = None,
    dry_run: bool = True,  # Default to dry_run=True for safety!
    force: bool = False,
):
    """Download CME futures data from Databento.

    Args:
        products: Specific products to download (default: all from config)
        dry_run: If True, show what would be downloaded without doing it (DEFAULT: True)
        force: If True, re-download even if data exists
    """
    import databento as db

    from utils import ML4T_DATA_PATH

    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise ValueError("DATABENTO_API_KEY not set. See API Key Setup section.")

    # Load config
    config = yaml.safe_load(config_path.read_text())

    if products is None:
        products = list(config["products"].keys())

    output_dir = ML4T_DATA_PATH / "futures" / "market" / "continuous" / "hourly"

    print("=== CME Futures Download ===")
    print(f"Products: {len(products)}")
    print(f"Tenors: {config['tenors']}")
    print(f"Date range: {config['default_start']} to {config['default_end']}")
    print(f"Output: {output_dir}")

    if dry_run:
        print("\n[DRY RUN] Would download:")
        for product in products:
            product_info = config["products"].get(product, {})
            start = product_info.get("start", config["default_start"])
            print(f"  {product}: {start} to {config['default_end']}")
        print("\nRun estimate_futures_cost() to see cost estimate.")
        print("Set dry_run=False to actually download.")
        return

    # Initialize client
    client = db.Historical()
    total_rows = 0

    print(f"\nDownloading {len(products)} products...")
    for product in products:
        product_info = config["products"].get(product, {})
        start = product_info.get("start", config["default_start"])

        # Check existing
        product_dir = output_dir / f"product={product}"
        if product_dir.exists() and not force:
            existing_years = list(product_dir.glob("year=*/data.parquet"))
            if existing_years:
                print(
                    f"  {product}: Already exists ({len(existing_years)} years). Use force=True to re-download."
                )
                continue

        # Build symbols for continuous contracts
        symbols = [f"{product}.{config['roll_type']}.{pos}" for pos in config["tenors"]]

        print(f"  {product}...", end=" ", flush=True)
        try:
            data = client.timeseries.get_range(
                dataset=config["dataset"],
                symbols=symbols,
                schema=config["schema"],
                start=start,
                end=config["default_end"],
                stype_in="continuous",
            )

            df = data.to_df()
            if len(df) == 0:
                print("WARNING (no data)")
                continue

            # Convert to polars and add metadata
            df_pl = pl.from_pandas(df.reset_index())
            df_pl = df_pl.with_columns(pl.lit(product).alias("product"))

            # Extract tenor from symbol (ES.v.0 -> 0)
            if "symbol" in df_pl.columns:
                df_pl = df_pl.with_columns(
                    pl.col("symbol")
                    .str.extract(rf"\.{config['roll_type']}\.(\d+)$", 1)
                    .cast(pl.Int8)
                    .alias("tenor")
                )

            # Partition by year
            df_pl = df_pl.with_columns(pl.col("ts_event").dt.year().alias("year"))

            for year in df_pl["year"].unique().sort().to_list():
                year_data = df_pl.filter(pl.col("year") == year)
                year_dir = output_dir / f"product={product}" / f"year={year}"
                year_dir.mkdir(parents=True, exist_ok=True)
                year_data.sort(["ts_event", "symbol"]).write_parquet(year_dir / "data.parquet")

            total_rows += len(df_pl)
            print(f"OK ({len(df_pl):,} rows)")

        except Exception as e:
            print(f"ERROR: {e}")

    print("\n=== Complete ===")
    print(f"Total rows: {total_rows:,}")
    print(f"Output: {output_dir}")


# %% [markdown]
# ### Estimate Cost (ALWAYS DO THIS FIRST!)

# %%
# Uncomment to estimate cost for all products
# estimate_futures_cost()

# Estimate for specific products
# estimate_futures_cost(products=["ES", "NQ", "CL", "GC"])

# %% [markdown]
# ### Download Data
#
# **WARNING**: This will consume Databento credits!

# %%
# Dry run (default) - shows what would be downloaded
download_futures_data(dry_run=True)

# %%
# Uncomment to actually download (after reviewing cost estimate!)
# download_futures_data(dry_run=False)

# Download specific products only
# download_futures_data(products=["ES", "NQ"], dry_run=False)

# %% [markdown]
# ## 4. Load and Explore
#
# Once downloaded, use the loader throughout the book:

# %%
from data import load_cme_futures

# Load daily continuous contracts (default)
df = load_cme_futures(frequency="daily")

print(f"Shape: {df.shape}")
print(f"Products: {df['product'].n_unique()}")
print(f"Date range: {df['session_date'].min()} to {df['session_date'].max()}")
print(f"Memory: {df.estimated_size('mb'):.1f} MB")

# %%
# Schema
df.schema

# %%
# Preview
df.head(10)

# %% [markdown]
# ### Coverage by Product

# %%
# Coverage and basic stats by product
coverage = (
    df.group_by("product")
    .agg(
        pl.col("session_date").min().alias("first_date"),
        pl.col("session_date").max().alias("last_date"),
        pl.len().alias("n_bars"),
        pl.col("volume").mean().alias("avg_volume"),
    )
    .sort("avg_volume", descending=True)
)
coverage

# %% [markdown]
# ### Hourly Data

# %%
# Load hourly data for specific products
hourly = load_cme_futures(frequency="hourly", products=["ES", "NQ"])
print(f"Hourly ES/NQ: {hourly.shape}")
print(f"Date range: {hourly['timestamp'].min()} to {hourly['timestamp'].max()}")

# %% [markdown]
# ## 5. Data Profile
#
# Profiles document the dataset structure, statistics, and quality metrics.

# %%
from utils import ML4T_DATA_PATH

# Check for existing profile
profile_path = ML4T_DATA_PATH / "futures" / "market" / "profile.json"

if profile_path.exists():
    profile = json.loads(profile_path.read_text())
    print("=== Futures Profile ===")
    print(f"Dataset: {profile['dataset']}")
    print(f"Rows: {profile['rows']:,}")
    print(f"Memory: {profile['memory_mb']:.1f} MB")
else:
    print(f"Profile not found at {profile_path}")
    print("Generate with: python generate_profiles.py --dataset cme_futures")

# %% [markdown]
# ## 6. Loader Options
#
# The loader supports filtering by frequency, products, tenors, and date range:

# %%
# Daily frequency (default)
daily = load_cme_futures(frequency="daily")
print(f"Daily data: {daily.shape}")

# %%
# Specific products
equities = load_cme_futures(products=["ES", "NQ", "YM", "RTY"])
print(f"Equity indices only: {equities.shape}")

# %%
# Specific tenor (front month only)
front_month = load_cme_futures(tenors=[0])
print(f"Front month only: {front_month.shape}")

# %%
# Date range
recent = load_cme_futures(start_date="2024-01-01")
print(f"2024 onwards: {recent.shape}")

# %%
# Combined filters
filtered = load_cme_futures(
    frequency="daily",
    products=["ES", "CL", "GC"],
    tenors=[0],
    start_date="2020-01-01",
    end_date="2023-12-31",
)
print(f"ES/CL/GC front month 2020-2023: {filtered.shape}")

# %% [markdown]
# ## 7. Documentation
#
# ### Databento
# - [Databento Documentation](https://databento.com/docs/)
# - [CME Globex Dataset](https://databento.com/docs/datasets/cme)
# - [Continuous Contracts](https://databento.com/docs/schemas/continuous)
#
# ### Continuous Contract Construction
#
# The data uses **volume-based roll** (`.v.` suffix):
# - Roll occurs when previous day's volume shows next contract > current
# - This is realistic for trading (you know yesterday's volume at today's open)
#
# Available tenors:
# - **V0**: Front month (nearest expiry)
# - **V1**: Second month
# - **V2**: Third month
#
# ### Product Categories
#
# | Category | Products | Description |
# |----------|----------|-------------|
# | Equity Index | ES, NQ, YM, RTY | S&P 500, NASDAQ-100, Dow, Russell 2000 |
# | Treasury | ZN, ZB, ZF, ZT | 10Y, 30Y, 5Y, 2Y notes/bonds |
# | Energy | CL, NG, HO, RB | Crude, natural gas, heating oil, gasoline |
# | Metals | GC, SI, HG, PL | Gold, silver, copper, platinum |
# | FX | 6E, 6J, 6B, 6A, 6C, 6S | EUR, JPY, GBP, AUD, CAD, CHF |
# | Agriculture | ZC, ZS, ZW, ZM, ZL | Corn, soybeans, wheat, meal, oil |
# | Livestock | LE, HE, GF | Live cattle, lean hogs, feeder cattle |

# %% [markdown]
# ## 8. Updating Data
#
# To update with the latest data:
#
# ```python
# # Estimate cost first!
# estimate_futures_cost()
#
# # Download updates (re-downloads full history)
# download_futures_data(dry_run=False)
#
# # Force re-download specific products
# download_futures_data(products=["ES", "NQ"], force=True, dry_run=False)
# ```
#
# **Note**: Databento downloads replace full history (no incremental updates).
# Plan updates strategically to minimize cost.

# %% [markdown]
# ## Summary
#
# | Item | Value |
# |------|-------|
# | Products | 30 core (+ 6 extension) |
# | Frequencies | Hourly (raw), Daily (derived) |
# | Coverage | 2011-2025 |
# | Provider | Databento (**PAID** - $125 free credit) |
# | Config | `config.yaml` |
# | Loader | `load_cme_futures(frequency, products, tenors, start_date, end_date)` |
#
# **CRITICAL**: Always run `estimate_futures_cost()` before downloading!
