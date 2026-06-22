# Factor Data (Fama-French, AQR)

Academic factor-return series used for factor attribution in backtest
tearsheets (Chs 16-20) and as explanatory regressors in Ch10-14 factor
modelling work. Two providers, both free, both daily and monthly.

## Fama-French (Ken French Data Library)

- **Source**: Ken French Data Library
  (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
- **Coverage**: 1926-07 → present (daily); 1926-07 → present (monthly).
- **Factors**: FF3 (Mkt-RF, SMB, HML, RF), FF5 (+ RMW, CMA), Momentum
  (MOM), plus developed-market FF3, size/B-M 25-portfolio sorts, and
  industry-return 5-portfolio sorts.
- **Size on disk**: ~1 MB total.
- **Runtime**: under 1 minute (small CSV pulls from Dartmouth).
- **API key**: not required.
- **License / attribution**: Factor series are distributed under Ken
  French's terms (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) —
  free for academic and personal use with attribution. Cite Fama &
  French (1993, 2015) when publishing.

## AQR (AQR Data Sets)

- **Source**: AQR Data Sets (https://www.aqr.com/Insights/Datasets).
- **Coverage**: Varies by series (QMJ 1957→; BAB 1931→; HML-Devil 1926→;
  VME 1972→; century premia 1800s→).
- **Factors**: QMJ (Quality Minus Junk), BAB (Betting Against Beta),
  HML-Devil (value, devil variant), VME (Value/Momentum Everywhere),
  century premia, credit premium, ESG frontier, TSMOM, 6 QMJ portfolios,
  25 VME portfolios.
- **Size on disk**: ~12 MB.
- **Runtime**: ~1-2 minutes (Excel workbook downloads).
- **API key**: not required.
- **License / attribution**: AQR permits use for personal research with
  attribution to the AQR Capital Management white-paper that introduced
  the factor. See https://www.aqr.com/Insights/Datasets (terms on each
  dataset page).

## Download

```bash
# Fama-French — core (ff3, ff5, mom, daily + monthly)
uv run python data/factors/ff_download.py

# Fama-French — all 70+ datasets from the library
uv run python data/factors/ff_download.py --all

# Fama-French — single dataset
uv run python data/factors/ff_download.py --dataset ff5

# AQR — all four primary factor sets
uv run python data/factors/aqr_download.py
```

Output layout under `$ML4T_DATA_PATH/factors/`:

```
fama-french/
├── ff3_daily.parquet
├── ff3_monthly.parquet
├── ff5_daily.parquet
├── ff5_monthly.parquet
├── mom_daily.parquet
├── mom_monthly.parquet
├── ff3_developed_monthly.parquet
├── ind_5_monthly.parquet
├── port_size_monthly.parquet
└── bp_me_monthly.parquet
aqr/
├── qmj_factors.parquet          qmj_factors_daily.parquet      qmj_6_portfolios.parquet
├── bab_factors.parquet          bab_factors_daily.parquet
├── hml_devil.parquet            hml_devil_daily.parquet
├── vme_factors.parquet          vme_portfolios.parquet
├── century_premia.parquet       credit_premium.parquet
├── esg_frontier.parquet         tsmom.parquet
├── metadata.json
└── source/                      # raw Excel / CSV archives
```

## Loading

```python
from data import load_ff_factors, load_aqr_factors

# Fama-French
ff5 = load_ff_factors(dataset="ff5", frequency="daily")
ff3 = load_ff_factors(dataset="ff3", frequency="monthly")
mom = load_ff_factors(dataset="mom", frequency="monthly")
ff = load_ff_factors(
    dataset="ff5", frequency="daily",
    start_date="2010-01-01", end_date="2023-12-31",
)

# AQR
qmj = load_aqr_factors(dataset="qmj")
bab = load_aqr_factors(dataset="bab")
vme = load_aqr_factors(dataset="vme")
hml = load_aqr_factors(dataset="hml_devil")
```

Schema (both loaders return canonical `timestamp` + per-factor float
columns; FF files include `RF` risk-free rate, AQR files include
per-geography columns).

## Consumers

### Fama-French
- **Ch16**: `09_performance_reporting.py` (factor attribution tab).
- **All 9 case studies** — `*_strategy_analysis.py` uses FF5 for
  factor-attribution tearsheets (`case_studies/utils/factor_attribution.py`).

### AQR
- **Ch10**: factor-family surveys (AQR QMJ / BAB primary references).
- **Ch14**: latent factor models use AQR factor returns as comparison
  benchmarks.
