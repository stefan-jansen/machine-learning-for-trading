# ETF Universe (Yahoo Finance)

100 diversified ETFs spanning nine thematic categories — the red-thread
universe for the Ch6 momentum strategy and the ETF case study that runs
through Ch11-20. Daily OHLCV back to 2006.

## Dataset

- **Source**: Yahoo Finance, pulled via the `yfinance` Python package.
- **Coverage**: 2006-01-01 → present, daily OHLCV.
- **Symbols**: 100 ETFs across 9 categories.
- **Size on disk**: ~29 MB.
- **Runtime**: ~2-3 minutes for a full refresh (yfinance uses public
  Yahoo Finance endpoints with light rate-limiting).
- **API key**: not required.
- **License / attribution**: Yahoo Finance data is free for personal
  and educational use (https://policies.yahoo.com/us/en/yahoo/terms/index.htm).
  Redistribution of the raw OHLCV is not permitted; derived analytics
  (returns, features, model outputs) are fine. When publishing results,
  cite Yahoo Finance as the source.

## Categories

| Group                    | Count | Symbols                                                                                          |
| ------------------------ | ----- | ------------------------------------------------------------------------------------------------ |
| US Equity — Broad        | 10    | SPY, QQQ, IWM, DIA, VTI, MDY, IJR, RSP, IVW, IVE                                                 |
| US Equity — Style        | 10    | VTV, VUG, MTUM, QUAL, VLUE, USMV, DVY, SDY, VIG, SCHD                                            |
| US Sectors               | 13    | XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, XLRE, VNQ, IYR                                 |
| International Developed  | 18    | EFA, VEA, VGK, IEFA, ACWI, ACWX, EWJ, EWG, EWU, EWT, EWH, EWQ, EWL, EWN, EWI, EWP, EWC, EWA      |
| Emerging Markets         | 11    | EEM, VWO, IEMG, FXI, MCHI, EWZ, EWY, EWW, INDA, EZA, THD                                         |
| Fixed Income             | 15    | AGG, BND, BNDX, TLT, IEF, SHY, GOVT, BIL, LQD, VCSH, HYG, JNK, TIP, EMB, MUB                     |
| Commodities              | 9     | GLD, IAU, SLV, PPLT, USO, UNG, DBC, GSG, DBA                                                     |
| Specialty                | 10    | IBB, XBI, SMH, SOXX, KRE, XME, OIH, XRT, ITB, ITA                                                |
| Currency                 | 4     | UUP, FXE, FXY, FXB                                                                               |

Full symbol list + category tagging: `config.yaml`.

## Download

```bash
uv run python data/etfs/market/download.py              # all 100 ETFs
uv run python data/etfs/market/download.py --symbol SPY # single symbol
uv run python data/etfs/market/download.py --dry-run    # plan only
```

Output layout under `$ML4T_DATA_PATH/etfs/`:

```
ohlcv_daily.parquet                  # consolidated 100-ETF daily OHLCV (loader target)
ohlcv/symbol=<TICKER>/data.parquet   # hive-partitioned per-symbol bars (provider-native)
etfs_dictionary.parquet              # symbol metadata (category, inception, AUM, expense ratio)
config.yaml                          # universe definition + category tags
```

## Loading

```python
from data import load_etfs

df = load_etfs()                                             # all 100 ETFs
df = load_etfs(symbols=["SPY", "QQQ", "IWM"])
df = load_etfs(start_date="2020-01-01", end_date="2024-12-31")
```

Schema (canonical):

| Column      | Type     | Description   |
| ----------- | -------- | ------------- |
| `symbol`    | String   | ETF ticker    |
| `timestamp` | Date     | Trading date  |
| `open`      | Float    | Opening price |
| `high`      | Float    | High price    |
| `low`       | Float    | Low price     |
| `close`     | Float    | Closing price |
| `volume`    | Int      | Trading volume|

## Consumers

- **Ch2**: `01_etf_eda.py`.
- **Ch6**: `01_etfs_setup.py` (strategy definition).
- **Ch8**: `01_price_volume_features.py`, `03_structural_cross_instrument_features.py`, `05_feature_selection.py`, `06_robustness_sensitivity.py`, `07_event_studies.py`.
- **Ch16**: `06_framework_parity.py`, `09_performance_reporting.py`, `11_sharpe_ratio_inference.py`.
- **Ch18**: `01_cost_taxonomy.py`, `03_market_impact_calibration.py`, `06_ml4t_execution_demo.py`.
- **`case_studies/etfs/`**: full pipeline from `01_feasibility_analysis.py` through
  `18_strategy_analysis.py` — the flagship reader case study.
