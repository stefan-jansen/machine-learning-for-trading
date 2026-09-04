# FX Strategy Research

Research workspace for FX strategy work against MetaTrader 5
(via `mt5_client` / Wine bridge on Linux).

**Strategy (declared):** H4 range breakout + ATR → binary baseline
`breakout` / `no_trade` (`false_breakout` deferred). FTMO $10k Challenge 2-Step.

## Repo / branch

| Item | Value |
|------|--------|
| Path | `PKM/FX Strategy Research/` (inside ML4T monorepo) |
| Branch | `fx-research` (base `c2c7b6ca`, FX commits only — no TabM) |
| Remote | `fork` → `git@github.com:Rezzaa13/machine-learning-for-trading.git` |
| Tracking | `fork/fx-research` |
| Config | [`config/setup.yaml`](config/setup.yaml) |

```bash
git switch fx-research
git log --oneline -5
git push fork fx-research
```

## Setup

- Project root: this folder
- Files: `mt5_client.py`, `test_connection.py`, `load_config.py`, `probe_symbols.py`, `pull_ohlcv_h4.py`, `feasibility_h4.py`, `.gitignore`, `README.md`, `config/setup.yaml`
- `feasibility_h4.py` — breadth + cost exceedance on pulled H4 (no MT5 needed)
- Linux Python needs `mt5linux` (do **not** use system `/usr/bin/python` 3.14)
- Reuse the existing MQL-PYTHON venv (Python 3.12 + `mt5linux==1.0.11`):

```bash
MQL_PY="/run/media/me2/shared-data/vaults/PKM/02 - Projects/MQL-PYTHON/.venv/bin/python"
```

- Linux run order:
  1. Start MT5 terminal: `mt5`
  2. Start Wine bridge (keep that terminal open):  
     `"/run/media/me2/shared-data/vaults/PKM/02 - Projects/MQL-PYTHON/scripts/start-mt5-bridge.sh"`
  3. Run scripts in a **separate** shell with `"$MQL_PY"`

## Progress log

### Stage 0 — Connection smoke test

**Goal:** Verify Python ↔ MT5 link without pulling any market data.

**Files:**
- `mt5_client.py` — cross-platform MT5 accessor (`get_mt5`)
- `test_connection.py` — initialize, print `version()` + `account_info()`, shutdown
- `.gitignore` — exclude `data/`, parquet/csv, venv, caches

**How to run:**

```bash
cd "/run/media/me2/shared-data/vaults/PKM/02 - Projects/machine-learning-for-trading/PKM/FX Strategy Research"
"$MQL_PY" test_connection.py
```

**Expected output:**
- `OK: connected` plus version and account info, or
- `FAIL: …` with the error

**Runs so far:**
| When | Interpreter | Result |
|------|-------------|--------|
| 2026-09-04 | `/usr/bin/python` (3.14) | `FAIL: No module named 'mt5linux'` |
| 2026-09-04 | MQL-PYTHON `.venv` (3.12) | `FAIL: [Errno 111] Connection refused` (bridge down) |
| 2026-09-04 | MQL-PYTHON `.venv` (3.12) + bridge up | `OK: connected` — version `(500, 6140, '21 Aug 2026')`, account logged in |

**Status:** complete — on `fx-research` / `fork`

### Stage 0.5 — Strategy setup config

**Goal:** Declare FTMO/MT5 research setup (single source of truth before data pull).

**File:** [`config/setup.yaml`](config/setup.yaml)

| Field | Choice |
|-------|--------|
| Idea | range breakout + ATR (Donchian lookback 20, ATR 14) |
| Baseline | binary `breakout` / `no_trade` (`false_breakout` deferred) |
| Timeframe | H4 (provisional — needs feasibility) |
| Universe | same 20 G10 pairs as `case_studies/fx_pairs` / `01_feasibility_analysis.py` (MT5 names: `EURUSD`, …) |
| Account | FTMO $10k Challenge 2-Step, leverage 100, `initial_cash: 10000` |
| Venue | MT5 bridge `127.0.0.1:18812`, server `FTMO-Server3` |

**Status:** complete — config + README on `fx-research` / `fork`

### Stage 1 — Symbol probe + H4 OHLCV pull

**Goal:** Confirm FTMO symbol names, then pull H4 bars (no labels yet).

**Files:**
- `load_config.py` — read `config/setup.yaml`, connect via `mt5_client`
- `probe_symbols.py` — resolve universe → `data/symbol_map.csv`
- `pull_ohlcv_h4.py` — H4 bars → `data/ohlcv_h4/{symbol}.csv`
- `feasibility_h4.py` — breadth + cost exceedance (offline on CSVs)

**How to run** (bridge up for probe/pull; feasibility is offline):

```bash
cd "/run/media/me2/shared-data/vaults/PKM/02 - Projects/machine-learning-for-trading/PKM/FX Strategy Research"
"$MQL_PY" probe_symbols.py
"$MQL_PY" pull_ohlcv_h4.py
"$MQL_PY" feasibility_h4.py
```

**Runs (2026-09-04):**
| Step | Result |
|------|--------|
| `probe_symbols.py` | **20/20** exact names (no suffix); spreads sampled on live ticks |
| `pull_ohlcv_h4.py` | **20/20** H4 CSVs under `data/ohlcv_h4/` (~5000 bars each, common window ~2023-06-19 → 2026-09-04) |
| `feasibility_h4.py` | common-window breadth **20/20**; median cost exceedance **h6=80.1%**, **h30=90.6%** → provisional **H4 keep** |

**Status:** Stage 1 complete (probe + pull + feasibility). Data gitignored.

### Next (planned)

- Stage 2: labels `breakout` / `no_trade` (Donchian+ATR from `setup.yaml`, point-in-time)
- Then baseline binary model checkpoint
