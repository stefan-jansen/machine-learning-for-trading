# FX Strategy Research

Research workspace for FX strategy work against MetaTrader 5
(via `mt5_client` / Wine bridge on Linux).

**Strategy:** H4 range breakout + ATR → binary baseline `breakout` / `no_trade`
(`false_breakout` deferred). FTMO $10k Challenge 2-Step.

**Focus pair (v1.1):** `USDJPY` only (simpler labels/baseline). Full 20-pair
H4 history remains under `data/ohlcv_h4/` for later expansion.

**Where we are:** Stage 0–1 complete. H4 kept. Universe narrowed to USDJPY.
Next = Stage 2 labels on USDJPY.

## Repo / branch

| Item | Value |
|------|--------|
| Path | `PKM/FX Strategy Research/` (inside ML4T monorepo) |
| Branch | `fx-research` (base `c2c7b6ca` — no TabM) |
| Tip | `git log -1` on `fx-research` |
| Remote | `fork` → `git@github.com:Rezzaa13/machine-learning-for-trading.git` |
| Tracking | `fork/fx-research` |
| Config | [`config/setup.yaml`](config/setup.yaml) |

```bash
git switch fx-research
git log --oneline -5
git push fork fx-research
```

## Layout

| Path | Role |
|------|------|
| `config/setup.yaml` | Single source of truth (universe, H4, FTMO, labels, costs) |
| `mt5_client.py` / `load_config.py` | Bridge accessor + YAML loader |
| `test_connection.py` | Stage 0 smoke test |
| `probe_symbols.py` | Resolve broker symbol names → `data/symbol_map.csv` |
| `pull_ohlcv_h4.py` | H4 OHLCV → `data/ohlcv_h4/{SYMBOL}.csv` (skips existing) |
| `feasibility_h4.py` | Offline breadth + cost exceedance |
| `data/` | **gitignored** — bars, symbol map, feasibility outputs |
| `.gitignore` | Excludes `data/`, parquet/csv, venv, caches |

## Setup

- Linux: use MQL-PYTHON venv (Python 3.12 + `mt5linux`) — **not** `/usr/bin/python` 3.14

```bash
MQL_PY="/run/media/me2/shared-data/vaults/PKM/02 - Projects/MQL-PYTHON/.venv/bin/python"
cd "/run/media/me2/shared-data/vaults/PKM/02 - Projects/machine-learning-for-trading/PKM/FX Strategy Research"
```

- Run order when talking to MT5:
  1. `mt5` (terminal)
  2. bridge (keep open):  
     `"/run/media/me2/shared-data/vaults/PKM/02 - Projects/MQL-PYTHON/scripts/start-mt5-bridge.sh"`
  3. scripts in a **separate** shell with `"$MQL_PY"`

## Progress log

### Stage 0 — Connection smoke test

**Goal:** Python ↔ MT5 without price data.

```bash
"$MQL_PY" test_connection.py
```

| When | Interpreter | Result |
|------|-------------|--------|
| 2026-09-04 | `/usr/bin/python` (3.14) | `FAIL: No module named 'mt5linux'` |
| 2026-09-04 | MQL `.venv` | `FAIL: Connection refused` (bridge down) |
| 2026-09-04 | MQL `.venv` + bridge | `OK` — MT5 `(500, 6140)`, FTMO account logged in |

**Status:** complete

### Stage 0.5 — Strategy setup config

**File:** [`config/setup.yaml`](config/setup.yaml) (`setup_version: v1.1`)

| Field | Choice |
|-------|--------|
| Idea | range breakout + ATR (Donchian 20, ATR 14) |
| Baseline | binary `breakout` / `no_trade` |
| Timeframe | H4 (kept after Stage 1) |
| Universe | **`USDJPY` only** (v1.1 focus; was 20 G10 in Stage 1 screen) |
| Account | FTMO $10k 2-Step, leverage 100, cash 10_000 |
| Venue | bridge `127.0.0.1:18812`, `FTMO-Server3` |

**Status:** complete (narrowed to USDJPY)

### Stage 1 — Probe, H4 pull, feasibility (full 20, then focus)

```bash
"$MQL_PY" probe_symbols.py
"$MQL_PY" pull_ohlcv_h4.py
"$MQL_PY" feasibility_h4.py
```

| Step | Result (2026-09-04) |
|------|---------------------|
| Probe | **20/20** exact names |
| Pull | **20/20** H4 CSVs (kept on disk; research focus = USDJPY) |
| Feasibility (panel) | H4 keep — median h6 exceedance ~80% |
| **USDJPY** (focus) | live spread ~3 pts; h6 exceedance **~92%**; med \|r\| h6 ~31 bps |

**Status:** complete

### Stage 1.5 — Fix single pair

**Choice:** `USDJPY` (user). Rationale: major, tight spread, strong H4 cost clearance;
simpler path to Stage 2 labels than a 20-pair panel.

**Status:** `setup.yaml` + README updated to `n_assets: 1`

### Next — Stage 2

- Labels on **USDJPY** H4 only: `breakout` / `no_trade` (Donchian+ATR, point-in-time)
- Then baseline binary model checkpoint
