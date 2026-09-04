# FX Strategy Research

Research workspace for FX strategy work against MetaTrader 5
(via `mt5_client` / Wine bridge on Linux).

**Strategy:** H4 range breakout + ATR → binary baseline `breakout` / `no_trade`
(`false_breakout` deferred). FTMO $10k Challenge 2-Step.

**Where we are:** Stage 0–1 complete. H4 **provisionally kept** after feasibility.
Next = Stage 2 labels.

## Repo / branch

| Item | Value |
|------|--------|
| Path | `PKM/FX Strategy Research/` (inside ML4T monorepo) |
| Branch | `fx-research` (base `c2c7b6ca` — no TabM) |
| Tip | `368648f4` — Stage 1 probe / H4 pull / feasibility |
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

**File:** [`config/setup.yaml`](config/setup.yaml)

| Field | Choice |
|-------|--------|
| Idea | range breakout + ATR (Donchian 20, ATR 14) |
| Baseline | binary `breakout` / `no_trade` |
| Timeframe | H4 (locked provisional after Stage 1) |
| Universe | 20 G10 pairs (same as `case_studies/fx_pairs`) — MT5 names |
| Account | FTMO $10k 2-Step, leverage 100, cash 10_000 |
| Venue | bridge `127.0.0.1:18812`, `FTMO-Server3` |

**Status:** complete

### Stage 1 — Probe, H4 pull, feasibility

```bash
"$MQL_PY" probe_symbols.py      # needs bridge
"$MQL_PY" pull_ohlcv_h4.py      # needs bridge; retries; skips existing CSVs
"$MQL_PY" feasibility_h4.py     # offline on data/
```

| Step | Result (2026-09-04) |
|------|---------------------|
| Probe | **20/20** exact (`EURUSD`, … — no suffix) |
| Pull | **20/20** files in `data/ohlcv_h4/` |
| Common window | ~2023-06-19 → 2026-09-04 (~99.8k panel rows) |
| Breadth (dev) | min/median/max **20/20/20**, under floor **0** |
| Cost exceedance | median **h6 = 80.1%**, **h30 = 90.6%** |
| Verdict | **OK — provisional H4 keep** |

Outputs (gitignored): `data/symbol_map.csv`, `data/ohlcv_h4/*.csv`,
`data/feasibility_h4_summary.txt`, `data/feasibility_h4_by_symbol.csv`

**Status:** complete — committed `368648f4` on `fork/fx-research`

### Next — Stage 2

- Build point-in-time labels: `breakout` / `no_trade` from Donchian+ATR in `setup.yaml`
- No look-ahead on range bands (`lag: 1`)
- Then baseline binary model checkpoint (not yet)
