# FX Strategy Research

Research workspace for FX strategy work against MetaTrader 5
(via `mt5_client` / Wine bridge on Linux).

## Repo / branch

| Item | Value |
|------|--------|
| Path | `PKM/FX Strategy Research/` (inside ML4T monorepo) |
| Branch | `fx-research` (base `c2c7b6ca`, FX commit only — no TabM) |
| Tip | `4aa21442` — `chore(fx-research): init FX Strategy Research workspace` |
| Remote | `fork` → `git@github.com:Rezzaa13/machine-learning-for-trading.git` |
| Tracking | `fork/fx-research` (pushed 2026-09-04) |

```bash
git switch fx-research
git push fork fx-research
```

## Setup

- Project root: this folder
- Files: `mt5_client.py`, `test_connection.py`, `.gitignore`, `README.md`
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

**Status:** Stage 0 complete — committed on `fx-research`, pushed to `fork`

### Next (planned)

- Stage 1: pull price / bar data into `data/` (gitignored)
