# FX Strategy Research

Research workspace for FX strategy work against MetaTrader 5
(via `mt5_client` / Wine bridge on Linux).

## Setup

- Project root: this folder
- Requires `mt5_client.py` in the same directory
- Linux Python needs `mt5linux` (do **not** use system `/usr/bin/python` 3.14)
- Reuse the existing MQL-PYTHON venv (Python 3.12 + `mt5linux==1.0.11`):

```bash
MQL_PY="/run/media/me2/shared-data/vaults/PKM/02 - Projects/MQL-PYTHON/.venv/bin/python"
```

- Linux run order:
  1. Start MT5 terminal: `mt5`
  2. Start Wine bridge: `"/run/media/me2/shared-data/vaults/PKM/02 - Projects/MQL-PYTHON/scripts/start-mt5-bridge.sh"`
  3. Run the smoke test below

## Progress log

### Stage 0 — Connection smoke test

**Goal:** Verify Python ↔ MT5 link without pulling any market data.

**Files:**
- `mt5_client.py` — cross-platform MT5 accessor (`get_mt5`)
- `test_connection.py` — initialize, print `version()` + `account_info()`, shutdown

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

**Status:** Stage 0 complete — connection smoke test passed

**Note:** keep the bridge terminal open while working; run the test in a *separate* shell.

### Next (planned)

- Pull price / bar data
