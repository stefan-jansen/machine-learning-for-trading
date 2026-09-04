"""Stage 1b — pull H4 OHLCV for resolved universe into data/ohlcv_h4/.

Requires data/symbol_map.csv from probe_symbols.py (ok=1 rows).
Writes one CSV per symbol: data/ohlcv_h4/{resolved}.csv
No labels / features — bars only.

Retries on mt5linux 'Terminal: Call failed' (common over the Wine bridge).
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import pandas as pd

from load_config import ROOT, load_setup, mt5_connect

SYMBOL_MAP = ROOT / "data" / "symbol_map.csv"
OUT_DIR = ROOT / "data" / "ohlcv_h4"
MAX_ATTEMPTS = 5
SLEEP_S = 1.5


def load_resolved_symbols() -> list[tuple[str, str]]:
    if not SYMBOL_MAP.is_file():
        raise FileNotFoundError(
            f"missing {SYMBOL_MAP} — run probe_symbols.py first"
        )
    pairs: list[tuple[str, str]] = []
    with SYMBOL_MAP.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("ok") == "1" and row.get("resolved"):
                pairs.append((row["declared"], row["resolved"]))
    if not pairs:
        raise RuntimeError("symbol_map.csv has no ok=1 rows")
    return pairs


def fetch_rates(mt5, symbol: str, tf: int, n_bars: int):
    mt5.symbol_select(symbol, True)
    time.sleep(0.3)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)
    if rates is not None and len(rates) > 0:
        return rates
    # Fallback: from a fixed calendar window (UTC)
    end = pd.Timestamp.now("UTC").to_pydatetime()
    start = (pd.Timestamp.now("UTC") - pd.Timedelta(days=int(n_bars / 6 * 1.5))).to_pydatetime()
    return mt5.copy_rates_range(symbol, tf, start, end)


def main() -> None:
    setup = load_setup()
    n_bars = int(setup.get("mt5", {}).get("lookback_bars", 5000))
    pairs = load_resolved_symbols()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mt5, _ = mt5_connect()
    try:
        tf = mt5.TIMEFRAME_H4
        print(f"OK: connected — pulling H4 × {n_bars} bars for {len(pairs)} symbols")
        n_ok = 0
        for declared, resolved in pairs:
            out = OUT_DIR / f"{resolved}.csv"
            if out.is_file() and out.stat().st_size > 0:
                print(f"SKIP: {declared:>8} → {resolved:<12} (already have {out.name})")
                n_ok += 1
                continue
            rates = None
            last_err = None
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    rates = fetch_rates(mt5, resolved, tf, n_bars)
                    if rates is not None and len(rates) > 0:
                        break
                    last_err = mt5.last_error()
                except Exception as exc:  # noqa: BLE001 — bridge can raise mid-call
                    last_err = exc
                    rates = None
                print(f"  retry {attempt}/{MAX_ATTEMPTS}: {resolved} — {last_err}")
                time.sleep(SLEEP_S * attempt)

            if rates is None or len(rates) == 0:
                print(f"FAIL: {declared} ({resolved}) — no rates ({last_err})")
                continue

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df.insert(0, "declared", declared)
            df.insert(1, "symbol", resolved)
            df.to_csv(out, index=False)
            t0, t1 = df["time"].iloc[0], df["time"].iloc[-1]
            print(
                f"OK: {declared:>8} → {resolved:<12} {len(df):5d} bars  "
                f"{t0} … {t1}  → {out.name}"
            )
            n_ok += 1
            time.sleep(0.5)

        print(f"\nOK: wrote {n_ok}/{len(pairs)} files under {OUT_DIR}")
        if n_ok < len(pairs):
            print("FAIL: partial pull — inspect FAIL lines above")
        else:
            print("OK: full universe pulled")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}")
