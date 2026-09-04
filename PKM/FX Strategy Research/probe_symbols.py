"""Stage 1a — resolve declared universe symbols on the live FTMO/MT5 terminal.

Reads symbols from config/setup.yaml, probes symbol_info / symbols_get,
prints a table, writes data/symbol_map.csv (gitignored via data/).
Does not pull OHLCV bars.
"""

from __future__ import annotations

import csv
from pathlib import Path

from load_config import ROOT, load_setup, mt5_connect

# Common retail / prop suffixes if bare name is missing
_SUFFIX_TRIES = ("", "m", ".m", "c", ".c", "i", ".i", ".pro", ".r", "r")


def _candidates(name: str) -> list[str]:
    out: list[str] = []
    for suf in _SUFFIX_TRIES:
        cand = f"{name}{suf}"
        if cand not in out:
            out.append(cand)
    return out


def resolve_symbol(mt5, declared: str) -> tuple[str | None, str]:
    """Return (resolved_name_or_None, note)."""
    for cand in _candidates(declared):
        info = mt5.symbol_info(cand)
        if info is not None:
            note = "exact" if cand == declared else f"suffix→{cand}"
            return cand, note

    # Fuzzy: any market-watch / terminal symbol containing the bare name
    matches = mt5.symbols_get(f"*{declared}*")
    if matches:
        names = sorted({m.name for m in matches})
        # Prefer shortest name that starts with declared
        preferred = [n for n in names if n.upper().startswith(declared.upper())]
        pick = min(preferred or names, key=len)
        return pick, f"symbols_get→{pick} (candidates={len(names)})"

    return None, "NOT_FOUND"


def main() -> None:
    setup = load_setup()
    declared = list(setup["universe"]["symbols"])
    out_dir = ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "symbol_map.csv"

    mt5, _ = mt5_connect()
    rows: list[dict] = []
    try:
        print(f"OK: connected — probing {len(declared)} declared symbols")
        print(
            f"{'declared':<10} {'resolved':<14} {'visible':>7} {'spread':>8} "
            f"{'digits':>6} {'trade':>6}  note"
        )
        n_ok = 0
        for name in declared:
            resolved, note = resolve_symbol(mt5, name)
            if resolved is None:
                print(f"{name:<10} {'—':<14} {'—':>7} {'—':>8} {'—':>6} {'—':>6}  {note}")
                rows.append(
                    {
                        "declared": name,
                        "resolved": "",
                        "ok": 0,
                        "visible": "",
                        "spread": "",
                        "digits": "",
                        "trade_mode": "",
                        "note": note,
                    }
                )
                continue

            # Ensure symbol is selected so quotes update
            mt5.symbol_select(resolved, True)
            info = mt5.symbol_info(resolved)
            tick = mt5.symbol_info_tick(resolved)
            spread = ""
            if tick is not None and info is not None and info.point:
                spread = f"{(tick.ask - tick.bid) / info.point:.1f}"
            visible = getattr(info, "visible", "")
            digits = getattr(info, "digits", "")
            trade_mode = getattr(info, "trade_mode", "")
            print(
                f"{name:<10} {resolved:<14} {str(visible):>7} {spread:>8} "
                f"{str(digits):>6} {str(trade_mode):>6}  {note}"
            )
            rows.append(
                {
                    "declared": name,
                    "resolved": resolved,
                    "ok": 1,
                    "visible": visible,
                    "spread": spread,
                    "digits": digits,
                    "trade_mode": trade_mode,
                    "note": note,
                }
            )
            n_ok += 1

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "declared",
                    "resolved",
                    "ok",
                    "visible",
                    "spread",
                    "digits",
                    "trade_mode",
                    "note",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nOK: {n_ok}/{len(declared)} resolved → {out_csv}")
        if n_ok < len(declared):
            print("FAIL: some symbols missing — fix setup.yaml or broker suffix before pull")
        else:
            print("OK: full universe resolved — safe to run pull_ohlcv_h4.py")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}")
