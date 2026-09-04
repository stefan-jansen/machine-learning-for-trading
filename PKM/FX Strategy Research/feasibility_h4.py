"""Stage 1c — H4 feasibility on pulled CSVs (no model fit).

Reads data/ohlcv_h4/*.csv + config/setup.yaml.
Asks: breadth at each H4 bar, and what fraction of |fwd returns|
clear a round-trip cost scaled from setup.yaml / bar spreads.

Writes:
  data/feasibility_h4_summary.txt
  data/feasibility_h4_by_symbol.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from load_config import ROOT, load_setup

OHLCV_DIR = ROOT / "data" / "ohlcv_h4"
OUT_TXT = ROOT / "data" / "feasibility_h4_summary.txt"
OUT_CSV = ROOT / "data" / "feasibility_h4_by_symbol.csv"


def is_major(symbol: str) -> bool:
    return "USD" in symbol.upper()


def round_trip_bps(symbol: str, setup: dict, bar_spread_pts: float | None, digits: int | None) -> float:
    """One-way mid of setup band × 2, optionally floored by median bar spread in bps."""
    bands = setup["costs"]["spread_bps"]
    one_way = bands["major_pairs"] if is_major(symbol) else bands["cross_pairs"]
    setup_rt = 2.0 * float(np.mean(one_way))
    if bar_spread_pts is None or not digits:
        return setup_rt
    # point size ≈ 10^{-digits}; spread column is in points
    # bps ≈ spread_points * point / price * 1e4 — use last close later; here approx via points
    # Prefer setup band; report both. Use setup for exceedance primary.
    return setup_rt


def load_panel() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(OHLCV_DIR.glob("*.csv")):
        df = pd.read_csv(path, parse_dates=["time"])
        if df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize("UTC")
        else:
            df["time"] = df["time"].dt.tz_convert("UTC")
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no CSVs in {OHLCV_DIR}")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    setup = load_setup()
    horizon = int(setup["labels"]["forward_horizon_bars"])
    research_horizons = []
    for name in setup["labels"].get("research_variants", []):
        # fwd_ret_6bar → 6
        try:
            research_horizons.append(int(name.rsplit("_", 1)[-1].replace("bar", "")))
        except ValueError:
            pass
    horizons = sorted(set([horizon, *research_horizons]))
    floor = int(setup["universe"].get("breadth_floor", setup["universe"]["n_assets"]))
    holdout_start = pd.Timestamp(setup["evaluation"]["holdout_start"], tz="UTC")

    panel = load_panel()
    symbols = sorted(panel["symbol"].unique())
    print(f"OK: loaded {len(panel):,} rows | {len(symbols)} symbols | horizons {horizons}")

    # Align to intersection so breadth isn't penalized by staggered pull windows
    t0 = panel.groupby("symbol")["time"].min().max()
    t1 = panel.groupby("symbol")["time"].max().min()
    panel = panel[(panel["time"] >= t0) & (panel["time"] <= t1)].copy()
    print(f"OK: common window {t0} … {t1} → {len(panel):,} rows")

    # --- Breadth ---
    breadth = panel.groupby("time")["symbol"].nunique().sort_index()
    # Development window only for decisions that shape the setup
    breadth_dev = breadth[breadth.index < holdout_start]
    under = int((breadth_dev < floor).sum()) if len(breadth_dev) else 0
    if len(breadth_dev):
        bmin, bmed, bmax = int(breadth_dev.min()), float(breadth_dev.median()), int(breadth_dev.max())
    else:
        bmin = bmed = bmax = float("nan")
    print(
        f"breadth (dev < {holdout_start.date()}): "
        f"min={bmin} max={bmax} median={bmed} | "
        f"under floor {floor}: {under}/{len(breadth_dev)}"
    )

    # --- Per-symbol cost exceedance ---
    rows = []
    lines = [
        "# H4 feasibility summary",
        f"symbols={len(symbols)} rows={len(panel):,} holdout_start={holdout_start.date()}",
        f"breadth_dev min/median/max={bmin}/{bmed}/{bmax} "
        f"under_floor_{floor}={under}/{len(breadth_dev)}",
        "",
    ]

    for sym, g in panel.groupby("symbol"):
        g = g.sort_values("time").drop_duplicates("time")
        g = g[g["time"] < holdout_start].copy()
        if len(g) < max(horizons) + 50:
            print(f"FAIL: {sym} too short in dev ({len(g)} bars)")
            continue
        mid_spread = float(g["spread"].median()) if "spread" in g else float("nan")
        rt = round_trip_bps(sym, setup, mid_spread, None)
        # Approximate bar-implied one-way bps from median spread points / close
        close = g["close"].astype(float)
        if "spread" in g and g["spread"].notna().any():
            # MT5 spread is in points; point ≈ 10^{-digits}. Infer digits from price magnitude.
            # Use (ask-bid)/mid ≈ spread_points * point / mid. point ≈ tick size from unique diffs —
            # simpler: bps ≈ spread / (10 ** digits) / mid * 1e4. digits from setup probe unused;
            # use: for JPY pairs price~100+, digits 3 → point 0.001; else digits 5 → 1e-5.
            point = 0.001 if close.median() > 20 else 1e-5
            bar_ow_bps = float((g["spread"] * point / close).median() * 1e4)
            bar_rt_bps = 2.0 * bar_ow_bps
        else:
            bar_rt_bps = float("nan")

        rets = {}
        exceed = {}
        for h in horizons:
            fwd = close.shift(-h) / close - 1.0
            abs_bps = fwd.abs() * 1e4
            # Primary cost yardstick: max(setup_rt, bar_rt) so we don't understate live spread
            cost = max(rt, bar_rt_bps) if np.isfinite(bar_rt_bps) else rt
            valid = abs_bps.dropna()
            frac = float((valid > cost).mean()) if len(valid) else float("nan")
            med = float(valid.median()) if len(valid) else float("nan")
            rets[h] = med
            exceed[h] = frac

        row = {
            "symbol": sym,
            "n_dev_bars": len(g),
            "setup_rt_bps": round(rt, 2),
            "bar_rt_bps": round(bar_rt_bps, 2) if np.isfinite(bar_rt_bps) else "",
            "median_spread_pts": round(mid_spread, 2) if np.isfinite(mid_spread) else "",
            "t0": str(g["time"].iloc[0]),
            "t1": str(g["time"].iloc[-1]),
        }
        for h in horizons:
            row[f"med_abs_bps_h{h}"] = round(rets[h], 2)
            row[f"exceed_cost_h{h}"] = round(exceed[h], 4)
        rows.append(row)

        line = (
            f"{sym}: setup_rt={rt:.1f}bps bar_rt={bar_rt_bps:.1f}bps | "
            + " ".join(f"h{h} med|r|={rets[h]:.1f}bps exceed={exceed[h]:.1%}" for h in horizons)
        )
        print(line)
        lines.append(line)

    by_sym = pd.DataFrame(rows).sort_values("symbol")
    by_sym.to_csv(OUT_CSV, index=False)

    # Aggregate verdict
    lines.append("")
    for h in horizons:
        col = f"exceed_cost_h{h}"
        med_ex = float(by_sym[col].median())
        lines.append(f"median exceedance h{h}: {med_ex:.1%} across {len(by_sym)} symbols")
        print(f"median exceedance h{h}: {med_ex:.1%}")

    primary_ex = float(by_sym[f"exceed_cost_h{horizon}"].median())
    cost_ok = primary_ex >= 0.15
    breadth_ok = under == 0 and len(breadth_dev) > 0
    if breadth_ok and cost_ok:
        verdict = (
            f"OK: provisional H4 keep — full breadth on common window, "
            f"primary h{horizon} median exceedance {primary_ex:.1%}"
        )
    elif cost_ok and not breadth_ok:
        verdict = (
            f"WARN: cost OK (h{horizon} exceedance {primary_ex:.1%}) but breadth "
            f"under floor on {under}/{len(breadth_dev)} stamps — inspect gaps"
        )
    elif breadth_ok and not cost_ok:
        verdict = (
            f"WARN: breadth OK but h{horizon} exceedance {primary_ex:.1%} low — "
            "longer hold / majors-only / different TF"
        )
    else:
        verdict = "FAIL: both breadth and cost look weak for declared H4 setup"
    lines.append("")
    lines.append(f"VERDICT: {verdict}")
    print(f"\nVERDICT: {verdict}")

    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nOK: wrote {OUT_TXT}")
    print(f"OK: wrote {OUT_CSV}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}")
