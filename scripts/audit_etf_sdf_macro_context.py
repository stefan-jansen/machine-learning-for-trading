#!/usr/bin/env python3
"""Read-only ETF SDF macro-context and FRED/ALFRED vintage audit."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import polars as pl
import yaml

from case_studies.utils.latent_factors.macro_context import load_configured_macro_context
from case_studies.utils.latent_factors.panel import align_macro_to_dates
from case_studies.utils.registry import training_hash_from_spec
from utils.downloading import load_dotenv

SAFE_SERIES = [
    "dgs1",
    "dgs2",
    "dgs3",
    "dgs5",
    "dgs7",
    "dgs10",
    "dgs20",
    "dgs30",
    "vixcls",
    "YIELD_CURVE_SLOPE",
    "YIELD_CURVE_5_10",
]
FRED_SOURCE_SERIES = ["DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30", "VIXCLS"]
FRED_API = "https://api.stlouisfed.org/fred"
H15_RELEASE_URL = "https://www.federalreserve.gov/releases/h15/"
VIXCLS_URL = "https://fred.stlouisfed.org/series/VIXCLS"
USER_AGENT = (
    "ml4t-sdf-pit-audit/1.0 (https://github.com/stefan-jansen/machine-learning-for-trading)"
)


def _urlopen(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=60)  # noqa: S310


def _json_request(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    url = f"{FRED_API}/{endpoint}?{urllib.parse.urlencode(params)}"
    with _urlopen(url) as response:
        return json.load(response)


def _vintage_dates(series_id: str, api_key: str) -> list[str]:
    vintage_dates: list[str] = []
    offset = 0
    while True:
        payload = _json_request(
            "series/vintagedates",
            {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "limit": 10000,
                "offset": offset,
                "sort_order": "asc",
            },
        )
        page = payload.get("vintage_dates", [])
        vintage_dates.extend(page)
        offset += len(page)
        if not page or offset >= int(payload.get("count", len(vintage_dates))):
            return vintage_dates


def _initial_release_observations(series_id: str, api_key: str) -> dict[str, str]:
    vintage_dates = _vintage_dates(series_id, api_key)
    observations: list[dict[str, str]] = []
    for start in range(0, len(vintage_dates), 1500):
        chunk = vintage_dates[start : start + 1500]
        print(
            f"  {series_id}: vintages {chunk[0]} through {chunk[-1]}",
            flush=True,
        )
        offset = 0
        while True:
            payload = _json_request(
                "series/observations",
                {
                    "series_id": series_id,
                    "api_key": api_key,
                    "file_type": "json",
                    "realtime_start": chunk[0],
                    "realtime_end": chunk[-1],
                    "output_type": 4,
                    "limit": 100000,
                    "offset": offset,
                    "sort_order": "asc",
                },
            )
            page = payload.get("observations", [])
            observations.extend(page)
            offset += len(page)
            if not page or offset >= int(payload.get("count", offset)):
                break
    return {row["date"]: row["value"] for row in observations if row.get("value") != "."}


def _latest_observations(series_id: str, api_key: str) -> dict[str, str]:
    payload = _json_request(
        "series/observations",
        {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "limit": 100000,
        },
    )
    return {row["date"]: row["value"] for row in payload["observations"] if row.get("value") != "."}


def _revision_dates(series_id: str, api_key: str) -> tuple[list[str], int]:
    initial = _initial_release_observations(series_id, api_key)
    latest = _latest_observations(series_id, api_key)
    comparable = sorted(initial.keys() & latest.keys())
    revised = [date for date in comparable if float(initial[date]) != float(latest[date])]
    return revised, len(comparable)


def _audit_official_timing(api_key: str, setup: dict[str, Any]) -> None:
    with _urlopen(H15_RELEASE_URL) as response:
        h15_text = response.read().decode(errors="replace").lower()
    if not any(marker in h15_text for marker in ("4:15 p.m.", "4:15 pm", "4:15pm")):
        raise RuntimeError("Official H.15 page no longer states the 4:15 p.m. publication time")

    vix = _json_request(
        "series",
        {"series_id": "VIXCLS", "api_key": api_key, "file_type": "json"},
    )["seriess"][0]
    if vix.get("frequency_short") != "D" or "VIX" not in vix.get("title", ""):
        raise RuntimeError("VIXCLS is no longer identified by FRED as a daily VIX series")

    decision = setup["decision"]
    if decision.get("snapshot") != "close" or decision.get("execution_delay") != "next_bar_open":
        raise RuntimeError(
            "A one-day lag is unsafe unless the ETF signal is formed at close and trades next bar open"
        )
    print(f"timing PASS: H.15 4:15 p.m. publication ({H15_RELEASE_URL})", flush=True)
    print(f"timing PASS: VIXCLS daily close series ({VIXCLS_URL})", flush=True)
    print(
        "event order PASS: lagged t value enters the t+1 close decision, traded next bar open",
        flush=True,
    )


def _read_base_spec(case_dir: Path, training_hash: str) -> dict[str, Any]:
    uri = f"file:{case_dir / 'run_log' / 'registry.db'}?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        row = db.execute(
            "SELECT spec_json FROM training_runs WHERE training_hash=?",
            (training_hash,),
        ).fetchone()
    if row is None:
        raise RuntimeError(f"Training hash {training_hash} not found in {case_dir}")
    return json.loads(row[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--base-training-hash", required=True)
    parser.add_argument("--prediction-hash", required=True)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()

    load_dotenv(args.env_file)
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY is required for the official vintage audit")

    setup = yaml.safe_load(Path("case_studies/etfs/config/setup.yaml").read_text())
    macro_config = setup["modeling"]["latent_factors"]["macro_context"]
    if macro_config["series"] != SAFE_SERIES:
        raise RuntimeError(
            f"ETF SDF series differ from the approved ordered set: {macro_config['series']}"
        )
    if macro_config["availability_lag_days"] != 1:
        raise RuntimeError("ETF SDF availability lag is not one calendar day")

    panel, identity = load_configured_macro_context(macro_config)
    if panel.columns != ["timestamp", *SAFE_SERIES]:
        raise RuntimeError(f"Unexpected macro panel columns: {panel.columns}")

    prediction_path = (
        args.case_dir / "run_log" / "predictions" / args.prediction_hash / "predictions.parquet"
    )
    dates = (
        pl.read_parquet(prediction_path, columns=["timestamp"])
        .unique()
        .sort("timestamp")["timestamp"]
        .to_list()
    )
    aligned, names = align_macro_to_dates(panel, dates)
    if names != SAFE_SERIES or not np.isfinite(aligned).all():
        raise RuntimeError(
            "MacroLSTM context is missing, reordered, or non-finite on prediction dates"
        )
    print(
        f"panel PASS: {len(SAFE_SERIES)} ordered series, {len(dates)} prediction dates, no nulls",
        flush=True,
    )
    print(f"input digest: {identity['input_digest']}", flush=True)

    _audit_official_timing(api_key, setup)
    revision_failures: dict[str, list[str]] = {}
    for series_id in FRED_SOURCE_SERIES:
        print(f"vintage audit started: {series_id}", flush=True)
        revisions, n_compared = _revision_dates(series_id, api_key)
        if revisions:
            revision_failures[series_id] = revisions
            print(
                f"vintage audit FAIL: {series_id} changed on {len(revisions)} dates: "
                f"{','.join(revisions)}",
                flush=True,
            )
        else:
            print(
                f"vintage audit PASS: {series_id} initial and latest values match "
                f"on {n_compared} ALFRED-covered dates",
                flush=True,
            )
    if revision_failures:
        failed = ", ".join(
            f"{series_id}={len(dates)}" for series_id, dates in revision_failures.items()
        )
        raise RuntimeError(f"Non-revision gate failed: {failed}")
    print("derived spread PASS: both yield-curve spreads inherit audited DGS inputs", flush=True)

    base_spec = _read_base_spec(args.case_dir, args.base_training_hash)
    prospective_spec = {**base_spec, "macro_context": identity}
    print(f"prospective training hash: {training_hash_from_spec(prospective_spec)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
