"""Generate synthetic intermediates for notebooks that would otherwise have no inputs.

Run once to enrich the test-data repo with minimal synthetic artifacts that let
the remaining skipped notebooks execute their code paths.

Usage:
    uv run python tests/generate_skip_data.py --output ~/ml4t/test-data

This generates, all of it under ``intermediates/``:
1. Engine divergence predictions (Ch16/07)
2. Signal quality synthesis data (Ch20/02)
3. MLOps registry and stub predictions (Ch26/03, Ch26/06)

Nothing here writes into ``data/``. Two generators used to, and both wrote where
nothing reads:

- ``generate_sec_10q_mda`` wrote ``alternative/text/sp500_10q_mda.parquet`` with a
  pre-canonical ``mda_text`` column, while ``load_sp500_10q_mda`` reads
  ``equities/fundamentals/10q/sp500/reference/all_10q_filings.parquet``. That file
  is production-sourced and declared as ``sec_filing_references`` in
  ``tests/create_test_data.py``, so Ch10/09 has been running against the real
  schema and the synthetic copy reached no reader.
- ``enrich_adv_columns`` added ``adv_21d`` to ``etfs/etf_universe.parquet`` and
  ``equities/us_equities.parquet``. Neither path exists: the fixture carries
  ``etfs/market/`` and ``equities/market/us_equities/``, so every run printed
  "SKIP (not found)". No notebook reads ``adv_21d`` from either file.

Keeping this script out of ``data/`` is what makes the fixture's producers
countable: ``tests/create_test_data.py`` derives from production and
``tests/generate_test_microstructure.py`` is synthetic, and
``tests/test_every_fixture_file_has_a_producer.py`` checks the fixture against
those two.

The FNSPID news fixture is not here either. It is subsampled from production by
``tests/create_test_data.py``, whose ``fnspid_news`` dataset bounds it by the
us_equities panel's date range so 07_news_return_signals' price join has dates to
land on; the synthetic generator that used to live here wrote 2022-2024, which is
past the end of that panel.
"""

import argparse
import json
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

np.random.seed(42)

SYMBOLS_ETF = ["SPY", "QQQ", "IWM", "TLT", "GLD", "XLF", "XLK", "XLE", "EFA", "VWO"]
SYMBOLS_EQ = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ"]


def generate_engine_divergence_predictions(intermediates_dir: Path):
    """Generate predictions with model column for Ch16/07 engine divergence."""
    out = intermediates_dir / "ch16_signal_method_comparison"
    out.mkdir(parents=True, exist_ok=True)

    dates = pl.date_range(date(2022, 1, 3), date(2023, 12, 29), "1d", eager=True)
    rows = []
    for d in dates:
        for sym in SYMBOLS_ETF[:5]:
            rows.append(
                {
                    "timestamp": d,
                    "symbol": sym,
                    "prediction": np.random.normal(0, 0.02),
                    "model": "ridge_a1.0",
                }
            )

    df = pl.DataFrame(rows)
    df.write_parquet(out / "predictions_with_model.parquet")
    print(f"  Engine divergence: {len(df)} rows -> {out}")


def generate_signal_quality_data(intermediates_dir: Path):
    """Generate synthesis data for Ch20/02 signal quality notebook."""
    # The notebook reads from Ch20/01 aggregate_synthesis outputs
    out = intermediates_dir / "ch20_synthesis"
    out.mkdir(parents=True, exist_ok=True)

    case_studies = [
        "etfs",
        "crypto_perps_funding",
        "nasdaq100_microstructure",
        "sp500_equity_option_analytics",
        "us_firm_characteristics",
        "fx_pairs",
        "cme_futures",
        "sp500_options",
        "us_equities_panel",
    ]
    models = ["linear/ridge", "gbm/leaves_15", "deep_learning/lstm", "tabular_dl/tabm_l"]

    # IC comparison data
    ic_rows = []
    for cs in case_studies:
        for model in models:
            ic_rows.append(
                {
                    "case_study": cs,
                    "source": model,
                    "ic_mean": np.random.uniform(-0.02, 0.06),
                    "ic_std": np.random.uniform(0.01, 0.04),
                    "n_folds": 5,
                }
            )

    ic_df = pl.DataFrame(ic_rows)
    ic_df.write_parquet(out / "ic_comparison.parquet")

    # Synthesis JSON
    synthesis = {
        "case_studies": {
            cs: {
                "champion": {
                    "source": "gbm/leaves_15",
                    "sharpe": float(np.random.uniform(-0.5, 2.0)),
                },
                "holdout": {
                    "ic": float(np.random.uniform(-0.02, 0.1)),
                    "sharpe": float(np.random.uniform(-1, 3)),
                },
            }
            for cs in case_studies
        }
    }
    (out / "all_synthesis.json").write_text(json.dumps(synthesis, indent=2))
    print(f"  Signal quality: IC comparison + synthesis -> {out}")


def generate_mlops_data(intermediates_dir: Path):
    """Generate the registry and stub predictions Ch26/03 and Ch26/06 read."""
    # Ch26/03 needs a linear/lasso validation run in registry
    out = intermediates_dir / "us_equities_panel" / "run_log"
    out.mkdir(parents=True, exist_ok=True)

    db_path = out / "registry.db"
    db = sqlite3.connect(str(db_path))
    db.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
            run_id TEXT PRIMARY KEY,
            entry_point TEXT,
            source TEXT,
            label TEXT,
            config_hash TEXT,
            created_at TEXT,
            ic_mean REAL,
            status TEXT DEFAULT 'completed'
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS prediction_sets (
            pred_id TEXT PRIMARY KEY,
            run_id TEXT,
            entry_point TEXT,
            source TEXT,
            label TEXT,
            config_hash TEXT,
            created_at TEXT,
            ic_mean REAL,
            n_rows INTEGER,
            pred_path TEXT
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS prediction_metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            pred_id TEXT,
            fold INTEGER,
            ic REAL,
            n_rows INTEGER
        )
    """)

    # Insert a few synthetic runs
    for i, (source, ic) in enumerate(
        [
            ("linear/ridge_a1.0", 0.025),
            ("linear/lasso_a0.01", 0.018),
            ("gbm/leaves_15_mae", 0.042),
        ]
    ):
        run_id = f"run_{i:03d}"
        pred_id = f"pred_{i:03d}"
        db.execute(
            "INSERT OR REPLACE INTO training_runs VALUES (?,?,?,?,?,?,?,?)",
            (
                run_id,
                "06_linear" if "linear" in source else "07_gbm",
                source,
                "fwd_ret_1d",
                f"hash_{i}",
                "2026-01-01T00:00:00",
                ic,
                "completed",
            ),
        )
        db.execute(
            "INSERT OR REPLACE INTO prediction_sets VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                pred_id,
                run_id,
                "06_linear" if "linear" in source else "07_gbm",
                source,
                "fwd_ret_1d",
                f"hash_{i}",
                "2026-01-01T00:00:00",
                ic,
                1000,
                f"predictions/{pred_id}.parquet",
            ),
        )
        for fold in range(5):
            db.execute(
                "INSERT INTO prediction_metrics (pred_id, fold, ic, n_rows) VALUES (?,?,?,?)",
                (pred_id, fold, ic + np.random.normal(0, 0.005), 200),
            )

    db.commit()
    db.close()
    print(f"  MLOps registry: 3 runs -> {db_path}")

    # Generate stub predictions for the registry entries
    preds_dir = out.parent / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    dates = pl.date_range(date(2023, 1, 2), date(2023, 12, 29), "1d", eager=True)
    for i in range(3):
        rows = []
        for d in dates:
            for sym in SYMBOLS_EQ[:5]:
                rows.append(
                    {
                        "timestamp": d,
                        "symbol": sym,
                        "prediction": np.random.normal(0, 0.02),
                        "fold": np.random.randint(0, 5),
                    }
                )
        df = pl.DataFrame(rows)
        df.write_parquet(preds_dir / f"pred_{i:03d}.parquet")
    print(f"  MLOps predictions: 3 files -> {preds_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic intermediates for skipped notebooks"
    )
    parser.add_argument("--output", required=True, help="Test data repo root")
    args = parser.parse_args()

    intermediates_dir = Path(args.output) / "intermediates"

    print("Generating synthetic intermediates for skipped notebooks...")
    print()

    print("[1/3] Engine divergence predictions (Ch16/07)...")
    generate_engine_divergence_predictions(intermediates_dir)

    print("[2/3] Signal quality synthesis data (Ch20/02)...")
    generate_signal_quality_data(intermediates_dir)

    print("[3/3] MLOps registry and predictions (Ch26/03, Ch26/06)...")
    generate_mlops_data(intermediates_dir)

    print()
    print("Done! Now commit changes to the test-data repo and update overrides.yaml.")


if __name__ == "__main__":
    main()
