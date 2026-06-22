"""Generate synthetic test data for currently-skipped notebooks.

Run once to enrich the test-data repo with minimal synthetic datasets
that allow the remaining skipped notebooks to execute their code paths.

Usage:
    uv run python tests/generate_skip_data.py --output ~/ml4t/test-data

This generates data for:
1. FNSPID news dataset (Ch10/07, Ch10/08)
2. SEC 10-Q MD&A text (Ch10/09)
3. ADV columns for Kyle lambda (Ch18/03)
4. Engine divergence predictions (Ch16/07)
5. Signal quality synthesis data (Ch20/02)
6. MLOps drift detection features (Ch26/02)
7. MLOps safe model rollout (Ch26/03)
8. MLOps MLflow registry (Ch26/06)
"""

import argparse
import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

np.random.seed(42)

SYMBOLS_ETF = ["SPY", "QQQ", "IWM", "TLT", "GLD", "XLF", "XLK", "XLE", "EFA", "VWO"]
SYMBOLS_EQ = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ"]


def generate_fnspid_news(data_dir: Path):
    """Generate synthetic FNSPID financial news data."""
    out = data_dir / "alternative" / "news" / "fnspid"
    out.mkdir(parents=True, exist_ok=True)

    headlines = [
        "{sym} reports strong quarterly earnings, beats estimates",
        "{sym} shares drop on weaker-than-expected revenue guidance",
        "{sym} announces major acquisition worth $2.5B",
        "Analysts upgrade {sym} citing improving margins",
        "{sym} CEO discusses expansion plans in earnings call",
        "Market volatility hits {sym} as sector rotates",
        "{sym} launches new product line targeting enterprise customers",
        "Institutional investors increase {sym} holdings in Q3",
        "{sym} faces regulatory scrutiny over data practices",
        "{sym} dividend increase signals management confidence",
    ]

    rows = []
    dates = pl.date_range(date(2022, 1, 3), date(2024, 12, 31), "1d", eager=True)
    for d in dates:
        # 2-5 news items per day
        n_items = np.random.randint(2, 6)
        for _ in range(n_items):
            sym = np.random.choice(SYMBOLS_EQ)
            headline = np.random.choice(headlines).format(sym=sym)
            rows.append(
                {
                    "ticker": sym,
                    "timestamp": d,
                    "title": headline,
                    "source": np.random.choice(["Reuters", "Bloomberg", "CNBC", "WSJ"]),
                }
            )

    df = pl.DataFrame(rows)
    df.write_parquet(out / "fnspid_sample.parquet")
    print(f"  FNSPID: {len(df)} news items -> {out / 'fnspid_sample.parquet'}")


def generate_sec_10q_mda(data_dir: Path):
    """Generate synthetic SEC 10-Q MD&A text data."""
    out = data_dir / "alternative" / "text"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for sym in SYMBOLS_EQ[:6]:
        for year in range(2019, 2024):
            for quarter in range(1, 5):
                month = quarter * 3 + 1
                if month > 12:
                    month = 1
                    year_f = year + 1
                else:
                    year_f = year
                filing_date = date(year_f, min(month, 12), 15)
                period_end = date(year, quarter * 3, 28)

                mda_text = (
                    f"Management's Discussion and Analysis for {sym}. "
                    f"During Q{quarter} {year}, revenue increased by {np.random.uniform(2, 15):.1f}% "
                    f"year-over-year. Operating margins improved to {np.random.uniform(15, 35):.1f}%. "
                    f"We continue to invest in R&D and expect continued growth. "
                    f"Key risks include market volatility and regulatory changes."
                )
                rows.append(
                    {
                        "symbol": sym,
                        "cik": str(np.random.randint(100000, 999999)),
                        "accession_no": f"0001234567-{year_f:04d}-{np.random.randint(10000, 99999):05d}",
                        "filing_date": filing_date,
                        "period_end": period_end,
                        "mda_text": mda_text,
                        "mda_word_count": len(mda_text.split()),
                        "mda_char_count": len(mda_text),
                    }
                )

    df = pl.DataFrame(rows)
    df.write_parquet(out / "sp500_10q_mda.parquet")
    print(f"  SEC 10-Q: {len(df)} filings -> {out / 'sp500_10q_mda.parquet'}")


def enrich_adv_columns(data_dir: Path):
    """Add adv_21d (21-day average daily volume) to datasets that need it.

    The Kyle lambda market impact calibration notebook (Ch18/03) reads
    adv_21d from equity price data. The test data doesn't have this computed.
    """
    datasets = [
        ("etfs", "etf_universe.parquet"),
        ("equities", "us_equities.parquet"),
    ]
    for subdir, filename in datasets:
        path = data_dir / subdir / filename
        if not path.exists():
            print(f"  ADV: SKIP {path} (not found)")
            continue
        df = pl.read_parquet(path)
        if "adv_21d" in df.columns:
            print(f"  ADV: SKIP {path} (already has adv_21d)")
            continue
        if "volume" not in df.columns:
            print(f"  ADV: SKIP {path} (no volume column)")
            continue

        # Compute rolling 21-day average volume per symbol
        sort_cols = ["symbol", "timestamp"] if "symbol" in df.columns else ["timestamp"]
        group_col = "symbol" if "symbol" in df.columns else None

        if group_col:
            df = df.sort(sort_cols).with_columns(
                pl.col("volume")
                .rolling_mean(window_size=21, min_samples=1)
                .over(group_col)
                .alias("adv_21d")
            )
        else:
            df = df.sort("timestamp").with_columns(
                pl.col("volume").rolling_mean(window_size=21, min_samples=1).alias("adv_21d")
            )

        df.write_parquet(path)
        print(f"  ADV: Added adv_21d to {path} ({len(df)} rows)")


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


def generate_mlops_data(intermediates_dir: Path, data_dir: Path):
    """Generate data for Ch26 MLOps notebooks (02, 03, 06)."""
    # Ch26/02 needs ETFs features with adv_21d — handled by enrich_adv_columns

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
        description="Generate synthetic test data for skipped notebooks"
    )
    parser.add_argument("--output", required=True, help="Test data repo root")
    args = parser.parse_args()

    root = Path(args.output)
    data_dir = root / "data"
    intermediates_dir = root / "intermediates"

    print("Generating synthetic test data for skipped notebooks...")
    print()

    print("[1/6] FNSPID news data (Ch10/07, Ch10/08)...")
    generate_fnspid_news(data_dir)

    print("[2/6] SEC 10-Q MD&A text (Ch10/09)...")
    generate_sec_10q_mda(data_dir)

    print("[3/6] ADV columns for Kyle lambda (Ch18/03)...")
    enrich_adv_columns(data_dir)

    print("[4/6] Engine divergence predictions (Ch16/07)...")
    generate_engine_divergence_predictions(intermediates_dir)

    print("[5/6] Signal quality synthesis data (Ch20/02)...")
    generate_signal_quality_data(intermediates_dir)

    print("[6/6] MLOps registry and predictions (Ch26/02-06)...")
    generate_mlops_data(intermediates_dir, data_dir)

    print()
    print("Done! Now commit changes to the test-data repo and update overrides.yaml.")


if __name__ == "__main__":
    main()
