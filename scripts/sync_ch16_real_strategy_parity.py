"""Build the Chapter 16 parity resource from accepted ml4t-backtest evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "16_strategy_simulation" / "resources" / "framework_parity_audit.json"
SOURCE_FILES = (
    "validation/REAL_STRATEGY_RESULTS.json",
    "validation/REAL_STRATEGY_PERFORMANCE.json",
    "validation/LARGE_SCALE_RESULTS.json",
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_digest(root: Path) -> str:
    git_root = Path(
        subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    relative_root = root.resolve().relative_to(git_root.resolve())
    tracked = subprocess.run(
        ["git", "-C", str(git_root), "ls-files", "-z", "--", relative_root.as_posix()],
        check=True,
        capture_output=True,
    ).stdout
    paths = sorted(
        git_root / value.decode("utf-8")
        for value in tracked.split(b"\0")
        if value and value.endswith(b".py")
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _check(checks: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [check for check in checks if check.get("name") == name]
    if len(matches) != 1 or matches[0].get("passed") is not True:
        raise ValueError(f"required large-scale check failed or is missing: {name}")
    return matches[0]


def _validate_sources(
    source_root: Path,
    correctness: dict[str, Any],
    performance: dict[str, Any],
    scale: dict[str, Any],
) -> None:
    scope = correctness.get("scope", {})
    records = correctness.get("records", [])
    required = [record for record in records if record.get("status") != "unsupported"]
    unsupported = [record for record in records if record.get("status") == "unsupported"]
    if scope.get("real_strategy_equivalence_gate_passed") is not True:
        raise ValueError("real-strategy equivalence evidence has not passed")
    if scope.get("required_pairs") != 12 or scope.get("unsupported_pairs") != 8:
        raise ValueError(
            "real-strategy evidence does not contain the 12 required and 8 unsupported pairs"
        )
    if len(required) != scope.get("required_pairs") or any(
        record.get("status") != "pass" for record in required
    ):
        raise ValueError("real-strategy required-pair matrix is incomplete or failed")
    if len(unsupported) != scope.get("unsupported_pairs"):
        raise ValueError("real-strategy unsupported-pair matrix is incomplete")
    for record in required:
        if record.get("negative_control", {}).get("detected") is not True:
            raise ValueError(
                f"negative control failed: {record['case_study']}/{record['framework']}"
            )
        surfaces = record.get("surfaces", {})
        if any(
            surfaces.get(name, {}).get("passed") is not True
            for name in ("fills", "equity", "terminal")
        ):
            raise ValueError(
                f"comparison surface failed: {record['case_study']}/{record['framework']}"
            )

    policy = correctness.get("comparison_policy", {})
    if policy.get("record_numeric_quantum") != "0.00000001":
        raise ValueError("real-strategy fill precision is not 1e-8")
    if policy.get("account_money_quantum") != "0.01":
        raise ValueError("real-strategy account precision is not one cent")
    if policy.get("rounding") != "ROUND_HALF_EVEN":
        raise ValueError("real-strategy monetary rounding policy differs")

    ml4t_provenance = correctness.get("provenance", {}).get("ml4t", {})
    if ml4t_provenance.get("dirty") is not False:
        raise ValueError("real-strategy evidence was generated from a dirty library tree")
    current_engine_digest = _tree_digest(source_root / "src" / "ml4t" / "backtest")
    if ml4t_provenance.get("engine_source_sha256") != current_engine_digest:
        raise ValueError("real-strategy evidence does not describe the current library engine")

    required_pairs = {(record["case_study"], record["framework"]) for record in required}
    measured_pairs = {
        (record["case_study"], record["framework"]) for record in performance.get("records", [])
    }
    if measured_pairs != required_pairs:
        raise ValueError("performance evidence does not cover the passing correctness matrix")
    if performance.get("correctness_evidence_generated_at") != correctness.get("generated_at"):
        raise ValueError("performance evidence refers to a different correctness run")
    if performance.get("provenance", {}).get("ml4t_engine_source_sha256") != current_engine_digest:
        raise ValueError("performance evidence does not describe the current library engine")
    if performance.get("timing_policy", {}).get("boundary") != "engine call only":
        raise ValueError("performance evidence does not use the engine-call timing boundary")
    if performance.get("timing_policy", {}).get("measured_processes") != 10:
        raise ValueError("performance evidence does not contain ten measured processes per side")
    for record in performance["records"]:
        if record.get("correctness_status") != "pass":
            raise ValueError("performance evidence includes a pair that did not pass correctness")
        for runner in ("framework_engine", "ml4t_engine"):
            samples = record.get(runner, {}).get("samples_seconds", [])
            if len(samples) != performance["timing_policy"]["measured_processes"]:
                raise ValueError(
                    f"wrong sample count for {record['case_study']}/{record['framework']}"
                )

    if scale.get("release_gate_passed") is not True:
        raise ValueError("large-scale evidence has not passed")
    if any(
        record.get("comparison", {}).get("passed") is not True
        for record in scale.get("frameworks", [])
    ):
        raise ValueError("large-scale framework matrix contains a failure")
    for record in scale.get("frameworks", []):
        comparison = record.get("comparison", {})
        if comparison.get("canonical_record_quantum") != "1E-8":
            raise ValueError("large-scale fill precision is not 1e-8")
        if comparison.get("canonical_money_quantum") != "0.01":
            raise ValueError("large-scale account precision is not one cent")


def build_snapshot(source_root: Path) -> dict[str, Any]:
    """Return a compact Chapter 16 snapshot from accepted library artifacts."""
    paths = {relative: source_root / relative for relative in SOURCE_FILES}
    correctness, performance, scale = (_load(paths[relative]) for relative in SOURCE_FILES)
    _validate_sources(source_root, correctness, performance, scale)

    current_library_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    required = [record for record in correctness["records"] if record["status"] != "unsupported"]
    unsupported = [record for record in correctness["records"] if record["status"] == "unsupported"]

    real_rows = []
    for record in required:
        equity = record["surfaces"]["equity"]
        fills = record["surfaces"]["fills"]
        terminal = record["surfaces"]["terminal"]
        real_rows.append(
            {
                "case_study": record["case_study"],
                "framework": record["framework"],
                "status": record["status"],
                "profile": record["profile"],
                "input_bundle_sha256": record["input_bundle_sha256"],
                "negative_control_detected": record["negative_control"]["detected"],
                "valuation_timestamps_match": equity["coverage_passed"],
                "fills": fills["framework_records"],
                "valuations": equity["coverage"]["shared_timestamps"],
                "fill_gap": fills["max_canonical_difference"],
                "equity_gap": equity["max_canonical_difference"],
                "equity_raw_gap": equity["max_raw_difference"],
                "terminal_gap": terminal["max_canonical_difference"],
                "terminal_raw_gap": terminal["max_raw_difference"],
            }
        )

    timing_rows = []
    for record in performance["records"]:
        timing_rows.append(
            {
                "case_study": record["case_study"],
                "framework": record["framework"],
                "framework_median_seconds": record["framework_engine"]["median_seconds"],
                "framework_ci95_seconds": record["framework_engine"]["ci_95_seconds"],
                "ml4t_median_seconds": record["ml4t_engine"]["median_seconds"],
                "ml4t_ci95_seconds": record["ml4t_engine"]["ci_95_seconds"],
                "framework_to_ml4t_ratio": record["framework_to_ml4t_median_ratio"],
            }
        )

    scale_rows = []
    for record in scale["frameworks"]:
        checks = record["comparison"]["checks"]
        scale_rows.append(
            {
                "framework": record["framework"],
                "intents": _check(checks, "order_intents")["expected_count"],
                "fills": _check(checks, "fills")["expected_count"],
                "trades": _check(checks, "trades")["expected_count"],
                "terminal_value": _check(checks, "final_value")["canonical_expected"],
                "status": "pass",
            }
        )

    source_artifacts = [
        {"path": relative, "sha256": _sha256(path)} for relative, path in paths.items()
    ]
    return {
        "schema_version": 2,
        "audit_generated_at": max(
            correctness["generated_at"], performance["generated_at"], scale["generated_at"]
        ),
        "library_commit": current_library_commit,
        "engine_commit": correctness["provenance"]["ml4t"]["commit"],
        "engine_source_sha256": correctness["provenance"]["ml4t"]["engine_source_sha256"],
        "evidence": {
            "correctness": "https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json",
            "performance": "https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_PERFORMANCE.json",
            "synthetic_stress": "https://github.com/ml4t/backtest/blob/main/validation/LARGE_SCALE_RESULTS.json",
            "source_artifacts": source_artifacts,
        },
        "comparison_policy": correctness["comparison_policy"],
        "protocol_scope": {
            "shared_inputs": "frozen model-derived targets and historical market data",
            "disabled_on_both_sides": ["transaction costs", "position rules"],
            "tested": [
                "target sizing",
                "order sequencing",
                "fills",
                "cash and margin",
                "funding where applicable",
                "valuation",
            ],
        },
        "scope": correctness["scope"],
        "frameworks": correctness["provenance"]["frameworks"],
        "real_strategy_records": real_rows,
        "unsupported_records": unsupported,
        "performance_policy": performance["timing_policy"],
        "performance_records": timing_rows,
        "synthetic_stress": {
            "generated_at": scale["generated_at"],
            "assets": scale["workload"]["recipe"]["assets"],
            "sessions": scale["workload"]["recipe"]["bars"],
            "bars": scale["workload"]["data_points"],
            "synthetic": True,
            "comparison_policy": {
                "record_numeric_quantum": str(
                    scale["frameworks"][0]["comparison"]["canonical_record_quantum"]
                ),
                "account_money_quantum": str(
                    scale["frameworks"][0]["comparison"]["canonical_money_quantum"]
                ),
            },
            "records": scale_rows,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path.home() / "ml4t" / "libraries" / "ml4t-backtest",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = build_snapshot(args.source_root.expanduser().resolve())
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != rendered:
            print(f"stale Chapter 16 parity resource: {OUTPUT}")
            return 1
        print(f"Chapter 16 parity resource is current: {OUTPUT}")
        return 0
    OUTPUT.write_text(rendered, encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
