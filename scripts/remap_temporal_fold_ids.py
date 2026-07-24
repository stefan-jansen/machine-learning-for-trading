"""Remap temporal-artifact fold metadata without changing feature values."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl
import yaml

from utils.artifact_specs import resolve_label_buffer
from utils.cv_splits import generate_cv_splits
from utils.modeling import ID_COLS, validate_temporal_fold_coverage


def _parse_mapping(items: list[str]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for item in items:
        try:
            source, target = (int(part) for part in item.split(":", maxsplit=1))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid fold mapping {item!r}; expected SOURCE:TARGET"
            ) from exc
        if source < 0 or target < 0:
            raise argparse.ArgumentTypeError("Negative holdout fold IDs cannot be remapped")
        if source in mapping:
            raise argparse.ArgumentTypeError(f"Duplicate source fold {source}")
        mapping[source] = target
    if len(set(mapping.values())) != len(mapping):
        raise argparse.ArgumentTypeError("Fold mapping targets must be unique")
    return mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value_hash(frame: pl.DataFrame) -> str:
    row_hashes = frame.drop("fold").hash_rows(seed=0).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def remap_artifact(
    *,
    case_study_id: str,
    source: Path,
    output: Path,
    mapping: dict[int, int],
) -> dict[str, object]:
    """Write a validated copy whose only changed column is ``fold``."""
    if source.resolve() == output.resolve():
        raise ValueError("Output must differ from the source artifact")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    case_dir = source.parent.parent
    setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())
    primary_label = setup["labels"]["primary"]
    label_buffer = resolve_label_buffer(case_study_id, primary_label, setup)
    if not label_buffer:
        raise ValueError(f"No label buffer configured for {primary_label!r}")

    financial = pl.read_parquet(case_dir / "features" / "financial.parquet")
    labels = pl.read_parquet(case_dir / "labels" / f"{primary_label}.parquet")
    join_keys = sorted(set(financial.columns) & set(labels.columns) & ID_COLS)
    dataset = financial.join(labels, on=join_keys, how="inner")
    date_col = "timestamp" if "timestamp" in dataset.columns else "date"
    splits = generate_cv_splits(
        dataset,
        case_study_id=case_study_id,
        label_buffer=label_buffer,
        date_col=date_col,
    )

    artifact = pl.read_parquet(source)
    source_folds = set(artifact.filter(pl.col("fold") >= 0)["fold"].unique().to_list())
    target_folds = {int(split["fold"]) for split in splits}
    if set(mapping) != source_folds or set(mapping.values()) != target_folds:
        raise ValueError(
            "Mapping must be a bijection from every nonnegative artifact fold "
            f"{sorted(source_folds)} to canonical folds {sorted(target_folds)}"
        )

    fold_expr = pl.col("fold")
    for source_fold, target_fold in mapping.items():
        fold_expr = (
            pl.when(pl.col("fold") == source_fold).then(pl.lit(target_fold)).otherwise(fold_expr)
        )
    remapped = artifact.with_columns(fold_expr.cast(artifact.schema["fold"]).alias("fold"))
    validate_temporal_fold_coverage(dataset, remapped, splits, date_col=date_col)

    value_hash = _value_hash(artifact)
    if artifact.height != remapped.height or value_hash != _value_hash(remapped):
        raise AssertionError("Remap changed artifact rows or non-fold values")

    output.parent.mkdir(parents=True, exist_ok=True)
    remapped.write_parquet(output)
    written = pl.read_parquet(output)
    if written.height != artifact.height or _value_hash(written) != value_hash:
        raise AssertionError("Written artifact failed the row/value preservation audit")

    return {
        "source": str(source),
        "output": str(output),
        "rows": artifact.height,
        "mapping": {str(key): value for key, value in sorted(mapping.items())},
        "source_sha256": _sha256(source),
        "output_sha256": _sha256(output),
        "non_fold_value_sha256": value_hash,
        "coverage": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-study", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--map", nargs="+", required=True, metavar="SOURCE:TARGET")
    args = parser.parse_args()
    mapping = _parse_mapping(args.map)
    audit = remap_artifact(
        case_study_id=args.case_study,
        source=args.source,
        output=args.output,
        mapping=mapping,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
