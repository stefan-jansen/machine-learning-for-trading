"""Fold preparation, done once and shared by every model family.

Preparing a fold is expensive and has nothing to do with the model that will be fitted on it.
Slicing the walk-forward window, replacing the model-based feature columns with that fold's own
values, dropping rows whose label is missing and subsampling the training rows depend only on the
case study, the label, the split set, the feature list and the sampling fraction. Every
configuration of every family that shares those five things wants exactly the same arrays.

Before this module each family carried its own copy of that procedure and each configuration ran
it again. On ``etfs`` the joined dataset loads in 1.9s and the eight folds take 10.1s to prepare;
the linear notebook alone repeated those 10.1s twenty-eight times, once per configuration, for
identical output. The families differed only in what they did to the arrays afterwards - linear
imputes and standardises to float64, gradient boosting casts to float32 and hands the missing
values to LightGBM - which is the cheap part.

So preparation splits in two. :func:`prepare_raw_folds` does the expensive, family-independent
half once and caches it. The adapters below - :func:`standardized_fold`, :func:`gbm_fold` - do the
cheap, family-specific half on top of it.

Numerics are pinned here rather than left to the caller. Arrays are made C-contiguous before any
reduction runs over them, because a sum over the same values in a different memory layout does not
give the same last bits: the polars and pandas paths of the previous implementation disagreed by
1.4e-11 in the standardised design matrix for that reason alone, which was enough to move a
data-derived hyperparameter and fork a training identity.

``FOLD_PREPARATION_VERSION`` is the declared behaviour of this module. Bump it when a change here
would change a fitted result. It enters the training identity in place of a hash of this file's
bytes, so refactoring, logging and comments do not invalidate registered results while a real
change to preparation does. ``tests/test_folds.py`` pins the digest of the prepared arrays and
fails when preparation changes, which is what makes the declared version enforceable rather than
a promise to remember.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

__all__ = [
    "FOLD_PREPARATION_VERSION",
    "PREPROCESSING_ID",
    "RawFold",
    "clear_fold_cache",
    "clear_memo",
    "fold_cache_key",
    "fold_set_bytes",
    "folds_built",
    "holds_in_memory",
    "gbm_fold",
    "prepare_gbm_folds_from_mds",
    "iter_raw_folds",
    "prepare_raw_folds",
    "prepare_standardized_folds",
    "split_frames",
    "standardized_fold",
    "training_labels_for_split",
]

# Declared behaviour of the preparation below. See the module docstring.
FOLD_PREPARATION_VERSION = 1

# Declared behaviour of :func:`standardized_fold`. Families that standardise record this string.
PREPROCESSING_ID = "median-imputer-standard-scaler/v1"

_CACHE_ENV = "ML4T_FOLD_CACHE"
_DISABLE_ENV = "ML4T_FOLD_CACHE_DISABLED"

# Prepared fold sets held for the life of the process, addressed by :func:`fold_cache_key`.
#
# This is what makes preparation cost once per notebook rather than once per configuration, and
# it does so for every caller rather than only the one that was restructured to share a load.
# Two entries: a notebook works through one label at a time, and holding more than the current
# and previous fold set would cost gigabytes to serve a lookup nothing performs.
_MEMO_LIMIT = 2
_RAW_MEMO: dict[str, list[RawFold]] = {}
_STANDARDIZED_MEMO: dict[str, list[dict[str, Any]]] = {}
_GBM_MEMO: dict[str, list[dict[str, Any]]] = {}

# A held fold set costs rows x features x 8 bytes per fold, which the case studies do not agree
# on within an order of magnitude: eight etfs folds are 0.9 GB, us_equities_panel's are 24.5 GB
# and nasdaq100_microstructure's 44.1 GB, against 125 GB of machine shared by up to three
# notebooks. So holding the set is a policy, not a default. Above this budget nothing is held and
# each fold is prepared, used and released in turn, which is what the runner's fold-major loop
# does anyway; below it the whole set is shared and no configuration rebuilds one.
_MEMO_BUDGET_ENV = "ML4T_FOLD_MEMO_BUDGET_BYTES"
_DEFAULT_MEMO_BUDGET = 8 * 1024**3


def _memoize(store: dict[str, Any], key: str, value: Any) -> Any:
    store[key] = value
    while len(store) > _MEMO_LIMIT:
        store.pop(next(iter(store)))
    return value


def clear_memo() -> None:
    """Drop every in-process fold set. For tests, and for freeing memory between labels."""
    _RAW_MEMO.clear()
    _STANDARDIZED_MEMO.clear()
    _GBM_MEMO.clear()


# Folds actually built from the dataset, as opposed to served from the memo or read from disk.
# A runner reports this so its diagnostics say how much preparation work a run did rather than
# how many times it asked for a fold, which are no longer the same number.
_BUILT = 0


def folds_built() -> int:
    """How many folds this process has prepared from the dataset."""
    return _BUILT


def memo_budget_bytes() -> int:
    override = os.environ.get(_MEMO_BUDGET_ENV)
    return int(override) if override else _DEFAULT_MEMO_BUDGET


def fold_set_bytes(mds: Any, splits: Sequence[dict[str, Any]]) -> int:
    """An upper bound on what holding this fold set would cost, before building it.

    Each fold's training rows are a subset of the dataset, so the full height per fold is an
    over-estimate - which is the safe direction for a memory decision, and cheap enough to make
    before the first fold exists.
    """
    height = getattr(getattr(mds, "dataset", None), "height", 0) or 0
    return int(height) * len(mds.feature_names) * 8 * max(1, len(splits))


def holds_in_memory(mds: Any, splits: Sequence[dict[str, Any]]) -> bool:
    """Whether a fold set this size may be held for reuse."""
    return fold_set_bytes(mds, splits) <= memo_budget_bytes()


# ---------------------------------------------------------------------------
# The prepared fold
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawFold:
    """One walk-forward fold, sliced and cleaned but not yet transformed for a model.

    ``X_train`` and ``X_val`` keep their missing values. Imputation belongs to the family that
    cannot handle them, not to preparation: LightGBM routes NaN down its own branch and would be
    given a median it never asked for.
    """

    fold: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    y_eval: np.ndarray | None
    meta: pl.DataFrame
    dates: np.ndarray
    entities: np.ndarray | None

    @property
    def n_train(self) -> int:
        return int(self.X_train.shape[0])

    @property
    def n_val(self) -> int:
        return int(self.X_val.shape[0])


# ---------------------------------------------------------------------------
# Cache addressing
# ---------------------------------------------------------------------------


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def fold_cache_key(
    *,
    case_study: str,
    label_col: str,
    eval_label_col: str | None,
    feature_names: Sequence[str],
    splits: Sequence[dict[str, Any]],
    input_lineage: dict[str, Any] | None,
    train_sample_frac: float,
    seed: int,
    design_dtype: str = "float64",
) -> str:
    """Address a prepared fold set by everything preparation actually depends on.

    The model, its family and its hyperparameters are deliberately absent: two configurations that
    agree on these inputs must share the arrays, which is the whole point.

    The design matrix's float type is part of the address. Without it a case study that switches
    precision reads back the cache written in the other one and fits at a precision it did not
    declare, which no error would report. It is written only when it is not ``float64``, so the
    eight case studies that declared no precision keep the addresses they already have;
    ``FOLD_PREPARATION_VERSION`` stays the one deliberate way to invalidate all of them.
    """
    payload = {
        "version": FOLD_PREPARATION_VERSION,
        "case_study": case_study,
        "label_col": label_col,
        "eval_label_col": eval_label_col,
        "feature_names": list(feature_names),
        "splits": [
            {
                "fold": int(split["fold"]),
                "train_start": str(split["train_start"]),
                "train_end": str(split["train_end"]),
                "val_start": str(split.get("val_start", split.get("test_start"))),
                "val_end": str(split.get("val_end", split.get("test_end"))),
            }
            for split in splits
        ],
        "input_lineage": input_lineage,
        "train_sample_frac": float(train_sample_frac),
        "seed": int(seed),
    }
    if design_dtype != "float64":
        payload["design_dtype"] = design_dtype
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:16]


def _fold_key(
    mds: Any,
    splits: Sequence[dict[str, Any]],
    *,
    train_sample_frac: float,
    seed: int,
) -> str:
    try:
        lineage = mds.input_lineage
    except (AttributeError, ValueError):
        # A dataset assembled by hand rather than loaded carries no lineage. It still addresses
        # a fold set within this process; it just cannot be told apart from another one built
        # the same way, so it never reaches disk.
        lineage = None
    return fold_cache_key(
        case_study=getattr(mds, "case_study_id", ""),
        label_col=mds.label_col,
        eval_label_col=mds.eval_label_col,
        feature_names=mds.feature_names,
        splits=splits,
        input_lineage=lineage,
        train_sample_frac=train_sample_frac,
        seed=seed,
        design_dtype=getattr(mds, "feature_dtype", "float64"),
    )


def _cache_root(case_study: str) -> Path | None:
    if os.environ.get(_DISABLE_ENV):
        return None
    override = os.environ.get(_CACHE_ENV)
    if override:
        return Path(override) / case_study
    from utils.paths import get_case_study_dir

    return get_case_study_dir(case_study) / "folds"


def clear_fold_cache(case_study: str) -> int:
    """Delete every cached fold set for *case_study*. Returns the number removed."""
    root = _cache_root(case_study)
    if root is None or not root.exists():
        return 0
    import shutil

    removed = 0
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            shutil.rmtree(entry)
            removed += 1
    return removed


# ---------------------------------------------------------------------------
# Cache persistence
# ---------------------------------------------------------------------------


def _fold_paths(directory: Path, fold_id: int) -> dict[str, Path]:
    stem = f"fold_{fold_id:03d}"
    return {
        "design": directory / f"{stem}.npz",
        "train": directory / f"{stem}_train.parquet",
        "val": directory / f"{stem}_val.parquet",
        "meta": directory / f"{stem}_meta.parquet",
    }


def _write_cache(directory: Path, key: str, folds: list[RawFold]) -> None:
    """Persist a prepared fold set, completed set or nothing.

    Only the design matrices go through ``npz``. Labels, timestamps and entity identifiers carry
    their own dtypes - an integral class, a timezone-aware timestamp, a string symbol - and a
    columnar format round-trips those exactly where a numpy object array cannot be loaded back
    at all without enabling pickle.
    """
    staging = _open_staging(directory)
    records = [_write_fold(staging, raw) for raw in folds]
    _close_staging(staging, directory, key, records)


def _open_staging(directory: Path) -> Path:
    """A clean partial directory to write a fold set into."""
    import shutil

    staging = directory.with_name(f".{directory.name}.partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return staging


def _write_fold(staging: Path, raw: RawFold) -> dict[str, int]:
    """Persist one fold and return only what the manifest needs to record about it.

    Returning the counts rather than the fold is what lets a streaming writer forget each fold
    as it goes: a manifest built from a list of folds pins the whole set in memory until the
    last one is written, which is the cost this exists to avoid.
    """
    paths = _fold_paths(staging, raw.fold)
    np.savez(paths["design"], X_train=raw.X_train, X_val=raw.X_val)
    pl.DataFrame({"y": raw.y_train}).write_parquet(paths["train"])
    val = {"y": raw.y_val, "date": raw.dates}
    if raw.y_eval is not None:
        val["y_eval"] = raw.y_eval
    if raw.entities is not None:
        val["entity"] = raw.entities
    pl.DataFrame(val).write_parquet(paths["val"])
    raw.meta.write_parquet(paths["meta"])
    return {"fold": int(raw.fold), "n_train": raw.n_train, "n_val": raw.n_val}


def _close_staging(staging: Path, directory: Path, key: str, records: list[dict[str, int]]) -> None:
    manifest = {
        "key": key,
        "version": FOLD_PREPARATION_VERSION,
        "folds": [record["fold"] for record in records],
        "n_train": {str(record["fold"]): record["n_train"] for record in records},
        "n_val": {str(record["fold"]): record["n_val"] for record in records},
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    # Rename last: a reader either sees a complete fold set or no directory at all.
    staging.rename(directory)


def _cached_fold_ids(directory: Path) -> list[int] | None:
    """The fold ids a complete cache entry holds, or ``None`` if it is unusable.

    Every path is checked before any fold is read, so a caller that reads them one at a time
    still gets all-or-nothing: a half-written entry is rejected up front rather than part way
    through, when some folds have already been handed out.
    """
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return None
    if manifest.get("version") != FOLD_PREPARATION_VERSION:
        return None
    fold_ids = [int(fold_id) for fold_id in manifest["folds"]]
    for fold_id in fold_ids:
        if not all(path.exists() for path in _fold_paths(directory, fold_id).values()):
            return None
    return fold_ids


def _read_fold(directory: Path, fold_id: int) -> RawFold:
    paths = _fold_paths(directory, fold_id)
    val = pl.read_parquet(paths["val"])
    with np.load(paths["design"], allow_pickle=False) as design:
        return RawFold(
            fold=fold_id,
            X_train=design["X_train"],
            y_train=pl.read_parquet(paths["train"])["y"].to_numpy(),
            X_val=design["X_val"],
            y_val=val["y"].to_numpy(),
            y_eval=val["y_eval"].to_numpy() if "y_eval" in val.columns else None,
            meta=pl.read_parquet(paths["meta"]),
            dates=val["date"].to_numpy(),
            entities=val["entity"].to_numpy() if "entity" in val.columns else None,
        )


def _read_cache(directory: Path) -> list[RawFold] | None:
    fold_ids = _cached_fold_ids(directory)
    if fold_ids is None:
        return None
    return [_read_fold(directory, fold_id) for fold_id in fold_ids]


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


def _boundary(dtype: pl.DataType, value: Any) -> pl.Expr:
    import datetime

    raw = str(value)
    if dtype == pl.Date:
        return pl.lit(datetime.date.fromisoformat(raw[:10]))
    if dtype.base_type() == pl.Datetime:
        try:
            parsed = datetime.datetime.fromisoformat(raw)
        except ValueError:
            parsed = datetime.datetime.fromisoformat(f"{raw[:10]}T00:00:00")
        return pl.lit(parsed).cast(dtype)
    return pl.lit(value).cast(dtype)


def _contiguous(array: np.ndarray, dtype: Any = np.float64) -> np.ndarray:
    """Fix dtype and memory layout so a reduction over these values is path-independent.

    numpy's pairwise summation walks the array in memory order, so the mean and standard
    deviation of the same values differ in their last bits between a C- and an F-contiguous
    copy. Pinning the layout here is what stops a downstream data-derived hyperparameter from
    depending on how the array happened to be built.
    """
    return np.ascontiguousarray(array, dtype=dtype)


def split_frames(mds: Any, split: dict[str, Any]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The training and validation rows of one fold, before anything is turned into an array.

    This is the single definition of which rows a fold contains. Anything that needs to know
    that - the design matrix, the labels, a hyperparameter derived from the labels - comes
    through here, because two implementations of it agree only by luck. The GBM runner used to
    derive Huber's threshold from its own pandas re-selection of the training rows, which matched
    this one until it did not, and gave one declared configuration two training identities.
    """
    dataset: pl.DataFrame = mds.dataset
    if not isinstance(dataset, pl.DataFrame):
        raise TypeError("fold preparation requires a polars dataset")

    fold_id = int(split["fold"])
    date_col = mds.date_col
    label_col = mds.label_col
    date_dtype = dataset.schema[date_col]
    val_start = split.get("val_start", split.get("test_start"))
    val_end = split.get("val_end", split.get("test_end"))

    train_df = dataset.filter(
        (pl.col(date_col) >= _boundary(date_dtype, split["train_start"]))
        & (pl.col(date_col) <= _boundary(date_dtype, split["train_end"]))
    )
    val_df = dataset.filter(
        (pl.col(date_col) >= _boundary(date_dtype, val_start))
        & (pl.col(date_col) <= _boundary(date_dtype, val_end))
    )
    if train_df.height == 0 or val_df.height == 0:
        raise ValueError(f"fold {fold_id} is empty (train={train_df.height}, val={val_df.height})")

    has_temporal = (
        mds.temporal_by_fold is not None and mds.temporal_keys and mds.temporal_feature_names
    )
    if has_temporal:
        from utils.modeling import fold_temporal_frame

        fold_temporal = fold_temporal_frame(
            mds.temporal_by_fold,
            fold_id,
            temporal_keys=mds.temporal_keys,
            schema=train_df.schema,
        )
        train_df = train_df.drop(mds.temporal_feature_names).join(
            fold_temporal, on=list(mds.temporal_keys), how="left"
        )
        val_df = val_df.drop(mds.temporal_feature_names).join(
            fold_temporal, on=list(mds.temporal_keys), how="left"
        )

    label_present = pl.col(label_col).is_not_null()
    if dataset.schema[label_col] in {pl.Float32, pl.Float64}:
        label_present = label_present & pl.col(label_col).is_not_nan()
    train_df = train_df.filter(label_present)
    val_df = val_df.filter(label_present)
    if train_df.height == 0 or val_df.height == 0:
        raise ValueError(
            f"fold {fold_id} has no labelled rows (train={train_df.height}, val={val_df.height})"
        )
    return train_df, val_df


def _subsample_index(n_rows: int, fold_id: int, train_sample_frac: float, seed: int) -> np.ndarray:
    """Which training rows a reduced run keeps. One definition, so every caller keeps the same."""
    keep = max(1, int(n_rows * train_sample_frac))
    rng = np.random.default_rng(seed + fold_id)
    return np.sort(rng.choice(n_rows, size=keep, replace=False))


def training_labels_for_split(
    mds: Any,
    split: dict[str, Any],
    *,
    train_sample_frac: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """The exact training labels one fold will be fitted on, without building its design matrix.

    For hyperparameters derived from the labels - Huber's threshold is the one in use - where
    materializing the features to read the labels would cost gigabytes for a standard deviation.
    """
    from utils.modeling import RANDOM_SEED

    if seed is None:
        seed = RANDOM_SEED
    train_df, _ = split_frames(mds, split)
    labels = np.ascontiguousarray(train_df[mds.label_col].to_numpy())
    if train_sample_frac < 1.0:
        labels = np.ascontiguousarray(
            labels[_subsample_index(labels.shape[0], int(split["fold"]), train_sample_frac, seed)]
        )
    return labels


def prepare_raw_folds(
    mds: Any,
    splits: Sequence[dict[str, Any]],
    *,
    train_sample_frac: float = 1.0,
    seed: int | None = None,
    use_cache: bool = True,
) -> list[RawFold]:
    """Slice, temporal-replace, clean and subsample every fold in *splits*.

    Family-independent and cached. Raises if any declared fold comes out empty: a silently
    missing fold changes what a result means and must never pass unnoticed.

    This is :func:`iter_raw_folds` collected into a list, and holds the whole set by
    definition. A caller that consumes folds one at a time and drops each as it goes should
    call the generator instead - that is what bounds the peak on a large panel.
    """
    from utils.modeling import RANDOM_SEED

    key = _fold_key(
        mds, splits, train_sample_frac=train_sample_frac, seed=RANDOM_SEED if seed is None else seed
    )
    if key in _RAW_MEMO:
        return _RAW_MEMO[key]
    folds = list(
        iter_raw_folds(
            mds, splits, train_sample_frac=train_sample_frac, seed=seed, use_cache=use_cache
        )
    )
    return _memoize(_RAW_MEMO, key, folds) if holds_in_memory(mds, splits) else folds


def iter_raw_folds(
    mds: Any,
    splits: Sequence[dict[str, Any]],
    *,
    train_sample_frac: float = 1.0,
    seed: int | None = None,
    use_cache: bool = True,
):
    """Yield each prepared fold as it is built, holding one at a time.

    The same preparation as :func:`prepare_raw_folds`, which is this collected into a list.

    Why it is a generator. Every family transforms a raw fold into a second full copy of the
    design matrix - standardised for the linear families, float32 for GBM - and both consumers
    already drop each raw fold as they cast it. That bounds what is held *afterwards* and does
    nothing about the peak, because the whole raw set was built before the first cast. Measured
    on us_equities_panel 07_gbm, 2026-08-18: a 20.10 GB dataset, a 31.03 GB float64 raw set and
    a 52.56 GB peak, settling to 35.56 GB once only the 15.52 GB float32 set remained. Feeding
    the consumer fold by fold removes the raw set from that peak.

    The disk cache is written the same way, one fold at a time into a staging directory that is
    renamed only after the manifest lands, so an interrupted run leaves no half-set a later run
    could read. Abandoning the generator early leaves that staging directory behind; the next
    write removes it.
    """
    from utils.modeling import ID_COLS, RANDOM_SEED

    if seed is None:
        seed = RANDOM_SEED
    if not 0.0 < train_sample_frac <= 1.0:
        raise ValueError("train_sample_frac must be in (0, 1]")

    key = _fold_key(mds, splits, train_sample_frac=train_sample_frac, seed=seed)
    if key in _RAW_MEMO:
        yield from _RAW_MEMO[key]
        return

    case_study = getattr(mds, "case_study_id", "")
    # No case study means no addressable location for the artefact and no way to invalidate it,
    # so such a dataset is prepared in memory only.
    directory = None
    if use_cache and case_study:
        root = _cache_root(case_study)
        directory = (root / key) if root is not None else None
    stale = False
    if directory is not None and directory.exists():
        cached_ids = _cached_fold_ids(directory)
        if cached_ids is not None and len(cached_ids) == len(splits):
            # Read one at a time for the same reason they are built one at a time.
            for fold_id in cached_ids:
                yield _read_fold(directory, fold_id)
            return
        # An entry written by an older layout, or left incomplete, is repaired rather than
        # left to make every future run miss the cache silently.
        stale = True

    dataset: pl.DataFrame = mds.dataset
    if not isinstance(dataset, pl.DataFrame):
        raise TypeError("fold preparation requires a polars dataset")

    feature_names = list(mds.feature_names)
    label_col = mds.label_col
    eval_label_col = mds.eval_label_col
    date_col = mds.date_col
    entity_col = mds.entity_cols[0] if mds.entity_cols else None
    date_dtype = dataset.schema[date_col]

    has_temporal = (
        mds.temporal_by_fold is not None and mds.temporal_keys and mds.temporal_feature_names
    )
    id_cols = [column for column in dataset.columns if column in ID_COLS]

    # The case study says what precision its design matrices are built in. Pinning float64
    # here regardless meant a case study that declares float32 paid for the wide form and then
    # cast back down: on nasdaq100_microstructure that is 2.9 GB per fold built to be halved.
    design_dtype = (
        np.float32 if getattr(mds, "feature_dtype", "float64") == "float32" else np.float64
    )

    staging = None
    records: list[dict[str, int]] = []
    if directory is not None and (stale or not directory.exists()):
        try:
            if stale:
                import shutil

                shutil.rmtree(directory)
            staging = _open_staging(directory)
        except OSError:
            # A cache that cannot be written is not a reason to fail a run.
            staging = None

    for split in splits:
        fold_id = int(split["fold"])
        train_df, val_df = split_frames(mds, split)

        # Only the design matrix has its dtype pinned. A classification label is integral and
        # coercing it to float would change what the family is asked to fit.
        X_train = _contiguous(train_df.select(feature_names).to_numpy(), design_dtype)
        X_val = _contiguous(val_df.select(feature_names).to_numpy(), design_dtype)
        y_train = np.ascontiguousarray(train_df[label_col].to_numpy())
        y_val = np.ascontiguousarray(val_df[label_col].to_numpy())
        y_eval = np.ascontiguousarray(val_df[eval_label_col].to_numpy()) if eval_label_col else None

        # Subsample training rows only. Validation is never sampled: out-of-sample IC is always
        # computed on the full slice, so the reduction changes cost and not what is measured.
        if train_sample_frac < 1.0:
            index = _subsample_index(X_train.shape[0], fold_id, train_sample_frac, seed)
            X_train = _contiguous(X_train[index], design_dtype)
            y_train = np.ascontiguousarray(y_train[index])

        global _BUILT
        _BUILT += 1
        raw = RawFold(
            fold=fold_id,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            y_eval=y_eval,
            meta=val_df.select(id_cols),
            dates=val_df[date_col].to_numpy(),
            entities=val_df[entity_col].to_numpy() if entity_col else None,
        )
        if staging is not None:
            try:
                records.append(_write_fold(staging, raw))
            except OSError:
                staging = None
        # Every local naming the fold's arrays is dropped before yielding, so a consumer that
        # holds nothing leaves one fold alive rather than two.
        del X_train, y_train, X_val, y_val, y_eval, train_df, val_df
        yield raw
        del raw

    if staging is not None and len(records) == len(splits):
        try:
            _close_staging(staging, directory, directory.name, records)
        except OSError:
            pass


def prepare_standardized_folds(
    mds: Any,
    splits: Sequence[dict[str, Any]],
    *,
    train_sample_frac: float = 1.0,
    seed: int | None = None,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Prepared folds, imputed and standardised, for the families that need that form.

    Held in process once per fold set. Every configuration of a standardising family sees the
    same arrays, which is correct as well as cheap: the transform depends on the training rows
    and not on the model, so computing it per configuration only invited them to disagree.
    """
    from utils.modeling import RANDOM_SEED

    key = _fold_key(
        mds, splits, train_sample_frac=train_sample_frac, seed=RANDOM_SEED if seed is None else seed
    )
    if key in _STANDARDIZED_MEMO:
        return _STANDARDIZED_MEMO[key]
    may_hold = holds_in_memory(mds, splits)
    # The standardised arrays are a second full copy of the design matrix, and on the large
    # panels one copy runs to tens of gigabytes. Taking the raw folds from the generator means
    # one is alive at a time on the way in as well as on the way out - `prepare_raw_folds` would
    # build the whole raw set first, and that set was the larger half of the peak.
    # The memo entry goes first, or it keeps the raw list alive through the loop.
    _RAW_MEMO.pop(key, None)
    standardized = [
        standardized_fold(raw)
        for raw in iter_raw_folds(
            mds, splits, train_sample_frac=train_sample_frac, seed=seed, use_cache=use_cache
        )
    ]
    return _memoize(_STANDARDIZED_MEMO, key, standardized) if may_hold else standardized


def prepare_gbm_folds_from_mds(
    mds: Any,
    splits: Sequence[dict[str, Any]],
    *,
    train_sample_frac: float = 1.0,
    seed: int | None = None,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Prepared folds cast to LightGBM's native precision, off the shared raw preparation.

    The expensive half - slicing, temporal replacement, cleaning, subsampling - is
    :func:`prepare_raw_folds`, the same one the standardising families use, so a case study pays
    for it once however many families it fits. The GBM-specific half is :func:`gbm_fold`, which
    is a cast.

    Held in process only when the whole set fits the memo budget, which is what bounds a
    fold-major run on a panel whose fold set runs to tens of gigabytes: there the memo declines
    and each fold is released as the next is built.
    """
    from utils.modeling import RANDOM_SEED

    key = _fold_key(
        mds, splits, train_sample_frac=train_sample_frac, seed=RANDOM_SEED if seed is None else seed
    )
    if key in _GBM_MEMO:
        return _GBM_MEMO[key]
    may_hold = holds_in_memory(mds, splits)
    task_type = getattr(mds, "task_type", "regression")
    class_values = getattr(mds, "class_values", None)
    # float32 is a second copy of the design matrix, so raw folds are taken one at a time and
    # dropped as they are cast. Taking them from the generator is what bounds the PEAK rather
    # than only what is held afterwards: `prepare_raw_folds` built every fold before the first
    # cast, so the whole raw set sat beside the growing cast set. Measured on us_equities_panel
    # 07_gbm, 2026-08-18, under the old arrangement: a 20.10 GB dataset, a 31.03 GB float64 raw
    # set, a 52.56 GB peak, settling to 35.56 GB once only the 15.52 GB float32 set remained.
    # The memo entry goes first, or it keeps the raw list alive through the loop.
    _RAW_MEMO.pop(key, None)
    cast = [
        gbm_fold(raw, task_type=task_type, class_values=class_values)
        for raw in iter_raw_folds(
            mds, splits, train_sample_frac=train_sample_frac, seed=seed, use_cache=use_cache
        )
    ]
    return _memoize(_GBM_MEMO, key, cast) if may_hold else cast


# ---------------------------------------------------------------------------
# Family adapters - the cheap half
# ---------------------------------------------------------------------------


def standardized_fold(raw: RawFold) -> dict[str, Any]:
    """Median-impute and standardise, for families that cannot take a missing value.

    Declared as :data:`PREPROCESSING_ID`. The imputer and the scaler are fitted on the training
    rows only and applied to validation, so no validation statistic reaches the fit.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    preprocessor = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    # The standardised matrix keeps the precision the raw fold was built in. Pinning float64
    # here upcast a single-precision design matrix straight back, so a case study that declared
    # float32 paid the precision loss and still carried the wide array through the fit.
    design_dtype = raw.X_train.dtype if raw.X_train.dtype == np.float32 else np.float64
    X_train = _contiguous(preprocessor.fit_transform(raw.X_train), design_dtype)
    X_val = _contiguous(preprocessor.transform(raw.X_val), design_dtype)
    # SimpleImputer drops a feature that is entirely missing across the training rows, so the
    # design matrix silently narrows while the recorded feature list still claims the full set.
    # A result fitted on different columns than it declares is worse than a failed run.
    if X_train.shape[1] != raw.X_train.shape[1]:
        raise ValueError(
            f"fold {raw.fold}: imputation dropped "
            f"{raw.X_train.shape[1] - X_train.shape[1]} feature column(s) that are entirely "
            "missing in this fold's training rows; the declared feature list no longer "
            "describes the design matrix"
        )
    # A column that is constant or entirely missing in training scales to a non-finite value.
    # Zero is the standardised mean, so substituting it keeps the column present and inert.
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "fold": raw.fold,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": raw.y_train,
        "y_val": raw.y_val,
        "y_eval": raw.y_eval,
        "meta_pl": raw.meta,
        "meta": None,
        "dates": raw.dates,
        "entities": raw.entities,
        "n_train": raw.n_train,
        "n_val": raw.n_val,
        "preprocessor": preprocessor,
    }


def gbm_fold(
    raw: RawFold,
    *,
    task_type: str = "regression",
    class_values: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Cast to LightGBM's native precision and leave missing values in place.

    No imputation and no scaling: a boosted tree splits on the ordering of a feature, which
    standardising does not change, and it routes a missing value down its own branch, which
    imputing a median would replace with a fabricated observation.

    Only the design matrix is cast. The labels stay float64, which is where this differs from the
    ``prepare_gbm_folds`` path it replaces: LightGBM converts a label to its own precision anyway,
    so the fit is identical either way (measured, squared error and huber alike, to zero
    difference), while ``y_eval`` is the target IC is measured against and the standardising
    families keep it float64. Casting it here would have made a GBM IC and a linear IC two
    slightly different measurements of the same quantity.
    """
    fold = {
        "fold": raw.fold,
        "X_train": np.ascontiguousarray(raw.X_train, dtype=np.float32),
        "X_val": np.ascontiguousarray(raw.X_val, dtype=np.float32),
        "y_train": raw.y_train,
        "y_val": raw.y_val,
        "y_eval": raw.y_eval,
        "meta_pl": raw.meta,
        "meta": None,
        "dates": raw.dates,
        "entities": raw.entities,
        "n_train": raw.n_train,
        "n_val": raw.n_val,
    }
    if task_type == "classification" and class_values:
        # LightGBM requires contiguous 0-indexed classes; the original values stay in y_*.
        lookup = {value: index for index, value in enumerate(class_values)}
        fold["y_train_lgb"] = np.array([lookup[value] for value in raw.y_train], dtype=np.int32)
        fold["y_val_lgb"] = np.array([lookup[value] for value in raw.y_val], dtype=np.int32)
    else:
        # Always present, so the caller passes fold["y_train_lgb"] to the booster without asking
        # what the task is. For a regression there is nothing to remap and it is the label itself.
        fold["y_train_lgb"] = raw.y_train
        fold["y_val_lgb"] = raw.y_val
    return fold
