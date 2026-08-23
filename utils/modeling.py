"""Shared modeling infrastructure for Ch11+ notebooks.

Provides:
- load_modeling_dataset(): Load features + temporal + labels, join, detect schema
- load_configs(): Load model configs for a label from label YAML + presets
- prepare_cv_folds(): Preprocess data into train/val folds (impute, scale)
- ModelingDataset: Container for joined data with detected schema

The cross-sectional IC itself is the library's: call
``ml4t.diagnostic.metrics.cross_sectional_ic`` against a polars frame of
(date, symbol, y_true, y_pred) directly. ``cross_sectional_ic_mean()`` below is
the adapter for callers holding aligned numpy arrays instead, which is what a
scikit-learn or Optuna objective has in hand inside a fold.

Usage:
    from utils.modeling import load_modeling_dataset, load_configs, prepare_cv_folds

    mds = load_modeling_dataset("etfs", "fwd_ret_21d")
    configs = load_configs("etfs", "fwd_ret_21d", family="linear")
    folds = prepare_cv_folds(mds.dataset.to_pandas(), mds.splits, ...)
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
import yaml

from utils.artifact_specs import (
    load_feature_spec,
    load_label_spec,
    resolve_label_buffer,
    resolve_label_horizon,
    resolve_market_semantics,
    resolve_storage_path,
)
from utils.cv_splits import earliest_train_start, generate_cv_splits, make_wf_config

RANDOM_SEED = 42
MIN_TEMPORAL_DATE_COVERAGE = 0.95  # Allow short calendar-edge gaps, not missing windows.
# Burn-in a temporal model cannot emit through, excused only at the start of a
# train window. Measured on crypto_perps_funding, whose GARCH and HMM features
# carry a 90-bar rolling z-score: 97/2024 dates (4.8%) for fold 0 and 119/1935
# (6.1%) for fold 1. A stale artifact whose fold IDs have shifted presents as a
# leading gap of roughly half the window, so this bound still rejects it.
MAX_TEMPORAL_WARMUP_FRACTION = 0.10


def cross_sectional_ic_mean(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: np.ndarray,
    entities: np.ndarray,
    *,
    min_obs: int = 10,
) -> float:
    """Mean cross-sectional Spearman IC over the dates where it is defined.

    The library computes the per-date series from two polars frames. Model
    evaluation reaches it holding four aligned numpy arrays instead - what a
    scikit-learn or Optuna objective has in hand inside a fold - so this adapts
    the one to the other. It is the only thing this function does: the
    correlation itself is ``ml4t.diagnostic``'s.

    A date carries no coefficient when fewer than ``min_obs`` entities are
    priced, or when every prediction or every return ties. ml4t-diagnostic 0.1.2
    and later report those as null; 0.1.1 returns NaN, which ``drop_nulls`` does
    not remove and which turns the mean of the whole series into NaN. Both are
    dropped here, so the value does not depend on which of the two is installed.

    Returns NaN when no date has a defined coefficient - a real answer to "what
    was the average IC", unlike 0.0, which reads as "measured, and it was zero".
    """
    # Local import: `ml4t.diagnostic` brings scikit-learn's OpenMP runtime up
    # transitively, and this module is imported by notebooks that must load a
    # gradient-boosting library first. See .github/scripts/check_openmp_import_order.py.
    from ml4t.diagnostic.metrics import cross_sectional_ic_series

    ic_per_date = cross_sectional_ic_series(
        pl.DataFrame({"timestamp": dates, "symbol": entities, "prediction": y_pred}),
        pl.DataFrame({"timestamp": dates, "symbol": entities, "forward_return": y_true}),
        pred_col="prediction",
        ret_col="forward_return",
        date_col="timestamp",
        entity_col="symbol",
        method="spearman",
        min_obs=min_obs,
    )
    defined = ic_per_date.drop_nulls("ic").filter(pl.col("ic").is_finite())
    return float(defined["ic"].mean()) if defined.height else float("nan")


def seed_everything(seed: int = RANDOM_SEED) -> None:
    """Set all random seeds for full reproducibility (CPU + GPU).

    Must be called before any stochastic operation. For per-fold
    reproducibility, call again at the start of each fold with
    ``seed_everything(RANDOM_SEED + fold_id)``.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # torch not needed for sklearn/numpy-only models (PCA, IPCA)


from utils.paths import get_case_study_dir

# Columns that are identifiers, not features
ID_COLS = {
    "date",
    "timestamp",
    "asset",
    "symbol",
    "stock_id",
    "product",
    "position",
    "instrument_id",
}

# Meta columns that may leak or are composites (not independent features)
META_LEAK = {
    "underlying_price",
    "instr_mid",
    "instr_bid",
    "instr_ask",
    "ls_signal",
    "risk_adj_score",
}


@dataclass
class ModelingDataset:
    """Container for a fully-joined modeling dataset with detected schema."""

    dataset: pl.DataFrame
    feature_names: list[str]
    label_col: str
    date_col: str
    entity_cols: list[str]
    join_cols: list[str]
    splits: list[dict[str, Any]]
    label_buffer: str
    # Which case study this was loaded from. Callers that persist or cache anything derived from
    # the dataset need it, and it is not otherwise recoverable from the frame.
    case_study_id: str = ""
    cv_config: Any = None  # WalkForwardConfig (optional, avoids hard import dep)
    task_type: str = "regression"  # "regression" or "classification"
    num_classes: int = 0  # 0 for regression, 2+ for classification
    class_values: list = field(default_factory=list)  # sorted unique values for classification
    # Per-fold temporal features, carrying a 'fold' column. Held lazily where the loader
    # can: on us_equities_panel this artifact is 68.7M rows, and materialising it cost
    # 8.7 GB for the whole run when every consumer wants one fold. Reach it through
    # ``fold_temporal_frame``, never by indexing it directly.
    temporal_by_fold: pl.LazyFrame | pl.DataFrame | pd.DataFrame | None = None
    temporal_keys: list[str] = field(default_factory=list)  # Join keys for temporal features
    temporal_feature_names: list[str] = field(default_factory=list)  # Temporal feature column names
    temporal_artifact_splits: list[dict[str, Any]] = field(default_factory=list)
    # The float type the design matrices are built in. Read from the case study's
    # ``features.storage_dtype``; fold preparation honours it rather than pinning float64.
    feature_dtype: str = "float64"
    # Continuous-return label that classification predictions are scored against.
    # None for regression labels. When set, the column lives in ``dataset`` and
    # downstream IC computation must use it instead of the binary ``label_col``.
    eval_label_col: str | None = None
    # Inputs ``input_lineage`` is derived from: the artifact paths and the
    # universe reduction, which are not otherwise recoverable from this object.
    lineage_inputs: dict[str, Any] = field(default_factory=dict, repr=False)
    _input_lineage: dict[str, Any] | None = field(default=None, init=False, repr=False)

    @property
    def input_lineage(self) -> dict[str, Any]:
        """Identity-defining input lineage, computed on first use.

        A training spec that persists results includes this payload so changed
        artifacts or CV windows cannot reuse an old training hash. It digests
        every input artifact, and those run to gigabytes - nasdaq100's feature
        parquet alone is 7 GB and takes ~12 s to hash - while only the notebooks
        that pass it to ``build_training_spec`` need it. Computing it here rather
        than in ``load_modeling_dataset`` keeps that cost off every other caller.
        """
        if self._input_lineage is None:
            if not self.lineage_inputs:
                raise ValueError(
                    "input_lineage is unavailable: this ModelingDataset was built without "
                    "lineage_inputs. Construct it via load_modeling_dataset()."
                )
            self._input_lineage = build_modeling_input_lineage(
                artifacts=self.lineage_inputs["artifacts"],
                feature_names=self.feature_names,
                splits=self.splits,
                label_buffer=self.label_buffer,
                task_type=self.task_type,
                eval_label_col=self.eval_label_col,
                max_symbols=self.lineage_inputs["max_symbols"],
                symbols=self.lineage_inputs["symbols"],
                feature_dtype=self.feature_dtype,
            )
        return self._input_lineage


def _sha256_file(path: Path) -> str:
    """Return a stable digest for an identity-defining input artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_modeling_input_lineage(
    *,
    artifacts: dict[str, Path],
    feature_names: list[str],
    splits: list[dict[str, Any]],
    label_buffer: str,
    task_type: str,
    eval_label_col: str | None,
    max_symbols: int,
    symbols: list[str] | None,
    feature_dtype: str = "float64",
) -> dict[str, Any]:
    """Build the portable input identity carried by persisted training runs.

    ``feature_dtype`` is part of the identity because the artifacts are not. The parquet files
    a case study reads are unchanged by a precision declaration, so without this a result fitted
    in double precision and one fitted in single resolve to the same training identity, and the
    registry serves the older one for a spec that asked for the other.

    It is written into the payload only when it is not ``float64``. A key added unconditionally
    changes the fingerprint of every case study, including the eight that declared nothing, and
    would invalidate every training run already registered against them. Omitting the default
    keeps those fingerprints exactly as they were and gives only the declaring case study a new
    one. Any future default must be added the same way, for the same reason.
    """
    split_fields = ("fold", "train_start", "train_end", "val_start", "val_end")

    def _normalize(key: str, value: Any) -> str:
        # str() on a pd.Timestamp renders "2019-01-07 00:00:00" or "...+00:00"
        # depending on whether the caller's boundaries are tz-aware, so the same
        # window read two ways would fingerprint differently.
        if key == "fold":
            return str(value)
        return pd.Timestamp(value).tz_localize(None).isoformat()

    normalized_splits = [
        {key: _normalize(key, split[key]) for key in split_fields if split.get(key) is not None}
        for split in splits
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifacts": {
            name: {"sha256": _sha256_file(path), "size": path.stat().st_size}
            for name, path in sorted(artifacts.items())
        },
        "feature_names": list(feature_names),
        "splits": normalized_splits,
        "label_buffer": label_buffer,
        "task_type": task_type,
        "eval_label_col": eval_label_col,
        "max_symbols": int(max_symbols),
        "symbols": sorted(symbols) if symbols else None,
    }
    if feature_dtype != "float64":
        payload["feature_dtype"] = feature_dtype
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


# ---------------------------------------------------------------------------
# CV / Protocol Configuration
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardConfig:
    """Config for walk-forward cross-validation.

    Compatible with ml4t.diagnostic.splitters.WalkForwardCV.
    """

    n_splits: int
    train_size: str
    test_size: str
    embargo_td: str
    label_horizon: str
    timestamp_col: str = "timestamp"
    calendar_id: str | None = None
    test_start: str | None = None
    test_end: str | None = None

    def model_dump(self) -> dict:
        """Return fields as a dict (Pydantic-compatible API)."""
        from dataclasses import asdict

        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        """Serialize config to a JSON file (Pydantic-compatible API)."""
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> WalkForwardConfig:
        """Deserialize config from a JSON file."""
        import json

        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def load_protocol(case_study_id: str) -> dict:
    """Load evaluation protocol from config/setup.yaml.

    Returns dict with n_splits, train_size, test_size, holdout,
    leakage_guards, calendar.
    """
    path = get_case_study_dir(case_study_id) / "config" / "setup.yaml"
    if not path.exists():
        msg = f"Setup not found: {path}"
        raise FileNotFoundError(msg)

    with open(path) as f:
        setup = yaml.safe_load(f)

    ev = setup.get("evaluation", {})
    labels = setup.get("labels", {})
    market_semantics = resolve_market_semantics(case_study_id, setup)

    protocol = {
        "n_splits": ev.get("n_splits", 5),
        "train_size": ev.get("train_size", "5Y"),
        "test_size": ev.get("val_size", "1Y"),
        "step_size": ev.get("step_size", "1Y"),
        "calendar": market_semantics.get("calendar") or ev.get("calendar", "NYSE"),
        "holdout": {
            "start": ev.get("holdout_start"),
            "end": ev.get("holdout_end"),
        },
        "leakage_guards": {},
    }

    # Parse label horizon from buffer string (e.g. "21D", "8h", "15T", "60m", "1M")
    buffer = labels.get("buffer", "21D")
    if buffer.endswith("D"):
        protocol["leakage_guards"]["label_horizon_days"] = int(buffer[:-1])
    elif buffer.endswith(("h", "H")):
        protocol["leakage_guards"]["label_horizon_hours"] = int(buffer[:-1])
    elif buffer.endswith(("T",)) or (buffer.endswith("m") and not buffer.endswith("M")):
        protocol["leakage_guards"]["label_horizon_minutes"] = int(buffer[:-1])
    elif buffer.endswith("M"):
        # Monthly: approximate as 30 calendar days.
        # pd.Timedelta rejects 'M' as ambiguous; ml4t-diagnostic needs a library fix
        # to support calendar-month durations natively (see ml4t-diagnostic-dev/bugs/).
        protocol["leakage_guards"]["label_horizon_days"] = int(buffer[:-1]) * 30

    return protocol


def _label_horizon_to_iso(leakage_guards: dict) -> str:
    """Convert label horizon to ISO 8601 duration string."""
    if "label_horizon_days" in leakage_guards:
        return f"P{leakage_guards['label_horizon_days']}D"
    if "label_horizon_hours" in leakage_guards:
        return f"PT{leakage_guards['label_horizon_hours']}H"
    if "label_horizon_minutes" in leakage_guards:
        return f"PT{leakage_guards['label_horizon_minutes']}M"
    return "P0D"


def get_cv_config(case_study_id: str) -> WalkForwardConfig:
    """Load walk-forward CV config from case study setup.yaml.

    This is the single entry point for CV configuration. Reads
    from ``case_studies/{id}/config/setup.yaml``.
    """
    protocol = load_protocol(case_study_id)
    embargo = _label_horizon_to_iso(protocol.get("leakage_guards", {}))
    holdout = protocol.get("holdout", {})

    return WalkForwardConfig(
        n_splits=protocol["n_splits"],
        train_size=protocol["train_size"],
        test_size=protocol["test_size"],
        embargo_td=embargo,
        label_horizon=embargo,
        calendar_id=protocol.get("calendar", "NYSE"),
        test_start=str(holdout["start"]) if holdout.get("start") else None,
        test_end=str(holdout["end"]) if holdout.get("end") else None,
    )


# Classification label prefixes
_CLASSIFICATION_PREFIXES = ("fwd_dir_", "fwd_class_", "fwd_tb_", "fwd_carry_")


def detect_label_type(label_col: str, label_series: pl.Series) -> tuple[str, int, list]:
    """Detect regression vs classification from label name and values.

    Returns (task_type, num_classes, class_values).
    - task_type: "regression" or "classification"
    - num_classes: 0 for regression, 2+ for classification
    - class_values: sorted unique values for classification, [] for regression
    """
    is_classification = any(label_col.startswith(p) for p in _CLASSIFICATION_PREFIXES)

    # Also classify if integer dtype with few unique values
    if not is_classification and label_series.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
        if label_series.drop_nulls().n_unique() <= 10:
            is_classification = True

    if not is_classification:
        return "regression", 0, []

    unique_vals = sorted(label_series.drop_nulls().unique().to_list())
    return "classification", len(unique_vals), unique_vals


def get_classification_eval_label(case_study_id: str, label: str) -> str:
    """Return the continuous return that a classification label is derived from.

    Classification predictions (probabilities or class-expected-values) must be
    scored against the underlying continuous return, not the binary/categorical
    label itself: ``Spearman(score, fwd_continuous_return)`` is the proper IC,
    while ``Spearman(score, binary_label)`` collapses to ``2·(AUC − 0.5)``.

    The mapping is declared per-case-study under
    ``labels.classification_eval_label`` in ``config/setup.yaml``. There is no
    runtime inference — every classification label must be registered
    explicitly.

    Parameters
    ----------
    case_study_id : str
        Case study identifier (e.g., ``"us_firm_characteristics"``).
    label : str
        Classification label name (e.g., ``"fwd_class_1m"``).

    Returns
    -------
    str
        Continuous-return label name (e.g., ``"fwd_ret_1m"``).

    Raises
    ------
    KeyError
        If ``labels.classification_eval_label[label]`` is missing from
        ``setup.yaml``.
    """
    from utils import CASE_STUDIES_DIR

    setup_path = CASE_STUDIES_DIR / case_study_id / "config" / "setup.yaml"
    setup = yaml.safe_load(setup_path.read_text())
    mapping = (setup.get("labels") or {}).get("classification_eval_label") or {}
    if label not in mapping:
        raise KeyError(
            f"labels.classification_eval_label[{label!r}] not declared in "
            f"case_studies/{case_study_id}/config/setup.yaml. Add an entry mapping "
            f"the binary/categorical label to its source continuous-return label "
            f"(e.g., {label}: fwd_ret_1m). IC for classification predictions is "
            f"computed against the continuous return — there is no silent fallback."
        )
    return str(mapping[label])


def get_direction_labels(case_study_id: str, label: str) -> tuple[str, ...]:
    """Return the classification labels cut from a given continuous-return label.

    ``labels.classification_eval_label`` read backwards. It declares which continuous return
    each classification label was derived from; inverting it says which direction labels exist
    for a return, which is what scoring a regression model by AUC needs. No separate
    declaration is added for the inverse - one mapping cannot disagree with itself.

    Returns an empty tuple when the case study declares no classification label for this
    return, which is the case for five of the nine.
    """
    from utils import CASE_STUDIES_DIR

    setup_path = CASE_STUDIES_DIR / case_study_id / "config" / "setup.yaml"
    setup = yaml.safe_load(setup_path.read_text())
    mapping = (setup.get("labels") or {}).get("classification_eval_label") or {}
    return tuple(sorted(k for k, v in mapping.items() if v == label))


def verify_artifact_sidecars(
    artifacts: Mapping[str, Path],
    *,
    require_sidecar: bool = True,
    values: bool = True,
) -> dict[str, str]:
    """Check each input artifact against the digest sidecar its producer wrote.

    ``case_studies/utils/artifact_digest.py`` writes ``<artifact>.digest.json``
    beside every artifact, recording the content hash, the row count and the key
    columns. Until this function existed nothing read one: stage 02 and stage 03
    wrote them and the chain stopped, so an upstream value change could not reach
    a downstream identity. What it closes is narrow and worth stating exactly - the
    registry records feature-set *names* and no digest of feature *values*, so a
    model trained on corrected features and one trained on the leaky version it
    replaced produce the identical training_hash unless something reads the values.

    Raises on a digest that disagrees with the file. Whether a *missing* sidecar
    also raises is the caller's to choose, because the two carry different
    evidence. A disagreeing sidecar is a value that moved without its record
    moving with it, which is the defect itself, and it is never acceptable. A
    missing sidecar means the producer recorded nothing, which cannot be told
    apart from a silent change but is also the state of every artifact written
    before the sidecar existed.

    Returns the verified digest per artifact, so the caller can carry it into the
    training spec. Artifacts skipped under ``require_sidecar=False``, and every
    artifact under ``values=False``, are absent from the mapping rather than
    present with a null.

    ``values`` chooses what the check costs, and the two settings answer different
    questions. With it, the parquet is read and every row hashed, which is what the
    recorded content digest is - order-independent, so it moves when and only when
    a value moves. Without it, only the row count is compared, which parquet
    metadata answers without reading a column. The cheap check cannot see a value
    change that preserves the row count, and it does see the case that a crash
    creates: ``write_artifact`` writes the parquet before its sidecar, so an
    interrupted regeneration leaves new data beside the previous record, and new
    data almost always has a different number of rows.

    ``load_modeling_dataset`` runs the cheap check on every load and the full one
    under ``verify_input_digests``; ``scripts/verify_artifact_sidecars.py`` runs the
    full one over every case study, once, after a regeneration. That split is about
    cost: hashing us_equities_panel's feature matrix is a multi-GB read plus a sort
    over 68M row hashes.

    Whether a *missing* sidecar raises is the caller's to choose, because presence
    and disagreement carry different evidence. A missing sidecar means the producer
    recorded nothing, which is the state of every artifact written before the
    sidecar existed, and requiring presence before those are regenerated fails for
    absence rather than for a changed value. Measured rather than assumed: on
    2026-08-09 the production artifacts carried 49 sidecars out of 51, both gaps in
    sp500_options, while the CI fixtures under
    ``ml4t/third-edition-test-data/intermediates`` carried **none** - 0 of 7 for
    cme_futures, 0 of 5 for etfs, 0 of 6 for fx_pairs, 0 of 6 for us_equities_panel.
    """
    from case_studies.utils.artifact_digest import read_digest, sidecar_path, value_digest

    verified: dict[str, str] = {}
    problems: list[str] = []
    for name, path in sorted(artifacts.items()):
        side = sidecar_path(path)
        if not side.exists():
            if require_sidecar:
                problems.append(f"{name}: no digest sidecar beside {path.name}")
            continue
        try:
            record = read_digest(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            problems.append(f"{name}: sidecar unreadable ({exc})")
            continue
        recorded = record.get("digest")
        if not recorded:
            problems.append(f"{name}: sidecar records no digest")
            continue

        rows = record.get("n_rows")
        if not values:
            if rows is None:
                continue
            try:
                actual_rows = cast(
                    pl.DataFrame,
                    pl.scan_parquet(path).select(pl.len()).collect(),
                ).item()
            except (OSError, pl.exceptions.PolarsError) as exc:
                problems.append(f"{name}: {path.name} unreadable ({exc})")
                continue
            if int(rows) != int(actual_rows):
                problems.append(
                    f"{name}: {path.name} holds {actual_rows} rows, its sidecar records {rows}"
                )
            continue

        try:
            frame = pl.read_parquet(path)
        except (OSError, pl.exceptions.PolarsError) as exc:
            problems.append(f"{name}: {path.name} unreadable ({exc})")
            continue
        actual = value_digest(frame)
        if actual != recorded:
            problems.append(f"{name}: {path.name} hashes {actual}, its sidecar records {recorded}")
            continue
        if rows is not None and int(rows) != frame.height:
            problems.append(
                f"{name}: {path.name} holds {frame.height} rows, its sidecar records {rows}"
            )
            continue
        verified[name] = recorded

    if problems:
        raise ValueError(
            "input artifacts do not match the digests their producers recorded, so a "
            "training run built on them would carry an identity that does not describe "
            "its inputs: " + "; ".join(problems)
        )
    return verified


def feature_storage_dtype(case_study_id: str) -> pl.DataType:
    """The float type a case study stores and fits its feature matrices in.

    ``float64`` unless ``features.storage_dtype`` in the case study's ``setup.yaml`` says
    otherwise. Declared per case study because only the large panels need the narrower type:
    ``nasdaq100_microstructure`` is 16,098,877 rows x 88 features, which is 10.9 GB of modeling
    dataset in double precision against 5.6 GB in single, while the small case studies fit
    comfortably either way and narrowing them would move their numbers for no gain.
    """
    # Read from the repository, not through ``get_case_study_dir``, which redirects to
    # ``ML4T_OUTPUT_DIR``. Every CI job points that at a scratch directory holding no
    # ``config/``, so going through the redirect would resolve every case study to the default
    # and silently answer a question about what a case study declares with "nothing".
    from utils import CASE_STUDIES_DIR

    setup_path = CASE_STUDIES_DIR / case_study_id / "config" / "setup.yaml"
    if not setup_path.exists():
        return pl.Float64
    declared = (yaml.safe_load(setup_path.read_text()) or {}).get("features", {})
    name = (declared or {}).get("storage_dtype", "float64")
    if name not in {"float32", "float64"}:
        raise ValueError(
            f"{case_study_id}: features.storage_dtype must be float32 or float64, got {name!r}"
        )
    return pl.Float32 if name == "float32" else pl.Float64


def load_modeling_dataset(
    case_study_id: str,
    primary_label: str,
    max_symbols: int = 0,
    symbols: list[str] | None = None,
    verify_input_digests: bool = False,
) -> ModelingDataset:
    """Load and join features + temporal + labels for a case study.

    This is the canonical data loading function for ALL Ch11+ modeling
    notebooks. It handles schema detection (date vs timestamp, asset vs
    stock_id vs product), temporal join-key casting, and universe reduction.

    Parameters
    ----------
    case_study_id : str
        Case study identifier (e.g., "etfs", "crypto_perps_funding").
    primary_label : str
        Label file stem (e.g., "fwd_ret_21d").
    max_symbols : int, default 0
        Universe reduction for fast development. 0 = all symbols.
    symbols : list of str, optional
        Explicit symbol whitelist. When given, the universe is restricted to
        exactly these symbols (intersected with what is available) and
        ``max_symbols`` is ignored. Used by tests to pin a small universe that
        is guaranteed to exist in the reduced test-data (e.g. the Darts base
        return series), rather than the top-by-history selection ``max_symbols``
        makes — which can pick symbols absent from a sampled data set.

    Returns
    -------
    ModelingDataset
        Container with joined dataset, detected schema, and CV splits.
    """
    case_dir = get_case_study_dir(case_study_id)
    financial_spec = load_feature_spec(case_study_id, "financial")
    temporal_spec = load_feature_spec(case_study_id, "model_based")
    label_spec = load_label_spec(case_study_id, primary_label)

    # Check prerequisites exist before loading
    features_path = resolve_storage_path(
        case_study_id, financial_spec, "features/financial.parquet"
    )
    label_path = resolve_storage_path(case_study_id, label_spec, f"labels/{primary_label}.parquet")

    missing = []
    if not features_path.exists():
        missing.append(("features/financial.parquet", "03_financial_features"))
    if not label_path.exists():
        missing.append((f"labels/{primary_label}.parquet", "02_labels"))
    if not (case_dir / "config" / "setup.yaml").exists():
        missing.append(("config/setup.yaml", None))

    if missing:
        print(f"\n  Missing prerequisites for '{case_study_id}' modeling:\n")
        for path, producer in missing:
            if producer is None:
                print(f"    {path}  (canonical hand-curated file — ensure committed)")
            else:
                print(f"    {path}  (run {producer}.py first)")
        first_producer = next((p for _, p in missing if p is not None), None)
        if first_producer is not None:
            print("\n  Example:")
            print(f"    uv run python case_studies/{case_study_id}/{first_producer}.py\n")
        raise FileNotFoundError(
            f"Missing prerequisites for '{case_study_id}': " + ", ".join(p for p, _ in missing)
        )

    # Load artifacts. The declared storage type is applied in the scan, so a case study that
    # fits in single precision never materialises the double-precision form on the way.
    storage_dtype = feature_storage_dtype(case_study_id)

    def _read(path: Path) -> pl.DataFrame:
        frame = pl.scan_parquet(path)
        if storage_dtype != pl.Float64:
            narrow = [n for n, t in frame.collect_schema().items() if t == pl.Float64]
            if narrow:
                frame = frame.with_columns([pl.col(c).cast(storage_dtype) for c in narrow])
        return frame.collect()

    features = _read(features_path)

    temporal_path = resolve_storage_path(
        case_study_id, temporal_spec, "features/model_based.parquet"
    )
    # Scanned, not read. This artifact is per-fold, so it is a multiple of the feature table:
    # 68.7M rows on us_equities_panel against the panel's 9.9M. Every consumer wants one fold,
    # so the fold predicate is pushed into the scan instead of the whole thing being held.
    temporal = pl.scan_parquet(temporal_path) if temporal_path.exists() else None
    if temporal is not None and storage_dtype != pl.Float64:
        narrow = [n for n, t in temporal.collect_schema().items() if t == pl.Float64]
        if narrow:
            temporal = temporal.with_columns([pl.col(c).cast(storage_dtype) for c in narrow])
    temporal_columns = temporal.collect_schema().names() if temporal is not None else []

    # Labels are deliberately not narrowed. ``features.storage_dtype`` covers the design
    # matrix; the label is the target IC and every metric are measured against, and
    # ``gbm_fold`` states that it stays float64 whatever the design matrix is cast to.
    labels = pl.read_parquet(label_path)

    # Auto-detect label column (the non-ID column in the label file)
    label_col = [c for c in labels.columns if c not in ID_COLS][0]

    # Detect date column from features
    feature_keys = sorted(set(features.columns) & ID_COLS)
    date_col = "timestamp" if "timestamp" in feature_keys else "date"
    alt_date = "timestamp" if date_col == "date" else "date"

    # Normalize date column names across DataFrames
    if alt_date in labels.columns and date_col not in labels.columns:
        labels = labels.rename({alt_date: date_col})
    if temporal is not None and alt_date in temporal_columns and date_col not in temporal_columns:
        temporal = temporal.rename({alt_date: date_col})
        temporal_columns = [date_col if c == alt_date else c for c in temporal_columns]

    # Detect join columns
    label_keys = sorted(set(labels.columns) & ID_COLS)
    join_cols = sorted(set(feature_keys) & set(label_keys))
    entity_cols = [c for c in join_cols if c != date_col]

    # Filter out constant entity columns (e.g. instrument_id='straddle_30d_atm')
    # that break cross-sectional IC computation by collapsing all entities into one group.
    # NOTE: join_cols retains ALL shared ID columns for data integrity during joins;
    # entity_cols is filtered separately for IC computation only.
    entity_cols = [c for c in entity_cols if features[c].n_unique() > 1]

    # Sort by cardinality descending so the primary entity (most unique values)
    # comes first. Important when downstream code uses entity_cols[0] for IC
    # (e.g., CME futures: 'product' has 30 values vs 'position' has 3).
    entity_cols = sorted(entity_cols, key=lambda c: features[c].n_unique(), reverse=True)

    # Join features + temporal (left join to keep all feature rows)
    temporal_by_fold_pd = None
    _temporal_keys = []
    _temporal_feature_names = []

    if temporal is not None:
        temporal_schema = temporal.collect_schema()
        _temporal_keys = sorted(set(temporal_columns) & set(feature_keys))
        casts = {
            k: features.schema[k]
            for k in _temporal_keys
            if temporal_schema[k] != features.schema[k]
        }
        if casts:
            temporal = temporal.cast(casts)

        if "fold" in temporal_columns:
            # Per-fold temporal features. The dataset carries one fold as a placeholder so the
            # schema is complete; the per-fold values are substituted at fold preparation time
            # by ``fold_temporal_frame``, which reads the fold it is asked for and no more.
            _temporal_feature_names = [
                c for c in temporal_columns if c not in set(_temporal_keys) | {"fold"}
            ]
            fold_ids = sorted(temporal.select(pl.col("fold").unique()).collect()["fold"].to_list())
            placeholder_dedup = (
                temporal.filter(pl.col("fold") == fold_ids[0])
                .drop("fold")
                .unique(subset=_temporal_keys, keep="last")
                .collect()
            )
            dataset = features.join(placeholder_dedup, on=_temporal_keys, how="left", suffix="_t")
            del placeholder_dedup

            # Kept lazy. Materialising it here is what made a run hold every fold at once.
            temporal_by_fold_pd = temporal
        else:
            # Legacy: single feature set, join directly
            temporal_dedup = temporal.unique(subset=_temporal_keys, keep="last").collect()
            dataset = features.join(temporal_dedup, on=_temporal_keys, how="left", suffix="_t")
            del temporal_dedup
    else:
        dataset = features

    # Inner-join with labels (drops rows without labels)
    dataset = dataset.join(labels, on=join_cols, how="inner")

    # Drop any meta columns that leaked in
    drop_cols = [c for c in dataset.columns if c in META_LEAK]
    if drop_cols:
        dataset = dataset.drop(drop_cols)

    # Optional universe reduction
    if symbols and entity_cols:
        primary_entity = entity_cols[0]
        dataset = dataset.filter(pl.col(primary_entity).is_in(list(symbols)))
    elif max_symbols > 0 and entity_cols:
        dataset = reduce_to_top_entities(dataset, entity_cols[0], max_symbols)

    # Feature columns = everything except IDs and label
    feature_names = [c for c in dataset.columns if c not in ID_COLS and c != label_col]

    # CV splits — read buffer from setup.yaml (explicit, handles non-standard labels)
    setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())
    label_buffer = resolve_label_buffer(case_study_id, primary_label, setup)
    if not label_buffer:
        raise ValueError(
            f"No explicit label buffer found for '{primary_label}' in "
            f"case_studies/{case_study_id}/config/setup.yaml. "
            f"Add buffer to labels.buffer (primary) or labels.variant_buffers (variants)."
        )
    # Fold identity belongs to the label contract. Deriving boundaries from the
    # feature-joined frame lets warm-up nulls or feature availability shift the
    # calendar and makes model selection disagree with canonical_window().
    splits = generate_cv_splits(
        labels,
        case_study_id=case_study_id,
        label_buffer=label_buffer,
        outcome_horizon=resolve_label_horizon(case_study_id, primary_label, setup),
        date_col=date_col,
    )
    temporal_artifact_splits: list[dict[str, Any]] = []
    if temporal_by_fold_pd is not None:
        assert temporal is not None
        validate_temporal_fold_coverage(
            dataset,
            temporal,
            splits,
            date_col=date_col,
        )
        # The artifact carries one fold set, built on the primary label's geometry, and
        # the rows above were selected from it by ``fold`` id. So the values this load
        # returns for fold F were fit on data through the *primary* label's train_end,
        # and a label whose own validation opens earlier than that is scored on sessions
        # its features already saw. Coverage does not see this: an artifact can cover
        # every date a variant is scored on and still have been fit past the start of
        # that window. Checked here rather than in a notebook because every model
        # notebook reaches its data through this call, and only the label being loaded
        # is checked, so the cost is one label timeline rather than all of them.
        from case_studies.utils.cv_window import (
            assert_variant_folds_are_out_of_sample,
            configured_labels,
            temporal_artifact_fold_boundaries,
        )

        configured = configured_labels(case_study_id)
        artifact_label = configured[0] if configured else primary_label
        temporal_artifact_splits = temporal_artifact_fold_boundaries(
            case_study_id,
            artifact_label,
            temporal_path,
        )
        if configured and primary_label != configured[0]:
            assert_variant_folds_are_out_of_sample(
                case_study_id, configured[0], variants=[primary_label]
            )

    # WalkForwardConfig for library integration
    # Normalize month-based buffers to days (pd.Timedelta rejects 'M' as ambiguous)
    wf_horizon = label_buffer
    if wf_horizon and wf_horizon.endswith("M") and wf_horizon[:-1].isdigit():
        wf_horizon = f"{int(wf_horizon[:-1]) * 30}D"
    try:
        cv_config = make_wf_config(case_study_id, label_horizon=wf_horizon, date_col=date_col)
    except Exception as exc:
        warnings.warn(f"WalkForwardConfig creation failed for {case_study_id}: {exc}", stacklevel=2)
        cv_config = None

    # Detect label type (regression vs classification)
    task_type, num_classes, class_values = detect_label_type(label_col, dataset[label_col])

    # Classification labels: load the continuous-return label they were derived
    # from so IC can be computed against returns rather than the binary target.
    eval_label_col: str | None = None
    eval_label_path: Path | None = None
    if task_type == "classification":
        eval_label_col = get_classification_eval_label(case_study_id, label_col)
        eval_label_path = resolve_storage_path(
            case_study_id,
            load_label_spec(case_study_id, eval_label_col),
            f"labels/{eval_label_col}.parquet",
        )
        if not eval_label_path.exists():
            raise FileNotFoundError(
                f"Eval label parquet missing for classification label {label_col!r}: "
                f"expected {eval_label_path}. Generate it via 02_labels.py or update "
                f"labels.classification_eval_label[{label_col}] in setup.yaml."
            )
        eval_labels = pl.read_parquet(eval_label_path)
        # Normalize date column name if needed
        if alt_date in eval_labels.columns and date_col not in eval_labels.columns:
            eval_labels = eval_labels.rename({alt_date: date_col})
        # Inner-join eval column on the same join_cols (drops rows missing eval)
        eval_join_cols = sorted(set(eval_labels.columns) & set(dataset.columns) & ID_COLS)
        eval_labels = eval_labels.select([*eval_join_cols, eval_label_col])
        dataset = dataset.join(eval_labels, on=eval_join_cols, how="left")
        # Refresh feature_names so the eval label column is not used as a feature
        feature_names = [
            c for c in dataset.columns if c not in ID_COLS and c not in {label_col, eval_label_col}
        ]

    input_artifacts = {
        "financial": features_path,
        "label": label_path,
    }
    if temporal_path.exists():
        input_artifacts["model_based"] = temporal_path
    if eval_label_path is not None:
        input_artifacts["eval_label"] = eval_label_path
    # Two depths, and the flag chooses between them rather than between checking
    # and not. The row-count comparison reads parquet metadata and no column, so it
    # runs always: write_artifact writes the parquet before its sidecar, and an
    # interrupted regeneration therefore leaves new data beside the previous
    # record, which almost always has a different number of rows. Comparing the
    # content digest is what catches a value change that keeps the row count, and
    # that means hashing every row - a multi-GB read plus a sort over 68M row
    # hashes on us_equities_panel, paid per load. It is the same work
    # scripts/verify_artifact_sidecars.py does once over everything after a
    # regeneration, which is where a sweep should pay it.
    verify_artifact_sidecars(
        input_artifacts,
        require_sidecar=verify_input_digests,
        values=verify_input_digests,
    )
    return ModelingDataset(
        dataset=dataset,
        feature_names=feature_names,
        label_col=label_col,
        date_col=date_col,
        entity_cols=entity_cols,
        join_cols=join_cols,
        splits=splits,
        label_buffer=label_buffer,
        case_study_id=case_study_id,
        cv_config=cv_config,
        task_type=task_type,
        num_classes=num_classes,
        class_values=class_values,
        temporal_by_fold=temporal_by_fold_pd,
        temporal_keys=_temporal_keys,
        temporal_feature_names=_temporal_feature_names,
        temporal_artifact_splits=temporal_artifact_splits,
        feature_dtype="float32" if storage_dtype == pl.Float32 else "float64",
        eval_label_col=eval_label_col,
        lineage_inputs={
            "artifacts": input_artifacts,
            "max_symbols": max_symbols,
            "symbols": symbols,
        },
    )


def reduce_to_top_entities(
    dataset: pl.DataFrame,
    primary_entity: str,
    max_symbols: int,
) -> pl.DataFrame:
    """Keep the ``max_symbols`` entities with the most rows, ties broken by name.

    Row counts tie readily on these panels, and a tie broken by frame order is
    not stable across runs or across callers. The entity name is the secondary
    key so that every caller reducing the same dataset to the same size gets the
    same universe. Without it a reduced stage-04 run and the reduced model
    notebooks downstream of it can choose different equal-history symbols, and
    the ones only a single side chose carry null temporal features - a wrong
    answer that runs clean rather than a failure.

    Production runs set ``max_symbols=0`` and never reach this.
    """
    top = (
        dataset.group_by(primary_entity)
        .len()
        .sort(["len", primary_entity], descending=[True, False])
        .head(max_symbols)
    )
    # implode: is_in against a bare Series of the same dtype is deprecated in
    # polars as ambiguous, and membership in the value set is what is meant.
    return dataset.filter(pl.col(primary_entity).is_in(top[primary_entity].implode()))


def _inclusive_end_of(boundary: str) -> pd.Timestamp:
    """Widen a configured end *date* to cover every bar of that session.

    ``setup.yaml`` configures ``holdout_end`` as a date. Parsed, it is that date
    at midnight, and every fold filter in this module is ``timestamp <= val_end``.
    On a daily panel the bar already sits at midnight and the two agree. On an
    intraday panel every bar of the final session sorts *after* midnight, so the
    whole session is excluded from every modeling path: nasdaq100_microstructure
    writes 2021-12-31 into its holdout fold - 38,610 rows added when the producer
    side was fixed - and no consumer scores one of them.

    A boundary that already carries a time of day is meant literally and is
    returned unchanged, so a config can still name an instant inside a session -
    including an explicit midnight. That is why this reads the configured *string*
    rather than the parsed Timestamp: "2021-12-31" and "2021-12-31 00:00:00" parse
    to the same instant, and only the first one means the whole day.
    """
    parsed = pd.Timestamp(boundary)
    names_a_time = any(sep in str(boundary) for sep in (" ", "T"))
    if names_a_time or parsed != parsed.normalize():
        return parsed
    return parsed + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)


def append_holdout_fold_if_needed(
    mds: ModelingDataset,
    prediction_split: str,
    case_study_id: str,
) -> None:
    """Append a holdout fold to ``mds.splits`` when ``prediction_split=='holdout'``.

    The holdout fold trains on everything available before ``holdout_start`` and
    validates on ``[holdout_start, holdout_end]``. It becomes fold N+1, so downstream
    code iterating ``mds.splits`` produces one holdout prediction set per
    (training run, config) pair without any other change to the training loop.

    "Everything available" is ``min(train_start)`` across the CV folds, not
    ``splits[0]["train_start"]``. ``generate_cv_splits`` steps backward from the holdout
    boundary, so fold 0 is the most *recent* fold and carries the *latest* training start.
    Measured on etfs: ``splits[0]`` starts 2008-01-02 where the earliest fold starts
    2005-01-03, so indexing the list built a holdout retrain that silently discarded three
    years. ``case_studies/etfs/04_model_based_features.py`` says so in its CV Fold Setup
    prose - "Indexing the list hands it the shortest window of the set, silently" - and
    this function cited that notebook while doing the thing it warns against.

    Idempotent — if the trailing fold already covers the holdout window
    (val_end matches setup.yaml's holdout_end), no fold is appended.

    Use case: nasdaq100 v4 winner-only holdout regeneration in Session 4
    of the v4 sweep program. The val sweep keeps ``prediction_split=
    'validation'`` (no holdout fold); the single retrain at the end uses
    ``prediction_split='holdout'``.
    """
    if prediction_split != "holdout":
        return
    if not mds.splits:
        msg = (
            "ModelingDataset.splits is empty; cannot derive train_start for "
            "the holdout fold. Verify that load_modeling_dataset produced "
            "at least one CV fold for this case study."
        )
        raise RuntimeError(msg)
    path = get_case_study_dir(case_study_id) / "config" / "setup.yaml"
    with open(path) as f:
        setup = yaml.safe_load(f)
    ev = setup.get("evaluation", {})
    holdout_start = str(ev.get("holdout_start", "")).strip()
    holdout_end = str(ev.get("holdout_end", "")).strip()
    if not holdout_start or not holdout_end:
        msg = (
            f"setup.yaml::evaluation.holdout_start/holdout_end missing for "
            f"case_studies/{case_study_id}; cannot append holdout fold."
        )
        raise KeyError(msg)
    # Use pd.Timestamp boundaries to match generate_cv_splits (which stores
    # train/val_start/end as Timestamps). Mixing raw strings with Timestamps
    # made the idempotency check below never match (str(Timestamp) != YAML
    # string) and risked a tz-naive/aware comparison on the pandas filter path.
    ho_start_ts = pd.Timestamp(holdout_start)
    ho_end_ts = _inclusive_end_of(holdout_end)
    # Any fold covering the holdout window, not just the trailing one: the CV
    # folds run newest first and only the appended holdout fold lands at the end,
    # so reading one position is a second place the ordering has to be right.
    already_covered = any(
        s.get("val_end") is not None
        and pd.Timestamp(s["val_end"]) == ho_end_ts
        and pd.Timestamp(s["val_start"]) == ho_start_ts
        for s in mds.splits
    )
    if already_covered:
        return
    holdout_fold = {
        "fold": len(mds.splits),
        "train_start": earliest_train_start(mds.splits),
        "train_end": ho_start_ts,
        "val_start": ho_start_ts,
        "val_end": ho_end_ts,
    }
    mds.splits.append(holdout_fold)
    # input_lineage digests the fold set and memoizes on first access, so a caller
    # that read it before this append would otherwise persist a spec whose lineage
    # describes a fold set that no longer exists.
    mds._input_lineage = None  # noqa: SLF001 — same module, and the cache is this call's to invalidate


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """Raised when model config loading or instantiation fails."""


# Model type directory → family mapping
_MODEL_TYPE_TO_FAMILY: dict[str, str] = {
    "ols": "linear",
    "ridge": "linear",
    "lasso": "linear",
    "elastic_net": "linear",
    "logistic": "linear",
    "lgb": "gbm",
    "lstm": "deep_learning",
    "tcn": "deep_learning",
    "tsmixer": "deep_learning",
    "nlinear": "deep_learning",
    "nbeats": "deep_learning",
    "patchtst": "deep_learning",
    "tabm": "tabular_dl",
    "pca": "latent_factors",
    "ipca": "latent_factors",
    "cae": "latent_factors",
    "sae": "latent_factors",
    "sdf": "latent_factors",
    "dml": "causal_dml",
}

# Model type directory → library mapping
_MODEL_TYPE_TO_LIBRARY: dict[str, str] = {
    "ols": "sklearn",
    "ridge": "sklearn",
    "lasso": "sklearn",
    "elastic_net": "sklearn",
    "logistic": "sklearn",
    "lgb": "lightgbm",
    "lstm": "pytorch",
    "tcn": "pytorch",
    "tsmixer": "pytorch",
    "nlinear": "pytorch",
    "nbeats": "darts",
    "patchtst": "pytorch",
    "tabm": "tabm",
    "pca": "sklearn",
    "ipca": "sklearn",
    "cae": "pytorch",
    "sae": "pytorch",
    "sdf": "pytorch",
    "dml": "causal_ml",
}

# Config names that override the directory-default library
_CONFIG_LIBRARY_OVERRIDES: dict[str, str] = {
    "tabpfn": "tabpfn",
    "tsmixer": "darts",
}


def _enrich_config(preset: dict, preset_path: Path) -> dict:
    """Inject config_name, family, library from the file path.

    These were previously stored in the YAML but are derivable from the
    file system location: ``config/{model_type}/{config_name}.yaml``.
    """
    config_name = preset_path.stem
    model_type = preset_path.parent.name
    preset["config_name"] = config_name
    preset.setdefault("family", _MODEL_TYPE_TO_FAMILY.get(model_type, model_type))
    preset.setdefault(
        "library",
        _CONFIG_LIBRARY_OVERRIDES.get(config_name, _MODEL_TYPE_TO_LIBRARY.get(model_type, "")),
    )
    return preset


def _find_preset(config_root: Path, name: str) -> Path | None:
    """Search all config/{model_type}/ subdirs for a preset YAML."""
    matches = list(config_root.glob(f"*/{name}.yaml"))
    return matches[0] if matches else None


def load_configs(
    case_study_id: str,
    label: str,
    family: str,
) -> list[dict[str, Any]]:
    """Load model configurations for a label and family.

    Reads the training menu file (e.g., ``config/training/fwd_ret_21d.yaml``) to
    get the list of config names for the given family, then loads each
    referenced preset from ``case_studies/config/{model_type}/{config_name}.yaml``.

    Parameters
    ----------
    case_study_id : str
        Case study identifier (e.g., "etfs").
    label : str
        Label name (e.g., "fwd_ret_21d"). Must match a YAML file in
        ``config/training/``.
    family : str
        Model family (e.g., "linear", "gbm", "deep_learning").

    Returns
    -------
    list[dict]
        Each dict has keys: config_name, family, library, model_class, params.

    Raises
    ------
    ConfigError
        If the training menu file or a referenced preset is missing.
    """
    case_dir = get_case_study_dir(case_study_id)
    label_config_path = case_dir / "config" / "training" / f"{label}.yaml"

    if not label_config_path.exists():
        raise ConfigError(
            f"No training config file found: {label_config_path}\n"
            f"Create it with the config names you want to run.\n"
            f"See case_studies/config/ for available presets."
        )

    label_config = yaml.safe_load(label_config_path.read_text())
    config_names = label_config.get(family, [])

    if not config_names:
        raise ConfigError(
            f"No '{family}' configs listed in {label_config_path}\n"
            f"Add config names under the '{family}:' key."
        )

    config_root = case_dir.parent / "config"
    configs = []
    for name in config_names:
        preset_path = _find_preset(config_root, name)
        if preset_path is None:
            raise ConfigError(
                f"Preset not found: {name}.yaml in {config_root}/*/\n"
                f"Referenced in {label_config_path} under '{family}'."
            )
        preset = yaml.safe_load(preset_path.read_text())
        _enrich_config(preset, preset_path)
        configs.append(preset)

    return configs


def resolve_linear_params(
    cfg: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, Any]:
    """Resolve a linear preset's constructor params for one training fold.

    Most presets carry an absolute ``alpha`` and are returned unchanged. L1
    presets (Lasso, ElasticNet) may instead specify ``alpha_frac`` — a fraction
    of the data-derived degeneracy boundary ``alpha_max`` (the smallest alpha
    that drives every coefficient to zero). Because ``alpha_max`` is computed
    per fold from the standardized design, ``alpha_frac`` auto-calibrates the
    regularization grid to each case study and fold, so a fraction < 1 always
    yields at least one non-zero coefficient. This avoids hardcoding absolute
    alphas that land above ``alpha_max`` for low-signal case studies (which
    would silently drop the entire L1 family as degenerate).

    For Lasso the degeneracy boundary is ``max|Xᵀ(y − ȳ)| / n``; for ElasticNet
    it is that quantity divided by ``l1_ratio`` (only the L1 term gates entry).

    Parameters
    ----------
    cfg : dict
        Preset config with a ``params`` dict (e.g. from :func:`load_configs`).
    X_train, y_train : np.ndarray
        Standardized training design and target for the current fold.

    Returns
    -------
    dict
        Constructor params with ``alpha`` resolved and ``alpha_frac`` removed.
    """
    params = dict(cfg["params"])
    frac = params.pop("alpha_frac", None)
    if frac is None:
        return params

    n = X_train.shape[0]
    residual = y_train - y_train.mean()
    alpha_max = float(np.max(np.abs(X_train.T @ residual))) / n
    l1_ratio = params.get("l1_ratio", 1.0)
    params["alpha"] = frac * alpha_max / l1_ratio
    return params


# ---------------------------------------------------------------------------
# CV fold preparation
# ---------------------------------------------------------------------------


def validate_temporal_fold_coverage(
    dataset: pl.DataFrame | pd.DataFrame,
    temporal_by_fold: pl.DataFrame | pd.DataFrame,
    splits: list[dict[str, Any]],
    *,
    date_col: str,
    min_date_coverage: float = MIN_TEMPORAL_DATE_COVERAGE,
    max_warmup_fraction: float = MAX_TEMPORAL_WARMUP_FRACTION,
) -> None:
    """Fail when a fold-specific temporal artifact does not cover a CV window.

    Coverage is measured on unique decision timestamps rather than entity keys.
    Temporal model fitting may legitimately skip individual symbols, whose
    missing values are imputed downstream, but a missing date range indicates
    that artifact fold IDs or windows do not match the canonical CV splits.

    A temporal model cannot emit a value before it has been fitted, so the start
    of a *training* window carries a burn-in prefix that no regeneration removes.
    A leading run of uncovered dates in a train window is therefore excused up to
    ``max_warmup_fraction`` of that window and coverage is measured on the
    remainder. Validation windows get no such allowance: every date a model is
    scored on must carry a temporal value. The excused prefix is bounded because
    a stale artifact whose fold IDs have shifted also presents as a leading gap,
    one that spans a large share of the window rather than a burn-in.
    """
    if not 0 < min_date_coverage <= 1:
        raise ValueError("min_date_coverage must be in (0, 1]")
    if not 0 <= max_warmup_fraction < 1:
        raise ValueError("max_warmup_fraction must be in [0, 1)")

    if isinstance(dataset, pl.DataFrame):
        dataset_dates = dataset.select(date_col).unique()[date_col].to_pandas()
    else:
        dataset_dates = dataset[date_col].drop_duplicates()
    temporal_pd = temporal_fold_index(temporal_by_fold, date_col).to_pandas()

    dataset_index = pd.DatetimeIndex(pd.to_datetime(dataset_dates, utc=True)).unique().sort_values()
    temporal_pd = temporal_pd.copy()
    temporal_pd[date_col] = pd.to_datetime(temporal_pd[date_col], utc=True)
    available_folds = set(temporal_pd["fold"].dropna().astype(int).unique())
    failures: list[str] = []

    for split in splits:
        fold_id = int(split["fold"])
        if fold_id not in available_folds:
            failures.append(f"fold {fold_id}: artifact fold is missing")
            continue

        fold_dates = pd.DatetimeIndex(
            temporal_pd.loc[temporal_pd["fold"] == fold_id, date_col].unique()
        )
        for window, start_key, end_key in (
            ("train", "train_start", "train_end"),
            ("validation", "val_start", "val_end"),
        ):
            start = pd.Timestamp(split[start_key])
            end = pd.Timestamp(split[end_key])
            start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
            end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
            expected = dataset_index[(dataset_index >= start) & (dataset_index <= end)]
            if len(expected) == 0:
                failures.append(f"fold {fold_id} {window}: dataset window is empty")
                continue
            present = expected.isin(fold_dates)
            warmup = 0
            if window == "train" and present.any() and not present[0]:
                leading = int(present.argmax())
                if leading <= max_warmup_fraction * len(expected):
                    warmup = leading
            scored = present[warmup:]
            covered = int(scored.sum())
            coverage = covered / len(scored)
            if coverage < min_date_coverage:
                excused = f", excusing a {warmup}-date warm-up prefix" if warmup else ""
                failures.append(
                    f"fold {fold_id} {window}: temporal date coverage "
                    f"{covered}/{len(scored)} ({coverage:.1%}){excused}"
                )

    if failures:
        detail = "; ".join(failures)
        raise ValueError(
            "Fold-specific temporal artifact is not aligned with the canonical CV splits: "
            f"{detail}. Regenerate the artifact with generate_cv_splits or migrate its fold IDs."
        )


def validate_temporal_split_geometry(
    requested_splits: list[dict[str, Any]],
    canonical_splits: list[dict[str, Any]],
    temporal_by_fold: pl.DataFrame | pd.DataFrame | None,
) -> None:
    """Require requests to use the fold geometry that fitted temporal features."""
    if temporal_by_fold is None:
        return
    fields = ("train_start", "train_end", "val_start", "val_end")
    canonical = {int(split["fold"]): split for split in canonical_splits}
    failures: list[str] = []
    for requested in requested_splits:
        fold = int(requested["fold"])
        fitted = canonical.get(fold)
        if fitted is None:
            failures.append(f"fold {fold} has no fitted temporal artifact geometry")
            continue
        mismatches = [
            field
            for field in fields
            if pd.Timestamp(requested[field]).tz_localize(None)
            != pd.Timestamp(fitted[field]).tz_localize(None)
        ]
        if mismatches:
            failures.append(f"fold {fold} differs in {mismatches}")
    if failures:
        raise ValueError(
            "Custom CV cannot reuse fold-specific temporal features fitted on different "
            f"boundaries: {'; '.join(failures)}. Use canonical folds or regenerate the "
            "temporal artifact for the requested geometry."
        )


def fold_temporal_frame(
    temporal_by_fold: Any,
    fold_id: int,
    *,
    temporal_keys: Sequence[str] | None = None,
    schema: Any = None,
) -> pl.DataFrame:
    """The temporal feature rows of one fold, as polars, whatever form the artifact is held in.

    One definition, because selecting a fold out of the temporal artifact was written four
    separate times - twice against pandas and twice against polars - and copies of a selection
    rule agree only by luck. It is also the only place that decides how much of the artifact is
    read: given a :class:`polars.LazyFrame` the fold predicate reaches the scan, so a run holds
    one fold's rows rather than every fold's.

    ``temporal_keys`` deduplicates, keeping the last row per key as the artifact writer intends.
    ``schema`` is the frame the result will be joined to; the join keys are cast to match it,
    because an artifact written with a different integer width joins to nothing and reports
    itself as missing values rather than as an error.
    """
    if temporal_by_fold is None:
        raise ValueError("no temporal artifact to select a fold from")

    if isinstance(temporal_by_fold, pl.LazyFrame):
        frame = temporal_by_fold.filter(pl.col("fold") == fold_id).drop("fold").collect()
    elif isinstance(temporal_by_fold, pl.DataFrame):
        frame = temporal_by_fold.filter(pl.col("fold") == fold_id).drop("fold")
    else:  # pandas, for artifacts and tests that still build one eagerly
        frame = pl.from_pandas(
            temporal_by_fold[temporal_by_fold["fold"] == fold_id].drop(columns=["fold"])
        )

    if temporal_keys:
        frame = frame.unique(subset=list(temporal_keys), keep="last")
    if schema is not None:
        casts = {
            key: schema[key]
            for key in (temporal_keys or [])
            if key in frame.columns and key in schema and frame.schema[key] != schema[key]
        }
        if casts:
            frame = frame.cast(casts)
    return frame


def temporal_fold_index(temporal_by_fold: Any, date_col: str) -> pl.DataFrame:
    """The distinct ``(date, fold)`` pairs an artifact carries, without reading its features.

    What every coverage check needs, and all it needs. Against a lazy artifact the projection
    reaches the scan, so this costs two columns rather than the whole table.
    """
    columns = [date_col, "fold"]
    if isinstance(temporal_by_fold, pl.LazyFrame):
        return temporal_by_fold.select(columns).unique().collect()
    if isinstance(temporal_by_fold, pl.DataFrame):
        return temporal_by_fold.select(columns).unique()
    return pl.from_pandas(temporal_by_fold[columns].drop_duplicates())


def replace_temporal_columns(
    dataset_pd: pd.DataFrame,
    mask: np.ndarray,
    temporal_by_fold: pd.DataFrame,
    temporal_keys: list[str],
    temporal_feature_names: list[str],
    fold_id: int,
) -> pd.DataFrame:
    """Replace temporal feature columns in a dataset slice with fold-specific values.

    Returns a copy of the masked rows with temporal columns overwritten.
    """
    rows = dataset_pd.loc[mask].copy()
    fold_temp = fold_temporal_frame(
        temporal_by_fold, fold_id, temporal_keys=temporal_keys
    ).to_pandas()

    # Drop old temporal columns and merge fold-specific ones
    rows = rows.drop(columns=temporal_feature_names, errors="ignore")
    rows = rows.merge(fold_temp, on=temporal_keys, how="left")
    return rows


def prepare_cv_folds(
    dataset_pd: pd.DataFrame,
    splits: list[dict[str, Any]],
    feature_names: list[str],
    label_col: str,
    date_col: str,
    entity_col: str | None,
    temporal_by_fold: pd.DataFrame | None = None,
    temporal_keys: list[str] | None = None,
    temporal_feature_names: list[str] | None = None,
    train_sample_frac: float = 1.0,
    eval_label_col: str | None = None,
) -> list[dict[str, Any]]:
    """Preprocess data into train/val folds with imputation and scaling.

    For each CV split: filter by date range, drop NaN labels, impute missing
    features (median), and standardize. Returns a list of fold dicts ready for
    model fitting.

    Parameters
    ----------
    dataset_pd : pd.DataFrame
        Full dataset (pandas) with features, label, date, entity columns.
    splits : list[dict]
        Walk-forward splits with fold, train_start, train_end, val_start, val_end.
    feature_names : list[str]
        Feature column names.
    label_col : str
        Target column name.
    date_col : str
        Date/timestamp column name.
    entity_col : str or None
        Entity column for cross-sectional IC. None for single-entity data.
    temporal_by_fold : pd.DataFrame, optional
        Per-fold temporal features with a 'fold' column. When provided, each
        fold's temporal feature columns are replaced with fold-specific values
        (fit on that fold's training data only — no look-ahead).
    temporal_keys : list[str], optional
        Join keys for temporal features (e.g., ['timestamp', 'symbol']).
    temporal_feature_names : list[str], optional
        Temporal feature column names to replace per fold.

    Returns
    -------
    list[dict]
        Each dict has keys: fold, X_train, X_val, y_train, y_val,
        meta (val metadata DataFrame), dates, entities, n_train, n_val.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    has_fold_temporal = temporal_by_fold is not None and temporal_keys and temporal_feature_names

    dates_series = dataset_pd[date_col]
    folds = []

    for split in splits:
        fold_id = split["fold"]
        val_start = split.get("val_start", split.get("test_start"))
        val_end = split.get("val_end", split.get("test_end"))

        train_mask = (dates_series >= split["train_start"]) & (dates_series <= split["train_end"])
        val_mask = (dates_series >= val_start) & (dates_series <= val_end)

        if has_fold_temporal:
            assert temporal_by_fold is not None
            assert temporal_keys is not None
            assert temporal_feature_names is not None
            train_rows = replace_temporal_columns(
                dataset_pd,
                train_mask,
                temporal_by_fold,
                temporal_keys,
                temporal_feature_names,
                fold_id,
            )
            val_rows = replace_temporal_columns(
                dataset_pd,
                val_mask,
                temporal_by_fold,
                temporal_keys,
                temporal_feature_names,
                fold_id,
            )
            X_train = train_rows[feature_names].values
            y_train = train_rows[label_col].values
            X_val = val_rows[feature_names].values
            y_val = val_rows[label_col].values
            y_eval = val_rows[eval_label_col].values if eval_label_col else None
            del train_rows, val_rows
        else:
            X_train = dataset_pd.loc[train_mask, feature_names].values
            y_train = dataset_pd.loc[train_mask, label_col].values
            X_val = dataset_pd.loc[val_mask, feature_names].values
            y_val = dataset_pd.loc[val_mask, label_col].values
            y_eval = dataset_pd.loc[val_mask, eval_label_col].values if eval_label_col else None

        if len(X_train) == 0 or len(X_val) == 0:
            print(f"  Fold {fold_id}: SKIP (empty train={len(X_train)}, val={len(X_val)})")
            continue

        # Drop rows where label is NaN
        train_valid = ~np.isnan(y_train)
        val_valid = ~np.isnan(y_val)
        X_train, y_train = X_train[train_valid], y_train[train_valid]
        X_val, y_val = X_val[val_valid], y_val[val_valid]
        if y_eval is not None:
            y_eval = y_eval[val_valid]

        # Optional train subsample (never touch val — OOS IC uses full val slice).
        # Walk-forward CV structure is preserved; only within-fold row density
        # is reduced. Seed tied to fold_id for reproducibility.
        if 0.0 < train_sample_frac < 1.0 and len(X_train) > 0:
            n_keep = max(1, int(len(X_train) * train_sample_frac))
            rng = np.random.default_rng(RANDOM_SEED + fold_id)
            keep_idx = rng.choice(len(X_train), size=n_keep, replace=False)
            keep_idx.sort()
            X_train = X_train[keep_idx]
            y_train = y_train[keep_idx]

        # Preprocessing: median imputation + standard scaling
        preprocessor = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        X_train_s = preprocessor.fit_transform(X_train)
        X_val_s = preprocessor.transform(X_val)
        X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_s = np.nan_to_num(X_val_s, nan=0.0, posinf=0.0, neginf=0.0)

        val_meta = dataset_pd.loc[val_mask].iloc[val_valid.nonzero()[0]]

        folds.append(
            {
                "fold": fold_id,
                "X_train": X_train_s,
                "X_val": X_val_s,
                "y_train": y_train,
                "y_val": y_val,
                "y_eval": y_eval,
                "meta": val_meta,
                "dates": val_meta[date_col].values,
                "entities": val_meta[entity_col].values if entity_col else None,
                "n_train": len(X_train),
                "n_val": len(X_val),
                "preprocessor": preprocessor,
            }
        )

    return folds


def prepare_single_fold(
    dataset: pl.DataFrame | pd.DataFrame,
    split: dict[str, Any],
    feature_names: list[str],
    label_col: str,
    date_col: str,
    entity_col: str | None,
    temporal_by_fold: pd.DataFrame | None = None,
    temporal_keys: list[str] | None = None,
    temporal_feature_names: list[str] | None = None,
    *,
    train_sample_frac: float = 1.0,
    eval_label_col: str | None = None,
) -> dict[str, Any] | None:
    """Preprocess a single CV fold — impute, scale, return arrays.

    Same logic as ``prepare_cv_folds`` but for ONE split at a time.
    Use this for large datasets where materializing all folds at once
    would exceed available memory.

    Accepts either Polars or pandas DataFrames. When given Polars, only
    the fold's rows are converted to numpy — avoiding a full-dataset
    pandas copy (~5GB for us_equities_panel).

    Parameters
    ----------
    train_sample_frac : float, optional
        Fraction of training rows to keep (1.0 = all). Same semantics
        as ``prepare_cv_folds`` / ``prepare_gbm_folds``: validation is
        never sampled, seed is tied to fold_id for reproducibility.

    Returns None if the fold is empty (no train or val rows).
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    fold_id = split["fold"]
    val_start = split.get("val_start", split.get("test_start"))
    val_end = split.get("val_end", split.get("test_end"))

    _has_fold_temporal = temporal_by_fold is not None and temporal_keys and temporal_feature_names

    if isinstance(dataset, pl.DataFrame):
        # Polars path — slice directly to numpy, no pandas intermediate
        # Cast string split boundaries to match the column's temporal dtype
        col_dtype = dataset.schema[date_col]
        _cast = pl.lit  # default: pass through
        if col_dtype == pl.Date:
            import datetime

            def _cast(s):
                return pl.lit(datetime.date.fromisoformat(str(s)[:10]))
        elif col_dtype in (pl.Datetime, pl.Datetime("us"), pl.Datetime("ns"), pl.Datetime("ms")):
            import datetime

            def _cast(s, _dt=col_dtype):
                # Cast to the column's exact dtype to preserve resolution + timezone
                # (e.g. crypto's timestamps are Datetime('ms', 'UTC'))
                try:
                    d = datetime.datetime.fromisoformat(str(s))
                except ValueError:
                    d = datetime.datetime.fromisoformat(f"{str(s)[:10]}T00:00:00")
                return pl.lit(d).cast(_dt)

        train_df = dataset.filter(
            (pl.col(date_col) >= _cast(split["train_start"]))
            & (pl.col(date_col) <= _cast(split["train_end"]))
        )
        val_df = dataset.filter(
            (pl.col(date_col) >= _cast(val_start)) & (pl.col(date_col) <= _cast(val_end))
        )

        if len(train_df) == 0 or len(val_df) == 0:
            print(f"  Fold {fold_id}: SKIP (empty train={len(train_df)}, val={len(val_df)})")
            return None

        # Replace temporal columns with fold-specific values if available
        if _has_fold_temporal:
            assert temporal_by_fold is not None
            assert temporal_keys is not None
            assert temporal_feature_names is not None
            fold_temp_pl = fold_temporal_frame(
                temporal_by_fold,
                fold_id,
                temporal_keys=temporal_keys,
                schema=train_df.schema,
            )

            for df_name in ("train_df", "val_df"):
                df = train_df if df_name == "train_df" else val_df
                df = df.drop(temporal_feature_names)
                df = df.join(fold_temp_pl, on=temporal_keys, how="left")
                if df_name == "train_df":
                    train_df = df
                else:
                    val_df = df

        # Drop rows where label is null
        train_df = train_df.filter(pl.col(label_col).is_not_null() & pl.col(label_col).is_not_nan())
        val_df = val_df.filter(pl.col(label_col).is_not_null() & pl.col(label_col).is_not_nan())

        X_train = train_df.select(feature_names).to_numpy()
        y_train = train_df[label_col].to_numpy()
        X_val = val_df.select(feature_names).to_numpy()
        y_val = val_df[label_col].to_numpy()
        y_eval = val_df[eval_label_col].to_numpy() if eval_label_col else None

        # Keep ID columns for val metadata (needed for IC computation + prediction assembly)
        id_cols = [c for c in dataset.columns if c in ID_COLS]
        val_meta_pl = val_df.select(id_cols)
        dates = val_df[date_col].to_numpy()
        entities = val_df[entity_col].to_numpy() if entity_col else None

        del train_df, val_df
    else:
        # Pandas path (used by prepare_cv_folds callers)
        dates_series = dataset[date_col]
        train_mask = (dates_series >= split["train_start"]) & (dates_series <= split["train_end"])
        val_mask = (dates_series >= val_start) & (dates_series <= val_end)

        if _has_fold_temporal:
            assert temporal_by_fold is not None
            assert temporal_keys is not None
            assert temporal_feature_names is not None
            train_rows = replace_temporal_columns(
                dataset,
                train_mask,
                temporal_by_fold,
                temporal_keys,
                temporal_feature_names,
                fold_id,
            )
            val_rows = replace_temporal_columns(
                dataset,
                val_mask,
                temporal_by_fold,
                temporal_keys,
                temporal_feature_names,
                fold_id,
            )
            X_train = train_rows[feature_names].values
            y_train = train_rows[label_col].values
            X_val = val_rows[feature_names].values
            y_val = val_rows[label_col].values
            y_eval = val_rows[eval_label_col].values if eval_label_col else None
        else:
            X_train = dataset.loc[train_mask, feature_names].values
            y_train = dataset.loc[train_mask, label_col].values
            X_val = dataset.loc[val_mask, feature_names].values
            y_val = dataset.loc[val_mask, label_col].values
            y_eval = dataset.loc[val_mask, eval_label_col].values if eval_label_col else None

        if len(X_train) == 0 or len(X_val) == 0:
            print(f"  Fold {fold_id}: SKIP (empty train={len(X_train)}, val={len(X_val)})")
            return None

        train_valid = ~np.isnan(y_train)
        val_valid = ~np.isnan(y_val)
        X_train, y_train = X_train[train_valid], y_train[train_valid]
        X_val, y_val = X_val[val_valid], y_val[val_valid]
        if y_eval is not None:
            y_eval = y_eval[val_valid]

        if _has_fold_temporal:
            val_meta_pd = val_rows.iloc[val_valid.nonzero()[0]]
        else:
            val_meta_pd = dataset.loc[val_mask].iloc[val_valid.nonzero()[0]]
        dates = val_meta_pd[date_col].values
        entities = val_meta_pd[entity_col].values if entity_col else None
        val_meta_pl = None  # Will use val_meta_pd below

    # Optional train subsample (never touch val — OOS IC uses full val slice).
    # Seed tied to fold_id for reproducibility.
    if 0.0 < train_sample_frac < 1.0 and len(X_train) > 0:
        n_keep = max(1, int(len(X_train) * train_sample_frac))
        rng = np.random.default_rng(RANDOM_SEED + fold_id)
        keep_idx = rng.choice(len(X_train), size=n_keep, replace=False)
        keep_idx.sort()
        X_train = X_train[keep_idx]
        y_train = y_train[keep_idx]

    # Preprocessing: median imputation + standard scaling
    preprocessor = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    X_train_s = preprocessor.fit_transform(X_train)
    X_val_s = preprocessor.transform(X_val)
    X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_s = np.nan_to_num(X_val_s, nan=0.0, posinf=0.0, neginf=0.0)

    n_train, n_val = len(X_train), len(X_val)
    del X_train, X_val  # Free unscaled arrays

    return {
        "fold": fold_id,
        "X_train": X_train_s,
        "X_val": X_val_s,
        "y_train": y_train,
        "y_val": y_val,
        "y_eval": y_eval,
        "meta_pl": val_meta_pl,  # Polars DataFrame (if Polars input)
        "meta": val_meta_pd if val_meta_pl is None else None,  # pandas (if pandas input)
        "dates": dates,
        "entities": entities,
        "n_train": n_train,
        "n_val": n_val,
        "preprocessor": preprocessor,
    }


# ---------------------------------------------------------------------------
# IC computation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def compute_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_values: list,
) -> dict[str, float]:
    """Compute classification metrics from predictions.

    Handles both binary and multiclass ordinal labels. The ``y_score``
    array is the expected value of class probabilities (``proba @ class_values``),
    which is how classification predictions are stored in this project.

    For binary {0, 1}: y_score IS p(class=1).
    For binary with other values: y_score is linearly transformed from probabilities.
    For multiclass ordinal: y_score is a weighted ranking score.

    Parameters
    ----------
    y_true : np.ndarray
        True class labels (integer values matching ``class_values``).
    y_score : np.ndarray
        Predicted scores (expected value of class probabilities).
    class_values : list
        Sorted unique class values (e.g. [0, 1] or [-1, 0, 1]).

    Returns
    -------
    dict[str, float]
        Metric name → value. Keys depend on binary vs multiclass.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    valid = np.isfinite(y_score) & np.isfinite(y_true)
    if valid.sum() < 10:
        return {}

    yt = y_true[valid]
    ys = y_score[valid]
    cv = sorted(class_values)
    n_classes = len(cv)

    metrics: dict[str, float] = {}

    if n_classes == 2:
        # Binary classification
        # Convert expected value back to p(positive_class)
        c0, c1 = cv[0], cv[1]
        # expected_value = p * c1 + (1 - p) * c0  =>  p = (ev - c0) / (c1 - c0)
        span = c1 - c0
        if span > 0:
            p1 = (ys - c0) / span
        else:
            return {}

        p1 = np.clip(p1, 1e-15, 1 - 1e-15)
        y_binary = (yt == c1).astype(int)
        y_pred_class = (p1 >= 0.5).astype(int)

        metrics["auc_roc"] = float(roc_auc_score(y_binary, p1))
        metrics["log_loss"] = float(log_loss(y_binary, p1))
        metrics["brier_score"] = float(brier_score_loss(y_binary, p1))
        metrics["accuracy"] = float(accuracy_score(y_binary, y_pred_class))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_binary, y_pred_class))

        # AUC-PR (average precision) — import separately since it can fail
        try:
            from sklearn.metrics import average_precision_score

            metrics["auc_pr"] = float(average_precision_score(y_binary, p1))
        except Exception as exc:
            warnings.warn(f"AUC-PR computation failed: {exc}", stacklevel=2)

    else:
        # Multiclass ordinal: assign to nearest class value
        y_pred_class_vals = np.array([cv[np.argmin(np.abs(np.array(cv) - s))] for s in ys])
        y_pred_class = y_pred_class_vals
        metrics["accuracy"] = float(accuracy_score(yt, y_pred_class))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(yt, y_pred_class))

        # AUC via ranking score against each binary split
        # For ordinal {-1,0,1}: treat expected value as ranking score
        # One-vs-rest AUC using the score as ordinal ranking
        for c in cv:
            y_bin = (yt == c).astype(int)
            if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
                # Higher score → more likely to be higher class
                score = ys if c == max(cv) else -ys if c == min(cv) else np.abs(ys)
                auc = float(roc_auc_score(y_bin, score))
                metrics[f"auc_class_{c}"] = auc

    return metrics


# Deprecated private aliases. Thirty notebook cells import these names with their leading
# underscore, which is what made them look private in the first place; each is a
# cross-module interface and is now public. The aliases keep those callers working until
# each notebook is re-executed with the public name, and go when the last one moves.
_replace_temporal_columns = replace_temporal_columns
