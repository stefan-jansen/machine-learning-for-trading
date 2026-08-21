"""A case study declares the precision it stores and fits its feature matrices in.

``nasdaq100_microstructure`` is 16,098,877 rows by 88 features. In double precision that is a
10.9 GB modeling dataset and a 58.9 GB peak through one gradient-boosting fold; in single it is
5.6 GB and 36.6 GB. LightGBM bins to uint8 regardless of what it is handed, and the linear
families fit standardised columns, so the extra mantissa was being paid for and discarded.

The declaration is per case study rather than global: the other eight fit comfortably in double
precision, and narrowing them would move their numbers for no gain. These tests hold that
boundary, and hold the cache to knowing which precision it holds - a fold set written in one and
read back in the other would fit at a precision nobody declared, and nothing would report it.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from case_studies.utils.folds import fold_cache_key, prepare_raw_folds
from utils.modeling import build_modeling_input_lineage, feature_storage_dtype

DECLARED_NARROW = {"nasdaq100_microstructure"}
EVERY_CASE_STUDY = [
    "cme_futures",
    "crypto_perps_funding",
    "etfs",
    "fx_pairs",
    "nasdaq100_microstructure",
    "sp500_equity_option_analytics",
    "sp500_options",
    "us_equities_panel",
    "us_firm_characteristics",
]


class TestTheDeclarationIsReadFromTheCaseStudy:
    @pytest.mark.parametrize("case_study", EVERY_CASE_STUDY)
    def test_only_the_case_studies_that_declare_it_are_narrowed(self, case_study):
        expected = pl.Float32 if case_study in DECLARED_NARROW else pl.Float64
        assert feature_storage_dtype(case_study) == expected

    def test_a_case_study_that_declares_nothing_stays_in_double_precision(self):
        assert feature_storage_dtype("etfs") == pl.Float64

    def test_an_unknown_case_study_does_not_silently_narrow(self):
        assert feature_storage_dtype("no_such_case_study") == pl.Float64


class TestThePrecisionIsPartOfTheFoldAddress:
    """A fold set written in one precision must not be served to a run that wants the other."""

    @staticmethod
    def _key(design_dtype: str) -> str:
        return fold_cache_key(
            case_study="nasdaq100_microstructure",
            label_col="fwd_ret_15m",
            eval_label_col=None,
            feature_names=["a", "b"],
            splits=[
                {
                    "fold": 0,
                    "train_start": "2020-01-01",
                    "train_end": "2020-06-01",
                    "val_start": "2020-06-02",
                    "val_end": "2020-07-01",
                }
            ],
            input_lineage={"artifacts": {}},
            train_sample_frac=1.0,
            seed=42,
            design_dtype=design_dtype,
        )

    def test_the_two_precisions_address_different_fold_sets(self):
        assert self._key("float32") != self._key("float64")


class _Dataset:
    """The minimum a fold preparation needs, built by hand so no artifact is required."""

    def __init__(self, feature_dtype: str):
        n = 400
        self.dataset = pl.DataFrame(
            {
                "timestamp": [f"2020-01-{d % 28 + 1:02d}" for d in range(n)],
                "symbol": [f"S{i % 4}" for i in range(n)],
                "f0": np.linspace(0.0, 1.0, n),
                "f1": np.linspace(1.0, 2.0, n),
                "y": np.linspace(-1.0, 1.0, n),
            }
        )
        self.feature_names = ["f0", "f1"]
        self.label_col = "y"
        self.eval_label_col = None
        self.date_col = "timestamp"
        self.entity_cols = ["symbol"]
        self.case_study_id = ""
        self.temporal_by_fold = None
        self.temporal_keys: list[str] = []
        self.temporal_feature_names: list[str] = []
        self.feature_dtype = feature_dtype
        self.splits = [
            {
                "fold": 0,
                "train_start": "2020-01-01",
                "train_end": "2020-01-20",
                "val_start": "2020-01-21",
                "val_end": "2020-01-28",
            }
        ]


class TestTheDesignMatrixIsBuiltInTheDeclaredPrecision:
    @pytest.mark.parametrize(
        ("declared", "expected"), [("float32", np.float32), ("float64", np.float64)]
    )
    def test_the_declaration_reaches_the_array(self, declared, expected):
        mds = _Dataset(declared)
        folds = prepare_raw_folds(mds, mds.splits, use_cache=False)
        assert folds[0].X_train.dtype == expected
        assert folds[0].X_val.dtype == expected

    def test_a_dataset_that_declares_nothing_keeps_double_precision(self):
        mds = _Dataset("float64")
        del mds.feature_dtype
        folds = prepare_raw_folds(mds, mds.splits, use_cache=False)
        assert folds[0].X_train.dtype == np.float64

    def test_single_precision_halves_the_design_matrix(self):
        wide = prepare_raw_folds(_Dataset("float64"), _Dataset("float64").splits, use_cache=False)
        narrow = prepare_raw_folds(_Dataset("float32"), _Dataset("float32").splits, use_cache=False)
        assert narrow[0].X_train.nbytes * 2 == wide[0].X_train.nbytes


class TestThePrecisionIsPartOfTheTrainingIdentity:
    """The fold cache is not the only place a precision switch could go unnoticed.

    The artifacts a case study reads are unchanged by a precision declaration, so if the
    declared type is absent from the training identity, a result fitted in double precision and
    one fitted in single address the same row: the registry serves the older fit for a spec that
    asked for the other, and the notebook prints numbers a rerun no longer produces.
    """

    @staticmethod
    def _lineage(feature_dtype: str) -> dict:
        from utils.modeling import build_modeling_input_lineage

        return build_modeling_input_lineage(
            artifacts={},
            feature_names=["a", "b"],
            splits=[
                {
                    "fold": 0,
                    "train_start": "2020-01-01",
                    "train_end": "2020-06-01",
                    "val_start": "2020-06-02",
                    "val_end": "2020-07-01",
                }
            ],
            label_buffer="15min",
            task_type="regression",
            eval_label_col=None,
            max_symbols=0,
            symbols=None,
            feature_dtype=feature_dtype,
        )

    def test_the_two_precisions_do_not_share_a_training_identity(self):
        assert self._lineage("float32")["fingerprint"] != self._lineage("float64")["fingerprint"]

    def test_the_declared_precision_is_recorded_not_merely_hashed(self):
        assert self._lineage("float32")["feature_dtype"] == "float32"

    def test_the_default_is_absent_from_the_payload_rather_than_written_as_float64(self):
        """A key written unconditionally rehashes the eight case studies that declared nothing."""
        assert "feature_dtype" not in self._lineage("float64")


class TestTheKnobDidNotInvalidateTheCaseStudiesThatDeclaredNothing:
    """The fingerprint a double-precision case study produces must be the one it always produced.

    ``PRE_PRECISION_FINGERPRINT`` was not computed from a reimplementation of the payload. It is
    what ``build_modeling_input_lineage`` at 719c9c14 - the last commit before the precision work
    - returns for ``FIXTURE`` below. Every registered training run predates that work, so if this
    value moves, the registry can no longer resolve any of them and every case study re-fits from
    scratch. The first cut of this change put the key in unconditionally and did exactly that.

    Raising ``schema_version`` in the payload also moves it, and that is a deliberate
    invalidation rather than the regression this class is named for. Re-pin the constant in the
    same commit as the bump, or the failure reads as a defect that did not happen.
    """

    PRE_PRECISION_FINGERPRINT = "97738b867c03d3dbdb59994f920855d08be1d9596e72eddf5f25cb81a270b720"

    FIXTURE = {
        "feature_names": ["alpha", "beta", "gamma"],
        "splits": [
            {
                "fold": 0,
                "train_start": "2018-01-01",
                "train_end": "2019-01-01",
                "val_start": "2019-01-02",
                "val_end": "2019-06-01",
            },
            {
                "fold": 1,
                "train_start": "2018-06-01",
                "train_end": "2019-06-01",
                "val_start": "2019-06-02",
                "val_end": "2019-12-01",
            },
        ],
        "label_buffer": "21D",
        "task_type": "regression",
        "eval_label_col": "fwd_ret_21d",
        "max_symbols": 0,
        "symbols": ["AAA", "BBB"],
    }

    @pytest.fixture
    def artifacts(self, tmp_path):
        path = tmp_path / "features.parquet"
        path.write_bytes(b"deterministic bytes for the artifact digest")
        return {"features": path}

    def _fingerprint(self, artifacts, **overrides):
        return build_modeling_input_lineage(artifacts=artifacts, **self.FIXTURE, **overrides)[
            "fingerprint"
        ]

    def test_calling_it_the_way_every_case_study_does_reproduces_the_old_fingerprint(
        self, artifacts
    ):
        assert self._fingerprint(artifacts) == self.PRE_PRECISION_FINGERPRINT

    def test_declaring_double_precision_explicitly_reproduces_it_too(self, artifacts):
        assert self._fingerprint(artifacts, feature_dtype="float64") == (
            self.PRE_PRECISION_FINGERPRINT
        )

    def test_the_case_study_that_declared_single_precision_gets_a_new_one(self, artifacts):
        assert self._fingerprint(artifacts, feature_dtype="float32") != (
            self.PRE_PRECISION_FINGERPRINT
        )


class TestTheFoldAddressIsPreservedForTheSameReason:
    """Fold preparation is the expensive half, and its cache is addressed the same way."""

    KEY = {
        "case_study": "etfs",
        "label_col": "fwd_ret_21d",
        "eval_label_col": None,
        "feature_names": ["alpha", "beta"],
        "splits": [
            {
                "fold": 0,
                "train_start": "2018-01-01",
                "train_end": "2019-01-01",
                "val_start": "2019-01-02",
                "val_end": "2019-06-01",
            }
        ],
        "input_lineage": {"fingerprint": "abc123"},
        "train_sample_frac": 1.0,
        "seed": 42,
    }

    PRE_PRECISION_KEY = "92376ca3e260204a"
    """What ``fold_cache_key`` at 719c9c14 returns for ``KEY``.

    ``FOLD_PREPARATION_VERSION`` is in this payload, so a deliberate bump moves this value too.
    ``tests/test_folds.py`` names ``GOLDEN_VERSION`` and ``PINNED_PREPARATION_DIGEST`` as the
    constants to update alongside such a bump; this is the third.
    """

    def test_the_default_addresses_the_cache_the_eight_already_wrote(self):
        """Against what 719c9c14 produced, not against another call into the same code.

        Comparing the default with an explicit ``float64`` passes whether or not the key is
        conditional, because both go down the same branch. Only the recorded value can fail.
        """
        assert fold_cache_key(**self.KEY) == self.PRE_PRECISION_KEY

    def test_declaring_double_precision_explicitly_addresses_it_too(self):
        assert fold_cache_key(**self.KEY, design_dtype="float64") == self.PRE_PRECISION_KEY

    def test_single_precision_addresses_a_different_one(self):
        assert fold_cache_key(**self.KEY, design_dtype="float32") != self.PRE_PRECISION_KEY
