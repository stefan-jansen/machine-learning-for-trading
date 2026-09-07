"""Every configuration catalog names the model it fits.

`model_class` was blank on every GBM and every TabM row of every published catalog, in
every case study: 0 of 25 LightGBM presets and 0 of 4 TabM presets declared it, against
11 of 11 for ridge. `CATALOG_COLUMNS` asks for the column, so a notebook rendering the
catalog printed an empty field to the reader.

The value each preset declares is the string the run records as
`computation.model.class`, read off the runner rather than repeated here, so the catalog
and the registry name the same object and a change to what a runner records fails until
the presets follow it.

These live apart from `test_research_configs.py` because they import
`case_studies.utils.gbm` and `case_studies.utils.tabular_dl` for those constants, and both
reach torch and lightgbm at module scope. That file runs in `test-unit`, which has neither.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from case_studies.research import Study, load_model_configs

_GBM_CASE_STUDIES = (
    "cme_futures",
    "crypto_perps_funding",
    "etfs",
    "fx_pairs",
    "nasdaq100_microstructure",
    "sp500_equity_option_analytics",
    "sp500_options",
    "us_equities_panel",
    "us_firm_characteristics",
)
_TABULAR_DL_CASE_STUDIES = tuple(
    case_study for case_study in _GBM_CASE_STUDIES if case_study != "nasdaq100_microstructure"
)


@pytest.mark.parametrize("case_study", _GBM_CASE_STUDIES)
def test_every_gbm_configuration_names_the_model_the_run_records(case_study: str) -> None:
    """The catalog and the registry must name the same object.

    `model_class` was blank on every GBM row in every case study, so a published configuration
    table printed an empty column while `computation.model.class` recorded
    ``lightgbm.Booster`` for the run that table describes. Reading the constant off the runner
    rather than repeating the string here is what makes this couple: a change to what the
    runner records fails this until the presets follow it.
    """
    from case_studies.utils.gbm import GBM_MODEL_CLASS

    catalog = load_model_configs(Study.open(case_study), "gbm")
    assert catalog.height > 0
    assert catalog.get_column("model_class").to_list() == [GBM_MODEL_CLASS] * catalog.height


@pytest.mark.parametrize("case_study", _TABULAR_DL_CASE_STUDIES)
def test_every_tabm_configuration_names_the_model_the_run_records(case_study: str) -> None:
    from case_studies.utils.tabular_dl import TABM_MODEL_CLASS

    catalog = load_model_configs(Study.open(case_study), "tabular_dl")
    assert catalog.height > 0
    assert catalog.get_column("model_class").to_list() == [TABM_MODEL_CLASS] * catalog.height


def test_the_tabular_dl_family_fits_two_models_and_the_catalog_separates_them() -> None:
    """`model_class` carries a real distinction inside `tabular_dl`, which is why declaring it
    is the fix and dropping the column is not.

    `tabpfn` is a preset in the same family fitted by a different model - `_run_tabpfn_fold`
    builds a `TabPFNRegressor`, and `_resolve_tabm_config` refuses it on the canonical path
    precisely because it is not TabM. No training menu declares it today, so a blank column
    hid a difference nothing else in the catalog shows.
    """
    from case_studies.utils.tabular_dl import TABM_MODEL_CLASS, TABPFN_MODEL_CLASS

    assert TABPFN_MODEL_CLASS != TABM_MODEL_CLASS
    preset_dir = Path(__file__).resolve().parents[1] / "case_studies" / "config" / "tabm"
    declared = {
        path.stem: (yaml.safe_load(path.read_text()) or {}).get("model_class")
        for path in sorted(preset_dir.glob("*.yaml"))
    }
    assert declared.pop("tabpfn") == TABPFN_MODEL_CLASS
    assert set(declared.values()) == {TABM_MODEL_CLASS}


def test_declaring_the_model_class_moves_no_training_identity() -> None:
    """A catalog column is a statement to the reader, not a declaration of what was fitted.

    `computation` is hashed whole, so a key that reached it would reprice every registered GBM
    and TabM row in every registry - 268 and 54 runs across the fleet. Both runners read a
    fixed set of fields off the preset, and `model_class` is on neither list, so the 268 and
    the 54 keep their identities.

    The failure mode this pins is the placement, not the value. Everything under a preset's
    `params` reaches the estimator - `gbm.py` takes `dict(config["params"])` straight to
    `lgb.train`, and `tabular_dl.py` spreads it into `computation.model.params` - so
    `model_class` declared one level lower would both change the identity and be handed to
    LightGBM as an unknown parameter.
    """
    from case_studies.utils.tabular_dl import _build_tabm_training_spec

    config_root = Path(__file__).resolve().parents[1] / "case_studies" / "config"
    presets = sorted((config_root / "lgb").glob("*.yaml")) + sorted(
        (config_root / "tabm").glob("*.yaml")
    )
    assert len(presets) == 29
    for path in presets:
        preset = yaml.safe_load(path.read_text())
        assert preset["model_class"], path
        assert "model_class" not in (preset.get("params") or {}), path

    def _spec(config: dict) -> dict:
        return _build_tabm_training_spec(
            config,
            label_col="fwd_ret_21d",
            n_folds=3,
            feature_names=["a", "b"],
            eval_label_col=None,
            task_type="regression",
            class_values=None,
            runtime_spec={"device": "cpu", "num_threads": 1, "seed": 42},
            seed=42,
        )

    declared = yaml.safe_load((config_root / "tabm" / "tabm_s.yaml").read_text())
    declared["config_name"] = "tabm_s"
    undeclared = {key: value for key, value in declared.items() if key != "model_class"}
    assert _spec(declared) == _spec(undeclared)
