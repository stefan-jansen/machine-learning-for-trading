"""Test-only model preset overrides.

Deliberately free of a pytest import: tests/generate_intermediates.py runs
standalone (`uv run python tests/generate_intermediates.py`, no dev extra
required) and imports this module, as does tests/conftest.py under pytest.
Two copies of this table previously drifted - a fix landing in one left the
other on stale values with no error to signal it.
"""

from pathlib import Path

import yaml

# Per-model-type overrides applied to copied preset YAMLs.
# Goal: minimal workload that still exercises the training loop + registry.
_TEST_PRESET_PATCHES: dict[str, dict] = {
    "lgb": {"max_iterations": 2, "checkpoint_interval": 1},
    # DL families: 2 epochs, checkpoint every epoch
    "lstm": {"n_epochs": 2, "checkpoint_interval": 1},
    "tsmixer": {"n_epochs": 2, "checkpoint_interval": 1},
    "tcn": {"n_epochs": 2, "checkpoint_interval": 1},
    "nlinear": {"n_epochs": 2, "checkpoint_interval": 1},
    "patchtst": {"n_epochs": 2, "checkpoint_interval": 1},
    # TabDL: 2 epochs
    "tabm": {"n_epochs": 2, "checkpoint_interval": 1},
    # Latent factors: 2 epochs
    "cae": {"n_epochs": 2, "checkpoint_interval": 1},
    "sdf": {"n_epochs": 2, "checkpoint_interval": 1},
    "sae": {"n_epochs": 2, "checkpoint_interval": 1},
    # IPCA's ALS needs far more sweeps on a fixture cross-section than on the
    # production one - 259-693 per fold on sp500_equity_option_analytics's
    # 21-asset panel at K=2 - but no max_iter patch is needed for that: the
    # pinned ml4t-models build defaults max_iter to 10,000 and nothing on
    # this path narrows it. factor_ridge/gamma_ridge are raised instead: the
    # fixture sits right at the K=2/K=3 identification boundary, where
    # convergence is sensitive enough to floating-point path that it
    # deterministically diverged between two machines on the unregularized
    # 1e-6 default (see tests/overrides.yaml's 11b_ipca entry for the
    # measured before/after). Regularizing is a conditioning fix, not a
    # bigger budget.
    "ipca": {"n_epochs": 2, "checkpoint_interval": 1, "factor_ridge": 1e-2, "gamma_ridge": 1e-2},
}


def _patch_presets_for_testing(config_dir: Path) -> None:
    """Patch copied preset YAMLs with reduced-workload values for testing."""
    for model_type, overrides in _TEST_PRESET_PATCHES.items():
        model_dir = config_dir / model_type
        if not model_dir.exists():
            continue
        for preset_path in model_dir.glob("*.yaml"):
            preset = yaml.safe_load(preset_path.read_text())
            if preset is None:
                continue
            preset.update(overrides)
            with open(preset_path, "w") as f:
                yaml.dump(preset, f, default_flow_style=False)


# Max configs per family in training menu YAMLs (keep tests fast but comprehensive).
# Only applied to families with homogeneous sweep configs (linear, gbm).
# DL/TabDL/latent/causal families are NOT trimmed because each config often
# maps to a dedicated notebook (e.g., 09_dl_lstm, 10_dl_tsmixer).
_MAX_CONFIGS_PER_FAMILY = 2
_TRIM_FAMILIES = {"linear", "gbm"}


def _trim_label_configs(cs_config_dir: Path) -> None:
    """Trim training menu YAMLs to at most _MAX_CONFIGS_PER_FAMILY for sweep families.

    The menus live at ``config/training/fwd_*.yaml``. An earlier copy of this
    function globbed ``config/fwd_*.yaml``, matched nothing, and silently left
    every sweep family at full width.
    """
    training_dir = cs_config_dir / "training"
    label_root = training_dir if training_dir.exists() else cs_config_dir
    for label_yaml in label_root.glob("fwd_*.yaml"):
        data = yaml.safe_load(label_yaml.read_text())
        if data is None or not isinstance(data, dict):
            continue
        trimmed = False
        for family, configs in data.items():
            if (
                family in _TRIM_FAMILIES
                and isinstance(configs, list)
                and len(configs) > _MAX_CONFIGS_PER_FAMILY
            ):
                data[family] = configs[:_MAX_CONFIGS_PER_FAMILY]
                trimmed = True
        if trimmed:
            with open(label_yaml, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
