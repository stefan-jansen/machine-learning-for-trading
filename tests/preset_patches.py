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
