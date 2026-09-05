"""What `_patch_presets_for_testing` writes into a copied preset.

The table was flat until PatchTST needed its window cut, and `lookback` lives
inside `params`. A shallow `dict.update` would have replaced the whole block -
`architecture` included, which `deep_learning.py` subscripts without a default -
and produced a network that is not the one the preset names. Nothing would have
raised; the run would simply have been fitting something else.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from tests.preset_patches import _TEST_PRESET_PATCHES, _patch_presets_for_testing


def _write(config_dir: Path, model_type: str, preset: dict) -> Path:
    model_dir = config_dir / model_type
    model_dir.mkdir(parents=True)
    path = model_dir / f"{model_type}.yaml"
    path.write_text(yaml.safe_dump(preset))
    return path


class TestNestedParams:
    def test_a_params_override_merges_and_keeps_its_siblings(self, tmp_path):
        path = _write(
            tmp_path,
            "patchtst",
            {
                "batch_size": 2048,
                "n_epochs": 100,
                "params": {
                    "architecture": "patchtst",
                    "d_model": 64,
                    "lookback": 60,
                    "n_heads": 4,
                    "patch_size": 16,
                },
            },
        )
        _patch_presets_for_testing(tmp_path)
        written = yaml.safe_load(path.read_text())

        assert written["params"]["lookback"] == 24
        assert written["params"] == {
            "architecture": "patchtst",
            "d_model": 64,
            "lookback": 24,
            "n_heads": 4,
            "patch_size": 16,
        }
        assert written["n_epochs"] == 2
        assert written["batch_size"] == 2048, "the batch is a production number and is not patched"

    def test_a_flat_entry_leaves_params_untouched(self, tmp_path):
        """nlinear declares no `params` override, so its block must survive whole."""
        params = {"architecture": "nlinear", "dropout": 0.1, "lookback": 60}
        path = _write(tmp_path, "nlinear", {"n_epochs": 100, "params": dict(params)})
        _patch_presets_for_testing(tmp_path)
        written = yaml.safe_load(path.read_text())

        assert written["params"] == params
        assert written["n_epochs"] == 2


class TestTableShape:
    def test_every_params_override_is_a_dict(self):
        """A scalar under `params` would silently replace the block it merges into."""
        for model_type, overrides in _TEST_PRESET_PATCHES.items():
            nested = overrides.get("params")
            assert nested is None or isinstance(nested, dict), model_type

    def test_no_entry_patches_architecture(self):
        """The patcher reduces a workload. Naming a different network is not that."""
        for model_type, overrides in _TEST_PRESET_PATCHES.items():
            assert "architecture" not in (overrides.get("params") or {}), model_type
