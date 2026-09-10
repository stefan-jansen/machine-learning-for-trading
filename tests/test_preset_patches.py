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

from tests.preset_patches import (
    _TEST_PRESET_PATCHES,
    _patch_presets_for_testing,
    _spread,
    _trim_label_configs,
)


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


class TestTheTrimSpansTheMenu:
    """What `_trim_label_configs` keeps out of a sweep family's menu.

    A training menu is a path: the linear family runs from ``ols`` through rising
    ridge and lasso penalties out to ``enet_f0.85``, and the gbm family from a
    default through wider leaf counts. The head of that path is the one two-config
    slice guaranteed to keep configurations that cannot be told apart, and the
    fixture took it for a year - crypto's ``fwd_ret_8h`` menu is 28 long and CI
    compared ``ols`` against ``ridge_a0.001``, whose rank IC agrees to every digit
    the registry stores because a penalty that small does not reorder predictions.
    """

    def _menu(self, tmp_path: Path, family: str, names: list[str]) -> Path:
        training = tmp_path / "training"
        training.mkdir(parents=True)
        path = training / "fwd_ret_8h.yaml"
        path.write_text(yaml.safe_dump({family: [{"name": n} for n in names]}))
        return path

    def test_the_kept_configs_are_the_ends_of_the_menu(self, tmp_path):
        path = self._menu(tmp_path, "linear", [f"c{i}" for i in range(28)])
        _trim_label_configs(tmp_path)

        kept = [entry["name"] for entry in yaml.safe_load(path.read_text())["linear"]]
        assert kept == ["c0", "c27"]

    def test_the_head_two_are_not_what_survives(self, tmp_path):
        """The regression this replaced: `configs[:2]` on the real crypto menu."""
        names = ["ols", "ridge_a0.001", "ridge_a0.01", "lasso_f0.015", "enet_f0.85"]
        path = self._menu(tmp_path, "linear", names)
        _trim_label_configs(tmp_path)

        kept = [entry["name"] for entry in yaml.safe_load(path.read_text())["linear"]]
        assert kept == ["ols", "enet_f0.85"]
        assert "ridge_a0.001" not in kept

    def test_a_menu_no_longer_than_the_budget_is_left_alone(self, tmp_path):
        path = self._menu(tmp_path, "gbm", ["default_mse", "leaves_63_huber"])
        _trim_label_configs(tmp_path)

        kept = [entry["name"] for entry in yaml.safe_load(path.read_text())["gbm"]]
        assert kept == ["default_mse", "leaves_63_huber"]

    def test_an_untrimmed_family_keeps_every_config(self, tmp_path):
        """DL configs map to dedicated notebooks, so the budget does not apply."""
        path = self._menu(tmp_path, "deep_learning", ["lstm", "tcn", "nlinear", "patchtst"])
        _trim_label_configs(tmp_path)

        kept = [entry["name"] for entry in yaml.safe_load(path.read_text())["deep_learning"]]
        assert len(kept) == 4

    def test_the_spread_is_evenly_spaced_for_any_budget(self):
        assert _spread(list(range(9)), 3) == [0, 4, 8]
        assert _spread(list(range(9)), 1) == [0]
        assert _spread([7], 2) == [7]
