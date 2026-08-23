"""Contracts for the shared model-request interface.

These are the behaviours three case studies had implemented differently in private copies before
`case_studies.research.configs` existed. Two of those copies fitted a smaller population without
complaining when a configuration name was mistyped, which is the first test here.
"""

from __future__ import annotations

import polars as pl
import pytest

from case_studies.research import (
    Study,
    declared_labels,
    load_model_configs,
    model_requests,
    narrows_declared_catalog,
    run_model_population,
    sweep_labels,
)


@pytest.fixture(scope="module")
def study() -> Study:
    return Study.open("etfs")


def test_unknown_configuration_name_raises(study: Study) -> None:
    """A mistyped configuration must fail, not silently shrink the population."""
    with pytest.raises(ValueError, match="not declared"):
        load_model_configs(
            study,
            "linear",
            labels=["fwd_ret_21d"],
            config_names=["ols", "ridge_a1e6"],
        )


def test_selection_returns_exactly_the_named_configurations(study: Study) -> None:
    selected = load_model_configs(
        study,
        "linear",
        labels=["fwd_ret_21d"],
        config_names=["ols", "ridge_a1000000.0"],
    )
    assert selected.get_column("config_name").to_list() == ["ols", "ridge_a1000000.0"]


def test_catalog_exposes_the_estimator_and_its_parameters(study: Study) -> None:
    """The reader sees Ridge(alpha=...), not an opaque preset name."""
    catalog = load_model_configs(study, "linear", labels=["fwd_ret_21d"])
    row = catalog.filter(pl.col("config_name") == "ridge_a1000000.0").row(0, named=True)
    assert row["model_class"] == "Ridge"
    assert row["params"] == "alpha=1000000.0"
    assert catalog.filter(pl.col("config_name") == "ols").row(0, named=True)["params"] == "defaults"


def test_labels_default_to_every_menu_declaring_the_family(study: Study) -> None:
    """The population follows the training menus rather than a constant beside them."""
    labels = declared_labels(study, "linear")
    assert set(labels) == {"fwd_ret_21d", "fwd_ret_5d"}
    everything = load_model_configs(study, "linear")
    assert set(everything.get_column("label").unique()) == set(labels)


def test_a_family_no_menu_declares_raises(study: Study) -> None:
    with pytest.raises(ValueError, match="declares"):
        declared_labels(study, "not_a_family")


def test_requests_carry_only_the_identity_columns(study: Study) -> None:
    """model_class and params are for the reader; passing them to study.model would raise."""
    catalog = load_model_configs(
        study,
        "linear",
        labels=["fwd_ret_21d"],
        config_names=["ols"],
    )
    (request,) = model_requests(study, catalog, execution_tier="canonical")
    assert (request.family, request.label, request.config_name) == ("linear", "fwd_ret_21d", "ols")


def test_population_refuses_to_mix_execution_tiers(study: Study) -> None:
    catalog = load_model_configs(
        study,
        "linear",
        labels=["fwd_ret_21d"],
        config_names=["ols", "ridge_a1000000.0"],
    )
    canonical = model_requests(study, catalog[:1], execution_tier="canonical")
    preview = model_requests(
        study,
        catalog[1:],
        execution_tier="preview",
        preview_reductions={"max_folds": 1},
    )
    with pytest.raises(ValueError, match="mix execution tiers"):
        run_model_population(study, [*canonical, *preview], population_name="mixed")


def test_a_preview_cannot_supersede_a_snapshot(study: Study) -> None:
    """A preview population is discarded with its workspace, so it has no lineage to extend."""
    catalog = load_model_configs(study, "linear", labels=["fwd_ret_21d"], config_names=["ols"])
    preview = model_requests(
        study,
        catalog,
        execution_tier="preview",
        preview_reductions={"max_folds": 1},
    )
    with pytest.raises(ValueError, match="cannot supersede"):
        run_model_population(
            study, preview, population_name="linear-preview", supersedes="6f061b802c3f"
        )


@pytest.mark.parametrize(
    "case_study",
    [
        "cme_futures",
        "crypto_perps_funding",
        "etfs",
        "fx_pairs",
        "nasdaq100_microstructure",
        "sp500_equity_option_analytics",
        "sp500_options",
        "us_equities_panel",
        "us_firm_characteristics",
    ],
)
@pytest.mark.parametrize("family", ["linear", "gbm", "tabular_dl", "deep_learning"])
def test_declared_labels_never_leaves_the_sweep(case_study: str, family: str) -> None:
    """A label the sweep does not fit must not reach a notebook that fits every declared label.

    `config/setup.yaml` says which labels the sweep fits; a training menu says what to fit for a
    label. `sp500_options` keeps full menus for four fixed-horizon labels that `02_labels` writes
    for the diagnostic notebooks and that `setup.yaml` dropped from the sweep, so a menu-driven
    default would fit 140 linear configurations instead of 28 and publish four out-of-sweep
    labels into the population `12_backtest` selects over.
    """
    opened = Study.open(case_study)
    in_sweep = set(sweep_labels(opened))
    try:
        labels = declared_labels(opened, family)
    except ValueError:
        return  # no sweep label declares this family, which is a legitimate answer
    assert set(labels) <= in_sweep, (
        f"{case_study} {family}: declared_labels returned labels outside the sweep: "
        f"{sorted(set(labels) - in_sweep)}"
    )
    assert labels, f"{case_study} {family}: declared_labels returned nothing"


def test_sp500_options_fits_only_the_label_its_sweep_declares() -> None:
    """The case that forced the rule, pinned so a menu edit cannot quietly restore it."""
    opened = Study.open("sp500_options")
    assert sweep_labels(opened) == ("ret_to_expiry",)
    assert declared_labels(opened, "linear") == ("ret_to_expiry",)
    assert load_model_configs(opened, "linear").height == 28
    assert set(load_model_configs(opened, "gbm").get_column("label")) == {"ret_to_expiry"}


def test_narrows_declared_catalog_catches_an_equal_sized_other_population() -> None:
    """Row counts are not enough to tell the canonical catalog from a different one.

    `sp500_options` keeps four out-of-sweep menus with exactly the 28 linear configurations the
    canonical menu has, so a count comparison passes `LABELS=["fwd_ret_5d"]` through and lets a
    run publish an entirely different member set under the canonical population name.
    """
    opened = Study.open("sp500_options")
    complete = load_model_configs(opened, "linear")
    other = load_model_configs(opened, "linear", labels=["fwd_ret_5d"])
    assert other.height == complete.height, "the fixture for this test needs equal sizes"
    assert not narrows_declared_catalog(opened, "linear", complete)
    assert narrows_declared_catalog(opened, "linear", other)


def test_narrows_declared_catalog_catches_a_configuration_subset() -> None:
    opened = Study.open("etfs")
    complete = load_model_configs(opened, "linear")
    subset = load_model_configs(opened, "linear", config_names=["ols"])
    assert not narrows_declared_catalog(opened, "linear", complete)
    assert narrows_declared_catalog(opened, "linear", subset)
