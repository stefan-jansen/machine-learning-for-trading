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
    run_model_population,
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
