"""What this case study's own workflow module covers that the shared tests do not.

`model_request_catalog` and `require_complete_canonical_requests` are this module's,
and their **refusal** paths are reached nowhere else - a notebook run cannot make a
guard fire without failing the run, and the notebook lane only ever exercises the
passing side. `tests/overrides.yaml` leaves `11a_pca`,
`11c_conditional_autoencoder`, `11d_stochastic_discount_factor` and
`11e_supervised_autoencoder` without a reduction, so those do resolve
`ExecutionTier.CANONICAL` and do reach the passing branch.

The declared-menu guard itself moved to `case_studies.research` and is tested
against all nine case studies in `tests/test_research_declared_menu_coverage.py`.
What remains here is one test that the thin wrappers route to it correctly, since
wiring is the only thing that can now break on this side.

`load_configs` resolves the training menus through `get_case_study_dir`, which
honours `ML4T_OUTPUT_DIR`; CI sets that workflow-wide and only the session-scoped
`seeded_output_dir` fixture populates it. These tests want the repo's committed
menus, not a trimmed copy, so the fixture below clears the redirect rather than
depending on whether an earlier module happened to seed it.
"""

from __future__ import annotations

import polars as pl
import pytest

from case_studies.sp500_equity_option_analytics.research_workflow import (
    configured_model_menu,
    model_request_catalog,
    published_labels,
    require_complete_canonical_requests,
    require_declared_menu_coverage,
)

CASE_STUDY_NAME = "sp500_equity_option_analytics"
PRIMARY = "fwd_ret_5d"
UNFITTED = {("deep_learning", "nlinear"): "no notebook fits it"}


@pytest.fixture(autouse=True)
def _read_the_committed_menus(monkeypatch):
    """Read `case_studies/`, not an `ML4T_OUTPUT_DIR` copy some other module seeded."""
    monkeypatch.delenv("ML4T_OUTPUT_DIR", raising=False)


def _covered() -> pl.DataFrame:
    """The declared menu minus the members no notebook fits."""
    excluded = pl.DataFrame(
        {"family": ["deep_learning"], "config_name": ["nlinear"]},
        schema={"family": pl.String, "config_name": pl.String},
    )
    return configured_model_menu().join(excluded, on=["family", "config_name"], how="anti")


def test_the_published_labels_lead_with_the_primary():
    labels = published_labels()
    assert labels[0] == PRIMARY
    assert len(set(labels)) == len(labels)


def test_the_menu_covers_every_label_that_declares_a_family():
    menu = configured_model_menu()
    assert set(menu.columns) == {"family", "config_name", "label"}
    assert menu.height == menu.unique().height
    # `causal_dml` is a declared entry in the same YAML and is not a predictive
    # family, so it must be absent by the adapter rule rather than by a list.
    assert "causal_dml" not in set(menu.get_column("family"))
    assert {"linear", "gbm", "tabular_dl", "deep_learning", "latent_factors"} == set(
        menu.get_column("family")
    )


def test_an_unknown_label_is_refused():
    with pytest.raises(ValueError, match="unknown labels"):
        model_request_catalog("linear", labels=["fwd_ret_nonesuch"])


def test_an_undeclared_configuration_is_refused():
    with pytest.raises(ValueError, match="not declared"):
        model_request_catalog("linear", labels=[PRIMARY], config_names=["ridge_a_nonesuch"])


def test_a_family_no_label_declares_is_refused():
    with pytest.raises(ValueError, match="no declared requests"):
        model_request_catalog("latent_factors", labels=["fwd_dir_5d"])


def test_a_partial_canonical_surface_is_refused():
    """The guard that stops a canonical run publishing a population short of its menu."""
    complete = model_request_catalog("linear")
    partial = complete.head(complete.height - 1)
    with pytest.raises(ValueError, match="complete declared request surface"):
        require_complete_canonical_requests(partial, family="linear", execution_tier="canonical")


def test_a_partial_surface_is_allowed_under_preview():
    complete = model_request_catalog("linear")
    require_complete_canonical_requests(complete.head(1), family="linear", execution_tier="preview")


def test_the_complete_surface_passes_canonically():
    complete = model_request_catalog("linear")
    require_complete_canonical_requests(complete, family="linear", execution_tier="canonical")


def test_a_label_that_declares_no_such_family_blames_the_label_not_the_name():
    """The mirror of the defect above, which the first fix for it introduced.

    `pca` is declared on all three return labels, so blaming the configuration name
    here would send a reader hunting a typo that does not exist. The cause is that
    `fwd_dir_5d` declares no latent factors at all. No notebook reaches this module
    today - `11a_pca.py` goes through `case_studies.utils.latent_factors.case_study` -
    so what is pinned here is the message, against the day one does.
    """
    with pytest.raises(ValueError, match="no declared requests for 'latent_factors'"):
        model_request_catalog("latent_factors", labels=["fwd_dir_5d"], config_names=["pca"])


def test_an_undeclared_name_names_the_labels_it_was_checked_against():
    with pytest.raises(ValueError, match="not declared") as excinfo:
        model_request_catalog("latent_factors", labels=[PRIMARY], config_names=["no_such_factor"])
    message = str(excinfo.value)
    assert "latent_factors" in message
    assert PRIMARY in message
    assert "no_such_factor" in message


def test_a_label_that_declares_a_different_family_is_skipped_not_refused():
    """A family the label does not declare is skipped, not refused.

    `published_labels()` includes `fwd_dir_5d`, which declares linear and gbm but no
    latent factors, so the mixed case - some labels declaring the family, some not - is
    what any caller passing the full label set gets. Tightening the skip into a refusal
    would make that call impossible. Only the refusal side was pinned before this.
    """
    catalog = model_request_catalog(
        "latent_factors", labels=[PRIMARY, "fwd_dir_5d"], config_names=["pca"]
    )
    assert catalog.get_column("label").to_list() == [PRIMARY]
    assert catalog.get_column("config_name").to_list() == ["pca"]


def test_the_default_latent_factor_call_spans_only_the_declaring_labels():
    """The same path as it is actually taken, with no labels argument at all."""
    catalog = model_request_catalog("latent_factors", config_names=["pca"])
    declaring = set(catalog.get_column("label"))
    assert declaring == {"fwd_ret_5d", "fwd_ret_10d", "fwd_ret_risk_adj_5d"}
    assert not declaring & {"fwd_dir_5d", "fwd_dir_10d"}


def test_an_empty_configuration_selection_names_itself():
    """`config_names=[]` used to be reported as the family declaring nothing."""
    with pytest.raises(ValueError, match="config_names is empty"):
        model_request_catalog("linear", labels=[PRIMARY], config_names=[])


def test_an_empty_label_list_means_every_published_label():
    """The asymmetry with `config_names=[]`, pinned so it stays deliberate.

    This case study's six model notebooks leave `LABELS = []` to mean "all", though
    each converts it with `LABELS or None` before calling, so no notebook reaches
    this branch - it exists for consistency with that parameters-cell convention.
    An empty `config_names` cannot be that convention, being only ever a literal in
    code, so one is widened and the other refused.
    """
    assert model_request_catalog("linear", labels=[]).equals(model_request_catalog("linear"))


def test_the_wrappers_route_to_the_shared_implementation():
    """Only the case-study binding. The guard's behaviour is the shared file's to prove.

    `tests/test_research_declared_menu_coverage.py` already asserts what the excluded
    frame contains for this case study; repeating it here would be a second copy that
    goes stale independently.
    """
    from case_studies.research import configured_model_menu as shared_menu
    from case_studies.research import require_declared_menu_coverage as shared_coverage

    assert configured_model_menu().equals(shared_menu(CASE_STUDY_NAME))
    assert require_declared_menu_coverage(_covered(), unfitted=UNFITTED).equals(
        shared_coverage(_covered(), case_study=CASE_STUDY_NAME, unfitted=UNFITTED)
    )
