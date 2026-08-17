"""The equity-option reader-facing workflow, on the branches CI never reaches.

`tests/overrides.yaml` gives every execution notebook of this case study a
`MAX_FOLDS`/`MAX_SYMBOLS` reduction, which selects `ExecutionTier.PREVIEW`. So the
canonical branches - the complete-surface requirement and the declared-menu
coverage assertion - are exercised nowhere else, which is exactly the shape of a
guard that cannot fail.
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

PRIMARY = "fwd_ret_5d"
UNFITTED = {("deep_learning", "nlinear"): "no notebook fits it"}


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


def test_declared_menu_coverage_accepts_the_full_population():
    unfitted = require_declared_menu_coverage(_covered(), unfitted=UNFITTED)
    # nlinear is declared on the three return labels and fitted by no notebook.
    assert unfitted.height == 3
    assert set(unfitted.get_column("config_name")) == {"nlinear"}
    assert set(unfitted.get_column("label")) == {
        "fwd_ret_5d",
        "fwd_ret_10d",
        "fwd_ret_risk_adj_5d",
    }
    assert unfitted.get_column("reason").is_not_null().all()


def test_a_missing_declared_model_fails_closed_and_names_itself():
    short = _covered().filter(~((pl.col("family") == "gbm") & (pl.col("label") == "fwd_dir_5d")))
    with pytest.raises(RuntimeError, match="omits declared models") as excinfo:
        require_declared_menu_coverage(short, unfitted=UNFITTED)
    # Counts cannot catch this: the message has to name the members.
    assert "fwd_dir_5d" in str(excinfo.value)
    assert "gbm" in str(excinfo.value)


def test_a_population_of_the_right_size_on_the_wrong_labels_still_fails():
    """Why coverage is compared on identity rather than on a row count."""
    covered = _covered()
    dropped = covered.filter(~((pl.col("family") == "gbm") & (pl.col("label") == "fwd_dir_5d")))
    # Put the count back by duplicating a label the menu does declare, so height
    # matches while the identities do not.
    padded = pl.concat(
        [
            dropped,
            covered.filter((pl.col("family") == "gbm") & (pl.col("label") == "fwd_dir_10d")),
        ]
    )
    assert padded.height == covered.height
    with pytest.raises(RuntimeError, match="omits declared models"):
        require_declared_menu_coverage(padded, unfitted=UNFITTED)


def test_a_produced_model_no_menu_declares_fails_closed():
    extra = pl.concat(
        [
            _covered(),
            pl.DataFrame(
                {"family": ["linear"], "label": [PRIMARY], "config_name": ["ridge_a_invented"]},
                schema={"family": pl.String, "label": pl.String, "config_name": pl.String},
            ),
        ]
    )
    with pytest.raises(RuntimeError, match="no menu declares"):
        require_declared_menu_coverage(extra, unfitted=UNFITTED)


def test_a_stale_exclusion_fails_closed():
    """An exclusion must not outlive the gap it describes, or it hides the next one."""
    with pytest.raises(ValueError, match="match no configured model"):
        require_declared_menu_coverage(
            _covered(), unfitted={**UNFITTED, ("gbm", "leaves_no_such"): "gone"}
        )


def test_an_exclusion_is_required_for_a_member_no_notebook_fits():
    """Without the entry, the same population must fail - the guard is not decorative."""
    with pytest.raises(RuntimeError, match="omits declared models"):
        require_declared_menu_coverage(_covered(), unfitted={})
