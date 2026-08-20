"""The declared-menu coverage guard, which is the only check that sees an unrequested model.

Each execution notebook checks that it requested its own complete surface, so none
of them can see a family member that no notebook requests at all. This guard runs
where the families reassemble - model analysis - and compares
`(family, label, config_name)` against the published YAML menus.

It lives in `case_studies.research` rather than in a per-case-study module because
every case study needs it and the failure it prevents is identical in each.
"""

from __future__ import annotations

import polars as pl
import pytest

from case_studies.research import configured_model_menu, require_declared_menu_coverage
from utils.paths import REPO_ROOT

# Anchored to REPO_ROOT, not the working directory. A cwd-relative glob collects
# nothing when pytest runs from anywhere else, and six parametrized tests then pass
# vacuously - which is the failure this module exists to catch, one level up.
CASE_STUDIES = sorted(p.parts[-3] for p in (REPO_ROOT / "case_studies").glob("*/config/setup.yaml"))
IDENTITY = ["family", "label", "config_name"]


@pytest.fixture(autouse=True)
def _read_the_committed_menus(monkeypatch):
    """`load_configs` honours ML4T_OUTPUT_DIR; these tests want the committed menus."""
    monkeypatch.delenv("ML4T_OUTPUT_DIR", raising=False)


def test_every_case_study_has_a_readable_menu():
    """A config the reader cannot resolve is a defect wherever it is."""
    assert CASE_STUDIES, "no case studies discovered"
    for case_study in CASE_STUDIES:
        menu = configured_model_menu(case_study)
        assert menu.height > 0, case_study
        assert set(menu.columns) == set(IDENTITY), case_study
        assert menu.height == menu.unique().height, f"{case_study} declares a duplicate member"


def test_causal_dml_is_excluded_by_the_adapter_rule_not_a_list():
    """`causal_dml` sits in the same YAML and is not a predictive family."""
    for case_study in CASE_STUDIES:
        families = set(configured_model_menu(case_study).get_column("family"))
        assert "causal_dml" not in families, case_study
        assert families, case_study


def test_an_unknown_label_is_refused():
    with pytest.raises(ValueError, match="unknown labels"):
        configured_model_menu(CASE_STUDIES[0], labels=["fwd_ret_nonesuch"])


@pytest.mark.parametrize("case_study", CASE_STUDIES)
def test_the_complete_menu_passes_its_own_coverage_check(case_study):
    declared = configured_model_menu(case_study)
    assert require_declared_menu_coverage(declared, case_study=case_study).is_empty()


@pytest.mark.parametrize("case_study", CASE_STUDIES)
def test_a_missing_member_fails_closed_and_names_itself(case_study):
    declared = configured_model_menu(case_study)
    dropped = declared.row(0, named=True)
    short = declared.filter(
        ~(
            (pl.col("family") == dropped["family"])
            & (pl.col("label") == dropped["label"])
            & (pl.col("config_name") == dropped["config_name"])
        )
    )
    with pytest.raises(RuntimeError, match="omits declared models") as excinfo:
        require_declared_menu_coverage(short, case_study=case_study)
    # Counts cannot catch this, so the message has to name the member.
    assert dropped["config_name"] in str(excinfo.value)
    assert case_study in str(excinfo.value)


@pytest.mark.parametrize("case_study", CASE_STUDIES)
def test_the_right_height_on_the_wrong_labels_still_fails(case_study):
    """Why coverage compares identity and not a row count.

    A plan of full menu length built on one label is the documented way this slips
    through, so restoring the height must not restore the verdict.
    """
    declared = configured_model_menu(case_study)
    labels = declared.get_column("label").unique().sort().to_list()
    if len(labels) < 2:
        pytest.skip(f"{case_study} declares one label; the swap needs two")
    dropped = declared.filter(pl.col("label") != labels[0])
    missing = declared.height - dropped.height
    # Restore the exact height with rows the menu does declare, so a count check
    # would pass. They collapse under the guard's own unique(), which is the point:
    # height is recoverable, identity is not.
    padding = declared.filter(pl.col("label") == labels[1]).head(1)
    padded = pl.concat([dropped, *([padding] * missing)])
    assert padded.height == declared.height
    with pytest.raises(RuntimeError, match="omits declared models"):
        require_declared_menu_coverage(padded, case_study=case_study)


@pytest.mark.parametrize("case_study", CASE_STUDIES)
def test_a_member_no_menu_declares_fails_closed(case_study):
    declared = configured_model_menu(case_study)
    invented = pl.DataFrame(
        {
            "family": ["linear"],
            "label": [declared.get_column("label")[0]],
            "config_name": ["ridge_a_invented"],
        },
        schema={"family": pl.String, "label": pl.String, "config_name": pl.String},
    )
    with pytest.raises(RuntimeError, match="no menu declares"):
        require_declared_menu_coverage(pl.concat([declared, invented]), case_study=case_study)


@pytest.mark.parametrize("case_study", CASE_STUDIES)
def test_a_stale_exclusion_fails_closed(case_study):
    """An exclusion outliving its gap hides the next one."""
    declared = configured_model_menu(case_study)
    with pytest.raises(ValueError, match="match no configured model"):
        require_declared_menu_coverage(
            declared, case_study=case_study, unfitted={("gbm", "leaves_no_such"): "gone"}
        )


def test_an_excluded_member_is_returned_with_its_reason_and_is_load_bearing():
    """sp500_equity_option_analytics declares `nlinear` and no notebook fits it."""
    case_study = "sp500_equity_option_analytics"
    declared = configured_model_menu(case_study)
    without = declared.filter(
        ~((pl.col("family") == "deep_learning") & (pl.col("config_name") == "nlinear"))
    )
    unfitted = {("deep_learning", "nlinear"): "no notebook fits it"}
    excluded = require_declared_menu_coverage(without, case_study=case_study, unfitted=unfitted)
    assert set(excluded.get_column("config_name")) == {"nlinear"}
    assert excluded.height == 3
    assert excluded.get_column("reason").is_not_null().all()
    # Remove the entry and the same population must fail: the exclusion is not decorative.
    with pytest.raises(RuntimeError, match="omits declared models"):
        require_declared_menu_coverage(without, case_study=case_study)


def test_a_population_missing_the_identity_columns_is_refused():
    case_study = CASE_STUDIES[0]
    declared = configured_model_menu(case_study).drop("config_name")
    with pytest.raises(ValueError, match="missing"):
        require_declared_menu_coverage(declared, case_study=case_study)
