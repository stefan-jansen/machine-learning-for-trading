"""What tests/smoke.yaml has to be true about, checked without executing a notebook.

Every check here reuses the constant the runtime itself enforces rather than restating it. A
reduction vocabulary written out twice drifts, and the drift is invisible until a run fails: all
four of the family mismatches this file would have caught on 2026-09-05 were found by paying for a
run each. Reading `_TABM_PREVIEW_FIELDS` rather than copying it means the check cannot go stale.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.pm_helpers import unusable_parameters

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE = yaml.safe_load((REPO_ROOT / "tests" / "smoke.yaml").read_text())

# The stage-01-05 notebooks are closed and locked; the smoke standard is for stages 06 and later.
FIRST_SMOKE_STAGE = 6


def _stage(stem: str) -> int:
    return int("".join(c for c in stem.split("_", 1)[0] if c.isdigit()))


def _smoke_notebooks() -> list[str]:
    keys = []
    for path in sorted((REPO_ROOT / "case_studies").glob("*/[0-9]*.py")):
        if _stage(path.stem) >= FIRST_SMOKE_STAGE:
            keys.append(f"case_studies/{path.parent.name}/{path.stem}")
    return keys


# Notebooks with no entry yet. This is a ratchet, not a permission: the test asserts the missing
# set is EXACTLY this, so declaring one means deleting its line and nothing can silently join the
# list. us_equities_panel is owned by another session and is listed for completeness.
UNDECLARED = {
    "case_studies/crypto_perps_funding/13_backtest",
    "case_studies/crypto_perps_funding/14_portfolio_management",
    "case_studies/crypto_perps_funding/15_risk_management",
    "case_studies/crypto_perps_funding/16_costs",
    "case_studies/crypto_perps_funding/19_strategy_analysis",
    "case_studies/etfs/13_model_analysis",
    "case_studies/etfs/14_backtest",
    "case_studies/etfs/15_portfolio_management",
    "case_studies/etfs/16_risk_management",
    "case_studies/etfs/17_costs",
    "case_studies/etfs/20_strategy_analysis",
    "case_studies/nasdaq100_microstructure/06_linear",
    "case_studies/nasdaq100_microstructure/07_gbm",
    "case_studies/nasdaq100_microstructure/08_dl_nlinear",
    "case_studies/nasdaq100_microstructure/09_dl_lstm",
    "case_studies/nasdaq100_microstructure/10_dl_tcn",
    "case_studies/nasdaq100_microstructure/11_dl_patchtst",
    "case_studies/nasdaq100_microstructure/12_causal_dml",
    "case_studies/nasdaq100_microstructure/13_model_analysis",
    "case_studies/nasdaq100_microstructure/14_backtest",
    "case_studies/nasdaq100_microstructure/15_portfolio_management",
    "case_studies/nasdaq100_microstructure/16_risk_management",
    "case_studies/nasdaq100_microstructure/17_costs",
    "case_studies/nasdaq100_microstructure/18_holdout_predictions",
    "case_studies/nasdaq100_microstructure/19_holdout_backtest",
    "case_studies/nasdaq100_microstructure/20_strategy_analysis",
    "case_studies/sp500_equity_option_analytics/06_linear",
    "case_studies/sp500_equity_option_analytics/07_gbm",
    "case_studies/sp500_equity_option_analytics/08_tabular_dl",
    "case_studies/sp500_equity_option_analytics/09_dl_lstm",
    "case_studies/sp500_equity_option_analytics/10_dl_patchtst",
    "case_studies/sp500_equity_option_analytics/11_latent_factors",
    "case_studies/sp500_equity_option_analytics/11a_pca",
    "case_studies/sp500_equity_option_analytics/11b_ipca",
    "case_studies/sp500_equity_option_analytics/11c_conditional_autoencoder",
    "case_studies/sp500_equity_option_analytics/11d_stochastic_discount_factor",
    "case_studies/sp500_equity_option_analytics/11e_supervised_autoencoder",
    "case_studies/sp500_equity_option_analytics/12_causal_dml",
    "case_studies/sp500_equity_option_analytics/13_model_analysis",
    "case_studies/sp500_equity_option_analytics/14_backtest",
    "case_studies/sp500_equity_option_analytics/15_portfolio_management",
    "case_studies/sp500_equity_option_analytics/16_risk_management",
    "case_studies/sp500_equity_option_analytics/17_costs",
    "case_studies/sp500_equity_option_analytics/18_holdout_predictions",
    "case_studies/sp500_equity_option_analytics/19_holdout_backtest",
    "case_studies/sp500_equity_option_analytics/20_strategy_analysis",
    "case_studies/sp500_options/06_linear",
    "case_studies/sp500_options/07_gbm",
    "case_studies/sp500_options/08_tabular_dl",
    "case_studies/sp500_options/09_deep_learning",
    "case_studies/sp500_options/09a_lstm",
    "case_studies/sp500_options/09b_patchtst",
    "case_studies/sp500_options/10_causal_dml",
    "case_studies/sp500_options/11_model_analysis",
    "case_studies/sp500_options/12_backtest",
    "case_studies/sp500_options/13_portfolio_management",
    "case_studies/sp500_options/14_risk_management",
    "case_studies/sp500_options/15_costs",
    "case_studies/sp500_options/16_holdout_predictions",
    "case_studies/sp500_options/17_holdout_backtest",
    "case_studies/sp500_options/18_strategy_analysis",
    "case_studies/sp500_options/90_ic_diagnostic",
    "case_studies/us_equities_panel/06_linear",
    "case_studies/us_equities_panel/07_gbm",
    "case_studies/us_equities_panel/08_tabular_dl",
    "case_studies/us_equities_panel/09_dl_nlinear",
    "case_studies/us_equities_panel/10_dl_lstm",
    "case_studies/us_equities_panel/11_dl_tsmixer",
    "case_studies/us_equities_panel/12_dl_weekly",
    "case_studies/us_equities_panel/13_latent_factors",
    "case_studies/us_equities_panel/13a_pca",
    "case_studies/us_equities_panel/13b_ipca",
    "case_studies/us_equities_panel/14_causal_dml",
    "case_studies/us_equities_panel/15_model_analysis",
    "case_studies/us_equities_panel/16_backtest",
    "case_studies/us_equities_panel/17_portfolio_management",
    "case_studies/us_equities_panel/18_costs",
    "case_studies/us_equities_panel/19_risk_management",
    "case_studies/us_equities_panel/20_strategy_analysis",
    "case_studies/us_firm_characteristics/06_gbm",
    "case_studies/us_firm_characteristics/07_tabular_dl",
    "case_studies/us_firm_characteristics/08_latent_factors",
    "case_studies/us_firm_characteristics/08a_ipca",
    "case_studies/us_firm_characteristics/08b_conditional_autoencoder",
    "case_studies/us_firm_characteristics/08c_stochastic_discount_factor",
    "case_studies/us_firm_characteristics/08d_supervised_autoencoder",
    "case_studies/us_firm_characteristics/09_causal_dml",
    "case_studies/us_firm_characteristics/10_model_analysis",
    "case_studies/us_firm_characteristics/11_backtest",
    "case_studies/us_firm_characteristics/12_portfolio_management",
    "case_studies/us_firm_characteristics/13_risk_management",
    "case_studies/us_firm_characteristics/14_costs",
    "case_studies/us_firm_characteristics/15_holdout_predictions",
    "case_studies/us_firm_characteristics/16_holdout_backtest",
    "case_studies/us_firm_characteristics/17_strategy_analysis",
}


# A preview run cannot create an official population - population.py::_refuse_preview_activation
# refuses it, because a population is canonical by definition and a reduced run must not publish
# one. So no smoke configuration can promise a row in these, and one that did would be asking for
# a failure it cannot avoid.
UNREACHABLE_TABLES = {"official_populations", "official_population_members"}


def test_every_stage_06_notebook_is_declared_or_recorded_as_undeclared() -> None:
    missing = {key for key in _smoke_notebooks() if key not in SMOKE}
    assert missing == UNDECLARED


def test_no_entry_names_a_notebook_that_does_not_exist() -> None:
    absent = [key for key in SMOKE if not (REPO_ROOT / f"{key}.py").exists()]
    assert absent == []


def test_a_block_names_its_issue_and_what_it_is_blocked_on() -> None:
    """A blocked entry is a notebook that CAN have a smoke run and does not have a working one.

    That is a different statement from an exemption, and the difference is why both exist: an
    exemption retires a notebook from the standard, a block keeps it inside the standard and
    records what has to be fixed. Neither may be a bare flag - an entry that says only
    "blocked" is indistinguishable from one nobody has got to.
    """
    for key, entry in SMOKE.items():
        if entry.get("blocked_by") is None:
            continue
        assert isinstance(entry["blocked_by"], int), f"{key}: blocked_by must be an issue number"
        assert len(entry.get("blocked_reason", "")) > 30, f"{key}: a block needs a reason"
        assert not entry.get("exempt"), f"{key}: blocked and exempt say different things"


def test_an_exemption_says_what_refuses_the_preview() -> None:
    """An exemption has to name the code that refuses, so a reader can check it.

    Without this an exemption is indistinguishable from a notebook nobody got to, and the
    difference is the whole value of the list.
    """
    for key, entry in SMOKE.items():
        if entry.get("exempt"):
            reason = entry.get("exempt_reason", "")
            assert len(reason) > 60, f"{key}: exemption needs a reason naming what refuses"


def test_every_declared_parameter_reaches_its_notebook() -> None:
    """Driven through the same helper as tests/overrides.yaml's gate.

    Fourteen blocks in overrides.yaml once declared MAX_FOLDS and MAX_SYMBOLS against notebooks
    that read neither, so runs described as reduced ran in full. A smoke configuration that
    misses this way is worse: the run passes, quickly, and proves nothing about the notebook.
    """
    unreachable = {
        key: unusable_parameters(REPO_ROOT / f"{key}.py", entry["parameters"])
        for key, entry in SMOKE.items()
        if entry.get("parameters")
        and unusable_parameters(REPO_ROOT / f"{key}.py", entry["parameters"])
    }
    assert unreachable == {}


def _allowed_reduction_fields(key: str) -> set[str] | None:
    """The reduction vocabulary the runtime will enforce for this notebook's family."""
    from case_studies.utils import causal, deep_learning, gbm, linear, tabular_dl
    from case_studies.utils.latent_factors import adapter as latent

    stem = key.rsplit("/", 1)[1]
    if "causal" in stem:
        return set(causal._DML_PREVIEW_FIELDS)
    if "tabular_dl" in stem or "tabm" in stem:
        return set(tabular_dl._TABM_PREVIEW_FIELDS)
    if "_dl_" in stem or stem.endswith(("_lstm", "_tcn", "_nlinear", "_patchtst", "_tsmixer")):
        return set(deep_learning._SEQUENCE_PREVIEW_FIELDS)
    if stem.endswith("_gbm"):
        return set(gbm._GBM_PREVIEW_FIELDS)
    if stem.endswith("_linear"):
        return set(linear._PREVIEW_FIELDS)
    for model, fields in latent._MODEL_PREVIEW_FIELDS.items():
        if stem.endswith(model) or model in stem:
            return set(fields)
    if stem.endswith(("_pca", "_autoencoder", "_discount_factor")):
        return set(latent._PREVIEW_FIELDS)
    return None


@pytest.mark.parametrize(
    "key", sorted(k for k, v in SMOKE.items() if v.get("parameters", {}).get("PREVIEW_REDUCTIONS"))
)
def test_a_reduction_uses_the_vocabulary_its_family_accepts(key: str) -> None:
    """Every family refuses a reduction key it does not declare, at run time, after loading data.

    Catching it here costs nothing. Catching it there cost four runs on 2026-09-05, all of which
    died on `train_sample_frac` - a key the linear and GBM runners accept and the TabM, sequence
    and latent ones do not.
    """
    allowed = _allowed_reduction_fields(key)
    if allowed is None:
        pytest.skip(f"no family mapping for {key}")
    declared = set(SMOKE[key]["parameters"]["PREVIEW_REDUCTIONS"])
    assert declared <= allowed, f"{key}: {sorted(declared - allowed)} not in {sorted(allowed)}"


@pytest.mark.parametrize(
    "key", sorted(k for k, v in SMOKE.items() if "causal" in k and v.get("parameters"))
)
def test_a_causal_reduction_can_actually_compute_its_refutation(key: str) -> None:
    """Below MIN_PLACEBO_DRAWS the permutation test is not run and refutation_p registers NULL.

    The row is written anyway, and the incomplete row then refuses to serve itself on the next
    run - so an under-powered reduction does not merely produce a weaker result, it leaves the
    workspace unable to retry without a manual delete. The recorded reduction in
    work/2026-08-21-stocktake/.../preview-model-analysis.sh asks for five draws and does exactly
    this.
    """
    from case_studies.utils.causal import MIN_PLACEBO_DRAWS

    parameters = SMOKE[key]["parameters"]
    # Two spellings, because the causal notebooks split into two shapes: cme, crypto and the
    # others declare the reduction as a PREVIEW_REDUCTIONS dict, while etfs and fx_pairs declare
    # N_PLACEBO, MAX_SAMPLES and CV_FOLDS as separate parameters. The floor is the same one.
    declared = parameters.get("PREVIEW_REDUCTIONS", {}).get("n_placebo")
    if declared is None:
        declared = parameters.get("N_PLACEBO")
    assert declared is not None, f"{key}: a causal smoke run has to declare its placebo count"
    assert declared >= MIN_PLACEBO_DRAWS


def test_no_entry_promises_a_table_a_preview_cannot_write() -> None:
    offenders = {
        key: sorted(set(entry.get("writes") or []) & UNREACHABLE_TABLES)
        for key, entry in SMOKE.items()
        if set(entry.get("writes") or []) & UNREACHABLE_TABLES
    }
    assert offenders == {}


def test_a_declared_notebook_records_what_it_measured() -> None:
    """A configuration with no measured runtime has not been run, and is a claim rather than a
    check. The number is what criteria 5 and 6 are assessed against."""
    unmeasured = [
        key
        for key, entry in SMOKE.items()
        if not entry.get("exempt")
        and entry.get("blocked_by") is None
        and entry.get("measured_s") is None
    ]
    assert unmeasured == []
