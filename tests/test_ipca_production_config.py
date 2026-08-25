from pathlib import Path

import yaml

CASE_STUDIES = (
    "etfs",
    "sp500_equity_option_analytics",
    "us_equities_panel",
    "us_firm_characteristics",
)
ITERATION_BUDGETS = {
    # Lower than its neighbours because it was measured rather than inherited: the eight ETF folds
    # converge between 53 and 110 alternations, so 1000 clears the slowest by roughly nine times
    # while still bounding a stuck fold to about 13 minutes at ~0.76s each. See the reasoning in
    # `git log -1 140b536c`.
    "etfs": 1000,
    "sp500_equity_option_analytics": 10000,
    "us_equities_panel": 10000,
    "us_firm_characteristics": 10000,
}


def test_production_ipca_uses_the_measured_convergent_solver_contract() -> None:
    for case_study in CASE_STUDIES:
        setup_path = Path("case_studies") / case_study / "config" / "setup.yaml"
        setup = yaml.safe_load(setup_path.read_text())
        config = setup["modeling"]["latent_factors"]["model_kwargs"]["ipca"]

        assert int(config["max_iter"]) == ITERATION_BUDGETS[case_study], case_study
        assert float(config["factor_ridge"]) == 0.01, case_study
        assert float(config["gamma_ridge"]) == 0.01, case_study
