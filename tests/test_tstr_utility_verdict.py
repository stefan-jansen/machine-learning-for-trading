"""The GReaT TSTR verdict must not read a majority-class collapse as high utility.

`05_synthetic_data/06_llm_tabular_great` divided two accuracies on a task whose positive
base rate is about 10%, so a synthetic-trained classifier that answered "no" for every row
scored 0.952 against the baseline's 0.953 and the notebook printed "HIGH utility" - while
printing, two lines above, a TSTR AUC of 0.297. The verdict now divides AUC.
"""

import ast
from pathlib import Path

NOTEBOOK = Path(__file__).parents[1] / "05_synthetic_data" / "06_llm_tabular_great.py"

# The two executions recorded in the issue, as (auc_trtr, auc_tstr, acc_trtr, acc_tstr).
RELEASE_RUN = (0.759, 0.697, 0.940, 0.872)
COLLAPSED_RUN = (0.750, 0.297, 0.953, 0.952)


def _load_verdict():
    """Lift ``tstr_utility_verdict`` out of the notebook without executing the notebook.

    It is lifted with every notebook-level function it calls, found rather than listed.
    ``tstr_utility_verdict`` delegates its thresholds to ``tstr_utility_level``, and a lift
    that took the verdict alone would exec a body whose call target is undefined: all seven
    cases here would fail with ``NameError`` instead of on what they assert, which is a
    suite that cannot fail for the right reason. Finding the callees means the next
    delegation does not have to be noticed by hand.
    """
    tree = ast.parse(NOTEBOOK.read_text())
    defined = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "tstr_utility_verdict" in defined, "the notebook no longer defines the verdict"

    wanted, queue = set(), ["tstr_utility_verdict"]
    while queue:
        name = queue.pop()
        if name in wanted:
            continue
        wanted.add(name)
        queue += [
            call.func.id
            for call in ast.walk(defined[name])
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in defined
        ]

    functions = [node for name, node in defined.items() if name in wanted]
    namespace: dict = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace["tstr_utility_verdict"]


def test_a_below_chance_tstr_auc_is_refused_rather_than_scored():
    auc_trtr, auc_tstr, _, _ = COLLAPSED_RUN
    verdict = _load_verdict()(auc_trtr, auc_tstr)
    assert "NO usable signal" in verdict
    assert "HIGH" not in verdict and "MODERATE" not in verdict


def test_the_collapse_the_accuracy_ratio_called_high_utility():
    """Accuracy divides to 99.8% on the run whose ranking is anti-predictive."""
    _, _, acc_trtr, acc_tstr = COLLAPSED_RUN
    assert acc_tstr / acc_trtr > 0.95, "the old gate's own arithmetic"
    auc_trtr, auc_tstr, _, _ = COLLAPSED_RUN
    assert "NO usable signal" in _load_verdict()(auc_trtr, auc_tstr)


def test_the_release_run_still_reads_as_moderate():
    """The 2026-07-14 render's verdict is unchanged: 0.697 / 0.759 is 91.8%."""
    auc_trtr, auc_tstr, _, _ = RELEASE_RUN
    assert "MODERATE utility" in _load_verdict()(auc_trtr, auc_tstr)


def test_a_preserved_ranking_reads_as_high():
    assert "HIGH utility" in _load_verdict()(0.750, 0.740)


def test_a_degraded_ranking_reads_as_limited():
    assert "LIMITED utility" in _load_verdict()(0.750, 0.560)


def test_a_constant_scoring_model_is_refused_not_scored():
    """A model emitting one probability for every row scores exactly 0.5.

    That is the collapse itself, and against a baseline barely above chance the ratio
    would read 96.2% and print HIGH. The refusal is therefore at 0.5, not below it.
    """
    verdict = _load_verdict()(0.520, 0.500)
    assert "NO usable signal" in verdict
    assert "HIGH" not in verdict and "MODERATE" not in verdict


def test_a_baseline_that_ranks_nothing_has_no_utility_to_preserve():
    """Dividing by a chance baseline would score an above-chance model against nothing."""
    verdict = _load_verdict()(0.500, 0.600)
    assert "no utility for the synthetic data to preserve" in verdict
    assert "HIGH" not in verdict
