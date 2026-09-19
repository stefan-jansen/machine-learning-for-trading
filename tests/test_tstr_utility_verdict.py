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


def _load(name: str):
    """Lift ``name`` and everything it calls out of the notebook, without executing it."""
    tree = ast.parse(NOTEBOOK.read_text())
    defined = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert name in defined, f"the notebook no longer defines {name}"

    wanted, queue = set(), [name]
    while queue:
        current = queue.pop()
        if current in wanted:
            continue
        wanted.add(current)
        queue += [
            call.func.id
            for call in ast.walk(defined[current])
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in defined
        ]

    functions = [node for fname, node in defined.items() if fname in wanted]
    namespace: dict = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace[name]


def _load_verdict():
    """``tstr_utility_verdict``, lifted with every notebook function it calls.

    It delegates its thresholds to ``tstr_utility_level``, and a lift that took the
    verdict alone would exec a body whose call target is undefined: every case here
    would fail with ``NameError`` instead of on what it asserts, which is a suite that
    cannot fail for the right reason. ``_load`` finds the callees rather than listing
    them, so the next delegation does not have to be noticed by hand.
    """
    return _load("tstr_utility_verdict")


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


# The verdict word alone, which is what the five-draw spread cell prints per draw. The
# notebook's own render exercises none of this: `tests/overrides.yaml` pins TSTR_DRAWS to 1
# with N_GENERATE 10, so every draw fails the `len(X_draw) > 10` gate and the cell prints
# "No draw produced a usable training set". Without these cases the thresholds and the
# tally are checked by a production run and nothing else.


def test_each_threshold_of_the_level_scale_is_reachable():
    level = _load("tstr_utility_level")
    assert level(0.750, 0.740) == "HIGH"
    assert level(0.750, 0.700) == "MODERATE"
    assert level(0.750, 0.560) == "LIMITED"
    assert level(0.750, 0.500) == "NONE"
    assert level(0.500, 0.600) == "NO BASELINE"


def test_a_chance_baseline_is_refused_before_the_ratio_is_taken():
    """0.6 / 0.5 is 1.2, so a ratio-first reading would call a chance baseline HIGH."""
    level = _load("tstr_utility_level")
    assert level(0.500, 0.600) == "NO BASELINE"
    assert level(0.490, 0.600) == "NO BASELINE"


def test_a_tstr_at_chance_is_refused_before_the_baseline_is_examined():
    """Both guards fire on (0.5, 0.5); the TSTR one is the answer, because the draw ranks
    nothing whatever the baseline did."""
    assert _load("tstr_utility_level")(0.500, 0.500) == "NONE"


def test_the_ratio_boundaries_are_exclusive():
    level = _load("tstr_utility_level")
    assert level(1.0, 0.95) == "MODERATE", "0.95 exactly is not above 0.95"
    assert level(1.0, 0.85) == "LIMITED", "0.85 exactly is not above 0.85"


def test_the_tally_counts_every_draw_and_names_each_verdict_once():
    tally = _load("tstr_level_tally")
    assert tally(["HIGH", "LIMITED", "HIGH", "NONE", "HIGH"]) == "HIGH x3, LIMITED x1, NONE x1"


def test_the_tally_orders_verdicts_by_the_draw_that_first_earned_one():
    """Not by count, and not by whatever order a set would produce.

    A single outlying draw is the finding; ordering by count would print it last behind
    the majority word, and a set would move it between runs at one seed - the defect this
    notebook was fixed for.
    """
    tally = _load("tstr_level_tally")
    assert tally(["NONE", "HIGH", "HIGH"]) == "NONE x1, HIGH x2"
    assert tally(["HIGH", "HIGH", "NONE"]) == "HIGH x2, NONE x1"


def test_the_tally_of_one_draw_is_that_draw():
    assert _load("tstr_level_tally")(["MODERATE"]) == "MODERATE x1"
