"""Gate: feature-selection notebooks must scope development analysis pre-holdout.

Feature selection, robustness sweeps, and feature evaluation are development
decisions; the sealed holdout (``case_studies/etfs/config/setup.yaml`` ->
``evaluation.holdout_start``) must not inform them (the rule taught in
``06_strategy_definition/02_cv_foundations``). Regenerating the selection
artifacts requires the full data downloads, so this test statically checks the
notebook sources instead: each must read the holdout boundary from
``setup.yaml`` and apply the holdout filter *before* its first IC computation.
"""

from __future__ import annotations

import ast
import re
from datetime import date
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent
SETUP_YAML = REPO_ROOT / "case_studies" / "etfs" / "config" / "setup.yaml"

# (notebook source, identifier marking its first IC computation). The filter
# must be applied earlier in the file than this marker so no IC ranking,
# BH-FDR discovery, stability selection, or importance ranking ever sees
# holdout rows.
HOLDOUT_SCOPED_NOTEBOOKS = [
    ("08_financial_features/05_feature_selection.py", "ic_by_date"),
    ("08_financial_features/06_robustness_sensitivity.py", "def compute_momentum_ic_series"),
    ("case_studies/etfs/02_labels.py", "cross_sectional_ic_series("),
    ("case_studies/etfs/05_evaluation.py", "ic_series_data = {feat"),
    ("case_studies/crypto_perps_funding/02_labels.py", "cross_sectional_ic_series("),
    ("case_studies/cme_futures/02_labels.py", "cross_sectional_ic_series("),
    ("case_studies/fx_pairs/02_labels.py", "cross_sectional_ic_series("),
    ("case_studies/us_firm_characteristics/02_labels.py", "cross_sectional_ic_series("),
    (
        "case_studies/sp500_equity_option_analytics/02_labels.py",
        "cross_sectional_ic_series(",
    ),
    ("case_studies/sp500_options/02_labels.py", "cross_sectional_ic_series("),
    ("case_studies/nasdaq100_microstructure/02_labels.py", "cross_sectional_ic_series("),
    ("case_studies/us_equities_panel/02_labels.py", "cross_sectional_ic_series("),
    ("case_studies/us_equities_panel/03_financial_features.py", "ic_results[feat] = "),
]


def load_holdout_start() -> str:
    setup = yaml.safe_load(SETUP_YAML.read_text())
    return setup["evaluation"]["holdout_start"]


def test_setup_yaml_declares_sealed_holdout() -> None:
    """The holdout window is declared, ISO-formatted, and non-empty."""
    setup = yaml.safe_load(SETUP_YAML.read_text())
    holdout_start = date.fromisoformat(setup["evaluation"]["holdout_start"])
    holdout_end = date.fromisoformat(setup["evaluation"]["holdout_end"])
    assert holdout_start < holdout_end


@pytest.mark.parametrize(
    ("rel_path", "first_ic_marker"),
    HOLDOUT_SCOPED_NOTEBOOKS,
    ids=[rel_path for rel_path, _ in HOLDOUT_SCOPED_NOTEBOOKS],
)
def test_holdout_filter_precedes_first_ic_computation(rel_path: str, first_ic_marker: str) -> None:
    source = (REPO_ROOT / rel_path).read_text()

    # The boundary must come from setup.yaml, not a hardcoded literal that can
    # silently drift from the case-study config.
    assert "setup.yaml" in source, f"{rel_path}: holdout boundary must be read from setup.yaml"
    assert "holdout_start" in source, (
        f"{rel_path}: must read evaluation.holdout_start from setup.yaml"
    )

    ic_pos = source.find(first_ic_marker)
    assert ic_pos != -1, (
        f"{rel_path}: first-IC marker {first_ic_marker!r} not found — "
        "update HOLDOUT_SCOPED_NOTEBOOKS if the notebook was restructured"
    )

    # Require the boundary to be COMPARED against, not merely bound. Every one of these
    # notebooks binds `HOLDOUT_START` from setup.yaml near the top, so searching for the
    # bare name finds that assignment, which precedes the IC whether or not any filter
    # exists — deleting the seal entirely used to pass here.
    #
    # The comparison is not always against the boundary name itself: one notebook narrows
    # via `end_date = min(END_DATE, HOLDOUT_START)` and then filters on `end_date`, which
    # is an equally sound seal. So taint the boundary name through assignments and accept a
    # comparison against anything derived from it.
    tree = ast.parse(source)
    tainted = {"HOLDOUT_START", "holdout_start_dt", "holdout_start"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        referenced = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
        if referenced & tainted:
            tainted |= {t.id for t in node.targets if isinstance(t, ast.Name)}

    seal_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} & tainted
    ]
    assert seal_lines, (
        f"{rel_path}: the holdout boundary is bound but never compared against. The seal must "
        "be an applied filter — `.filter(pl.col('_label_end') < HOLDOUT_START)` or a narrowed "
        "end date — not merely a constant read from setup.yaml"
    )
    ic_line = source.count("\n", 0, ic_pos) + 1
    assert min(seal_lines) < ic_line, (
        f"{rel_path}: the holdout filter must be applied before the first IC "
        f"computation so selection decisions never see the sealed holdout "
        f"(earliest comparison at line {min(seal_lines)}, first IC at line {ic_line})"
    )


# Notebooks that read a label series spanning the FULL sample and must therefore
# purge on the label endpoint rather than the signal date. Filtering
# ``timestamp < holdout_start`` looks sealed and is not: the forward labels of
# the last ``horizon`` pre-holdout dates are computed from holdout prices, so
# the IC ranking and BH-FDR below still see the holdout.
#
# The sibling notebooks in ``08_financial_features`` are deliberately absent:
# they truncate the price frame first and compute forward returns on the
# truncated frame, so the boundary rows come out null and drop. That is a
# different, equally sound mechanism, and asserting this one against them would
# be wrong. Seven case studies have not been audited for this class of leak; add
# their notebooks here as each is checked.
# ``case_studies/etfs/03_financial_features.py`` was in these lists and is in
# none of them now: ``rules/stages/03-financial-features.md`` removes the IC
# screen from stage 03, so that notebook reads no label and has no endpoint to
# purge. Its own seal is a different check -- it rebuilds the panel with the
# holdout withheld and requires every emitted feature value to agree.
# ``case_studies/us_firm_characteristics/02_labels.py`` is deliberately absent, and it is the
# one case study where that is a property of the data rather than a gap. The Chen-Pelger-Zhu
# release pairs the characteristics observed at the close of month t-1 with the return earned
# over month t and dates the row by month t, so a row's label resolves on its own timestamp
# and the label endpoint IS the signal date -- the notebook derives it as such and filters on
# it. Shifting the timestamp forward by the horizon to satisfy the pattern below would purge a
# month that is already sealed and would drop every firm's final observation, whenever it
# fell. The alignment is measured rather than assumed: 02_labels correlates ``ST_REV``, the
# short-term-reversal characteristic, against the label at three candidate lags and only the
# previous row's return carries it.
LABEL_ENDPOINT_PURGED_NOTEBOOKS = [
    "case_studies/us_equities_panel/02_labels.py",
    "case_studies/etfs/02_labels.py",
    "case_studies/etfs/05_evaluation.py",
    "case_studies/crypto_perps_funding/02_labels.py",
    "case_studies/cme_futures/02_labels.py",
    "case_studies/fx_pairs/02_labels.py",
    "case_studies/sp500_equity_option_analytics/02_labels.py",
    "case_studies/sp500_options/02_labels.py",
    "case_studies/nasdaq100_microstructure/02_labels.py",
]

# Of the four above, only etfs/05 uses a market-wide calendar; the other three
# derive the endpoint PER SYMBOL. The two are equivalent exactly while no symbol misses a
# session inside the label window before the boundary, which is a property of the
# data and not of the code -- so it is executed, not asserted here:
# ``test_the_two_purges_agree_on_the_shipped_label_panel`` runs both filters over
# the label panel and requires the same rows, and
# ``test_a_gapped_calendar_separates_the_two_purges`` builds a panel where they
# disagree so that check cannot pass vacuously. 58 of the 99 ETFs do not trade
# every session across the full panel -- they list late -- so "the panel is
# dense" would be the wrong justification for the exemption; the tested
# equivalence at the boundary is the right one. Converting 05 to the per-symbol
# form regardless would remove the dependence on that data property; it is
# deferred because editing 05 forces the case study's evaluation to be
# re-executed on a newer feature vintage.
# ``entity_col`` names the column the endpoint must be shifted within, which is the
# entity a label may not cross. It is ``symbol`` for seven of the nine case studies and
# ``product`` for cme_futures.
# ``sp500_equity_option_analytics/02_labels`` is deliberately absent: its entity is the
# security (``sec_id``), not the ticker. A ticker there is reassigned to another company
# after a merger or a spin-off, and ``adj_factor`` restarts with the new security, so
# shifting within ``symbol`` would be the under-purge this list exists to prevent rather
# than the fix for it. It is in both lists above, where the mechanism is what is checked.
# ``nasdaq100_microstructure/02_labels`` is deliberately absent from this third list while
# being present in the two above. Its labels are intraday and may not cross an overnight
# gap, so the entity a label may not cross is the compound ``["symbol", "session_date"]``
# and its endpoint is derived with ``.over(GROUP_COLS)``. The regex below binds a single
# quoted column name, so it cannot express a compound key -- listing the notebook here
# would produce a false red against a seal that is strictly stronger than the per-symbol
# one this test checks.
# ``etfs/03_financial_features`` is deliberately absent. It was listed here while it
# purged on a shifted label endpoint; public #447 replaced that mechanism with a seal
# that rebuilds the whole panel with the holdout withheld and compares all 57 columns,
# which is checked by executing the notebook rather than by reading its source. A
# source pattern cannot see the stronger check, so listing it here only produces a
# false red. See agent-workspace #141.
# ``sp500_options/02_labels`` is deliberately absent, and for the opposite reason to
# ``etfs/05``: its endpoint is not an approximation that needs the per-symbol form. The
# hold-to-expiry label settles on the expiration written into the contract and the
# fixed-horizon variants on the exit session recorded in the round-trip artifact, so
# ``_label_end`` is the exact resolution date of each individual trade rather than a
# calendar shift of the signal date. The notebook checks the recorded exit dates against a
# shift of the panel calendar by the declared horizon, which is what
# ``test_holdout_purge_is_on_the_label_endpoint`` above matches on.
# ``us_equities_panel/02_labels`` is deliberately absent from this third list while being in
# the two above. Its endpoint is not shifted within ``symbol`` at all: the row is looked up at
# the session numbered one higher in that stock's own series, so the entity boundary is part of
# the join key rather than of a window function, and a stock that missed the closing session
# gets no endpoint instead of a later one. That is what this list's per-symbol requirement is
# for, reached by a construction the regex below cannot express.
PER_SYMBOL_ENDPOINT_NOTEBOOKS = [
    ("case_studies/etfs/02_labels.py", "symbol"),
    ("case_studies/crypto_perps_funding/02_labels.py", "symbol"),
    ("case_studies/cme_futures/02_labels.py", "product"),
    ("case_studies/fx_pairs/02_labels.py", "symbol"),
]


def test_us_equities_panel_03_restricts_the_evaluation_on_the_label_endpoint() -> None:
    """``us_equities_panel/03_financial_features`` needs its own check, not a list entry.

    It belongs in ``HOLDOUT_SCOPED_NOTEBOOKS`` and cannot join
    ``LABEL_ENDPOINT_PURGED_NOTEBOOKS``: the winsorization figure draws a
    counterfactual bound from development rows with
    ``filter(pl.col("timestamp") < HOLDOUT_START)``, which is a per-date quantile
    that crosses no boundary but which that list's leaky-filter regex would reject.

    Membership of the scoped list alone is a vacuous gate here, and this test exists
    because the review of the commit that added it said so: the winsorization
    comparison against ``HOLDOUT_START`` precedes the first IC computation on its own,
    so deleting the endpoint restriction entirely would leave that check green. What
    has to hold is the mechanism -- the endpoint comes from the label's own horizon,
    the evaluation frame is restricted on it, and that happens before any IC is
    computed.
    """
    rel_path = "case_studies/us_equities_panel/03_financial_features.py"
    source = (REPO_ROOT / rel_path).read_text()

    assert re.search(r"pl\.col\(\"session\"\)\s*-\s*horizon", source), (
        f"{rel_path}: the label endpoint must be looked up at the session numbered "
        "PRIMARY_HORIZON higher, as 02_labels writes it, not read off the next row"
    )
    assert "rows_sessions_ahead(raw_df, PRIMARY_HORIZON)" in source, (
        f"{rel_path}: the endpoint must be derived for the declared primary horizon, "
        "not for a bare integer that can drift from setup.yaml"
    )

    restriction = source.find('.filter(pl.col("_label_end") < HOLDOUT_START)')
    assert restriction != -1, (
        f"{rel_path}: the evaluation frame must be restricted to rows whose label "
        "window closes strictly before the holdout. A filter on the observation date "
        "reads holdout prices while appearing not to."
    )
    first_ic = source.find("ic_results[feat] = ")
    assert first_ic != -1, f"{rel_path}: first-IC marker not found -- update this test"
    assert restriction < first_ic, (
        f"{rel_path}: the endpoint restriction must be applied before the first IC "
        f"computation (restriction at char {restriction}, first IC at {first_ic})"
    )


@pytest.mark.parametrize("rel_path", LABEL_ENDPOINT_PURGED_NOTEBOOKS, ids=lambda p: p)
def test_holdout_purge_is_on_the_label_endpoint(rel_path: str) -> None:
    """Regression gate for the 2026-07-21 revert.

    ``test_holdout_filter_precedes_first_ic_computation`` passes on a bare
    signal-date filter -- it only checks that the string ``holdout_start``
    appears before the first IC computation. That is why the seal could be
    removed from ``03_financial_features`` and stay green for five days. This
    test checks the mechanism instead: the label endpoint is derived by shifting
    the calendar by the label horizon, and the evaluation frame is filtered on
    that endpoint.
    """
    source = (REPO_ROOT / rel_path).read_text()

    assert "_label_end" in source, (
        f"{rel_path}: no label-endpoint column. Purge on the endpoint of the "
        "forward-label window (shift the dense calendar by the label horizon), "
        "not on the signal date -- see case_studies/etfs/05_evaluation.py"
    )
    # Two mechanisms move a row to its label's endpoint by the declared horizon, and the
    # second is the stronger one. Shifting rows is correct only where the entity trades every
    # session inside the window; looking the row up at the session numbered `horizon` higher
    # is correct whether it does or not, and returns nothing rather than a later date where
    # the entity missed that session. ``us_equities_panel/02_labels`` uses the second after
    # agent-workspace #218, so the pattern accepts either -- what it still refuses is a bare
    # integer, which can drift from the horizon declared in setup.yaml.
    assert re.search(
        r"\.shift\(-\s*[A-Za-z_]*horizon|pl\.col\(\"session\"\)\s*-\s*[A-Za-z_]*horizon",
        source,
        re.IGNORECASE,
    ), (
        f"{rel_path}: the label endpoint must be moved by the declared label horizon - by a "
        "shift of the entity's rows or by a lookup on the session counter - and not by a bare "
        "integer that can drift from the label config"
    )

    # The endpoint must actually gate a frame, not merely be computed.
    assert re.search(r"filter\(\s*pl\.col\(\"_label_end\"\)\s*<", source) or re.search(
        r"filter\(\s*pl\.col\(\"timestamp\"\)\s*<=\s*last_signal_date", source
    ), (
        f"{rel_path}: the label endpoint is computed but never filtered on. "
        "The evaluation frame must be restricted to signal dates whose label "
        "window closes strictly before the holdout."
    )

    # ...and the leaking filter must be gone. Without this, deriving
    # `last_signal_date` and then never using it satisfies the assertion above,
    # which is exactly the state the 2026-07-21 revert would be restored to.
    leaky = re.search(
        r"filter\(\s*pl\.col\(\"timestamp\"\)\s*<=?\s*(?:date\.fromisoformat\()?"
        r"[A-Za-z_]*holdout_start",
        source,
        re.IGNORECASE,
    )
    assert leaky is None, (
        f"{rel_path}: found a signal-date filter against the holdout boundary "
        f"({leaky.group(0) if leaky else ''!r}). That filter looks sealed and is "
        "not -- the forward labels of the last `horizon` dates it keeps are "
        "computed from holdout prices. Filter on the label endpoint instead."
    )


def test_selected_features_artifact_stops_before_holdout() -> None:
    """If the Ch8 selection artifact has been regenerated locally, its feature
    panel must not extend into the sealed holdout."""
    pl = pytest.importorskip("polars")

    # Production location of get_output_dir(8, "feature_selection") from
    # 05_feature_selection.py: {chapter_dir}/output/{strategy_id}/.
    artifact = (
        REPO_ROOT
        / "08_financial_features"
        / "output"
        / "feature_selection"
        / "features_selected.parquet"
    )
    if not artifact.exists():
        pytest.skip("selection artifact not generated locally (requires full data)")

    holdout_start = date.fromisoformat(load_holdout_start())
    max_ts = pl.scan_parquet(artifact).select(pl.col("timestamp").max()).collect().item()
    assert max_ts is not None
    max_date = max_ts.date() if hasattr(max_ts, "date") else max_ts
    assert max_date < holdout_start, (
        f"features_selected.parquet extends to {max_date}, at/after the sealed "
        f"holdout start {holdout_start} — feature selection saw holdout data"
    )


def _declared_parameters(notebook: str) -> dict:
    """Return the literal parameter assignments from a notebook's `parameters` cell.

    Parsed rather than matched as text. These notebooks moved onto the shared research boundary,
    where the device policy is one entry in an ``OVERRIDES`` mapping instead of a ``TRAIN_DEVICE``
    module constant, and a substring assertion on the old spelling passes or fails on how the line
    is written rather than on what it declares.
    """
    path = REPO_ROOT / "case_studies" / "crypto_perps_funding" / f"{notebook}.py"
    declared: dict = {}
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
            continue
        try:
            declared[node.targets[0].id] = ast.literal_eval(node.value)
        except ValueError:
            continue
    return declared


def _plans_through_shared_boundary(notebook: str) -> bool:
    """True when the notebook builds its run through the shared boundary rather than by hand."""
    path = REPO_ROOT / "case_studies" / "crypto_perps_funding" / f"{notebook}.py"
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "research_workflow" in node.module:
            if {"plan_model_catalog", "run_model_plan"} & {a.name for a in node.names}:
                return True
        # the causal notebook takes the other entry point on the same boundary
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "causal"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "study"
        ):
            return True
    return False


def _guards_executed_identity(notebook: str) -> bool:
    """True when the notebook refuses a result whose spec differs from the one it resolved."""
    path = REPO_ROOT / "case_studies" / "crypto_perps_funding" / f"{notebook}.py"
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Compare) or not isinstance(node.ops[0], ast.NotEq):
            continue
        rendered = ast.unparse(node)
        if "spec" in rendered and rendered.count(".spec") == 2:
            return True
    return False


def test_crypto_dml_plans_through_the_boundary_that_seals_the_holdout() -> None:
    """The DML run must take its sample and its identity from the shared boundary.

    It used to assert five identifiers in this notebook's source - `ESTIMATOR_CONTRACT`,
    `modeling_input_fingerprint`, `INPUT_FINGERPRINT`, and two inline holdout comparisons. None of
    them exists here any more: the notebook plans through `plan_model_catalog` / `run_model_plan`,
    and the fold geometry and holdout seal those five lines hand-rolled are now the boundary's,
    exercised by the fold tests above against the resolved splits rather than against the text.

    What still has to be true of this notebook is that it does not do any of it itself.
    """
    assert _plans_through_shared_boundary("11_causal_dml")

    declared = _declared_parameters("11_causal_dml")
    assert declared["EXECUTION_TIER"] == "canonical"
    assert declared["PREVIEW_REDUCTIONS"] == {}, (
        "a committed notebook must declare no reductions; a preview supplies them"
    )
    assert _guards_executed_identity("11_causal_dml"), (
        "the notebook must refuse a result whose spec differs from the one it resolved - the "
        "identity binding the removed input-fingerprint assertions stood for"
    )


def test_crypto_folds_purge_the_24h_buffer_the_variant_label_configures() -> None:
    """``fwd_ret_24h`` folds leave 24 hours between training and validation.

    The guard this replaces read the purge out of the notebook's own source. Both
    model notebooks now take their folds from the shared boundary, so the property
    is checked where it is decided: the configured buffer must reach
    ``generate_cv_splits``, and no fold may see a label whose outcome window
    reaches into the sealed holdout.
    """
    import pandas as pd
    import polars as pl

    from utils.cv_splits import generate_cv_splits
    from utils.modeling import resolve_label_buffer

    case_study = "crypto_perps_funding"
    root = REPO_ROOT / "case_studies" / case_study
    setup = yaml.safe_load((root / "config" / "setup.yaml").read_text())
    buffer = resolve_label_buffer(case_study, "fwd_ret_24h", setup)
    assert buffer == "24H"

    holdout_start = pd.Timestamp(setup["evaluation"]["holdout_start"])
    horizon = pd.Timedelta(buffer.lower())  # pandas deprecated the capital unit
    timeline = pl.DataFrame(
        {"timestamp": pd.date_range("2019-01-01", holdout_start, freq="8h", inclusive="left")}
    )
    splits = generate_cv_splits(
        timeline,
        case_study_id=case_study,
        label_buffer=buffer,
        outcome_horizon=buffer,
    )
    assert splits

    for split in splits:
        train_end = pd.Timestamp(split["train_end"])
        val_start = pd.Timestamp(split["val_start"])
        val_end = pd.Timestamp(split["val_end"])
        assert val_start - train_end >= horizon, (
            f"fold {split['fold']} leaves {val_start - train_end} between training and "
            f"validation, less than the {buffer} the label needs"
        )
        assert val_end + horizon <= holdout_start, (
            f"fold {split['fold']} validates to {val_end}, whose 24-hour outcome window "
            f"reaches past the sealed holdout start {holdout_start.date()}"
        )


def test_crypto_tabm_requires_cuda_and_plans_through_the_boundary() -> None:
    """The TABM grid must be GPU-only, and must say so where the runner reads it.

    The device policy moved from a `TRAIN_DEVICE` module constant into the `OVERRIDES` mapping the
    shared boundary resolves, so this reads the declaration rather than the spelling of the line.
    """
    declared = _declared_parameters("08_tabular_dl")
    assert declared["OVERRIDES"]["device"] == "cuda"
    assert declared["OVERRIDES"]["class_weight"] == "balanced"
    assert _plans_through_shared_boundary("08_tabular_dl")


def test_crypto_lstm_requires_cuda_and_plans_through_the_boundary() -> None:
    """The LSTM grid must be GPU-only, and must say so where the runner reads it.

    The device policy moved from a `TRAIN_DEVICE` module constant into the `OVERRIDES` mapping the
    shared boundary resolves, so this reads the declaration rather than the spelling of the line.
    """
    declared = _declared_parameters("09_dl_lstm")
    assert declared["OVERRIDES"]["device"] == "cuda"
    assert _plans_through_shared_boundary("09_dl_lstm")


def test_crypto_tcn_requires_cuda_and_plans_through_the_boundary() -> None:
    """The TCN grid must be GPU-only, and must say so where the runner reads it.

    The device policy moved from a `TRAIN_DEVICE` module constant into the `OVERRIDES` mapping the
    shared boundary resolves, so this reads the declaration rather than the spelling of the line.
    """
    declared = _declared_parameters("10_dl_tcn")
    assert declared["OVERRIDES"]["device"] == "cuda"
    assert _plans_through_shared_boundary("10_dl_tcn")


@pytest.mark.parametrize(
    ("rel_path", "entity_col"),
    PER_SYMBOL_ENDPOINT_NOTEBOOKS,
    ids=[rel_path for rel_path, _ in PER_SYMBOL_ENDPOINT_NOTEBOOKS],
)
def test_label_endpoint_is_derived_per_symbol(rel_path: str, entity_col: str) -> None:
    """The endpoint must be shifted within symbol, as the label generator does.

    A market-wide cutoff is only equivalent while every symbol trades every
    session. A symbol that misses one would have its own label window end later
    than the market-wide endpoint, and reach into the holdout.
    """
    source = (REPO_ROOT / rel_path).read_text()
    # The match must bind to `_label_end`. Without that anchor the label
    # generator's own `close.shift(-horizon).over("symbol")` in 02_labels
    # satisfies the pattern, so the endpoint derivation could be deleted and
    # this test would still pass -- the same vacuous-gate failure that let the
    # 2026-07-21 revert stay green.
    assert re.search(
        r"\.shift\(-\s*[A-Za-z_]*horizon\s*\)\s*\.over\(\s*\"" + entity_col + r"\"\s*\)"
        r"\s*\.alias\(\s*\"_label_end\"\s*\)",
        source,
        re.IGNORECASE,
    ), (
        f"{rel_path}: the label endpoint must be derived as shift(-horizon)"
        f'.over("{entity_col}").alias("_label_end"), matching 02_labels; a market-wide '
        "cutoff under-purges an entity with a gapped calendar"
    )


# The two purges, transcribed from the notebooks that run them, so the
# equivalence the 05 exemption rests on is executable.


def _kept_market_wide(label_df, horizon: int, holdout_start: date) -> set[tuple]:
    """``05_evaluation``: one calendar for the whole market, then a date cutoff."""
    import polars as pl

    last_signal_date = (
        label_df.select("timestamp")
        .unique()
        .sort("timestamp")
        .with_columns(pl.col("timestamp").shift(-horizon).alias("_label_end"))
        .filter(pl.col("_label_end") < holdout_start)["timestamp"]
        .max()
    )
    kept = label_df.filter(pl.col("timestamp") <= last_signal_date)
    return set(zip(kept["timestamp"].to_list(), kept["symbol"].to_list()))


def _kept_per_symbol(label_df, horizon: int, holdout_start: date) -> set[tuple]:
    """``03_financial_features``: each symbol's own label window."""
    import polars as pl

    kept = (
        label_df.sort(["symbol", "timestamp"])
        .with_columns(pl.col("timestamp").shift(-horizon).over("symbol").alias("_label_end"))
        .filter(pl.col("_label_end") < holdout_start)
    )
    return set(zip(kept["timestamp"].to_list(), kept["symbol"].to_list()))


def test_a_gapped_calendar_separates_the_two_purges() -> None:
    """Without this, the equivalence test below could pass on any panel at all.

    ``GAPPED`` stops trading four sessions before the boundary and resumes inside
    the holdout, so its label window from 01-02 closes on 01-09 -- inside. The
    market-wide cutoff keeps that row because the market's own calendar closes
    01-02's window on 01-05.
    """
    pl = pytest.importorskip("polars")
    holdout_start = date(2020, 1, 8)
    horizon = 3
    dense = [date(2020, 1, day) for day in range(1, 11)]
    gapped = [date(2020, 1, day) for day in (1, 2, 3, 4, 9, 10)]
    label_df = pl.DataFrame(
        {
            "timestamp": dense + gapped,
            "symbol": ["DENSE"] * len(dense) + ["GAPPED"] * len(gapped),
        }
    )

    market_wide = _kept_market_wide(label_df, horizon, holdout_start)
    per_symbol = _kept_per_symbol(label_df, horizon, holdout_start)

    assert per_symbol < market_wide, (
        "the gapped calendar must make the two purges disagree, or the "
        "equivalence check on the real panel proves nothing"
    )
    under_purged = market_wide - per_symbol
    assert (date(2020, 1, 2), "GAPPED") in under_purged
    # Every row the market-wide cutoff keeps and the per-symbol purge drops has
    # its own label window closing at or after the boundary: that is the leak.
    with_endpoints = label_df.sort(["symbol", "timestamp"]).with_columns(
        pl.col("timestamp").shift(-horizon).over("symbol").alias("_label_end")
    )
    endpoints = {
        (timestamp, symbol): endpoint for timestamp, symbol, endpoint in with_endpoints.iter_rows()
    }
    for row in under_purged:
        endpoint = endpoints[row]
        assert endpoint is None or endpoint >= holdout_start


def evaluation_notebook_label() -> tuple[str, int]:
    """The label file and horizon ``05_evaluation`` actually uses.

    05 used to hardcode both, and this helper read the two literals so the
    equivalence check below ran on the notebook's values rather than the
    configured ones. It no longer hardcodes them: it takes the label from
    ``setup.yaml`` and the horizon from ``resolve_label_buffer``, so there is one
    source of truth and a config change moves the notebook with it.

    What has to be checked therefore moves too. The risk was 05 purging on a
    label the case study no longer selects; that is now prevented by
    construction, but only while 05 really does bind both to the config. So this
    asserts the binding structurally and resolves the values the same way the
    notebook does, and the caller keeps its teeth by comparing the
    buffer-derived horizon against the one implied by the label's own name --
    two independent config entries that a typo can still separate.
    """
    tree = ast.parse((REPO_ROOT / "case_studies" / "etfs" / "05_evaluation.py").read_text())

    # Parsed rather than pattern-matched: a text match anywhere in the file is
    # satisfied by a call the label frame is not built from.
    bindings: dict[str, ast.expr] = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            bindings[target.id] = node.value

    def _calls_named(node: ast.AST, func: str) -> bool:
        return any(
            isinstance(sub, ast.Call)
            and (
                (isinstance(sub.func, ast.Name) and sub.func.id == func)
                or (isinstance(sub.func, ast.Attribute) and sub.func.attr == func)
            )
            for sub in ast.walk(node)
        )

    missing = {"PRIMARY_LABEL", "HAC_MAXLAGS", "LABEL_HORIZON"} - set(bindings)
    assert not missing, (
        "05_evaluation no longer binds "
        f"{sorted(missing)}; read its label and horizon from wherever it now "
        "takes them"
    )

    # The label must come from setup.yaml, not from a literal reintroduced here.
    primary_src = ast.dump(bindings["PRIMARY_LABEL"])
    assert "'labels'" in primary_src and "'primary'" in primary_src, (
        "05_evaluation's PRIMARY_LABEL is no longer read from "
        "SETUP['labels']['primary'], so it can now name a label setup.yaml does "
        "not select"
    )
    assert not isinstance(bindings["PRIMARY_LABEL"], ast.Constant), (
        "05_evaluation hardcodes PRIMARY_LABEL again; it must be read from setup.yaml"
    )

    # ...and the HAC bandwidth from the label's configured buffer.
    assert _calls_named(bindings["HAC_MAXLAGS"], "match") and "LABEL_BUFFER" in {
        sub.id for sub in ast.walk(bindings["HAC_MAXLAGS"]) if isinstance(sub, ast.Name)
    }, (
        "05_evaluation's HAC_MAXLAGS is no longer derived from LABEL_BUFFER, so "
        "the HAC correction and the label can drift apart"
    )
    assert _calls_named(bindings.get("LABEL_BUFFER", ast.Constant(None)), "resolve_label_buffer"), (
        "05_evaluation no longer resolves LABEL_BUFFER through resolve_label_buffer"
    )

    horizon_alias = bindings["LABEL_HORIZON"]
    assert isinstance(horizon_alias, ast.Name) and horizon_alias.id == "HAC_MAXLAGS", (
        "05_evaluation's purge horizon is no longer HAC_MAXLAGS itself, so this "
        "helper would report a horizon the notebook does not use"
    )

    def calls_to(node: ast.AST, method: str) -> list[ast.Call]:
        return [
            sub
            for sub in ast.walk(node)
            if isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == method
        ]

    def names_in(node: ast.AST) -> set[str]:
        return {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)}

    # The label frame must be read through the constant, in every statement that
    # builds it -- not merely somewhere in the file.
    label_reads = [
        read
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "label_df" for t in node.targets)
        for read in calls_to(node.value, "read_parquet")
    ]
    assert label_reads and all("PRIMARY_LABEL" in names_in(read) for read in label_reads), (
        "05_evaluation must load label_df through PRIMARY_LABEL; a hardcoded "
        "path there would leave the constant, and this check, describing a file "
        "the notebook does not read"
    )

    # ...and the endpoint it purges on must be that constant's shift.
    endpoint_aliases = [
        call
        for call in calls_to(tree, "alias")
        if call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "_label_end"
    ]
    assert endpoint_aliases, "05_evaluation no longer derives a _label_end column"
    for alias in endpoint_aliases:
        assert isinstance(alias.func, ast.Attribute)
        shifts = calls_to(alias.func.value, "shift")
        assert shifts and all("LABEL_HORIZON" in names_in(shift) for shift in shifts), (
            "05_evaluation must shift the calendar by LABEL_HORIZON to derive "
            "_label_end; another horizon there purges a window this check is not "
            "measuring"
        )
    # Resolved through the same function 05 calls, so a label spec that overrides
    # setup.yaml moves this check with the notebook rather than silently past it.
    #
    # The setup mapping is read from SETUP_YAML rather than through
    # `load_setup_config`, which resolves the case-study directory from the
    # environment: under CI's ML4T_OUTPUT_DIR it returns a different file and the
    # helper died on `KeyError: 'labels'`. This test is a static check on the
    # committed source, so it must read the committed config.
    from utils.artifact_specs import resolve_label_buffer

    setup = yaml.safe_load(SETUP_YAML.read_text())
    primary = str(setup["labels"]["primary"])
    buffer = resolve_label_buffer("etfs", primary, setup)
    assert buffer, f"no label buffer configured for {primary}"
    match = re.match(r"^(\d+)", str(buffer))
    assert match, f"label buffer {buffer!r} does not start with an integer"
    return f"{primary}.parquet", int(match.group(1))


def test_the_two_purges_agree_on_the_shipped_label_panel() -> None:
    """The 05 exemption in ``PER_SYMBOL_ENDPOINT_NOTEBOOKS``, executed.

    05 uses a market-wide cutoff, which under-purges a symbol whose calendar has
    a gap inside the label window before the boundary. Whether any symbol does is
    a property of the label panel, so check the panel rather than reasoning about
    it. Skips where the labels have not been generated; runs wherever they have.
    """
    pl = pytest.importorskip("polars")
    setup = yaml.safe_load(SETUP_YAML.read_text())
    label_col = setup["labels"]["primary"]
    configured_horizon = int(label_col.rsplit("_", 1)[-1].removesuffix("d"))

    # Run on what 05 reads, and require that to be what setup.yaml declares.
    label_file, horizon = evaluation_notebook_label()
    assert (label_file, horizon) == (f"{label_col}.parquet", configured_horizon), (
        f"05_evaluation evaluates {label_file} at horizon {horizon} while "
        f"setup.yaml declares {label_col} at horizon {configured_horizon}. Its "
        "purge is sealing a label the case study no longer selects; update 05 "
        "before relying on the equivalence below."
    )

    artifact = REPO_ROOT / "case_studies" / "etfs" / "labels" / label_file
    if not artifact.exists():
        pytest.skip(f"{artifact.relative_to(REPO_ROOT)} not generated locally")

    label_df = pl.read_parquet(artifact).select(["timestamp", "symbol"])
    holdout_start = date.fromisoformat(setup["evaluation"]["holdout_start"])

    market_wide = _kept_market_wide(label_df, horizon, holdout_start)
    per_symbol = _kept_per_symbol(label_df, horizon, holdout_start)

    under_purged = sorted(market_wide - per_symbol)[:5]
    assert market_wide == per_symbol, (
        f"05_evaluation's market-wide cutoff and 03's per-symbol purge no longer "
        f"keep the same rows on {label_col}: "
        f"{len(market_wide - per_symbol)} rows kept only by the market-wide "
        f"cutoff (e.g. {under_purged}), {len(per_symbol - market_wide)} only by "
        "the per-symbol purge. The exemption of 05_evaluation.py from "
        "PER_SYMBOL_ENDPOINT_NOTEBOOKS rested on that equivalence, so convert 05 "
        "to the per-symbol form -- derive _label_end with "
        '.shift(-LABEL_HORIZON).over("symbol") as 02 and 03 do.'
    )


def test_the_label_and_horizon_are_resolved_from_the_configured_primary_label() -> None:
    """The label and its horizon come from one config read, never from a literal.

    A hardcoded 21 works until ``labels.primary`` changes, and then every window
    that depends on the decision cycle is silently wrong for the label in force.
    """
    source = (REPO_ROOT / "case_studies" / "etfs" / "03_financial_features.py").read_text()

    assert 'setup["labels"]["primary"]' in source, (
        "03_financial_features must take the label column from setup.yaml "
        "labels.primary, not from a filename literal"
    )
    hardcoded = re.search(r"^LABEL_HORIZON\s*=\s*\d+", source, re.MULTILINE)
    assert hardcoded is None, (
        f"03_financial_features hardcodes the label horizon ({hardcoded.group(0) if hardcoded else ''}). "
        "Derive it from the configured primary label instead."
    )
    assert re.search(r'read_parquet\([^)]*"fwd_ret_\d+d\.parquet"', source) is None, (
        "03_financial_features names a label file literally; read the configured "
        "primary label's parquet instead"
    )
    # Two further assertions used to live here, requiring the resolved horizon to
    # reach the endpoint purge and the Newey-West lag. Both belonged to the stage-03
    # IC screen, which ``rules/stages/03-financial-features.md`` removes: the horizon
    # is still resolved from the configured label, but now the only thing that reads
    # it is the persistence figure, which has to look at least one decision cycle out.
    assert "resolve_label_horizon(" in source, (
        "the decision cycle must be resolved from the configured primary label"
    )


def test_the_narrative_states_the_configured_horizon() -> None:
    """The prose names a horizon in days; it has to be the configured one, or the
    notebook explains one label while computing another."""
    setup = yaml.safe_load(SETUP_YAML.read_text())
    horizon = int(setup["labels"]["primary"].rsplit("_", 1)[-1].removesuffix("d"))
    source = (REPO_ROOT / "case_studies" / "etfs" / "03_financial_features.py").read_text()

    stated = {int(n) for n in re.findall(r"(\d+)-day (?:forward|label|return)", source)}
    assert stated <= {horizon}, (
        f"03_financial_features describes {sorted(stated - {horizon})}-day labels "
        f"while setup.yaml configures a {horizon}-day horizon "
        f"({setup['labels']['primary']})"
    )


def test_the_holdout_fold_trains_on_everything_before_the_seal(tmp_path, monkeypatch) -> None:
    """The holdout retrain is the one fit the sealed holdout ever sees, so the window it
    trains on has to be everything available before the seal.

    ``generate_cv_splits`` steps backward from the holdout boundary: fold 0 is the most
    recent fold and carries the *latest* training start, and the list runs newest to
    oldest. ``append_holdout_fold_if_needed`` took ``splits[0]["train_start"]``, which is
    the shortest window of the set, not the longest. On etfs that is 2008-01-02 against an
    earliest fold start of 2005-01-03 - three years dropped from the retrain, silently,
    and only on the holdout path where nothing else would show it.
    """
    import pandas as pd

    import utils.modeling as modeling
    from utils.modeling import ModelingDataset, append_holdout_fold_if_needed

    case_dir = tmp_path / "cs"
    (case_dir / "config").mkdir(parents=True)
    (case_dir / "config" / "setup.yaml").write_text(
        yaml.safe_dump({"evaluation": {"holdout_start": "2018-01-02", "holdout_end": "2018-12-31"}})
    )
    monkeypatch.setattr(modeling, "get_case_study_dir", lambda _cs: case_dir)

    # Newest fold first, as generate_cv_splits emits them.
    splits = [
        {
            "fold": i,
            "train_start": pd.Timestamp(f"{2008 - i}-01-02"),
            "train_end": pd.Timestamp(f"{2017 - i}-11-29"),
            "val_start": pd.Timestamp(f"{2017 - i}-12-29"),
            "val_end": pd.Timestamp(f"{2018 - i}-11-29"),
        }
        for i in range(3)
    ]
    mds = ModelingDataset.__new__(ModelingDataset)
    mds.splits = splits
    mds._input_lineage = object()

    append_holdout_fold_if_needed(mds, "holdout", "whatever")

    holdout = mds.splits[-1]
    assert holdout["fold"] == 3
    assert holdout["train_start"] == pd.Timestamp("2006-01-02"), (
        "the holdout fold must start at the earliest train_start across folds, "
        f"not splits[0]'s {splits[0]['train_start']}"
    )
    assert holdout["train_end"] == pd.Timestamp("2018-01-02")
    assert mds._input_lineage is None, "the memoized lineage describes the pre-append fold set"


def test_the_holdout_fold_covers_every_bar_of_the_final_configured_session(
    tmp_path, monkeypatch
) -> None:
    """An intraday panel loses its whole last session to a midnight upper bound.

    ``holdout_end`` is configured as a date. Parsed it is that date at midnight,
    and every fold filter here is ``timestamp <= val_end``, so on minute bars the
    entire final session sorts after the bound and is written but never scored.
    Measured on nasdaq100_microstructure: the producer side was fixed in
    ``04_model_based_features`` and the holdout fold went 19,800,687 to 19,839,297
    rows, none of which the consumer could read.
    """
    import pandas as pd

    import utils.modeling as modeling
    from utils.modeling import ModelingDataset, append_holdout_fold_if_needed

    case_dir = tmp_path / "cs"
    (case_dir / "config").mkdir(parents=True)
    (case_dir / "config" / "setup.yaml").write_text(
        yaml.safe_dump({"evaluation": {"holdout_start": "2021-01-04", "holdout_end": "2021-12-31"}})
    )
    monkeypatch.setattr(modeling, "get_case_study_dir", lambda _cs: case_dir)

    mds = ModelingDataset.__new__(ModelingDataset)
    mds.splits = [
        {
            "fold": 0,
            "train_start": pd.Timestamp("2020-01-02 09:30"),
            "train_end": pd.Timestamp("2020-12-30 15:58"),
            "val_start": pd.Timestamp("2020-12-31 09:32"),
            "val_end": pd.Timestamp("2020-12-31 15:58"),
        }
    ]
    mds._input_lineage = object()

    append_holdout_fold_if_needed(mds, "holdout", "whatever")
    holdout = mds.splits[-1]

    # The decision minutes of the final configured session, as an intraday panel
    # carries them. Not one of these is at midnight.
    final_session = pd.date_range("2021-12-31 09:32", "2021-12-31 15:58", freq="2min")
    in_window = (final_session >= holdout["val_start"]) & (final_session <= holdout["val_end"])
    assert in_window.all(), (
        f"{(~in_window).sum()} of {len(final_session)} bars of the final configured "
        f"session fall outside [{holdout['val_start']}, {holdout['val_end']}]"
    )
    # And it stops there: the session after the configured end stays out.
    next_session = pd.Timestamp("2022-01-03 09:32")
    assert next_session > holdout["val_end"]


def test_a_holdout_end_naming_an_instant_is_taken_literally() -> None:
    """A config that names a time of day means that time, not the end of the day."""
    import pandas as pd

    from utils.modeling import _inclusive_end_of

    assert _inclusive_end_of("2021-12-31 15:58") == pd.Timestamp("2021-12-31 15:58")
    assert _inclusive_end_of("2021-12-31") > pd.Timestamp("2021-12-31 23:59:59")
    assert _inclusive_end_of("2021-12-31") < pd.Timestamp("2022-01-01")

    # An explicitly configured midnight is an instant, not a day. It parses to the
    # same Timestamp as the bare date, so the configured string is what decides.
    assert _inclusive_end_of("2021-12-31 00:00:00") == pd.Timestamp("2021-12-31")
