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
    ("case_studies/etfs/03_financial_features.py", "ic_matrix = np.full"),
    ("case_studies/etfs/02_labels.py", "def compute_ic_by_date"),
    ("case_studies/etfs/05_evaluation.py", "ic_series_data = {feat"),
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

    filter_pos = source.upper().find("HOLDOUT_START")
    assert 0 <= filter_pos < ic_pos, (
        f"{rel_path}: the holdout filter must be applied before the first IC "
        "computation so selection decisions never see the sealed holdout"
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
# be wrong. Widening the audit to the other eight case studies is tracked in
# ``issues/2026-07-18-evaluation-holdout-leak-sibling-sweep``.
LABEL_ENDPOINT_PURGED_NOTEBOOKS = [
    "case_studies/etfs/02_labels.py",
    "case_studies/etfs/03_financial_features.py",
    "case_studies/etfs/05_evaluation.py",
]

# Of the three above, 02 and 03 derive the endpoint PER SYMBOL; 05 still uses a
# market-wide calendar. The two are equivalent exactly while no symbol misses a
# session inside the label window before the boundary, which is a property of the
# data and not of the code -- so it is executed, not asserted here:
# ``test_the_two_purges_agree_on_the_shipped_label_panel`` runs both filters over
# the label panel and requires the same rows, and
# ``test_a_gapped_calendar_separates_the_two_purges`` builds a panel where they
# disagree so that check cannot pass vacuously. 58 of the 99 ETFs do not trade
# every session across the full panel -- they list late -- so "the panel is
# dense" would be the wrong justification for the exemption; the tested
# equivalence at the boundary is the right one. Converting 05 to the per-symbol
# form regardless is tracked in
# ``issues/2026-07-18-evaluation-holdout-leak-sibling-sweep`` §3.
PER_SYMBOL_ENDPOINT_NOTEBOOKS = [
    "case_studies/etfs/02_labels.py",
    "case_studies/etfs/03_financial_features.py",
]


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
    assert re.search(r"\.shift\(-\s*[A-Za-z_]*horizon", source, re.IGNORECASE), (
        f"{rel_path}: the label endpoint must be shifted by the declared label "
        "horizon, not by a bare integer that can drift from the label config"
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


def test_crypto_dml_seals_the_label_endpoint_and_hashes_current_inputs() -> None:
    """The DML sample and cache identity must bind the corrected 8-hour lineage."""
    source = (REPO_ROOT / "case_studies" / "crypto_perps_funding" / "11_causal_dml.py").read_text()

    assert 'ESTIMATOR_CONTRACT = "panel_time_v3"' in source
    assert "modeling_input_fingerprint(" in source
    assert '"input_fingerprint": INPUT_FINGERPRINT' in source
    assert "holdout_cutoff - label_horizon" in source
    assert "max() + LABEL_HORIZON >= HOLDOUT_CUTOFF" in source


def test_crypto_gbm_requires_cuda_and_hashes_current_inputs() -> None:
    """The corrected GBM grid must be GPU-only and content-addressed."""
    source = (REPO_ROOT / "case_studies" / "crypto_perps_funding" / "07_gbm.py").read_text()

    assert 'TRAIN_DEVICE = "cuda"' in source
    assert "modeling_input_fingerprint(" in source
    assert '"device": TRAIN_DEVICE' in source
    assert '"input_fingerprint": INPUT_FINGERPRINT' in source
    assert source.count("extra_params=IDENTITY_PARAMS") == 2
    assert "CPU fallback" not in source


@pytest.mark.parametrize("notebook", ["06_linear.py", "07_gbm.py"])
def test_crypto_model_guard_uses_active_24h_label_buffer(notebook: str) -> None:
    """Crypto model guards must purge the endpoint for the selected label horizon."""
    source = (REPO_ROOT / "case_studies" / "crypto_perps_funding" / notebook).read_text()
    setup = yaml.safe_load(
        (REPO_ROOT / "case_studies/crypto_perps_funding/config/setup.yaml").read_text()
    )

    assert setup["labels"]["variant_buffers"]["fwd_ret_24h"] == "24H"
    assert "pd.Timedelta(mds.label_buffer)" in source
    assert 'split["val_end"] + label_horizon < holdout_start' in source
    assert 'split["val_end"] + timedelta(hours=8)' not in source


def test_crypto_tabm_requires_cuda_and_hashes_current_inputs() -> None:
    """The corrected TabM grid must be GPU-only and content-addressed."""
    source = (REPO_ROOT / "case_studies" / "crypto_perps_funding" / "08_tabular_dl.py").read_text()

    assert 'TRAIN_DEVICE = "cuda"' in source
    assert "modeling_input_fingerprint(" in source
    assert '"device": TRAIN_DEVICE' in source
    assert '"input_fingerprint": INPUT_FINGERPRINT' in source
    assert source.count("extra_params=IDENTITY_PARAMS") == 2
    assert "identity_params=IDENTITY_PARAMS" in source
    assert "current CPU" not in source


def test_crypto_lstm_requires_cuda_and_hashes_current_inputs() -> None:
    """The corrected LSTM run must be GPU-only and content-addressed."""
    source = (REPO_ROOT / "case_studies" / "crypto_perps_funding" / "09_dl_lstm.py").read_text()

    assert 'TRAIN_DEVICE = "cuda"' in source
    assert "modeling_input_fingerprint(" in source
    assert '"device": TRAIN_DEVICE' in source
    assert '"input_fingerprint": INPUT_FINGERPRINT' in source
    assert "extra_params=IDENTITY_PARAMS" in source
    assert "identity_params=IDENTITY_PARAMS" in source
    assert "current CPU" not in source


def test_crypto_tcn_requires_cuda_and_hashes_current_inputs() -> None:
    """The corrected TCN run must be GPU-only and content-addressed."""
    source = (REPO_ROOT / "case_studies" / "crypto_perps_funding" / "10_dl_tcn.py").read_text()

    assert 'TRAIN_DEVICE = "cuda"' in source
    assert "modeling_input_fingerprint(" in source
    assert '"device": TRAIN_DEVICE' in source
    assert '"input_fingerprint": INPUT_FINGERPRINT' in source
    assert source.count("extra_params=IDENTITY_PARAMS") == 2
    assert "identity_params=IDENTITY_PARAMS" in source
    assert "current CPU" not in source


@pytest.mark.parametrize("rel_path", PER_SYMBOL_ENDPOINT_NOTEBOOKS, ids=lambda p: p)
def test_label_endpoint_is_derived_per_symbol(rel_path: str) -> None:
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
        r"\.shift\(-\s*[A-Za-z_]*horizon\s*\)\s*\.over\(\s*\"symbol\"\s*\)"
        r"\s*\.alias\(\s*\"_label_end\"\s*\)",
        source,
        re.IGNORECASE,
    ), (
        f"{rel_path}: the label endpoint must be derived as shift(-horizon)"
        '.over("symbol").alias("_label_end"), matching 02_labels; a market-wide '
        "cutoff under-purges a symbol with a gapped calendar"
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

    05 hardcodes both, so the equivalence check has to run on its values rather
    than the configured ones -- otherwise a config change would leave the check
    validating a purge the notebook does not perform. Requiring them to equal the
    configured label is what makes the hardcoding safe, and is the assertion that
    fails if `labels.primary` moves and 05 is not moved with it.
    """
    source = (REPO_ROOT / "case_studies" / "etfs" / "05_evaluation.py").read_text()
    # Anchored to the end of the value: `HAC_MAXLAGS = 21 * 2` must not be read
    # as 21, which is how a helper that parses a prefix reports a horizon the
    # notebook does not use.
    comment = r"[ \t]*(?:#.*)?$"
    label_file = re.search(rf'^PRIMARY_LABEL_FILE = "([^"]+)"{comment}', source, re.MULTILINE)
    maxlags = re.search(rf"^HAC_MAXLAGS = (\d+){comment}", source, re.MULTILINE)
    assert label_file and maxlags, (
        "05_evaluation no longer declares PRIMARY_LABEL_FILE and HAC_MAXLAGS as "
        "plain literals; read its label and horizon from wherever it now takes them"
    )
    assert re.search(rf"^LABEL_HORIZON = HAC_MAXLAGS{comment}", source, re.MULTILINE), (
        "05_evaluation's purge horizon is no longer HAC_MAXLAGS alone; this helper "
        "reports the wrong horizon for the equivalence check"
    )
    # ...and both must be what the notebook reads and shifts by, not constants it
    # declares and then ignores.
    assert re.search(
        r'read_parquet\(\s*CASE_DIR\s*/\s*"labels"\s*/\s*PRIMARY_LABEL_FILE\s*\)', source
    ), "05_evaluation declares PRIMARY_LABEL_FILE but does not load the labels with it"
    assert re.search(r"shift\(\s*-\s*LABEL_HORIZON\s*\)", source), (
        "05_evaluation declares LABEL_HORIZON but does not shift the calendar by it"
    )
    return label_file.group(1), int(maxlags.group(1))


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
        "to the per-symbol form -- see issues/"
        "2026-07-18-evaluation-holdout-leak-sibling-sweep §3."
    )


def test_the_label_and_horizon_are_resolved_from_the_configured_primary_label() -> None:
    """One config read must drive both the endpoint purge and the HAC lag.

    A hardcoded 21 works until ``labels.primary`` changes, and then the purge and
    the Newey-West lag are silently wrong for the label actually being evaluated.
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
    # The one resolved value has to reach both places that depend on it.
    assert 'shift(-LABEL_HORIZON).over("symbol")' in source, (
        "the holdout purge must shift by the resolved horizon"
    )
    assert "label_horizon=LABEL_HORIZON" in source, (
        "the Newey-West lag must be set from the resolved horizon, not a literal"
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
