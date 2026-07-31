"""Gate-0 leakage detectors — the methodology ratchet.

Default-deny AST scans for point-in-time / look-ahead leak classes. Comments and
string literals are invisible to ``ast``, so teaching commentary that *names* a
forbidden pattern (e.g. a markdown cell warning "do not use ``hmm.predict`` as a
feature") does not trip these — only executable code does.

Each detector classifies every occurrence into one of three buckets:

* **APPROVED** (an ``*_ALLOWLIST``) — reviewed and confirmed PIT-safe or purely
  pedagogical, with a one-line justification. Seeded only from chapters already
  audited for correctness (Ch01–11).
* **PENDING** (a ``*_PENDING``) — a *known, un-audited* occurrence in a chapter not
  yet re-passed under Gate 0. This is a **backlog, not an approval**: it awaits that
  chapter's adversarial pass, where the worker must either fix it or move it to the
  allowlist with a justification. It keeps the suite green on a baseline so a *new*
  leak-shaped call fails loudly the moment it appears — the standard lint-baseline
  ratchet. Invariant: **no audited chapter (Ch01–11) may sit in PENDING** — audited
  chapters must be resolved via real approvals, never parked in the backlog.
* **VIOLATION** — anything in neither bucket. The per-notebook sign-off worker must
  resolve it (fix the leak, or propose an allowlist entry the manager lands).

Every Gate-0 finding becomes (a) a new allowlist key here and (b) a sibling sweep, so a
fixed leak cannot silently return and the same bug cannot pass in a sibling notebook.
Design: ``agents/notebooks/CHECKLIST.md`` §Gate 0,
``agents/work/2026-07-17-signoff-process-redesign/SPEC.md`` §A.3.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
# One glob per chapter dir; case studies are Program 2 and are scanned separately.
NOTEBOOK_GLOB = "[0-9][0-9]_*/*.py"


def _notebooks() -> list[Path]:
    return sorted(REPO_ROOT.glob(NOTEBOOK_GLOB))


def _rel(p: Path) -> str:
    return p.relative_to(REPO_ROOT).as_posix()


def _rel_of_key(key: object) -> str:
    """The notebook relpath a bucket key refers to (keys are either the relpath
    string or a tuple whose first element is the relpath)."""
    return key if isinstance(key, str) else key[0]  # type: ignore[index]


def _is_audited(rel: str) -> bool:
    """Ch01–11 are the correctness-audited chapters that seed the allowlists."""
    return int(rel[:2]) <= 11


def _assert_ratchet(kind: str, live: set, approved: dict, pending: dict) -> None:
    """Shared ratchet check: buckets disjoint; no audited chapter parked in PENDING;
    no un-bucketed live occurrence (the gate); no dead bucket entry (hygiene)."""
    approved_keys, pending_keys = set(approved), set(pending)

    overlap = approved_keys & pending_keys
    assert not overlap, f"{kind}: keys in BOTH approved and pending:\n" + "\n".join(
        f"  {k}" for k in sorted(map(str, overlap))
    )

    audited_pending = [k for k in pending_keys if _is_audited(_rel_of_key(k))]
    assert not audited_pending, (
        f"{kind}: audited chapter(s) parked in PENDING — resolve via a real approval "
        "or a fix, never the backlog:\n"
        + "\n".join(f"  {k}" for k in sorted(map(str, audited_pending)))
    )

    violations = [k for k in live if k not in approved_keys and k not in pending_keys]
    assert not violations, (
        f"{kind}: un-reviewed occurrence(s) — fix the leak, or (manager lands) an "
        "allowlist entry with a justification:\n"
        + "\n".join(f"  {k}" for k in sorted(map(str, violations)))
    )

    dead = [k for k in (approved_keys | pending_keys) if k not in live]
    assert not dead, f"{kind}: stale bucket entry (occurrence no longer present):\n" + "\n".join(
        f"  {k}" for k in sorted(map(str, dead))
    )


# --------------------------------------------------------------------------- #
# Meta-test — the detectors must scan every notebook, or a leak hides in the gap
# --------------------------------------------------------------------------- #
def test_detectors_cover_every_notebook() -> None:
    """``NOTEBOOK_GLOB`` is flat (``chapter/*.py``); a notebook in a nested subdir
    would silently escape every detector. Assert every jupytext-paired notebook is in
    the scanned set — an unscanned notebook defeats the whole ratchet."""
    scanned = {_rel(p) for p in _notebooks()}
    paired = {
        p.relative_to(REPO_ROOT).as_posix()
        for p in REPO_ROOT.glob("[0-9][0-9]_*/**/*.py")
        if p.with_suffix(".ipynb").exists()
    }
    missed = sorted(paired - scanned)
    assert not missed, (
        "notebook(s) not scanned by the leakage detectors (a nested dir escaped "
        f"NOTEBOOK_GLOB={NOTEBOOK_GLOB!r}):\n" + "\n".join(f"  {m}" for m in missed)
    )


# --------------------------------------------------------------------------- #
# Detector — global-decoder / smoothed output used in a notebook (SPEC §A.3)
#
# Viterbi/global decoding (``hmm.predict``) and smoothing (``hmm.predict_proba``,
# ``hmm.decode``, ``res.smoothed_marginal_probabilities``) look at the *whole*
# observation sequence, including the future, to label each timestep. Using such
# labels/probabilities as ML features or backtest signals is look-ahead; the forward
# algorithm (filtered probabilities, ≤ t) is the PIT-safe substitute. This was the
# 11_hmm_regimes → 13_regime_as_feature same-day miss.
#
# Recall comes from TWO passes, not a receiver-name guess:
#   1. taint — collect names bound from a known HMM/Markov constructor, so
#      ``model = GaussianHMM(...); model.predict(X)`` is caught even though "model"
#      carries no hint (jupytext .py is linear top-to-bottom, so single-file taint is
#      sound). sklearn ``clf.predict`` on a test fold is NOT tainted and does not trip.
#   2. hint fallback — receiver name contains hmm/markov/switch (covers pickled loads).
# Plus an attribute scan for ``smoothed_*`` Markov results (an attribute access, not a
# Call — invisible to a call-only scan). Precision is high: these names are unambiguous.
# --------------------------------------------------------------------------- #

_DECODER_METHODS = {"predict", "predict_proba", "decode"}
_DECODER_RECEIVER_HINTS = ("hmm", "markov", "switch")
_HMM_CONSTRUCTORS = {
    "GaussianHMM",
    "GMMHMM",
    "MultinomialHMM",
    "CategoricalHMM",
    "PoissonHMM",
    "MarkovRegression",
    "MarkovAutoregression",
}
_SMOOTHED_ATTRS = {"smoothed_marginal_probabilities", "smoothed_state", "smoothed_probabilities"}

# key = (notebook relpath, receiver expression, method | smoothed-attr); value = reason.
DECODER_ALLOWLIST: dict[tuple[str, str, str], str] = {
    (
        "05_synthetic_data/05_diffusion_ts.py",
        "hmm",
        "predict",
    ): "Ch05: labels regimes of the full historical series to *condition a generator*; "
    "not a PIT feature for a backtest. Signed 2026-07-11.",
    (
        "09_model_based_features/11_hmm_regimes.py",
        "hmm2",
        "predict",
    ): "Ch09 teaching: demonstrates HMM state decoding on the toy/SPY series; "
    "the notebook explicitly contrasts this with filtered probs. Signed 2026-07-11.",
    (
        "09_model_based_features/11_hmm_regimes.py",
        "hmm2",
        "predict_proba",
    ): "Ch09 teaching: shows smoothed probabilities *as the look-ahead mistake* next "
    "to filtered probs; plotted, never used as a feature. Signed 2026-07-11.",
    (
        "09_model_based_features/11_hmm_regimes.py",
        "toy_hmm",
        "predict_proba",
    ): "Ch09 teaching: toy-HMM smoothed vs filtered comparison in the forward-algorithm "
    "walkthrough. Signed 2026-07-11.",
    (
        "09_model_based_features/11_hmm_regimes.py",
        "msar_result",
        "smoothed_marginal_probabilities",
    ): "Ch09 teaching: MS-AR smoothed probs computed *for diagnostics/comparison only*, "
    "plotted next to filtered (causal) probs; the notebook labels smoothed as full-sample "
    "and never feeds it to a model. Signed 2026-07-11.",
}

# Backlog: known, UN-AUDITED occurrences awaiting that chapter's Gate-0 pass. NOT approvals.
DECODER_PENDING: dict[tuple[str, str, str], str] = {
    (
        "21_rl_execution_hedging/rl_calibration.py",
        "model",
        "predict",
    ): "Ch21 helper, UN-AUDITED: HMM/Markov `model.predict` surfaced by taint. Resolve at "
    "the Ch21 Gate-0 pass (fix or approve with a downstream-sink argument). NOT approved.",
}


def _decoder_occurrences(tree: ast.AST) -> set[tuple[str, str, str]]:
    """(receiver_expr, method|attr) decoder uses: HMM-tainted or hint receivers, plus
    ``smoothed_*`` attribute accesses. Returns keys without the relpath prefix."""
    tainted: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            ctor = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else None
            )
            if ctor in _HMM_CONSTRUCTORS:
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        tainted.add(tgt.id)

    found: set[tuple[str, str, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in _DECODER_METHODS:
                receiver = ast.unparse(node.func.value)
                hinted = any(h in receiver.lower() for h in _DECODER_RECEIVER_HINTS)
                if hinted or receiver in tainted:
                    found.add(("", receiver, node.func.attr))
        elif isinstance(node, ast.Attribute) and node.attr in _SMOOTHED_ATTRS:
            found.add(("", ast.unparse(node.value), node.attr))
    return found


def test_no_unallowlisted_global_decoder_features() -> None:
    """No HMM/Markov global-decode or smoothed output may appear in a notebook unless
    it is allowlisted (reviewed teaching/non-PIT use) or in the un-audited backlog."""
    live: set[tuple[str, str, str]] = set()
    for nb in _notebooks():
        try:
            tree = ast.parse(nb.read_text())
        except SyntaxError as exc:  # pragma: no cover - a broken notebook is its own failure
            raise AssertionError(f"{_rel(nb)}: could not parse ({exc})") from exc
        rel = _rel(nb)
        for _, receiver, method in _decoder_occurrences(tree):
            live.add((rel, receiver, method))
    _assert_ratchet("global-decoder/smoothed", live, DECODER_ALLOWLIST, DECODER_PENDING)


# --------------------------------------------------------------------------- #
# Detector — hand-rolled random / shuffled split that bypasses the sanctioned CV
#
# Purge + embargo are enforced *centrally* by ``generate_cv_splits`` (embargo =
# ``label_buffer``, asserted by ``tests/test_cv_splits.py``). The per-notebook leak
# surface is a notebook that *skips* that generator and cuts its own split with a random
# shuffler — ``train_test_split``, ``ShuffleSplit``, ``StratifiedShuffleSplit``, or
# ``KFold``/``StratifiedKFold`` with ``shuffle=True`` (shuffled KFold is ShuffleSplit in
# a trenchcoat) — interleaving future and past across the boundary. Plain (contiguous)
# KFold is intentionally NOT banned: it is a milder overlap problem the central policy
# handles, and banning it is noisy.
#
# Random splitting is legitimate only for genuinely i.i.d. data (per-document text
# classification). Those uses are allowlisted after review; everything else must go
# through ``generate_cv_splits``.
# --------------------------------------------------------------------------- #

_RANDOM_SPLITTERS = {"train_test_split", "ShuffleSplit", "StratifiedShuffleSplit"}
_KFOLD_SPLITTERS = {"KFold", "StratifiedKFold"}

# key = notebook relpath; value = justification (the whole notebook is reviewed for this
# class, so line moves within it don't churn the allowlist).
RANDOM_SPLIT_ALLOWLIST: dict[str, str] = {
    "06_strategy_definition/02_cv_foundations.py": "Ch06 teaching: KFold(shuffle=True) on "
    "np.arange(100) synthetic indices, plotted as fold bars to *demonstrate why shuffling "
    "destroys temporal structure* (the notebook's own comment). No model, no features. "
    "Signed 2026-07-11.",
    "10_text_feature_engineering/03_sentiment_evolution.py": "Ch10: stratified split of "
    "Financial PhraseBank *sentences* by sentiment label — i.i.d. documents, not a time "
    "series. Signed 2026-07-11.",
    "10_text_feature_engineering/04_bert_finetuning.py": "Ch10: stratified train/val/test "
    "split of labeled documents for BERT fine-tuning — i.i.d. documents. Signed 2026-07-11.",
    "10_text_feature_engineering/05_financial_ner_finetuning.py": "Ch10: HF-datasets "
    "train_test_split of the NER corpus — i.i.d. sentences/tokens. Signed 2026-07-11.",
}

# Backlog: known, UN-AUDITED occurrences awaiting that chapter's Gate-0 pass. NOT approvals.
RANDOM_SPLIT_PENDING: dict[str, str] = {
    "23_knowledge_graphs/06_gnn_feature_engineering.py": "Ch23, UN-AUDITED: KFold(shuffle="
    "True) in GNN feature engineering. Resolve at the Ch23 Gate-0 pass. NOT approved.",
}


def _has_random_split(tree: ast.AST) -> bool:
    """True if the notebook calls a random/shuffled splitter that bypasses the sanctioned CV."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name in _RANDOM_SPLITTERS:
            return True
        if name in _KFOLD_SPLITTERS and any(
            kw.arg == "shuffle" and isinstance(kw.value, ast.Constant) and kw.value.value is True
            for kw in node.keywords
        ):
            return True
    return False


def test_no_hand_rolled_random_time_split() -> None:
    """Temporal notebooks must split via ``generate_cv_splits`` (purge+embargo), not a
    random/shuffled splitter. Allowlist genuinely i.i.d. uses with a reason."""
    live: set[str] = set()
    for nb in _notebooks():
        try:
            tree = ast.parse(nb.read_text())
        except SyntaxError as exc:  # pragma: no cover
            raise AssertionError(f"{_rel(nb)}: could not parse ({exc})") from exc
        if _has_random_split(tree):
            live.add(_rel(nb))
    _assert_ratchet("random/shuffled-split", live, RANDOM_SPLIT_ALLOWLIST, RANDOM_SPLIT_PENDING)


# --------------------------------------------------------------------------- #
# Detector — future-aware fill / centered window (a leak class the taxonomy omitted)
#
# ``bfill``/backward fill, ``rolling(center=True)``, and backward/bidirectional
# interpolation pull *future* values into *past* rows. On a financial time series these
# are almost never legitimate in feature code (a centered smoother is fine for an ex-post
# plot, not for a model input). High precision, small allowlist — worth the ratchet.
# --------------------------------------------------------------------------- #

FUTURE_FILL_ALLOWLIST: dict[tuple[str, str], str] = {}

# Backlog: known, UN-AUDITED occurrences awaiting that chapter's Gate-0 pass. NOT approvals.
# Empty since 2026-07-31. Both entries were `bfill` in Ch17, parked UN-AUDITED
# pending the Ch17 pass; #428's reviewed generation replaced both notebooks and
# both now fill forward, so the occurrences are gone and the ratchet's own
# dead-entry check requires the rows to go with them. The list had been stale -
# and this test red on `main` - since that PR landed.
FUTURE_FILL_PENDING: dict[tuple[str, str], str] = {}


def _future_fill_occurrences(tree: ast.AST) -> set[tuple[str, str]]:
    """(op-label,) future-aware fill / centered-window uses. Returns keys without relpath."""
    found: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        method = node.func.attr
        if method in ("bfill", "backfill"):
            found.add(("", "bfill"))
        elif method == "fillna":
            for kw in node.keywords:
                if (
                    kw.arg == "method"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value in ("bfill", "backfill")
                ):
                    found.add(("", "fillna_bfill"))
        elif method == "rolling":
            for kw in node.keywords:
                if (
                    kw.arg == "center"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                ):
                    found.add(("", "rolling_center"))
        elif method == "interpolate":
            for kw in node.keywords:
                if (
                    kw.arg == "limit_direction"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value in ("backward", "both")
                ):
                    found.add(("", "interpolate_backward"))
    return found


def test_no_future_aware_fill_or_centered_window() -> None:
    """No backward fill / centered window in a notebook unless allowlisted (ex-post plot
    or static-series fill) or in the un-audited backlog."""
    live: set[tuple[str, str]] = set()
    for nb in _notebooks():
        try:
            tree = ast.parse(nb.read_text())
        except SyntaxError as exc:  # pragma: no cover
            raise AssertionError(f"{_rel(nb)}: could not parse ({exc})") from exc
        rel = _rel(nb)
        for _, op in _future_fill_occurrences(tree):
            live.add((rel, op))
    _assert_ratchet("future-aware-fill", live, FUTURE_FILL_ALLOWLIST, FUTURE_FILL_PENDING)
