"""The cross-case-study feature inventory must not read an unwired checkout as an empty one.

`case_study_feature_summary` reported "No features" for any case study whose
`financial.parquet` it could not open. A worktree links each case study's `features/`
directory in from the artifact store, so a checkout missing that link produced a clean,
plausible, wrong answer: 7 of 9 case studies, and a feature-count range taken over the
seven it happened to see.
"""

import ast
import os
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).parents[1] / "08_financial_features" / "case_study_feature_summary.py"


def _load_resolver(case_studies_root: Path):
    """Lift ``resolve_feature_panel`` out of the notebook without executing the notebook."""
    tree = ast.parse(NOTEBOOK.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_feature_panel"
    )
    namespace: dict = {"get_case_study_dir": lambda cs: case_studies_root / cs}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace["resolve_feature_panel"]


@pytest.fixture
def resolve(tmp_path):
    return _load_resolver(tmp_path)


def _case_study(root: Path, name: str, *, features: bool, panel: bool) -> Path:
    case_dir = root / name
    case_dir.mkdir()
    if features:
        (case_dir / "features").mkdir()
        if panel:
            (case_dir / "features" / "financial.parquet").write_bytes(b"PAR1")
    return case_dir


def test_a_panel_on_disk_is_readable(resolve, tmp_path):
    _case_study(tmp_path, "etfs", features=True, panel=True)
    state, path = resolve("etfs")
    assert state == "readable"
    assert path == tmp_path / "etfs" / "features" / "financial.parquet"


def test_a_case_study_with_an_empty_features_dir_is_awaiting(resolve, tmp_path):
    _case_study(tmp_path, "etfs", features=True, panel=False)
    assert resolve("etfs")[0] == "awaiting"


def test_a_case_study_this_checkout_does_not_link_is_unreachable(resolve, tmp_path):
    """The defect: this is the state the notebook used to report as "No features"."""
    _case_study(tmp_path, "etfs", features=False, panel=False)
    assert resolve("etfs")[0] == "unreachable"


def test_the_two_absences_are_distinguishable(resolve, tmp_path):
    """`awaiting` and `unreachable` are the two readings the old `None` collapsed."""
    _case_study(tmp_path, "etfs", features=True, panel=False)
    _case_study(tmp_path, "us_equities_panel", features=False, panel=False)
    assert resolve("etfs")[0] != resolve("us_equities_panel")[0]


def _load_refusal(display_names: dict[str, str]):
    """Lift ``refuse_a_partial_view`` out of the notebook without executing the notebook."""
    tree = ast.parse(NOTEBOOK.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "refuse_a_partial_view"
    )
    namespace: dict = {
        "os": os,
        "DISPLAY_NAMES": display_names,
        "display_path": str,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace["refuse_a_partial_view"]


@pytest.fixture
def refuse(monkeypatch):
    monkeypatch.delenv("ML4T_OUTPUT_DIR", raising=False)
    return _load_refusal({"etfs": "ETFs", "us_equities_panel": "US Equities"})


def test_the_canonical_store_refuses_a_case_study_it_cannot_see(refuse, tmp_path):
    """The committed 7-of-9 render is the run this refusal exists to stop."""
    with pytest.raises(RuntimeError) as excinfo:
        refuse(
            [("us_equities_panel", tmp_path / "us_equities_panel" / "features" / "f.parquet")], 9
        )
    message = str(excinfo.value)
    assert "1 of 9" in message
    assert "US Equities" in message


def test_every_unreachable_case_study_is_named(refuse, tmp_path):
    with pytest.raises(RuntimeError) as excinfo:
        refuse(
            [
                ("etfs", tmp_path / "etfs" / "features" / "f.parquet"),
                ("us_equities_panel", tmp_path / "us_equities_panel" / "features" / "f.parquet"),
            ],
            9,
        )
    message = str(excinfo.value)
    assert "ETFs" in message and "US Equities" in message


def test_a_fully_visible_checkout_does_not_refuse(refuse):
    refuse([], 9)


def test_a_redirected_output_root_reports_instead_of_refusing(monkeypatch, tmp_path):
    """pytest points every case study at a scratch root that holds nothing by design."""
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(tmp_path))
    refuse_under_redirect = _load_refusal({"etfs": "ETFs"})
    refuse_under_redirect([("etfs", tmp_path / "etfs" / "features" / "f.parquet")], 9)
