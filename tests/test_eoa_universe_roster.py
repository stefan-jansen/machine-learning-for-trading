"""The sp500_equity_option_analytics roster is a property of the dataset, not of a run's window.

`config/setup.yaml` declares `universe.n_assets: 633` under `eligibility_rule:
sp500_with_options`, and stages 02, 03 and 04 enforce it by deriving the roster from the
option-surface extract and bounding the share bars to it. Two things can go wrong and both
did, at different times:

- Passing no roster at all, which is how the label and model-based artifacts came to carry
  five names with no listed options.
- Deriving the roster from the surface *after* the requested date window, which makes the
  declaration fail on any run that narrows `START_DATE` - a documented parameter that
  `tests/overrides.yaml` is allowed to set. It passes at the full window, which is why it
  survived a re-run and was caught only by review.

The count itself is checked **here and not in the notebooks**. `universe.n_assets` describes the
production extract; the reduced one CI runs the pipeline against carries 23 names by design, so an
assertion on the count inside a notebook fails there for being small rather than for being wrong -
which is exactly what it did, taking three notebooks red. A declaration about the dataset belongs
in a test over the dataset, and this file skips when that dataset is absent.

Moving the count out of the notebooks did not put it anywhere that runs, though. Three of these
tests assert an **absolute** property of the production extract - 633 names, and a share extract
wider than the roster - and `ml4t/third-edition-test-data` is the same reduced 23-name extract, so
they fail there for the same reason the notebooks did. They carry `@pytest.mark.production_extract`
and CI deselects them; they are verified locally against the real data and nowhere else. The other
four assert *relations* that hold on any extract - every roster name has bars, a narrowed window
holds fewer names than the declaration, the roster is not read off the requested window - and those
are the ones a CI job can actually gate.
"""

import ast

import polars as pl
import pytest
import yaml

from data import load_sp500_daily_bars, load_sp500_options_surface
from data.exceptions import DataNotFoundError
from utils.paths import get_case_study_dir

CASE_STUDY_ID = "sp500_equity_option_analytics"
ROSTER_STAGES = ("02_labels", "03_financial_features", "04_model_based_features")


def _window(start: str, end: str) -> pl.Expr:
    return pl.col("timestamp").is_between(pl.lit(start).str.to_date(), pl.lit(end).str.to_date())


@pytest.fixture(scope="module")
def declared_n_assets() -> int:
    setup = yaml.safe_load(
        (get_case_study_dir(CASE_STUDY_ID) / "config" / "setup.yaml").read_text()
    )
    return setup["universe"]["n_assets"]


# The module docstring says this file skips when the dataset is absent, and it did
# not: an absent dataset makes the loader raise DataNotFoundError, so the
# is_empty() check below was never reached and all seven tests errored at setup
# instead. The distinction only shows up in a checkout with no data at all, which
# is every CI job outside the case-study matrix - and this file ran in no job, so
# nothing ever exercised the guard.
def _or_skip(load, what: str) -> pl.DataFrame:
    try:
        frame = load()
    except DataNotFoundError:
        pytest.skip(f"no {what} in this data checkout")
    if frame.is_empty():
        pytest.skip(f"no {what} in this data checkout")
    return frame


@pytest.fixture(scope="module")
def surface(populated_data_dir):
    return _or_skip(load_sp500_options_surface, "sp500 option surface")


@pytest.fixture(scope="module")
def bars(populated_data_dir):
    return _or_skip(load_sp500_daily_bars, "sp500 daily bars")


@pytest.mark.production_extract
def test_the_unbounded_roster_is_the_declared_universe(surface, declared_n_assets):
    """What `n_assets` claims is what the surface extract holds."""
    roster = set(surface["symbol"].unique().to_list())
    assert len(roster) == declared_n_assets


def test_every_roster_name_has_share_bars(surface, bars):
    """A name that can be ranked must have a price to trade at."""
    roster = set(surface["symbol"].unique().to_list())
    assert not roster - set(bars["symbol"].unique().to_list())


@pytest.mark.production_extract
def test_the_bars_carry_names_the_universe_does_not(surface, bars):
    """The direction the original guard did not check.

    Both guards passing is not enough: the share extract is wider than the roster, and it is
    the surplus that reached the artifacts. If this ever comes back empty the bound has
    stopped doing anything and the test above is carrying the whole check.
    """
    roster = set(surface["symbol"].unique().to_list())
    surplus = set(bars["symbol"].unique().to_list()) - roster
    assert surplus, "the share extract no longer carries a name outside the roster"
    assert not any(sym in roster for sym in surplus)


@pytest.mark.parametrize(
    ("start", "end"), [("2020-01-01", "2020-12-31"), ("2021-06-01", "2021-12-31")]
)
def test_a_narrowed_window_holds_fewer_names_than_the_declaration(
    surface, declared_n_assets, start, end
):
    """Why the roster may not be read off the requested window.

    This is the defect itself, stated as data rather than as an argument: inside a shorter
    window the surface carries strictly fewer names than `n_assets`, so a notebook deriving
    its roster after applying `START_DATE` asserts against a number that cannot hold.
    """
    windowed = surface.filter(_window(start, end))["symbol"].n_unique()
    assert 0 < windowed < declared_n_assets


@pytest.mark.production_extract
@pytest.mark.parametrize(
    ("start", "end"), [("2020-01-01", "2020-12-31"), ("2017-01-01", "2021-12-31")]
)
def test_the_roster_bounds_a_narrowed_panel_without_shrinking_itself(
    surface, bars, declared_n_assets, start, end
):
    """The shape stages 02, 03 and 04 use: roster off the whole extract, panel off the window."""
    roster = sorted(surface["symbol"].unique().to_list())
    assert len(roster) == declared_n_assets

    panel = bars.filter(_window(start, end) & pl.col("symbol").is_in(roster))
    assert not panel.is_empty()
    assert set(panel["symbol"].unique().to_list()) <= set(roster)
    assert panel["timestamp"].min() >= pl.Series([start]).str.to_date()[0]
    assert panel["timestamp"].max() <= pl.Series([end]).str.to_date()[0]


def _roster_source_call(stem: str) -> ast.Call:
    """The `load_sp500_options_surface(...)` call whose result the notebook's ROSTER reads.

    Found by walking from the assignment rather than by matching text, so reformatting the
    cell cannot make the check pass or fail.
    """
    tree = ast.parse((get_case_study_dir(CASE_STUDY_ID) / f"{stem}.py").read_text())
    names: dict[str, ast.expr] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names[target.id] = node.value

    assert "ROSTER" in names, f"{stem} has no ROSTER assignment"
    seen: set[str] = set()
    frontier = [names["ROSTER"]]
    while frontier:
        node = frontier.pop()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                func = sub.func
                if isinstance(func, ast.Name) and func.id == "load_sp500_options_surface":
                    return sub
            if isinstance(sub, ast.Name) and sub.id in names and sub.id not in seen:
                seen.add(sub.id)
                frontier.append(names[sub.id])
    raise AssertionError(f"{stem}'s ROSTER does not come from load_sp500_options_surface")


@pytest.mark.parametrize("stem", ROSTER_STAGES)
def test_the_roster_is_not_read_off_the_requested_window(stem):
    """The regression guard for the defect the tests above only describe.

    Everything above is a statement about the extract and would keep passing if a notebook
    went back to deriving its roster from a date-filtered frame. This reads the notebook: the
    call feeding `ROSTER` must take no date arguments, because a roster narrowed to the
    window cannot satisfy `universe.n_assets` and the assertion beside it would fail on any
    run that sets `START_DATE`.
    """
    call = _roster_source_call(stem)
    passed = {kw.arg for kw in call.keywords} | ({"<positional>"} if call.args else set())
    assert not passed, f"{stem} derives ROSTER from a surface load bounded by {sorted(passed)}"
