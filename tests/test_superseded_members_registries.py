"""`superseded_members` against six lineage shapes a simple fixture does not reach.

The unit suite in `test_population_supersedes.py` builds its own populations, and five
successive forms of this function passed it while being wrong - three returning an empty
set, one over-retiring, one under-retiring systematically. Every one was caught by a real
registry or by review, none by the unit suite alone. So the shapes those registries had
are what this file is for, and each is kept here as a lineage it builds rather than as a
number read off a machine:

| shape | measured on | retires | what it catches |
|---|---|---:|---|
| `transitive_closure` | `etfs` 2026-08-25 | 30 | Four generations under one name. A form that finds the first edge and stops under-reports. |
| `independent_chains` | `us_firm_characteristics` 2026-08-25 | 135 | Four single-hop chains with uneven counts (30/3/30/72), so a form that stops at one chain misses the rest and an over-retiring form cannot coincidentally match. |
| `doubled_catalog` | `sp500_equity_option_analytics` 2026-08-25 | 123 | A doubled catalog: 915 prediction rows = 792 live + 123 retired, zero overlap. |
| `edge_outlives_rows` | `crypto_perps_funding` 2026-08-25 | 400 | The refit deleted generation A's `prediction_sets` rows and only the `official_population_members` record survives, so a form reading supersession out of `prediction_sets` returns empty here and correct on the others. |
| `dead_name_in_force` | `cme_futures` 2026-08-25 | 656 | A live population lists retired members. Four dead names left by two renames are in force under their own names and hold 178 identities that `linear` and `gbm` have refit past, so the global reading - retired by someone, listed by nobody in force - un-retires all 178 and the per-name reading does not. |
| `re_listed_generation` | `nasdaq100_microstructure` 2026-08-27 | 150 | A superseded generation that retires nothing. `linear-validation-v1` grew 16 members to 61 and re-listed all 16, so those 16 are superseded and still published; only `gbm-validation-v1` retires (150 of 150). A form comparing whole generations rather than members returns 166. |

**Why the shapes are built rather than read.** These were six assertions against the
canonical registries under `~/ml4t/artifacts`, which are shared, mutable and reset before
every retrain, so each count went red the moment anyone retrained and every agent learned to
read a red `test-unit` as environmental. That is the habit that lets a real failure through.
It is not a hypothetical drift either: measured 2026-09-03, `cme_futures` and
`crypto_perps_funding` hold **no supersedes edge at all** - their run logs were reset, every
population is an unsuperseded tip, and `superseded_members` correctly returns nothing there.
Two of the six shapes had ceased to exist in the registries the file was reading, so those
assertions had stopped testing the function some time before anyone noticed. A shape that is
built is fixed by construction, runs on a CI runner that has no `~/ml4t/artifacts`, and stays
the shape it was measured as.

**Why a built shape is not the fixture that let five wrong forms through.** That suite built
*simple* populations - one name, one edge - and the wrong forms agree with the right one
there. What separates them is structure, and structure is what is transcribed below.
`test_every_wrong_form_is_separated_by_some_shape` holds the four wrong readings as
executable functions and requires each to be caught, so the discriminating power is asserted
rather than asserted-about: delete a shape and that test says which reading stops being
covered.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from case_studies.research.population import superseded_members, superseded_members_at


@dataclass(frozen=True)
class Generation:
    """One published generation: what it is called, what it lists, what it replaced."""

    name: str
    population_hash: str
    supersedes_hash: str | None
    members: tuple[str, ...]


@dataclass(frozen=True)
class Shape:
    """A lineage, and the answer it must produce.

    ``registered`` is what ``prediction_sets`` holds. It is not the union of the members:
    a refit may delete the rows of the generation it replaced and keep only the membership
    record, which is the whole of ``edge_outlives_rows``.
    """

    why: str
    generations: tuple[Generation, ...]
    retired: frozenset[str]
    registered: frozenset[str]


def _ids(prefix: str, count: int, *, start: int = 0) -> tuple[str, ...]:
    """``count`` distinct twelve-character member hashes, stable across runs."""
    return tuple(f"{prefix}{index:08d}" for index in range(start, start + count))


def _chain(name: str, generations: list[tuple[str, tuple[str, ...]]]) -> list[Generation]:
    """A single name's generations, each superseding the one before it."""
    built: list[Generation] = []
    previous: str | None = None
    for population_hash, members in generations:
        built.append(Generation(name, population_hash, previous, members))
        previous = population_hash
    return built


def _shape_transitive_closure() -> Shape:
    """`etfs`: one name, four generations, each replacing the last outright."""
    a, b, c, d = (_ids("e", 10, start=index * 10) for index in range(4))
    generations = _chain(
        "etfs-gbm-validation-v1",
        [("e0000000a", a), ("e0000000b", b), ("e0000000c", c), ("e0000000d", d)],
    )
    return Shape(
        why="a form that follows one edge and stops reports 10 of the 30",
        generations=tuple(generations),
        retired=frozenset(a + b + c),
        registered=frozenset(a + b + c + d),
    )


def _shape_independent_chains() -> Shape:
    """`us_firm_characteristics`: four names, one edge each, uneven counts."""
    generations: list[Generation] = []
    retired: set[str] = set()
    registered: set[str] = set()
    for index, (family, count) in enumerate(
        [("cae", 30), ("ipca", 3), ("sae", 30), ("tabular_dl", 72)]
    ):
        old = _ids(f"u{index}o", count)
        new = _ids(f"u{index}n", count)
        generations += _chain(
            f"us_firm_characteristics-{family}-validation-v1",
            [(f"u{index}0000000", old), (f"u{index}1000000", new)],
        )
        retired |= set(old)
        registered |= set(old) | set(new)
    return Shape(
        why="a form that stops at the first chain reports at most 72 of the 135",
        generations=tuple(generations),
        retired=frozenset(retired),
        registered=frozenset(registered),
    )


def _shape_doubled_catalog() -> Shape:
    """`sp500_equity_option_analytics`: both generations complete and current."""
    old = _ids("s0", 123)
    new = _ids("s1", 792)
    generations = _chain(
        "sp500_equity_option_analytics-sdf-validation-v1",
        [("s000000000", old), ("s100000000", new)],
    )
    return Shape(
        why="915 catalog rows are 792 live plus 123 retired, and a sweep on the catalog "
        "alone takes both",
        generations=tuple(generations),
        retired=frozenset(old),
        registered=frozenset(old) | frozenset(new),
    )


def _shape_edge_outlives_rows() -> Shape:
    """`crypto_perps_funding`: the refit deleted the rows and kept the edge."""
    old = _ids("c0", 400)
    new = _ids("c1", 400)
    generations = _chain(
        "crypto_perps_funding-gbm-validation-v1",
        [("c000000000", old), ("c100000000", new)],
    )
    return Shape(
        why="nothing of generation A is left in prediction_sets, so a form reading "
        "supersession out of the rows returns nothing retired",
        generations=tuple(generations),
        retired=frozenset(old),
        registered=frozenset(new),
    )


def _shape_dead_name_in_force() -> Shape:
    """`cme_futures`: two renames left four names in force listing retired identities.

    The dead names were never superseded - a rename does not supersede anything - so they
    stand behind the members they listed on the day they were published, including the 178
    that `linear` and `gbm` have since refit past.
    """
    linear_old, linear_new = _ids("m0o", 328), _ids("m0n", 328)
    gbm_old, gbm_new = _ids("m1o", 328), _ids("m1n", 328)
    generations = _chain(
        "cme_futures-linear-validation-v1",
        [("m000000000", linear_old), ("m010000000", linear_new)],
    ) + _chain(
        "cme_futures-gbm-validation-v1",
        [("m100000000", gbm_old), ("m110000000", gbm_new)],
    )
    # The four leftovers, in force, between them listing 178 of the retired identities.
    leftovers = [
        ("cme_futures-linear-validation-v0", "m200000000", linear_old[:50]),
        ("cme_futures-linear-preflight-v1", "m210000000", linear_old[50:89]),
        ("cme_futures-gbm-validation-v0", "m300000000", gbm_old[:50]),
        ("cme_futures-gbm-preflight-v1", "m310000000", gbm_old[50:89]),
    ]
    generations += [
        Generation(name, population_hash, None, members)
        for name, population_hash, members in leftovers
    ]
    return Shape(
        why="the global reading un-retires the 178 identities a dead name still lists",
        generations=tuple(generations),
        retired=frozenset(linear_old + gbm_old),
        registered=frozenset(linear_old + linear_new + gbm_old + gbm_new),
    )


def _shape_re_listed_generation() -> Shape:
    """`nasdaq100_microstructure`: one edge retires everything, the other nothing."""
    gbm_old = _ids("n0o", 150)
    gbm_new = _ids("n0n", 500)
    linear_old = _ids("n1o", 16)
    linear_new = linear_old + _ids("n1n", 45)
    generations = _chain(
        "nasdaq100_microstructure-gbm-validation-v1",
        [("n000000000", gbm_old), ("n010000000", gbm_new)],
    ) + _chain(
        "nasdaq100_microstructure-linear-validation-v1",
        [("n100000000", linear_old), ("n110000000", linear_new)],
    )
    return Shape(
        why="retiring superseded generations wholesale reports 166, which retires 16 "
        "identities linear-validation-v1 still publishes",
        generations=tuple(generations),
        retired=frozenset(gbm_old),
        registered=frozenset(gbm_old + gbm_new + linear_new),
    )


SHAPES: dict[str, Shape] = {
    "transitive_closure": _shape_transitive_closure(),
    "independent_chains": _shape_independent_chains(),
    "doubled_catalog": _shape_doubled_catalog(),
    "edge_outlives_rows": _shape_edge_outlives_rows(),
    "dead_name_in_force": _shape_dead_name_in_force(),
    "re_listed_generation": _shape_re_listed_generation(),
}

# The counts the registries were measured at, kept because the table in the docstring is
# the record of what each shape is. A shape edited without its row being edited fails here
# rather than quietly measuring something else.
RETIRED_PREDICTIONS = {
    "transitive_closure": 30,
    "independent_chains": 135,
    "doubled_catalog": 123,
    "edge_outlives_rows": 400,
    "dead_name_in_force": 656,
    "re_listed_generation": 150,
}


def _write_registry(root: Path, shape: Shape) -> Path:
    """Write `shape` into a registry at `root`, in the schema the readers query."""
    run_log = root / "run_log"
    run_log.mkdir(parents=True, exist_ok=True)
    db_path = run_log / "registry.db"
    with sqlite3.connect(db_path) as db:
        db.executescript(
            """
            CREATE TABLE official_populations (
                population_hash  TEXT PRIMARY KEY,
                name             TEXT NOT NULL,
                member_kind      TEXT NOT NULL,
                snapshot_json    TEXT NOT NULL,
                supersedes_hash  TEXT REFERENCES official_populations(population_hash),
                created_at       TEXT NOT NULL
            );
            CREATE TABLE official_population_members (
                population_hash TEXT NOT NULL REFERENCES official_populations(population_hash),
                member_hash     TEXT NOT NULL,
                ordinal         INTEGER NOT NULL,
                PRIMARY KEY (population_hash, ordinal),
                UNIQUE (population_hash, member_hash)
            );
            CREATE TABLE prediction_sets (
                prediction_hash     TEXT PRIMARY KEY,
                training_hash       TEXT NOT NULL,
                checkpoint_value    INTEGER,
                checkpoint_kind     TEXT,
                split               TEXT NOT NULL,
                created_at          TEXT NOT NULL
            );
            """
        )
        for generation in shape.generations:
            db.execute(
                "INSERT INTO official_populations VALUES (?, ?, 'prediction', '{}', ?, "
                "'2026-08-25T00:00:00')",
                (generation.population_hash, generation.name, generation.supersedes_hash),
            )
            db.executemany(
                "INSERT INTO official_population_members VALUES (?, ?, ?)",
                [
                    (generation.population_hash, member, ordinal)
                    for ordinal, member in enumerate(generation.members)
                ],
            )
        db.executemany(
            "INSERT INTO prediction_sets VALUES (?, 't', NULL, NULL, 'validation', "
            "'2026-08-25T00:00:00')",
            [(member,) for member in sorted(shape.registered)],
        )
    return root


@pytest.fixture
def registry(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    """The case directory for the shape named by the test's parameter."""
    return _write_registry(tmp_path / request.param, SHAPES[request.param])


class _SingleRootStudy:
    """The shape `superseded_members` needs, pointed at one registry root.

    A production case study has no workspace: `study.root` and `study.release_case_root`
    are the same directory, which is the configuration these registries are in.
    """

    read_only = False

    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def release_case_root(self) -> Path:
        return self.root


# The four readings that are wrong, written out so a shape's claim to separate one is
# executable rather than a comment. Each takes the lineage in the form `_lineage` returns.


def _by_name(
    rows: list[tuple[str, str, str | None]],
) -> dict[str, list[tuple[str, str | None]]]:
    grouped: dict[str, list[tuple[str, str | None]]] = {}
    for population_hash, name, supersedes in rows:
        grouped.setdefault(name, []).append((population_hash, supersedes))
    return grouped


def _wrong_one_edge_only(rows, members) -> frozenset[str]:
    """Retire only the generation the tip directly replaced."""
    retired: set[str] = set()
    for generations in _by_name(rows).values():
        superseded = {s for _, s in generations if s is not None}
        for population_hash, supersedes in generations:
            if population_hash not in superseded and supersedes is not None:
                retired |= members.get(supersedes, set())
    return frozenset(retired)


def _wrong_first_chain_only(rows, members) -> frozenset[str]:
    """Answer for one name and stop."""
    for name in sorted(_by_name(rows)):
        answer = _wrong_whole_generation(
            [row for row in rows if row[1] == name],
            members,
        )
        if answer:
            return answer
    return frozenset()


def _wrong_global(rows, members) -> frozenset[str]:
    """Retired by someone, and listed by nobody in force - across all names at once."""
    superseded = {row[2] for row in rows if row[2] is not None}
    in_force = {row[0] for row in rows if row[0] not in superseded}
    listed = set().union(*(members.get(h, set()) for h in in_force)) if in_force else set()
    retired_somewhere = (
        set().union(*(members.get(h, set()) for h in superseded)) if superseded else set()
    )
    return frozenset(retired_somewhere - listed)


def _wrong_whole_generation(rows, members) -> frozenset[str]:
    """Retire every member of a superseded generation, re-listed or not."""
    superseded = {row[2] for row in rows if row[2] is not None}
    return frozenset(
        set().union(*(members.get(h, set()) for h in superseded)) if superseded else set()
    )


def _wrong_from_prediction_rows(rows, members, registered: frozenset[str]) -> frozenset[str]:
    """Derive supersession from the rows rather than from the membership record."""
    return frozenset(_wrong_whole_generation(rows, members) & registered)


# Each wrong reading, and the one shape whose job is to separate it. Naming the shape is
# what keeps a shape load-bearing: flatten `re_listed_generation` and the whole-generation
# row fails, rather than another shape quietly covering for it.
WRONG_FORMS = {
    "one edge only": (_wrong_one_edge_only, "transitive_closure"),
    "first chain only": (_wrong_first_chain_only, "independent_chains"),
    "global reading": (_wrong_global, "dead_name_in_force"),
    "whole generation": (_wrong_whole_generation, "re_listed_generation"),
}


def _lineage_of(shape: Shape) -> tuple[list[tuple[str, str, str | None]], dict[str, set[str]]]:
    rows = [(g.population_hash, g.name, g.supersedes_hash) for g in shape.generations]
    members = {g.population_hash: set(g.members) for g in shape.generations}
    return rows, members


@pytest.mark.parametrize("registry", sorted(SHAPES), indirect=True)
def test_retired_prediction_count(registry: Path) -> None:
    """The retired set is exactly the one the shape was measured to have.

    Both the identities and the count: a count alone is satisfied by an over-retiring form
    that happens to land on the same total, which is the coincidence
    `independent_chains` exists to rule out.
    """
    shape = SHAPES[registry.name]
    retired = superseded_members(_SingleRootStudy(registry), member_kind="prediction")
    assert retired == shape.retired
    assert len(retired) == RETIRED_PREDICTIONS[registry.name]


@pytest.mark.parametrize("registry", sorted(SHAPES), indirect=True)
def test_retired_members_were_registered_and_are_not_still_published(registry: Path) -> None:
    """The invariant behind the counts, asserted on every shape.

    Retirement is a claim about identities the registry knows, and nothing a name still
    stands behind may be in it. `dead_name_in_force` is why the second half is stated
    against the generations of the *publishing* name rather than against every population
    in force: a population left over by a rename does list retired identities, and that is
    the reading this function exists to refuse.
    """
    shape = SHAPES[registry.name]
    rows, members = _lineage_of(shape)
    retired = superseded_members(_SingleRootStudy(registry), member_kind="prediction")

    every_member = set().union(*members.values())
    assert retired <= every_member

    for generations in _by_name(rows).values():
        superseded = {s for _, s in generations if s is not None}
        published = set().union(
            *(members[h] for h, _ in generations if h not in superseded),
            set(),
        )
        assert not (published & retired & set().union(*(members[h] for h in superseded), set())), (
            "a name is listing identities it also retired, which no reading of the lineage allows"
        )


@pytest.mark.parametrize(
    ("form", "wrong", "shape_name"), [(k, *v) for k, v in sorted(WRONG_FORMS.items())]
)
def test_a_named_shape_separates_each_wrong_form(form: str, wrong, shape_name: str) -> None:
    """Each wrong reading is caught, by the shape that is here to catch it.

    This is what a built lineage has to earn that a read one had for free. Flattening a
    shape while editing it shows up here as the reading it was carrying, rather than as
    six tests that still pass because another shape happens to cover it.
    """
    shape = SHAPES[shape_name]
    assert wrong(*_lineage_of(shape)) != shape.retired, (
        f"{shape_name} no longer distinguishes the {form!r} reading from the right one"
    )


def test_the_prediction_row_reading_is_separated_by_the_deleted_generation() -> None:
    """`edge_outlives_rows` is the only shape where the rows and the record disagree.

    A form deriving supersession from `prediction_sets` returns nothing retired there and
    is right everywhere else, so it is the one wrong reading that needs a shape built
    around a deletion rather than around a lineage.
    """
    shape = SHAPES["edge_outlives_rows"]
    rows, members = _lineage_of(shape)
    assert _wrong_from_prediction_rows(rows, members, shape.registered) == frozenset()
    assert shape.retired


def test_a_retired_generation_and_the_live_catalog_do_not_overlap() -> None:
    """What the doubling looks like from the catalog's side.

    `doubled_catalog` holds both generations as complete, current rows, so the count a
    backtest would sweep is the sum. Asserting the disjointness rather than the totals is
    what says the retired set is a clean second generation and not an arbitrary subset.

    `len(registered - retired) == len(registered) - len(retired)` follows from
    `retired <= registered` by counting alone: it holds for any subset whatsoever and
    cannot fail, so it asserts nothing. What is worth checking is that nothing this name
    still publishes is also something it retired.
    """
    shape = SHAPES["doubled_catalog"]
    rows, members = _lineage_of(shape)
    retired = shape.retired

    assert retired
    assert retired <= shape.registered
    assert len(shape.registered) == 915
    superseded = {row[2] for row in rows if row[2] is not None}
    in_force = {row[0] for row in rows if row[0] not in superseded}
    live = set().union(*(members[h] for h in in_force))
    assert len(live) == 792
    assert not (live & retired)


def test_an_edge_outlives_the_rows_it_retired(tmp_path: Path) -> None:
    """`edge_outlives_rows`: the refit deleted the rows and kept the membership record.

    Nothing of generation A is left in `prediction_sets`; only
    `official_population_members` still lists it, which is what keeps the edge readable as
    history. A form that derives supersession from the prediction rows returns empty here
    and correct everywhere else, which is the failure this case exists to catch.
    """
    shape = SHAPES["edge_outlives_rows"]
    root = _write_registry(tmp_path / "edge_outlives_rows", shape)
    retired = superseded_members(_SingleRootStudy(root), member_kind="prediction")
    assert retired
    with sqlite3.connect(root / "run_log" / "registry.db") as db:
        registered = {row[0] for row in db.execute("SELECT prediction_hash FROM prediction_sets")}
    assert not (retired & registered)


def test_a_dead_name_still_in_force_does_not_un_retire_what_it_lists(tmp_path: Path) -> None:
    """`dead_name_in_force` measures the per-name rule against a lineage that defeats the global one.

    `linear` and `gbm` were published under earlier names before the current one, and those
    earlier names were never superseded - a rename leaves them in force, listing members
    their producer has since refit past. So the two readings disagree by construction, and
    the disagreement is exactly the overlap asserted below: identities that are retired
    under the name that published them and listed by a population nothing supersedes.

    The assertion is on the overlap being non-empty, not on its size, because the size is a
    property of how many configurations were fitted and would move under a re-run; that a
    dead name lists retired members at all is the property this shape contributes.
    """
    shape = SHAPES["dead_name_in_force"]
    root = _write_registry(tmp_path / "dead_name_in_force", shape)
    rows, members = _lineage_of(shape)
    retired = superseded_members(_SingleRootStudy(root), member_kind="prediction")
    assert retired

    superseded = {row[2] for row in rows if row[2] is not None}
    in_force = {row[0] for row in rows if row[0] not in superseded}
    listed_by_something_in_force = set().union(*(members[h] for h in in_force))

    # The global reading keeps only what nothing in force still lists. The per-name reading
    # is what `superseded_members` returns, and here the two differ.
    assert _wrong_global(rows, members) < retired, "no live population lists a retired identity"

    un_retired = retired & listed_by_something_in_force
    assert len(un_retired) == 178
    holders = {name for h, name, _ in rows if h in in_force and members[h] & un_retired}
    assert holders, "the overlap has no holder, so this shape no longer tests the rule"
    assert all(
        name not in ("cme_futures-linear-validation-v1", "cme_futures-gbm-validation-v1")
        for name in holders
    ), "a name is listing identities it also retired, which no reading of the lineage allows"


def test_a_superseded_generation_whose_members_are_all_re_listed_retires_none(
    tmp_path: Path,
) -> None:
    """The case that separates member-wise comparison from whole-generation comparison.

    `re_listed_generation` carries two supersedes edges. `gbm-validation-v1` replaced 150
    members with a disjoint 500 and retires all 150. `linear-validation-v1` grew 16 members
    to 61 and kept every one of the 16, so its superseded generation retires nothing at all
    - the name moved past that *snapshot*, not past those identities.

    A form that retires a superseded generation wholesale returns 166 here.
    """
    shape = SHAPES["re_listed_generation"]
    root = _write_registry(tmp_path / "re_listed_generation", shape)
    rows, members = _lineage_of(shape)
    retired = superseded_members(_SingleRootStudy(root), member_kind="prediction")
    assert len(retired) == 150

    superseded = {row[2] for row in rows if row[2] is not None}
    assert len(superseded) == 2, "both names must have refit for this case to be the one described"
    wholesale = _wrong_whole_generation(rows, members)
    assert len(wholesale) == 166
    assert retired < wholesale, (
        "retiring superseded generations wholesale would retire 16 identities that "
        "linear-validation-v1 still publishes"
    )


@pytest.mark.parametrize("registry", sorted(SHAPES), indirect=True)
def test_superseded_members_at_answers_for_the_root_it_is_given(registry: Path) -> None:
    """The root-based form a notebook uses when it holds a directory and no study.

    `14_backtest` reads its catalog through `prediction_rows_at(CASE_DIR)` precisely so that
    no `Study.open` runs and no `activate()` re-points the notebook mid-run. Asking the
    lineage question through a study there would answer for whichever registry the activation
    selected, so the retired set and the catalog it filters could describe different
    registries. Same reduction, same answer, one named root.
    """
    assert superseded_members_at(registry, member_kind="prediction") == superseded_members(
        _SingleRootStudy(registry), member_kind="prediction"
    )


def test_superseded_members_at_is_empty_for_a_root_holding_no_registry(tmp_path: Path) -> None:
    """A reader's clean clone has no registry, and that is not an error.

    The distinction that matters is between "no generation was ever written" and "the query
    could not read". The first is answered with nothing; the second must raise, because a
    swallowed failure here is indistinguishable from a clean filter and would silently admit
    every retired row.
    """
    assert superseded_members_at(tmp_path, member_kind="prediction") == frozenset()
