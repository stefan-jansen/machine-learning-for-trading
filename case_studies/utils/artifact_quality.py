"""Descriptive statistics and coverage for what a production stage just wrote.

Three questions, asked at the end of the notebook that produces an artifact, about
the frame it is about to write:

1. **Is it the size it should be?** Not "are the rows it has non-null" - a stage that
   emits a thousand rows where a million were expected reports no missing values and
   is wrong. ``coverage_against`` compares the produced keys to the universe the
   caller declares.
2. **What is in each column?** ``profile_columns`` gives one row per column: nulls,
   zeros, non-finites, distinct count, quantiles and a tail ratio.
3. **What is unusual?** ``flag_columns`` marks the profile rows that cross a
   threshold, with the reason attached.

Nothing here raises. A zero share of 0.5 is a defect in a return and correct in a
binary direction label; a 45% null share is a defect in a price and expected in a
term-structure feature that needs two expiries. Only the notebook author knows
which, so this surfaces the number and the author signs off on it in prose. The
gates that *do* refuse live in ``coverage.py`` and ``notebook_contracts.py``.

Polars in, polars out. Pandas only if a caller is at a visualization boundary.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

__all__ = [
    "coverage_against",
    "flag_columns",
    "profile_columns",
    "quality_report",
    "render_quality_report",
]

#: Quantiles reported for every numeric column. The outer pair is what makes a
#: heavy tail visible; the inner three are the shape.
DEFAULT_QUANTILES: tuple[float, ...] = (0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999)

#: Advisory thresholds. Deliberately loose - these decide what a reader is asked to
#: look at, not what is acceptable. Tighten per notebook by passing your own.
DEFAULT_RULES: dict[str, float] = {
    "null_share": 0.20,
    "zero_share": 0.60,
    "tail_ratio": 25.0,
}


def _numeric_columns(frame: pl.DataFrame, exclude: Sequence[str]) -> list[str]:
    excluded = set(exclude)
    return [c for c, dtype in frame.schema.items() if dtype.is_numeric() and c not in excluded]


def profile_columns(
    frame: pl.DataFrame,
    *,
    key_columns: Sequence[str] = (),
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> pl.DataFrame:
    """One row per numeric column: how much is there, and what shape is it.

    ``key_columns`` are excluded - a symbol id or a fold index has no distribution
    worth reporting. Non-finite values are counted separately from nulls: a null is
    an absent observation and an infinity is a computed one that overflowed, and the
    two have different causes even though most downstream code treats them alike.

    ``tail_ratio`` is ``|q99.9| / |q75|``, which is a scale-free way to see a column
    whose extreme values are orders of magnitude past its body. It is null where the
    denominator is zero rather than infinite, so the column stays sortable.
    """
    columns = _numeric_columns(frame, key_columns)
    if not columns:
        return pl.DataFrame(schema={"column": pl.String})

    rows: list[dict[str, object]] = []
    height = frame.height
    for name in columns:
        series = frame[name]
        finite = series.drop_nulls()
        n_nonfinite = int(finite.is_infinite().sum() or 0) + int(finite.is_nan().sum() or 0)
        clean = finite.filter(finite.is_finite())
        row: dict[str, object] = {
            "column": name,
            "rows": height,
            "n_null": int(series.null_count()),
            "null_share": series.null_count() / height if height else None,
            "n_nonfinite": n_nonfinite,
            "n_zero": int((clean == 0).sum() or 0),
            "zero_share": (float((clean == 0).sum() or 0) / clean.len()) if clean.len() else None,
            "n_distinct": int(clean.n_unique()),
            "constant": clean.n_unique() <= 1 if clean.len() else None,
            "mean": float(clean.mean()) if clean.len() else None,
            "std": float(clean.std()) if clean.len() > 1 else None,
        }
        for q in quantiles:
            row[f"q{q:g}"] = float(clean.quantile(q)) if clean.len() else None
        body = row.get("q0.75")
        tail = row.get("q0.999")
        row["tail_ratio"] = (
            abs(tail) / abs(body) if body not in (None, 0) and tail is not None else None
        )
        rows.append(row)
    return pl.DataFrame(rows)


def coverage_against(
    produced: pl.DataFrame,
    expected: pl.DataFrame,
    *,
    keys: Sequence[str],
) -> dict[str, pl.DataFrame]:
    """How much of the declared universe the stage actually produced.

    ``expected`` is the universe the caller declares this stage owes a row for - the
    label artifact restricted to a fold's validation window, a trading calendar
    crossed with the symbol universe, the feature panel a model is scored on. Both
    frames are reduced to their distinct ``keys`` before comparison, so passing a
    frame with extra columns is fine.

    Returns ``summary`` with the counts, plus ``missing`` and ``unexpected`` carrying
    the key tuples themselves. The tuples are what make the number readable: a
    handful of symbols absent entirely and every symbol missing its first weeks give
    the same percentage and are a universe restriction and a warm-up respectively.
    Group ``missing`` by the entity column to tell them apart.
    """
    key_list = list(keys)
    mismatched = [
        (k, expected.schema[k], produced.schema[k])
        for k in key_list
        if expected.schema[k] != produced.schema[k]
    ]
    if mismatched:
        detail = "; ".join(
            f"{k}: expected {want} but produced {got}" for k, want, got in mismatched
        )
        raise ValueError(
            "key dtypes differ between the produced frame and the declared universe, so "
            f"every row would read as missing - {detail}. Cast explicitly in the notebook: "
            "a silent cast here is how a Date label artifact and a Datetime prediction set "
            "come to report full coverage of nothing."
        )
    got = produced.select(key_list).unique()
    want = expected.select(key_list).unique()
    missing = want.join(got, on=key_list, how="anti")
    unexpected = got.join(want, on=key_list, how="anti")
    summary = pl.DataFrame(
        [
            {
                "expected": want.height,
                "produced": got.height,
                "missing": missing.height,
                "unexpected": unexpected.height,
                "coverage": (want.height - missing.height) / want.height if want.height else None,
            }
        ]
    )
    return {"summary": summary, "missing": missing, "unexpected": unexpected}


def flag_columns(
    profile: pl.DataFrame,
    *,
    rules: dict[str, float] | None = None,
) -> pl.DataFrame:
    """The profile rows a reader should look at, with the reason attached.

    A flag is a request for a sentence of explanation, not a failure. Every rule is a
    ceiling on a column of ``profile``; a column crossing none of them is absent from
    the result. Non-finite values and constants are always flagged, because neither
    has a legitimate reading that does not need saying out loud.
    """
    active = DEFAULT_RULES | (rules or {})
    reasons: list[pl.Expr] = [
        pl.when(pl.col("n_nonfinite") > 0)
        .then(pl.format("{} non-finite values", pl.col("n_nonfinite")))
        .otherwise(None),
        pl.when(pl.col("constant")).then(pl.lit("constant over every row")).otherwise(None),
    ]
    for column, ceiling in active.items():
        if column not in profile.columns:
            continue
        reasons.append(
            pl.when(pl.col(column) > ceiling)
            .then(pl.format(f"{column} {{}} above {ceiling:g}", pl.col(column).round(4)))
            .otherwise(None)
        )
    flagged = profile.with_columns(pl.concat_list(reasons).list.drop_nulls().alias("flags")).filter(
        pl.col("flags").list.len() > 0
    )
    return flagged.select(
        "column", pl.col("flags").list.join("; ").alias("why"), pl.exclude("column", "flags")
    )


def quality_report(
    frame: pl.DataFrame,
    *,
    name: str,
    key_columns: Sequence[str] = (),
    expected: pl.DataFrame | None = None,
    keys: Sequence[str] | None = None,
    rules: dict[str, float] | None = None,
) -> dict[str, pl.DataFrame]:
    """Profile, flags and (when an expected universe is given) coverage, in one call.

    Returns the three frames rather than printing them, so the notebook decides how
    to display each and writes its own sign-off around them.
    """
    profile = profile_columns(frame, key_columns=key_columns)
    report: dict[str, pl.DataFrame] = {
        "profile": profile,
        "flags": flag_columns(profile, rules=rules),
    }
    if expected is not None:
        report.update(
            {
                f"coverage_{part}": value
                for part, value in coverage_against(
                    frame, expected, keys=keys or list(key_columns)
                ).items()
            }
        )
    report["name"] = name
    return report


def render_quality_report(report: dict, *, max_rows: int = 80) -> None:
    """Print a ``quality_report`` as a notebook reader should read it: shortfall first.

    Coverage leads because it is the question the other tables cannot answer - a frame
    whose every column profiles cleanly is still wrong if it is a tenth of the universe.
    The flags come next as the shortlist a sign-off has to speak to, and the full profile
    last, so nothing is hidden but nothing has to be scanned to find the problem.
    """
    print(f"=== {report['name']} ===")

    summary = report.get("coverage_summary")
    if summary is not None:
        row = summary.row(0, named=True)
        share = "n/a" if row["coverage"] is None else f"{row['coverage']:.2%}"
        print(
            f"coverage {share}: {row['produced']:,} of {row['expected']:,} declared keys, "
            f"{row['missing']:,} missing, {row['unexpected']:,} unexpected"
        )
        missing = report.get("coverage_missing")
        if missing is not None and missing.height:
            entity = missing.columns[0]
            per_entity = missing.group_by(entity).len().sort("len", descending=True)
            print(f"  {per_entity.height} of the declared {entity}s are short; worst:")
            with pl.Config(tbl_rows=10, tbl_hide_dataframe_shape=True):
                print(per_entity.head(10))

    flags = report["flags"]
    if flags.height == 0:
        print("no column crossed a threshold")
    else:
        print(f"{flags.height} column(s) to speak to:")
        with pl.Config(tbl_rows=max_rows, tbl_cols=4, fmt_str_lengths=110):
            print(flags.select("column", "why", "rows", "n_null"))

    with pl.Config(tbl_rows=max_rows, tbl_cols=14, fmt_str_lengths=28):
        print(report["profile"])
