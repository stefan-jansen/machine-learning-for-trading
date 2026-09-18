"""A cross-case-study insight table must rank the live generation, not the registry's history.

A registry keeps every generation because the record of what was superseded is evidence, and
supersession is recorded one layer up in ``official_populations``. A reader that joins
``training_runs`` to ``prediction_metrics`` therefore sees retired rows beside live ones and
cannot tell them apart on the numbers, since a refit that only re-declared an input leaves
bit-identical metrics.

Ranking over that is not a neutral error. A generation is usually refitted because it was
short or narrow; a narrower sample is an easier one, so the stale rows run toward the *top* of
the ranking. Measured on ``cme_futures`` 2026-09-18: the retired ``deep_learning`` generation
covers 27,326 of 38,262 declared (entity, session) pairs, 71.4%, and outranked the complete
refit that replaced it. Two cells of the book's Table 14.3 came from retired rows.

Both readers behind those tables are covered here, because they are separate query paths:
``insight_chapter._raw_primary_candidates`` (Chapters 12, 13, 14) and
``model_analysis.load_metrics_from_registry`` (Chapters 12, 13, 14's supervised side).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import polars as pl
import pytest

from case_studies.utils.insight_chapter import (
    SYMMETRY_TABLE_SCHEMA,
    collect_rank1_per_cs,
    discover_symmetry_pairs,
)
from case_studies.utils.model_analysis import load_metrics_from_registry
from case_studies.utils.registry import (
    register_fold_metrics,
    register_prediction_set,
    register_training_run,
)

RETIRED_IC = 0.40  # the easier, shorter generation: it wins on score alone
LIVE_IC = 0.12


def _register(case_dir: Path, config_name: str, daily_ic: float, lineage: str) -> str:
    training_hash = register_training_run(
        "test",
        {
            "family": "deep_learning",
            "label": "fwd_ret_5d",
            "config_name": config_name,
            "params": {"input_lineage": lineage},
            "seed": 42,
            "n_folds": 2,
        },
        case_dir=case_dir,
    )
    prediction_hash = register_prediction_set(
        "test",
        training_hash,
        checkpoint_value=5,
        checkpoint_kind="epoch",
        split="validation",
        metrics={
            "ic_mean": daily_ic,
            "ic_mean_daily": daily_ic,
            "ic_std": 0.01,
            "ic_n_days": 100,
            "ic_se_hac": 0.02,
            "ic_ci_lo": daily_ic - 0.04,
            "ic_ci_hi": daily_ic + 0.04,
            "ic_t_hac": daily_ic / 0.02,
            "ic_p_hac": 0.01,
            "ic_hac_lag": 1,
            "auc_mean_daily": 0.5 + daily_ic,
            "auc_se_hac": 0.01,
            "auc_ci_lo": 0.5 + daily_ic - 0.01,
            "auc_ci_hi": 0.5 + daily_ic + 0.01,
            "auc_n_days": 100,
        },
        case_dir=case_dir,
    )
    register_fold_metrics(
        "test",
        prediction_hash,
        {0: {"ic": daily_ic}, 1: {"ic": daily_ic}},
        case_dir=case_dir,
    )
    return prediction_hash


def _publish(case_dir: Path, population_hash: str, members: list[str], supersedes: str | None):
    snapshot = {
        "schema_version": 1,
        "name": "test-deep_learning-validation-v1",
        "member_kind": "prediction",
        "members": members,
        "supersedes": supersedes,
    }
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        db.execute(
            "INSERT INTO official_populations "
            "(population_hash, name, member_kind, snapshot_json, supersedes_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                population_hash,
                snapshot["name"],
                "prediction",
                json.dumps(snapshot),
                supersedes,
                "2026-01-01T00:00:00+00:00",
            ),
        )


def _two_generations(case_dir: Path) -> tuple[str, str]:
    retired = _register(case_dir, "seq_model", RETIRED_IC, "narrow")
    live = _register(case_dir, "seq_model", LIVE_IC, "complete")
    _publish(case_dir, "pop000000000a", [retired], supersedes=None)
    _publish(case_dir, "pop000000000b", [live], supersedes="pop000000000a")
    return retired, live


def test_the_shared_metrics_loader_drops_a_retired_generation(tmp_path, monkeypatch) -> None:
    retired, live = _two_generations(tmp_path)
    monkeypatch.setattr("case_studies.utils.model_analysis.get_case_study_dir", lambda _: tmp_path)

    metrics = load_metrics_from_registry("test", families=["deep_learning"])

    assert metrics["prediction_hash"].to_list() == [live]
    assert metrics.sort("ic_mean", descending=True)["ic_mean"][0] == LIVE_IC
    # Without the filter the retired row wins on score, which is the defect.
    history = load_metrics_from_registry("test", families=["deep_learning"], include_retired=True)
    assert set(history["prediction_hash"].to_list()) == {retired, live}
    assert history.sort("ic_mean", descending=True)["prediction_hash"][0] == retired


def test_the_rank1_selector_drops_a_retired_generation(tmp_path, monkeypatch) -> None:
    retired, live = _two_generations(tmp_path)
    monkeypatch.setattr("case_studies.utils.insight_chapter.get_case_study_dir", lambda _: tmp_path)
    monkeypatch.setattr(
        "case_studies.utils.insight_chapter._resolve_label", lambda *_args, **_kw: "fwd_ret_5d"
    )

    selected = collect_rank1_per_cs(["test"], "deep_learning")

    assert selected.height == 1
    assert selected["prediction_hash"].to_list() == [live]
    assert selected["ic_mean_daily"].to_list() == [LIVE_IC]
    assert retired not in selected["prediction_hash"].to_list()


def test_the_rank1_selector_carries_the_auc_columns_the_book_tables_print(
    tmp_path, monkeypatch
) -> None:
    """Chapter 11's "Native AUC" and Chapter 12's "Classifier -> AUC" are these columns.

    They sit in the same ``prediction_metrics`` row as the IC. The query named the seven
    ``ic_*`` columns and none of the five ``auc_*`` ones, so both notebooks published a
    column neither could reproduce from its own output.
    """
    _two_generations(tmp_path)
    monkeypatch.setattr("case_studies.utils.insight_chapter.get_case_study_dir", lambda _: tmp_path)
    monkeypatch.setattr(
        "case_studies.utils.insight_chapter._resolve_label", lambda *_args, **_kw: "fwd_ret_5d"
    )

    selected = collect_rank1_per_cs(["test"], "deep_learning")

    for column in ("auc_mean_daily", "auc_se_hac", "auc_ci_lo", "auc_ci_hi", "auc_n_days"):
        assert column in selected.columns, column
    assert selected["auc_mean_daily"].to_list() == [0.5 + LIVE_IC]


def _declare(case_dir: Path, mapping: dict[str, str]) -> None:
    """Write the `labels.classification_eval_label` block the discovery reads."""
    config = case_dir / "config"
    config.mkdir(parents=True, exist_ok=True)
    (config / "setup.yaml").write_text(
        "labels:\n  classification_eval_label:\n"
        + "".join(f"    {direction}: {regression}\n" for direction, regression in mapping.items())
    )


def _register_pair(case_dir: Path, regression: str, direction: str, domain: list[int]) -> None:
    for label in (regression, direction):
        training_hash = register_training_run(
            "test",
            {"family": "gbm", "label": label, "config_name": "c", "params": {}, "seed": 1},
            case_dir=case_dir,
        )
        register_prediction_set(
            "test",
            training_hash,
            checkpoint_value=1,
            split="validation",
            metrics={"ic_mean": 0.1, "ic_mean_daily": 0.1, "ic_std": 0.01},
            case_dir=case_dir,
        )
    labels_dir = case_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({direction: domain}).write_parquet(labels_dir / f"{direction}.parquet")


def test_symmetry_pairs_are_discovered_and_a_skip_is_named(tmp_path, monkeypatch) -> None:
    """A hand-written pair list cannot report what has gone missing from it.

    Chapter 12's literal lost its ``us_firm_characteristics`` entry in the 2026-07-31
    chapter-tree restore and went on reporting "3 of 3 matched cells" - true of its own
    list and false of the corpus. Discovery replaces the list; a ternary label is excluded
    by measuring its domain and naming it, not by absence.
    """
    _register_pair(tmp_path, "fwd_ret_1m", "fwd_class_1m", [0, 1])
    _register_pair(tmp_path, "fwd_ret_15m", "fwd_dir_15m", [-1, 0, 1])
    _declare(tmp_path, {"fwd_class_1m": "fwd_ret_1m", "fwd_dir_15m": "fwd_ret_15m"})
    monkeypatch.setattr("case_studies.utils.insight_chapter.get_case_study_dir", lambda _: tmp_path)

    pairs, skipped = discover_symmetry_pairs(["test"], "gbm")

    assert pairs == {"test": [("fwd_ret_1m", "fwd_class_1m")]}
    assert len(skipped) == 1
    assert "fwd_dir_15m" in skipped[0]
    assert "[-1, 0, 1]" in skipped[0]


def test_two_direction_labels_on_one_regression_label_are_both_examined(
    tmp_path, monkeypatch
) -> None:
    """Crypto's shape, and the reason discovery reads the declaration and not the names.

    ``crypto_perps_funding`` maps both ``fwd_dir_8h`` and ``fwd_dir_8h_3c`` onto
    ``fwd_ret_8h``. A rule that builds the direction name from the regression suffix
    finds the binary one and never sees the ternary one, so it cannot report skipping
    it - the exact silence this function exists to remove.
    """
    _register_pair(tmp_path, "fwd_ret_8h", "fwd_dir_8h", [0, 1])
    _register_pair(tmp_path, "fwd_ret_8h", "fwd_dir_8h_3c", [-1, 0, 1])
    _declare(tmp_path, {"fwd_dir_8h": "fwd_ret_8h", "fwd_dir_8h_3c": "fwd_ret_8h"})
    monkeypatch.setattr("case_studies.utils.insight_chapter.get_case_study_dir", lambda _: tmp_path)

    pairs, skipped = discover_symmetry_pairs(["test"], "gbm")

    assert pairs == {"test": [("fwd_ret_8h", "fwd_dir_8h")]}
    assert len(skipped) == 1
    assert "fwd_dir_8h_3c" in skipped[0]


def test_a_direction_label_with_no_surface_on_disk_is_named_not_dropped(
    tmp_path, monkeypatch
) -> None:
    _register_pair(tmp_path, "fwd_ret_1m", "fwd_class_1m", [0, 1])
    _declare(tmp_path, {"fwd_class_1m": "fwd_ret_1m"})
    (tmp_path / "labels" / "fwd_class_1m.parquet").unlink()
    monkeypatch.setattr("case_studies.utils.insight_chapter.get_case_study_dir", lambda _: tmp_path)

    pairs, skipped = discover_symmetry_pairs(["test"], "gbm")

    assert pairs == {}
    assert skipped == ["test/fwd_class_1m: no label surface on disk"]


def test_discovery_can_return_nothing_and_the_shared_table_schema_still_renders(
    tmp_path, monkeypatch
) -> None:
    """An empty result must render, not raise, and this pins the notebooks' own schema.

    Both notebooks build their symmetry table as
    ``pl.DataFrame(sym_rows, schema=SYMMETRY_TABLE_SCHEMA)`` and then select
    ``short_name``, ``reg_label`` and ``dir_label``. Built from an empty list with only
    numeric ``schema_overrides`` - what they did before 2026-09-18 - the frame has no
    string columns and that selection raises ``ColumnNotFoundError``, so the run dies on
    the one path where the skip reasons are the whole output.

    This asserts against the shared constant the notebooks import, not against a copy, so
    dropping a string column from it fails here rather than in a render.
    """
    _register_pair(tmp_path, "fwd_ret_1m", "fwd_class_1m", [-1, 0, 1])
    _declare(tmp_path, {"fwd_class_1m": "fwd_ret_1m"})
    monkeypatch.setattr("case_studies.utils.insight_chapter.get_case_study_dir", lambda _: tmp_path)

    pairs, skipped = discover_symmetry_pairs(["test"], "gbm")
    assert pairs == {}
    assert len(skipped) == 1

    # The notebooks' construction, verbatim, on the empty result discovery just returned.
    sym_df = pl.DataFrame([], schema=SYMMETRY_TABLE_SCHEMA)
    rendered = sym_df.select(
        "short_name",
        "reg_label",
        "dir_label",
        pl.col("cls_score_ic").round(4).alias("A_ic"),
        pl.col("cls_score_auc").round(4).alias("native_auc"),
        pl.col("reg_score_auc").round(4).alias("B_auc"),
    )
    assert rendered.is_empty()
    assert sym_df.is_empty()
    assert [f"{r['short_name']} · {r['dir_label']}" for r in sym_df.iter_rows(named=True)] == []

    # The construction it replaced, as the negative control.
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        pl.DataFrame([], schema_overrides={"cls_score_ic": pl.Float64}).select("short_name")


def test_the_shared_table_schema_covers_every_column_both_notebooks_select() -> None:
    """A column either notebook renders has to be in the schema, or an empty run dies.

    Read from the notebook sources rather than listed here: a list would be a third copy
    of the column set and would go stale the first time a column is added.
    """
    import re
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    for notebook in (
        "11_ml_pipeline/07_case_study_insights.py",
        "12_gradient_boosting/12_case_study_insights.py",
    ):
        source = (repo / notebook).read_text()
        block = source[source.index("sym_df = pl.DataFrame(sym_rows") :]
        block = block[: block.index("\n# %%", block.index("sym_df.select("))]
        selected = set(re.findall(r'pl\.col\("(\w+)"\)', block)) | set(
            re.findall(r'^\s+"(\w+)",$', block, re.M)
        )
        missing = {c for c in selected if c.startswith(("cls_", "reg_", "short_", "dir_", "n_b"))}
        missing -= set(SYMMETRY_TABLE_SCHEMA)
        assert not missing, f"{notebook} renders columns absent from the schema: {sorted(missing)}"
