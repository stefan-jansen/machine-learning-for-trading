"""A training run may not join a population fitted on a different vintage of its inputs.

A training run pins each input artifact by whole-file sha256 and then fits on whatever is on
disk, and until this guard nothing compared the two. Regenerate a stage-03 or stage-04
artifact and the next run registers against a vintage no prior member of the population was
fitted under, silently, and the mixture is found later by comparing the registry to the disk
by hand - if at all.

Measured across the nine production registries on 2026-09-07, before the guard existed:
`fx_pairs` holds 138 training runs pinning one `model_based` sha, and **zero** of them match
the file on disk. That state is safe only for as long as nobody runs a modelling notebook in
that worktree, and nothing enforced it.

ml4t/agent-workspace#987.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from case_studies.utils.registry.registration import (
    declare_artifact_supersession,
    register_training_run,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


def _spec(*, model_based: str, config_name: str = "ridge_default", label: str = "fwd_ret_21d"):
    """A resolved training spec in the shape six of the seven producers build.

    ``config_name`` also varies a hashed model parameter: the training identity is the
    computation, and a second run distinguished only by its config name resolves to the same
    hash and is served from the registry rather than registered.
    """
    return {
        "identity_version": 2,
        "execution_tier": "canonical",
        "family": "linear",
        "label": label,
        "seed": 42,
        "config_name": config_name,
        "computation": {
            "feature_names": ["momentum"],
            "model": {"class": "Ridge", "config": config_name},
            "input_data_spec": {
                "schema_version": 1,
                "artifacts": {
                    "financial": {"sha256": "f" * 64, "size": 11},
                    "label": {"sha256": "1" * 64, "size": 12},
                    "model_based": {"sha256": model_based, "size": 13},
                },
                "fingerprint": model_based[:16],
            },
        },
    }


def _registered(case_dir: Path) -> int:
    with closing(sqlite3.connect(case_dir / "run_log" / "registry.db")) as db:
        return db.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0]


def test_a_run_on_the_registered_vintage_registers(tmp_path: Path) -> None:
    """The control. Nothing about the guard blocks fitting on the artifact on disk."""
    register_training_run("etfs", _spec(model_based=SHA_A), case_dir=tmp_path)
    register_training_run(
        "etfs", _spec(model_based=SHA_A, config_name="ridge_wide"), case_dir=tmp_path
    )
    assert _registered(tmp_path) == 2


def test_a_run_on_a_regenerated_artifact_is_refused(tmp_path: Path) -> None:
    """The defect: the second run fits on a file the first was never fitted on."""
    register_training_run("etfs", _spec(model_based=SHA_A), case_dir=tmp_path)

    with pytest.raises(ValueError, match="model_based") as raised:
        register_training_run(
            "etfs", _spec(model_based=SHA_B, config_name="ridge_wide"), case_dir=tmp_path
        )

    # The message has to name both shas, because the author's next step is deciding whether
    # the regeneration was deliberate - and that is answered by which file is on disk.
    assert SHA_A in str(raised.value)
    assert SHA_B in str(raised.value)
    assert _registered(tmp_path) == 1, "the refusal must leave no row behind"


def test_the_refusal_lands_before_anything_is_written(tmp_path: Path) -> None:
    """It refuses before the fit, so nothing about the new vintage reaches disk.

    `register_training_run` writes an immutable `spec.json` under the training directory
    before it inserts. A guard that ran after that would leave an orphaned artifact
    directory for a run that was refused, which the next registration then reads as a
    conflicting spec.
    """
    register_training_run("etfs", _spec(model_based=SHA_A), case_dir=tmp_path)
    before = sorted(p.name for p in (tmp_path / "run_log" / "training").iterdir())

    with pytest.raises(ValueError, match="model_based"):
        register_training_run(
            "etfs", _spec(model_based=SHA_B, config_name="ridge_wide"), case_dir=tmp_path
        )

    assert sorted(p.name for p in (tmp_path / "run_log" / "training").iterdir()) == before


def test_a_declared_supersession_lets_the_new_vintage_register(tmp_path: Path) -> None:
    """Superseding an artifact on purpose is legitimate, so the refusal has an override."""
    register_training_run("etfs", _spec(model_based=SHA_A), case_dir=tmp_path)
    declare_artifact_supersession(
        "etfs", "model_based", sha256=SHA_B, supersedes_sha256=SHA_A, case_dir=tmp_path
    )
    register_training_run(
        "etfs", _spec(model_based=SHA_B, config_name="ridge_wide"), case_dir=tmp_path
    )
    assert _registered(tmp_path) == 2


def test_a_declaration_naming_an_unregistered_sha_is_refused(tmp_path: Path) -> None:
    """A mistyped predecessor must not be recorded as though it had unblocked something.

    This is a function an author calls by hand with a hash copied out of an error message,
    which is the same reason `declare_causal_supersedes` validates its predecessor.
    """
    register_training_run("etfs", _spec(model_based=SHA_A), case_dir=tmp_path)
    with pytest.raises(ValueError, match="nothing to supersede"):
        declare_artifact_supersession(
            "etfs", "model_based", sha256=SHA_B, supersedes_sha256="c" * 64, case_dir=tmp_path
        )


def test_a_declaration_does_not_wave_through_a_third_vintage(tmp_path: Path) -> None:
    """The declaration names one sha, so it retires that one and nothing else.

    Without this the override would be a bypass with a hash attached: declare once and every
    later regeneration of the same artifact registers unremarked.
    """
    register_training_run("etfs", _spec(model_based=SHA_A), case_dir=tmp_path)
    declare_artifact_supersession(
        "etfs", "model_based", sha256=SHA_B, supersedes_sha256=SHA_A, case_dir=tmp_path
    )
    register_training_run(
        "etfs", _spec(model_based=SHA_B, config_name="ridge_wide"), case_dir=tmp_path
    )

    with pytest.raises(ValueError, match="model_based"):
        register_training_run(
            "etfs", _spec(model_based="c" * 64, config_name="ridge_deep"), case_dir=tmp_path
        )


def test_a_second_regeneration_registers_on_a_declared_chain(tmp_path: Path) -> None:
    """A regeneration of a regeneration, which is the same operation done twice.

    Declaring that C replaces B is the whole of what an author can say: A already names B as
    its successor, and `declare_artifact_supersession` refuses a second successor for one
    sha, so C -> A cannot be declared. But the population still holds the run fitted on A - a
    declaration retires a vintage for later runs, it does not delete the runs that used it -
    so reading one edge demanded a declaration that cannot be made, and the error named it.
    """
    register_training_run("etfs", _spec(model_based=SHA_A), case_dir=tmp_path)
    declare_artifact_supersession(
        "etfs", "model_based", sha256=SHA_B, supersedes_sha256=SHA_A, case_dir=tmp_path
    )
    register_training_run(
        "etfs", _spec(model_based=SHA_B, config_name="ridge_wide"), case_dir=tmp_path
    )

    sha_c = "c" * 64
    declare_artifact_supersession(
        "etfs", "model_based", sha256=sha_c, supersedes_sha256=SHA_B, case_dir=tmp_path
    )
    register_training_run(
        "etfs", _spec(model_based=sha_c, config_name="ridge_deep"), case_dir=tmp_path
    )
    assert _registered(tmp_path) == 3


def test_a_chain_does_not_retire_a_vintage_outside_it(tmp_path: Path) -> None:
    """The control: ancestry is followed, not assumed. A sha nobody declared stays refused."""
    register_training_run("etfs", _spec(model_based=SHA_A), case_dir=tmp_path)
    declare_artifact_supersession(
        "etfs", "model_based", sha256=SHA_B, supersedes_sha256=SHA_A, case_dir=tmp_path
    )
    register_training_run(
        "etfs", _spec(model_based=SHA_B, config_name="ridge_wide"), case_dir=tmp_path
    )
    # `d` replaces `c`, which no run was ever fitted on, so it retires nothing in this
    # population and the run reading it is still joining two vintages.
    register_training_run(
        "etfs", _spec(model_based="c" * 64, label="fwd_ret_5d"), case_dir=tmp_path
    )
    declare_artifact_supersession(
        "etfs", "model_based", sha256="d" * 64, supersedes_sha256="c" * 64, case_dir=tmp_path
    )
    with pytest.raises(ValueError, match="model_based"):
        register_training_run(
            "etfs", _spec(model_based="d" * 64, config_name="ridge_deep"), case_dir=tmp_path
        )


def test_another_label_is_compared_against_its_own_runs(tmp_path: Path) -> None:
    """The comparison is per label, because `label` names a different file for each one.

    `labels/<label>.parquet` is registered under the artifact name `label`, so a comparison
    across labels would refuse the first run of every new label on the grounds that its own
    label artifact is not the previous label's.
    """
    register_training_run("etfs", _spec(model_based=SHA_A), case_dir=tmp_path)

    other = _spec(model_based=SHA_A, label="fwd_ret_5d")
    other["computation"]["input_data_spec"]["artifacts"]["label"] = {"sha256": "2" * 64, "size": 9}
    register_training_run("etfs", other, case_dir=tmp_path)
    assert _registered(tmp_path) == 2


def _latent_spec(*, model_based: str, config_name: str = "pca_5", prefix: str = "sha256:"):
    """The shape the latent adapter records: a `files` list of {role, sha256} (#891).

    The `sha256:` prefix is the adapter's own, and it is what made these look like a
    different contract rather than the same one written differently.
    """
    spec = _spec(model_based=model_based, config_name=config_name)
    spec["family"] = "latent_factors"
    spec["computation"]["input_data_spec"] = {
        "schema_version": 1,
        "files": [
            {"role": "financial", "sha256": f"{prefix}{'f' * 64}"},
            {"role": "label", "sha256": f"{prefix}{'1' * 64}"},
            {"role": "model_based", "sha256": f"{prefix}{model_based}"},
        ],
    }
    return spec


def _sequence_spec(*, model_based: str, config_name: str = "tcn_default"):
    """The shape `deep_learning` records: `mds.input_lineage` nested one level too deep.

    `case_studies/utils/deep_learning.py` builds the payload as
    `{"input_data_spec": mds.input_lineage, ...}`, so the artifacts mapping lands at
    `computation.input_data_spec.input_data_spec.artifacts`. A nesting slip rather than a
    different contract, and it put 77 runs outside the guard.
    """
    spec = _spec(model_based=model_based, config_name=config_name)
    spec["family"] = "deep_learning"
    spec["computation"]["input_data_spec"] = {
        "input_data_spec": spec["computation"]["input_data_spec"],
        "lookback": 32,
    }
    return spec


def test_a_latent_run_on_the_registered_vintage_registers(tmp_path: Path) -> None:
    """The control for the `files` shape: the prefix is stripped, not compared."""
    register_training_run("etfs", _spec(model_based=SHA_A), case_dir=tmp_path)
    register_training_run("etfs", _latent_spec(model_based=SHA_A), case_dir=tmp_path)
    assert _registered(tmp_path) == 2


def test_a_latent_run_on_a_regenerated_artifact_is_refused(tmp_path: Path) -> None:
    """What #1137 changes. This registered silently before: `_input_artifact_shas` read
    `artifacts` only, returned an empty mapping for a `files` spec, and the guard returns
    early on an empty mapping - so 42 latent runs were vintage-checked by nothing."""
    register_training_run("etfs", _spec(model_based=SHA_A), case_dir=tmp_path)
    with pytest.raises(ValueError, match="model_based"):
        register_training_run("etfs", _latent_spec(model_based=SHA_B), case_dir=tmp_path)
    assert _registered(tmp_path) == 1


def test_an_unprefixed_files_sha_compares_the_same(tmp_path: Path) -> None:
    """The prefix is cosmetic, so its absence must not make a run unrefusable."""
    register_training_run("etfs", _spec(model_based=SHA_A), case_dir=tmp_path)
    with pytest.raises(ValueError, match="model_based"):
        register_training_run("etfs", _latent_spec(model_based=SHA_B, prefix=""), case_dir=tmp_path)


def test_a_sequence_run_on_the_registered_vintage_registers(tmp_path: Path) -> None:
    """The control for the nested shape."""
    register_training_run("etfs", _spec(model_based=SHA_A), case_dir=tmp_path)
    register_training_run("etfs", _sequence_spec(model_based=SHA_A), case_dir=tmp_path)
    assert _registered(tmp_path) == 2


def test_a_sequence_run_on_a_regenerated_artifact_is_refused(tmp_path: Path) -> None:
    """The other half of #1137, and the larger one: 77 deep_learning runs."""
    register_training_run("etfs", _spec(model_based=SHA_A), case_dir=tmp_path)
    with pytest.raises(ValueError, match="model_based"):
        register_training_run("etfs", _sequence_spec(model_based=SHA_B), case_dir=tmp_path)
    assert _registered(tmp_path) == 1


def test_a_sequence_run_is_what_a_later_run_is_compared_against(tmp_path: Path) -> None:
    """Reading a shape is only half of covering it.

    A widened reader that still wrote nothing comparable would leave the population exactly
    as unchecked, because `_registered_artifact_shas` builds the set every later run is
    measured against from the same function. So the direction that matters is this one: a
    sequence run registers FIRST, and an ordinary run on a different vintage is then refused
    against it.
    """
    register_training_run("etfs", _sequence_spec(model_based=SHA_A), case_dir=tmp_path)
    with pytest.raises(ValueError, match="model_based"):
        register_training_run("etfs", _spec(model_based=SHA_B), case_dir=tmp_path)
    assert _registered(tmp_path) == 1


def test_the_latent_eval_label_role_is_compared_against_the_other_families(
    tmp_path: Path,
) -> None:
    """Two words for one artifact is checked never, however wide the reader gets.

    The guard compares per NAME. The latent adapter's `files` list calls the evaluation
    label `evaluation_label` and gbm, linear and tabular_dl call it `eval_label`, so the
    same file was pinned under two names and a latent run's pin was only ever compared
    against other latent runs'. Measured before the alias landed, the only live case is
    `us_firm_characteristics/fwd_class_1m`, where all four families pin 04c53aa6a847 - so
    folding the names together refuses nothing that registers today and starts comparing
    what it was not comparing.
    """
    ordinary = _spec(model_based=SHA_A)
    ordinary["computation"]["input_data_spec"]["artifacts"]["eval_label"] = {
        "sha256": "e" * 64,
        "size": 14,
    }
    register_training_run("etfs", ordinary, case_dir=tmp_path)

    latent = _latent_spec(model_based=SHA_A, config_name="pca_eval")
    latent["computation"]["input_data_spec"]["files"].append(
        {"role": "evaluation_label", "sha256": "sha256:" + "9" * 64}
    )
    with pytest.raises(ValueError, match="eval_label"):
        register_training_run("etfs", latent, case_dir=tmp_path)
    assert _registered(tmp_path) == 1


def test_the_alias_does_not_refuse_the_vintage_the_population_holds(tmp_path: Path) -> None:
    """The other half, and the one that says the alias is safe rather than merely strict."""
    ordinary = _spec(model_based=SHA_A)
    ordinary["computation"]["input_data_spec"]["artifacts"]["eval_label"] = {
        "sha256": "e" * 64,
        "size": 14,
    }
    register_training_run("etfs", ordinary, case_dir=tmp_path)

    latent = _latent_spec(model_based=SHA_A, config_name="pca_eval")
    latent["computation"]["input_data_spec"]["files"].append(
        {"role": "evaluation_label", "sha256": "sha256:" + "e" * 64}
    )
    register_training_run("etfs", latent, case_dir=tmp_path)
    assert _registered(tmp_path) == 2


def test_a_files_list_pinning_one_role_twice_is_refused(tmp_path: Path) -> None:
    """A spec defect rather than a second vintage, so the later record must not win quietly."""
    latent = _latent_spec(model_based=SHA_A)
    latent["computation"]["input_data_spec"]["files"].append(
        {"role": "financial", "sha256": "sha256:" + "7" * 64}
    )
    with pytest.raises(ValueError, match="two shas for role"):
        register_training_run("etfs", latent, case_dir=tmp_path)


def test_a_run_pinning_no_artifacts_at_all_is_not_blocked(tmp_path: Path) -> None:
    """A spec carrying none of the three shapes still pins nothing and is not refused.

    The guard returns early on an empty mapping, and that has to stay true: a family whose
    inputs this cannot see is a weaker check for that family rather than a refusal it can
    never satisfy. With all three shapes read, nothing in the nine live registries is in
    this state - but a spec that predates them, or a future producer, can be.
    """
    register_training_run("etfs", _spec(model_based=SHA_A), case_dir=tmp_path)
    bare = _spec(model_based=SHA_B, config_name="bare")
    bare["computation"]["input_data_spec"] = {"schema_version": 1, "fingerprint": "none"}
    register_training_run("etfs", bare, case_dir=tmp_path)
    assert _registered(tmp_path) == 2


def test_recording_a_supersession_declares_it_in_the_registry(tmp_path: Path) -> None:
    """One author action leaves both records, so the refusal names a command that finishes.

    `scripts/record_artifact_supersession.py` establishes that a new artifact extended the
    old one rather than replacing it, fold by fold, while both files are still on disk. That
    is exactly the case where a new training run should be allowed to join the population
    fitted on the old vintage, so the script now declares it here too.
    """
    import hashlib
    import subprocess
    import sys

    import polars as pl

    from case_studies.utils.artifact_digest import write_artifact

    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _frame(folds: range) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "fold": [f for f in folds for _ in range(2)],
                "symbol": [s for _ in folds for s in ("AAA", "BBB")],
                "feature": [float(f * 10 + i) for f in folds for i in range(2)],
            }
        )

    features = tmp_path / "features"
    features.mkdir()
    old = features / "superseded.parquet"
    write_artifact(
        _frame(range(3)), old, keys=["fold", "symbol"], written_by="t", fold_column="fold"
    )
    new = features / "model_based.parquet"
    write_artifact(
        _frame(range(4)), new, keys=["fold", "symbol"], written_by="t", fold_column="fold"
    )

    register_training_run("etfs", _spec(model_based=_sha(old)), case_dir=tmp_path)

    repo_root = Path(__file__).resolve().parent.parent
    done = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "record_artifact_supersession.py"),
            "--superseded",
            str(old),
            "--current",
            str(new),
        ],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert done.returncode == 0, done.stderr
    assert "supersedes" in done.stdout, done.stdout

    # The point of the wiring: no second command between the recorder and a run that fits
    # on the file it just established as an extension.
    register_training_run(
        "etfs", _spec(model_based=_sha(new), config_name="ridge_wide"), case_dir=tmp_path
    )
    assert _registered(tmp_path) == 2
