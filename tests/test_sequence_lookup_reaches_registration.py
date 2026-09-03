"""A notebook that looks up its own fit has to build the identity the fit was registered under.

``run_dl_cv`` builds every registration identity through ``sequence_identity_params``
(``deep_learning.py``), and hands the result to ``build_training_spec`` as ``extra_params``. A
notebook that wants to find an existing fit - to skip a refit, or to rebuild a checkpoint curve
after one - has to arrive at the same ``training_hash``, and transcribing the builder's fields
into a literal is the way that stops being true. ``darts_forecasting.py`` states the consequence
directly: "any field in one and not the other means the lookup can never find the registration".

That happened, and the failure is not a cache miss costing a refit. #620 made ``device``
identity-bearing; ``etfs/09_dl_lstm`` and ``10_dl_tsmixer`` hand-wrote their lookup dicts without
it. The config trains, registers under the device-qualified hash, and the post-training rebuild
finds nothing - so the run dies with "Training completed but registered checkpoints are
incomplete" after paying for the fit.

``test_sequence_identity_roundtrip.py`` pins what the builder puts in the identity and says
explicitly that it does not check the call sites, because a test comparing the builder against a
transcription written in the test file would be comparing that file to itself. This is the other
half: it reads the notebooks, so it goes green because a notebook changed.

The check is which function the argument comes from rather than which fields it contains. Asserting
the field list would need updating every time one is added, which is the same transcription defect
one layer out.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BUILDER = "sequence_identity_params"
SPEC = "build_training_spec"


def _notebooks() -> list[Path]:
    """Case-study notebooks that both register sequence fits and look them up.

    ``run_dl_cv`` is what registers through the builder, and ``build_training_spec`` is what a
    hand-rolled lookup calls. A notebook doing only one of the two has no pair to disagree.
    """
    return [
        path
        for path in sorted(REPO.glob("case_studies/*/[0-9]*.py"))
        if "run_dl_cv" in (source := path.read_text(encoding="utf-8")) and SPEC in source
    ]


def _builder_names(tree: ast.Module) -> set[str]:
    """Names bound to a ``sequence_identity_params`` call anywhere in the module."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if (func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)) != BUILDER:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                bound.add(target.id)
    return bound


def _transcribed_lookups(path: Path) -> list[int]:
    """Line numbers of ``build_training_spec`` calls whose identity is not the builder's."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bound = _builder_names(tree)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)) != SPEC:
            continue
        for keyword in node.keywords:
            if keyword.arg != "extra_params":
                continue
            value = keyword.value
            if isinstance(value, ast.Call):
                called = value.func
                name = called.id if isinstance(called, ast.Name) else getattr(called, "attr", None)
                if name == BUILDER:
                    continue
            elif isinstance(value, ast.Name) and value.id in bound:
                continue
            offenders.append(node.lineno)
    return offenders


@pytest.mark.parametrize("path", _notebooks(), ids=lambda p: p.relative_to(REPO).as_posix())
def test_the_lookup_identity_comes_from_the_builder(path: Path) -> None:
    offenders = _transcribed_lookups(path)
    assert not offenders, (
        f"{path.relative_to(REPO)} builds a lookup spec at line(s) {offenders} without going "
        f"through {BUILDER}, so any identity-bearing field the builder adds - device, among "
        "others - is missing from the hash it queries and the lookup can never find the "
        "registration."
    )


def test_the_check_finds_a_transcribed_dict(tmp_path: Path) -> None:
    """The guard fails on the shape it exists to catch, not only on the shape it has now."""
    path = tmp_path / "09_dl_example.py"
    path.write_text(
        "spec = build_training_spec('deep_learning', 'lstm', 'y', "
        "extra_params={'batch_size': 512})\n",
        encoding="utf-8",
    )
    assert _transcribed_lookups(path) == [1]


def test_a_name_bound_to_the_builder_passes(tmp_path: Path) -> None:
    path = tmp_path / "09_dl_example.py"
    path.write_text(
        "params = sequence_identity_params(cfg, device=d)\n"
        "spec = build_training_spec('deep_learning', 'lstm', 'y', extra_params=params)\n",
        encoding="utf-8",
    )
    assert _transcribed_lookups(path) == []
