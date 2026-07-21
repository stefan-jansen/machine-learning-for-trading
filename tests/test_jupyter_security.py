"""Regression guards for the unauthenticated local Jupyter containers."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
JUPYTER_DOCKERFILES = (
    "envs/ml4t/Dockerfile",
    "envs/py312/Dockerfile",
    "envs/benchmark/Dockerfile",
    "envs/benchmark/Dockerfile.full",
)
JUPYTER_RUNTIME_CONFIGS = (*JUPYTER_DOCKERFILES, "docker-compose.yml")

UNSAFE_JUPYTER_PATTERNS = (
    re.compile(r"\b(?:ServerApp|NotebookApp)\.allow_origin\s*=\s*['\"]\*['\"]"),
    re.compile(r"\b(?:ServerApp|NotebookApp)\.disable_check_xsrf\s*=\s*True\b"),
    re.compile(r"--(?:ServerApp|NotebookApp)\.allow_origin(?:=|\s+)['\"]?\*"),
    re.compile(r"--(?:ServerApp|NotebookApp)\.disable_check_xsrf(?:=|\s+)True\b"),
)


def _unsafe_jupyter_lines(text: str) -> list[str]:
    executable = [line.split("#", 1)[0] for line in text.splitlines()]
    return [
        line.strip() for line in executable if any(p.search(line) for p in UNSAFE_JUPYTER_PATTERNS)
    ]


def test_jupyter_images_keep_origin_and_xsrf_checks_enabled() -> None:
    offenders = []
    for relative in JUPYTER_RUNTIME_CONFIGS:
        text = (REPO_ROOT / relative).read_text()
        offenders.extend(f"{relative}: {line}" for line in _unsafe_jupyter_lines(text))

    assert not offenders, (
        "Local Jupyter runs without a token, so origin and XSRF checks must remain enabled:\n  "
        + "\n  ".join(offenders)
    )


def test_jupyter_security_guard_ignores_safe_settings_and_comments() -> None:
    safe = """
# c.ServerApp.disable_check_xsrf = True
c.ServerApp.disable_check_xsrf = False
c.ServerApp.allow_origin = 'http://127.0.0.1:8888'
"""
    unsafe = """
c.ServerApp.disable_check_xsrf = True
jupyter lab --ServerApp.allow_origin='*'
"""
    assert _unsafe_jupyter_lines(safe) == []
    assert len(_unsafe_jupyter_lines(unsafe)) == 2
