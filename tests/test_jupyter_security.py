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
    re.compile(r"(?<!-)\b(?:ServerApp|NotebookApp)\.allow_origin\s*=\s*['\"]\*['\"]"),
    re.compile(
        r"(?<!-)\b(?:ServerApp|NotebookApp)\.disable_check_xsrf\s*=\s*['\"]?"
        r"(?:true|1|yes|on)['\"]?(?=\s|$|[,}\]])",
        re.IGNORECASE,
    ),
    re.compile(r"--(?:ServerApp|NotebookApp)\.allow_origin(?:=|\s+)['\"]?\*"),
    re.compile(
        r"--(?:ServerApp|NotebookApp)\.disable_check_xsrf(?:=|\s+)['\"]?"
        r"(?:true|1|yes|on)['\"]?(?=\s|$|[,}\]])",
        re.IGNORECASE,
    ),
)
ORIGIN_PATTERN = re.compile(
    r"(?:\b(?:ServerApp|NotebookApp)\.allow_origin_pat\s*=\s*|"
    r"--(?:ServerApp|NotebookApp)\.allow_origin_pat(?:=|\s+))"
    r"[rRuU]?(?P<quote>['\"])(?P<pattern>.*?)(?P=quote)"
)


def _matches_unrelated_origins(pattern: str) -> bool:
    try:
        compiled = re.compile(pattern)
    except re.error:
        return False
    origins = ("https://evil.example", "http://unrelated.invalid:9999")
    return all(compiled.match(origin) for origin in origins)


def _unsafe_jupyter_lines(text: str) -> list[str]:
    executable = " ".join(line.split("#", 1)[0] for line in text.splitlines())
    normalized = re.sub(r"\\\s+", " ", executable)
    normalized = re.sub(r"\s+", " ", normalized)
    offenders = [
        match.group(0)
        for pattern in UNSAFE_JUPYTER_PATTERNS
        for match in pattern.finditer(normalized)
    ]
    offenders.extend(
        f"allow_origin_pat={match.group('pattern')!r}"
        for match in ORIGIN_PATTERN.finditer(normalized)
        if _matches_unrelated_origins(match.group("pattern"))
    )
    return offenders


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
    safe = r"""
# c.ServerApp.disable_check_xsrf = True
c.ServerApp.disable_check_xsrf = False
c.ServerApp.allow_origin = 'http://127.0.0.1:8888'
c.ServerApp.allow_origin_pat = r'https://.*\.example\.com'
"""
    unsafe = """
c.ServerApp.disable_check_xsrf = True
jupyter lab --ServerApp.allow_origin='*'
c.ServerApp.allow_origin_pat = '.*'
jupyter lab --ServerApp.disable_check_xsrf=true
jupyter lab --NotebookApp.disable_check_xsrf=1
jupyter lab --ServerApp.allow_origin_pat='.+'
jupyter lab \\
  --NotebookApp.disable_check_xsrf \\
  'true'
"""
    assert _unsafe_jupyter_lines(safe) == []
    assert len(_unsafe_jupyter_lines(unsafe)) == 7
