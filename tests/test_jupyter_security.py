"""Regression guards for the unauthenticated local Jupyter containers."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
JUPYTER_DOCKERFILES = (
    "envs/ml4t/Dockerfile",
    "envs/py312/Dockerfile",
    "envs/benchmark/Dockerfile",
    "envs/benchmark/Dockerfile.full",
)


def test_jupyter_images_keep_origin_and_xsrf_checks_enabled() -> None:
    forbidden = ("allow_origin", "disable_check_xsrf")
    offenders = []
    for relative in JUPYTER_DOCKERFILES:
        text = (REPO_ROOT / relative).read_text()
        for setting in forbidden:
            if setting in text:
                offenders.append(f"{relative}: {setting}")

    assert not offenders, (
        "Local Jupyter runs without a token, so origin and XSRF checks must remain enabled:\n  "
        + "\n  ".join(offenders)
    )
