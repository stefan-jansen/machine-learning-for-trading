"""Regression tests for the local RoboRev installation and PR gate."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE = REPO_ROOT / "scripts" / "roborev_pr_gate.sh"
INSTALLER = REPO_ROOT / "scripts" / "install_local_roborev.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _fake_environment(tmp_path: Path, *, branch: str, open_review: bool) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    hooks = tmp_path / "hooks"
    bin_dir.mkdir()
    hooks.mkdir()

    _write_executable(
        bin_dir / "git",
        f"""#!/usr/bin/env bash
set -euo pipefail
case "$1 $2" in
  "rev-parse --show-toplevel") printf '%s\\n' "{tmp_path}" ;;
  "rev-parse --git-path") printf '%s\\n' "{hooks}/pre-push" ;;
  "branch --show-current") printf '%s\\n' "{branch}" ;;
  *) printf 'unexpected git invocation: %s\\n' "$*" >&2; exit 2 ;;
esac
""",
    )
    listing = "Found 1 open job(s):\\n\\nJob #42" if open_review else "No open jobs found."
    _write_executable(
        bin_dir / "roborev",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "{tmp_path}/roborev.log"
if [[ "$1 $2 $3" == "fix --open --list" ]]; then
  printf '%b\\n' "{listing}"
fi
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "ML4T_GIT_BIN": str(bin_dir / "git"),
            "ML4T_ROBOREV_BIN": str(bin_dir / "roborev"),
        }
    )
    return env


def test_gate_passes_after_branch_review_with_no_open_findings(tmp_path: Path) -> None:
    env = _fake_environment(tmp_path, branch="feature", open_review=False)

    result = subprocess.run([GATE], env=env, text=True, capture_output=True, check=False)

    assert result.returncode == 0
    assert "PR gate passed" in result.stdout
    invocations = (tmp_path / "roborev.log").read_text()
    assert "review --branch --base main --agent codex --panel none --min-severity low --wait" in (
        invocations
    )
    assert "fix --open --list --branch feature" in invocations


def test_gate_blocks_every_open_severity(tmp_path: Path) -> None:
    env = _fake_environment(tmp_path, branch="feature", open_review=True)

    result = subprocess.run([GATE], env=env, text=True, capture_output=True, check=False)

    assert result.returncode == 1
    assert "resolve every open RoboRev finding" in result.stderr


def test_gate_rejects_direct_main_push(tmp_path: Path) -> None:
    env = _fake_environment(tmp_path, branch="main", open_review=False)

    result = subprocess.run([GATE], env=env, text=True, capture_output=True, check=False)

    assert result.returncode == 1
    assert "Direct pushes from main are not allowed" in result.stderr
    assert not (tmp_path / "roborev.log").exists()


def test_installer_is_idempotent_and_refuses_foreign_hook(tmp_path: Path) -> None:
    env = _fake_environment(tmp_path, branch="feature", open_review=False)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    gate = scripts / GATE.name
    gate.write_bytes(GATE.read_bytes())
    gate.chmod(0o755)

    first = subprocess.run([INSTALLER], env=env, text=True, capture_output=True, check=False)
    hook = tmp_path / "hooks" / "pre-push"
    first_content = hook.read_text()
    second = subprocess.run([INSTALLER], env=env, text=True, capture_output=True, check=False)

    assert first.returncode == 0
    assert second.returncode == 0
    assert hook.read_text() == first_content
    assert "# ml4t roborev pre-push hook v1" in first_content

    hook.write_text("#!/usr/bin/env bash\nprintf 'foreign hook\\n'\n")
    rejected = subprocess.run([INSTALLER], env=env, text=True, capture_output=True, check=False)

    assert rejected.returncode == 1
    assert "Refusing to replace an existing pre-push hook" in rejected.stderr
    assert hook.read_text() == "#!/usr/bin/env bash\nprintf 'foreign hook\\n'\n"


@pytest.mark.parametrize("script", [GATE, INSTALLER])
def test_scripts_pass_bash_syntax_check(script: Path) -> None:
    subprocess.run(["bash", "-n", script], check=True)
