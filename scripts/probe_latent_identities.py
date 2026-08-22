"""Resolve latent-factor training identities without executing anything.

Used to measure which identities a config edit moves, by running it before and after
the edit and diffing. Resolve-only: no training, no artifact written.

    uv run python -m scripts.probe_latent_identities cme_futures fwd_ret_5d pca sdf
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print(__doc__)
        return 2
    case_study, label, *config_names = argv[1:]

    from case_studies.research.workspace import Study

    study = Study.regenerate(case_study, release_root=REPO)
    for config_name in config_names:
        request = study.model(
            family="latent_factors",
            label=label,
            config_name=config_name,
            execution_tier="canonical",
        )
        resolved = request.resolve()
        print(f"{config_name:8s} {resolved.identity}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
