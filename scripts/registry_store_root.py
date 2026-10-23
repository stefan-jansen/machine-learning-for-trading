#!/usr/bin/env python3
"""Where the case-study registries are, for a script that has to read them before a run.

One function, because two pre-run checkers need the same answer and both had it wrong in the
same way.

**The defect this exists to end.** `scripts/check_supersedes_literals.py` and
`scripts/check_input_artifact_vintage.py` each defaulted to a hardcoded
`~/ml4t/artifacts/case_studies`. That path was the artifact store until the 2026-09-24 storage
consolidation moved every registry to `~/Dropbox/ml4t/case-studies/<cs>/` and deleted it. Neither
script failed afterwards: a missing registry is a legitimate state - a reader's clone has none,
because `run_log/` is gitignored - so both report it and exit zero. The supersedes checker printed
`no registry on disk` for all nine case studies and signed off with `0 live generation(s) declared
nowhere`, and the vintage checker printed `0 artifact/label pair(s)`. Three weeks of runs were
gated by two checks that could not see anything.

What it cost, on 2026-10-19: `us_equities_panel/08_tabular_dl` fitted 48 folds over 13 h 49 m and
was then refused at the freeze, `a changed candidate set named
'us-equities-fwd-ret-1d-tabular-dl-v1' must explicitly supersedes f30d95e1fe5d`. Pointed at the
store, the supersedes checker names that lineage and its diagnostics twin, and prints the launch
parameter that declares them.

**So the root is derived from a registry that exists, never chosen in advance.** A
`--case-study` worktree symlinks `case_studies/<cs>/run_log` out to the canonical store, so the
directory a notebook opens and the directory beside the source are two different trees. Resolving
one of those links and taking its grandparent names the store wherever it has been moved to, which
is the property the hardcoded path did not have. A plain checkout has a real `run_log` or none;
both answer `<repo>/case_studies`, where its registries would be.

`scripts/artifact_audit.py` carries the same derivation over `features`/`labels` for the artifact
tree rather than the registry, with its own test. It is deliberately left where it is: it answers a
different question and returns `None` where this returns a path to report against.
"""

from __future__ import annotations

from pathlib import Path


def registry_store_root(repo_root: Path) -> Path:
    """The tree holding `<case study>/run_log/registry.db`, resolved through the checkout."""
    case_studies = repo_root / "case_studies"
    linked = next(iter(sorted(case_studies.glob("*/run_log"))), None)
    # <root>/<case study>/run_log
    return linked.resolve().parents[1] if linked is not None else case_studies
