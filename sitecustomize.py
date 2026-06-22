"""Make numbered chapter directories importable from any working directory.

Chapter directories (e.g. ``25_live_trading``) are prefixed with a number so
the repo lists them in reading order. That prefix makes them invalid Python
package names, so their per-chapter helper modules (``async_utils``,
``limit_orderbook``, ``rl_environments`` …) cannot be imported from the repo
root — only when the chapter directory happens to be the working directory
(which Jupyter Lab and the test harness arrange, but a bare ``python`` /
``docker compose run`` from the repo root does not).

Python imports ``sitecustomize`` automatically at interpreter startup whenever
it is found on the path. This module is loaded in both supported environments:

* Docker — the repo is bind-mounted at ``/app`` with ``PYTHONPATH=/app``, so
  ``/app/sitecustomize.py`` is on the startup path.
* Local editable install — declared as a top-level ``py-module`` in
  ``pyproject.toml``, so the editable finder resolves ``sitecustomize`` to this
  file and ``site`` imports it at startup.

It appends every ``NN_*`` chapter directory to ``sys.path`` (append, not
insert, so nothing shadows stdlib or installed packages). Chapter helper module
names do not collide, so a flat append is unambiguous.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent

for _chapter_dir in sorted(_REPO_ROOT.glob("[0-9][0-9]_*")):
    if _chapter_dir.is_dir():
        _entry = str(_chapter_dir)
        if _entry not in sys.path:
            sys.path.append(_entry)
