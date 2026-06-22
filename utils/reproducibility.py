"""One-call seed initialization for reproducible notebook runs.

Notebooks that produce any random output should call ``set_global_seeds()``
in their preamble, between imports and the first computation. Monte Carlo
demos that *want* per-run variability should still call it, with the seed
declared in their parameters cell so readers can change it explicitly.
"""

from __future__ import annotations

import os
import random


def set_global_seeds(seed: int = 42) -> None:
    """Seed Python ``random``, NumPy, Torch (CPU+CUDA), and ``PYTHONHASHSEED``.

    Polars and pandas operations that need a seed accept it per-call
    (e.g. ``df.sample(seed=seed)``) — there is no global polars seed to set.
    Scikit-learn estimators take ``random_state=`` per-instance.

    Returns ``None`` rather than the seed so that a bare
    ``set_global_seeds(SEED)`` in a notebook preamble does not render a
    spurious ``42`` execute_result. Notebooks that want to echo the seed back
    to the reader should ``print(SEED)`` explicitly.

    ``PYTHONHASHSEED`` is set on ``os.environ`` for the benefit of subprocesses
    spawned after the call; it has **no effect** on hash randomization in the
    currently running interpreter (that value is read once at startup). For
    end-to-end hash determinism the kernel must be launched with
    ``PYTHONHASHSEED=<seed>`` already set in the environment.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
