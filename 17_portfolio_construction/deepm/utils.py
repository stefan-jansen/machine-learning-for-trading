"""Small utilities shared across deepm modules."""

from __future__ import annotations

import os
import random
from collections.abc import Iterator

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for numpy, random, and torch."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def causal_attention_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Return a causal mask for torch.nn.MultiheadAttention.

    The mask is True where attention is disallowed (i.e., future positions).
    """
    return torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)


def to_float_tensor(x: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def iter_batches(
    indices: np.ndarray, batch_size: int, *, drop_last: bool = False
) -> Iterator[np.ndarray]:
    """Yield index batches."""
    n: int = int(indices.shape[0])
    for start in range(0, n, batch_size):
        end = start + batch_size
        if end > n and drop_last:
            break
        yield indices[start:end]
