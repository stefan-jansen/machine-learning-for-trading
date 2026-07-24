"""Macroeconomic graph prior utilities.

Builds an asset-level adjacency matrix from macro group labels and
cross-group edges for adjacency-masked attention in DeePM.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MacroGraph:
    """Asset-level macro graph."""

    assets: list[str]
    groups: list[str]
    adjacency: np.ndarray  # Boolean (N, N)


def build_macro_adjacency(
    *,
    assets: Sequence[str],
    asset_to_group: dict[str, str],
    cross_group_edges: Iterable[tuple[str, str]] = (),
    include_self_loops: bool = True,
) -> MacroGraph:
    """Build a boolean adjacency matrix from group labels and group edges.

    Parameters
    ----------
    assets: Asset identifiers matching the price panel columns.
    asset_to_group: Mapping from asset -> macro group label.
    cross_group_edges: Undirected (group_a, group_b) edges.
    include_self_loops: If True, sets A[i,i] = True.
    """
    assets_list = [str(a) for a in assets]
    groups = [str(asset_to_group.get(a, "UNKNOWN")) for a in assets_list]

    n = len(assets_list)
    adj = np.zeros((n, n), dtype=bool)

    if include_self_loops:
        np.fill_diagonal(adj, True)

    group_to_indices: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        group_to_indices.setdefault(g, []).append(i)

    for indices in group_to_indices.values():
        idx = np.array(indices, dtype=int)
        adj[np.ix_(idx, idx)] = True

    for g1, g2 in cross_group_edges:
        idx1 = group_to_indices.get(g1, [])
        idx2 = group_to_indices.get(g2, [])
        if not idx1 or not idx2:
            continue
        a = np.array(idx1, dtype=int)
        b = np.array(idx2, dtype=int)
        adj[np.ix_(a, b)] = True
        adj[np.ix_(b, a)] = True

    return MacroGraph(assets=assets_list, groups=groups, adjacency=adj)


def adjacency_to_attn_mask(adjacency: np.ndarray) -> np.ndarray:
    """Convert boolean adjacency to attention mask (True = disallowed)."""
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square (N,N)")
    return ~adjacency
