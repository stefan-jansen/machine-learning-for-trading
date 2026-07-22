"""DeePM-style deep portfolio management for end-to-end allocation.

This package implements the key components from:
- Zhang, Zohren & Roberts (2020): Differentiable Sharpe ratio optimization
- Wood, Roberts & Zohren (2026): DeePM regime-robust portfolio management

Components:
- configs: Dataclass configurations for features, model, and training
- features: Price → feature panel engineering (vol-normalized returns, MACD, z-scores)
- graph: Macro graph prior construction (adjacency-masked attention)
- dataset: Windowed PyTorch datasets for training and inference
- losses: Differentiable Sharpe ratio and SoftMin robust objective
- model: Full DeePM policy network (FiLM, V-VSN, LSTM, cross-sectional attention)
- train: Training loop with early stopping
- inference: Rolling-window inference for backtesting
"""

from __future__ import annotations

from . import configs, dataset, features, graph, inference, losses, model, train, utils

__all__ = [
    "configs",
    "dataset",
    "features",
    "graph",
    "inference",
    "losses",
    "model",
    "train",
    "utils",
]
