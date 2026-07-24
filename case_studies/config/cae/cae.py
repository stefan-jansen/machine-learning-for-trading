"""Conditional Autoencoder for Asset Pricing (GKX 2021).

Architecture:
- BetaNetwork: characteristics -> per-stock factor loadings
- FactorNetwork: managed portfolio returns -> factor returns
- ConditionalAutoencoder: predicted return = dot(betas, factors)

Reference: Gu, Kelly, Xiu (2021) "Autoencoder Asset Pricing Models"
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BetaNetwork(nn.Module):
    """Maps stock characteristics to per-stock factor loadings."""

    def __init__(self, n_characteristics: int, n_factors: int, hidden_units: tuple = (32,)):
        super().__init__()

        layers: list[nn.Module] = []
        in_features = n_characteristics

        for units in hidden_units:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.ReLU())
            in_features = units

        layers.append(nn.Linear(in_features, n_factors))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FactorNetwork(nn.Module):
    """Maps managed portfolio returns to factor returns (linear, no activation)."""

    def __init__(self, n_instruments: int, n_factors: int):
        super().__init__()
        self.linear = nn.Linear(n_instruments, n_factors, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ConditionalAutoencoder(nn.Module):
    """Conditional Autoencoder for Asset Pricing (GKX Architecture).

    Architecture:
    - Beta Network: characteristics -> ReLU -> factor loadings (with L1 regularization)
    - Factor Network: managed portfolios -> factor returns (linear)
    - Output: dot product of betas and factors

    Args:
        n_characteristics: Number of stock characteristics (L)
        n_instruments: Number of managed portfolio instruments (L+1 typically)
        n_factors: Number of latent factors (K)
        hidden_units: Tuple of hidden layer sizes for BetaNetwork
    """

    def __init__(
        self,
        n_characteristics: int,
        n_instruments: int,
        n_factors: int = 6,
        hidden_units: tuple = (32,),
    ):
        super().__init__()

        self.n_factors = n_factors
        self.n_characteristics = n_characteristics
        self.n_instruments = n_instruments

        self.beta_net = BetaNetwork(n_characteristics, n_factors, hidden_units)
        self.factor_net = FactorNetwork(n_instruments, n_factors)

    def forward(self, characteristics: torch.Tensor, portfolios: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            characteristics: (batch_size, n_characteristics) per-stock
            portfolios: (batch_size, n_instruments) managed portfolio features

        Returns:
            predicted_returns: (batch_size,)
        """
        betas = self.beta_net(characteristics)  # (batch, n_factors)
        factors = self.factor_net(portfolios)  # (batch, n_factors)
        pred = (betas * factors).sum(dim=1)
        return pred

    def get_betas(self, characteristics: torch.Tensor) -> torch.Tensor:
        """Extract factor loadings."""
        return self.beta_net(characteristics)

    def get_factors(self, portfolios: torch.Tensor) -> torch.Tensor:
        """Extract factor returns."""
        return self.factor_net(portfolios)


def l1_regularization(model: ConditionalAutoencoder, lambda_l1: float) -> torch.Tensor:
    """L1 regularization on BetaNetwork hidden Dense weights only.

    Matches the GKX (2021) reference, which applies ``kernel_regularizer='L1'``
    only to the *hidden* Dense layers. We exclude biases (sparsity is not
    meaningful) and the final output ``Linear(in, n_factors)`` layer (factor
    loadings should not be pushed to zero by an external regularizer — the
    autoencoder itself learns their magnitudes).
    """
    layers = [m for m in model.beta_net.network if isinstance(m, nn.Linear)]
    if len(layers) <= 1:
        # No hidden layers (CA0 — direct linear projection); no L1 to apply.
        return torch.zeros((), device=next(model.parameters()).device)
    hidden_layers = layers[:-1]  # exclude the final output Linear
    l1 = sum(layer.weight.abs().sum() for layer in hidden_layers)
    return lambda_l1 * l1
