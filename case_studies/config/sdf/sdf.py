"""Stochastic Discount Factor Network (Chen, Pelger, Zhu 2024).

Adversarial architecture:
- SDFNetwork: learns portfolio weights from characteristics (+ optional macro LSTM)
- MomentNetwork: adversary learns instrument portfolios to expose pricing failures
- 3-phase training: unconditional warmup -> adversarial rounds

When n_macro_features=0, the LSTM branch is omitted (case study usage).
When n_macro_features>0, full LSTM processes macro indicators (teaching notebook).

Reference: Chen, Pelger, Zhu (2024) "Deep Learning in Asset Pricing"
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SDFNetwork(nn.Module):
    """SDF Network: learns portfolio weights from asset characteristics + optional macro state.

    Architecture:
    - Optional MacroLSTM processes macro features to extract economic regime state
    - FFN combines asset features (+ macro state) to produce per-stock weights
    - SDF = 1 + sum(weights * returns)

    Args:
        n_asset_features: Number of asset characteristics
        n_macro_features: Number of macro features (0 to disable LSTM)
        state_dim: LSTM hidden state dimension
        hidden_dim: FFN hidden layer size
        dropout: Dropout rate
    """

    def __init__(
        self,
        n_asset_features: int,
        n_macro_features: int = 0,
        state_dim: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.use_macro = n_macro_features > 0

        if self.use_macro:
            # Paper-faithful CPZ uses dropout only inside the SDF FFN — the
            # macro LSTM input is fed raw. The previous implementation added
            # `nn.Dropout` on the macro path which is not in the published
            # spec; removed to match the reference.
            self.lstm = nn.LSTM(
                input_size=n_macro_features,
                hidden_size=state_dim,
                batch_first=True,
            )

        ffn_input_dim = n_asset_features + (state_dim if self.use_macro else 0)
        self.ffn = nn.Sequential(
            nn.Linear(ffn_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        asset_features: torch.Tensor,  # (T, N, F_asset)
        macro_features: torch.Tensor | None = None,  # (T, F_macro)
        mask: torch.Tensor | None = None,  # (T, N)
        h0: torch.Tensor | None = None,
        c0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, torch.Tensor | None]]:
        """Forward pass producing SDF weights.

        Returns:
            weights: (n_valid,) weight per valid observation
            (h_n, c_n): LSTM states (None if no macro)
        """
        if self.use_macro and macro_features is None:
            raise ValueError(
                "macro_features must be provided when n_macro_features > 0; "
                "the SDF LSTM branch has no fallback."
            )
        if (h0 is None) != (c0 is None):
            raise ValueError("h0 and c0 must be provided together (or both None).")

        T, N, F = asset_features.shape

        if mask is None:
            mask = torch.ones(T, N, dtype=torch.bool, device=asset_features.device)

        if self.use_macro and macro_features is not None:
            if h0 is None:
                h0 = torch.zeros(1, 1, self.state_dim, device=asset_features.device)
                c0 = torch.zeros(1, 1, self.state_dim, device=asset_features.device)

            macro_seq = macro_features.unsqueeze(0)  # (1, T, F_macro)
            macro_states, (h_n, c_n) = self.lstm(macro_seq, (h0, c0))
            macro_states = macro_states.squeeze(0)  # (T, state_dim)
            macro_tiled = macro_states.unsqueeze(1).expand(-1, N, -1)

            asset_flat = asset_features[mask]
            macro_flat = macro_tiled[mask]
            ffn_input = torch.cat([asset_flat, macro_flat], dim=1)
        else:
            asset_flat = asset_features[mask]
            ffn_input = asset_flat
            h_n = c_n = None

        weights = self.ffn(ffn_input).squeeze(-1)
        return weights, (h_n, c_n)


class MomentNetwork(nn.Module):
    """Moment Network (adversary): learns instruments for adversarial moment conditions.

    Finds test asset portfolios where E[M * R * Z] != 0, exposing SDF pricing failures.

    Args:
        n_asset_features: Number of asset characteristics
        n_macro_features: Number of macro features (0 to disable LSTM)
        n_instruments: Number of learned instruments
        state_dim: LSTM hidden state dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        n_asset_features: int,
        n_macro_features: int = 0,
        n_instruments: int = 8,
        state_dim: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_instruments = n_instruments
        self.use_macro = n_macro_features > 0

        if self.use_macro:
            # Paper-faithful CPZ moment net has no FFN hidden layers and no
            # macro-input dropout; LSTM input is fed raw.
            self.lstm = nn.LSTM(
                input_size=n_macro_features,
                hidden_size=state_dim,
                batch_first=True,
            )

        ffn_input_dim = n_asset_features + (state_dim if self.use_macro else 0)
        self.ffn = nn.Sequential(
            nn.Linear(ffn_input_dim, n_instruments),
            nn.Tanh(),
        )

    def forward(
        self,
        asset_features: torch.Tensor,  # (T, N, F_asset)
        macro_features: torch.Tensor | None = None,  # (T, F_macro)
        h0: torch.Tensor | None = None,
        c0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, torch.Tensor | None]]:
        """Forward pass producing instruments.

        Returns:
            instruments: (n_instruments, T, N)
            (h_n, c_n): LSTM states (None if no macro)
        """
        if self.use_macro and macro_features is None:
            raise ValueError(
                "macro_features must be provided when n_macro_features > 0; "
                "the moment-network LSTM branch has no fallback."
            )
        if (h0 is None) != (c0 is None):
            raise ValueError("h0 and c0 must be provided together (or both None).")

        T, N, F = asset_features.shape

        if self.use_macro and macro_features is not None:
            if h0 is None:
                h0 = torch.zeros(1, 1, self.state_dim, device=asset_features.device)
                c0 = torch.zeros(1, 1, self.state_dim, device=asset_features.device)

            macro_seq = macro_features.unsqueeze(0)
            macro_states, (h_n, c_n) = self.lstm(macro_seq, (h0, c0))
            macro_states = macro_states.squeeze(0)
            macro_tiled = macro_states.unsqueeze(1).expand(-1, N, -1)

            ffn_input = torch.cat([asset_features, macro_tiled], dim=2)
        else:
            ffn_input = asset_features
            h_n = c_n = None

        instruments = self.ffn(ffn_input)  # (T, N, n_instruments)
        instruments = instruments.permute(2, 0, 1)  # (n_instruments, T, N)
        return instruments, (h_n, c_n)


# ---------------------------------------------------------------------------
# SDF construction and loss functions
# ---------------------------------------------------------------------------


def get_segment_ids(mask: torch.Tensor) -> torch.Tensor:
    """Create segment IDs mapping valid observations to time steps.

    Args:
        mask: (T, N) boolean mask

    Returns:
        segment_ids: (n_valid,) time step index per valid observation
    """
    T, N = mask.shape
    time_ids = torch.arange(T, device=mask.device).unsqueeze(1).expand(-1, N)
    return time_ids[mask]


def construct_sdf(weights: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Construct SDF from weights and returns.

    SDF_t = 1 + sum_i(w_i * r_i) for each time step.

    Args:
        weights: (n_valid,) SDF weights
        returns: (T, N) asset returns
        mask: (T, N) valid observations

    Returns:
        sdf: (T,) SDF value per time step
    """
    T, N = returns.shape
    returns_flat = returns[mask]
    segment_ids = get_segment_ids(mask)

    weighted_returns = weights * returns_flat
    sdf_values = torch.zeros(T, device=weights.device)
    sdf_values.scatter_add_(0, segment_ids, weighted_returns)

    return 1 + sdf_values


def unconditional_loss(
    weights: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    n_obs_per_asset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unconditional pricing loss: E[M * R * 1]^2 with constant instrument Z=1.

    Returns:
        loss: scalar MSE
        sdf: (T,)
    """
    T, N = returns.shape
    mask_float = mask.float()

    sdf = construct_sdf(weights, returns, mask)
    sdf_expanded = sdf.unsqueeze(1)

    instruments = torch.ones(1, T, N, device=weights.device)
    sample_moments = returns * mask_float * sdf_expanded * instruments

    weighted_moments = sample_moments.sum(dim=1) / n_obs_per_asset.clamp(min=1)
    n_obs_norm = n_obs_per_asset / n_obs_per_asset.max()

    loss = (weighted_moments.pow(2) * n_obs_norm).mean()
    return loss, sdf


def conditional_loss(
    weights: torch.Tensor,
    instruments: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    n_obs_per_asset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Conditional pricing loss: E[M * R * Z]^2 with learned instruments.

    Args:
        weights: (n_valid,)
        instruments: (n_instruments, T, N)
        returns: (T, N)
        mask: (T, N)
        n_obs_per_asset: (N,)

    Returns:
        loss: scalar MSE
        sdf: (T,)
    """
    T, N = returns.shape
    n_instruments = instruments.shape[0]
    mask_float = mask.float()

    sdf = construct_sdf(weights, returns, mask)
    sdf_expanded = sdf.unsqueeze(1)

    sample_moments = returns * mask_float * sdf_expanded * instruments

    weighted_moments = sample_moments.sum(dim=1) / n_obs_per_asset.clamp(min=1)
    n_obs_norm = n_obs_per_asset / n_obs_per_asset.max()
    n_obs_tiled = n_obs_norm.unsqueeze(0).expand(n_instruments, -1)

    loss = (weighted_moments.pow(2) * n_obs_tiled).mean()
    return loss, sdf


def compute_sharpe(sdf: torch.Tensor) -> torch.Tensor:
    """Sharpe ratio of SDF portfolio (1 - M).

    Uses population standard deviation (``unbiased=False``) for a fixed-
    convention validation metric across folds. Empty / constant SDF series
    return 0 rather than NaN so the trainer's checkpoint comparison stays
    deterministic.
    """
    portfolio_return = 1 - sdf
    if portfolio_return.numel() == 0:
        return torch.zeros((), device=sdf.device, dtype=sdf.dtype)
    mean = portfolio_return.mean()
    std = portfolio_return.std(unbiased=False).clamp(min=1e-8)
    out = mean / std
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))
