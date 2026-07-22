"""Loss functions for DeePM-style end-to-end portfolio learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class LossOutput:
    """Outputs returned by the robust Sharpe loss."""

    loss: torch.Tensor
    sharpe_pool: torch.Tensor
    softmin_sharpe: torch.Tensor
    objective: torch.Tensor
    net_returns: torch.Tensor


def compute_net_portfolio_returns(
    *,
    p: torch.Tensor,
    y_fwd1: torch.Tensor,
    vol_scale: torch.Tensor,
    mask: torch.Tensor,
    costs: torch.Tensor | None,
    gamma_cost: float,
) -> torch.Tensor:
    """Compute net portfolio returns (Eq. 13).

    Parameters
    ----------
    p: Risk weights in (-1, 1), shape (B, T, N).
    y_fwd1: Vol-scaled forward returns, shape (B, T, N).
    vol_scale: Volatility scaling, shape (B, T, N).
    mask: Availability mask, shape (B, T, N).
    costs: Per-asset cost coefficients, shape (N,) or (N, 1). None to skip.
    gamma_cost: Global cost scaling factor.

    Returns
    -------
    Net portfolio return series, shape (B, T).
    """
    b, t, n = p.shape

    w = vol_scale * p
    w_prev = torch.cat(
        [torch.zeros((b, 1, n), device=w.device, dtype=w.dtype), w[:, :-1, :]], dim=1
    )

    gross = (mask * p * y_fwd1).sum(dim=-1)
    n_t = mask.sum(dim=-1).clamp(min=1.0)
    gross = gross / n_t

    if costs is None:
        return gross

    if (costs.ndim == 2 and costs.shape[1] == 1) or (costs.ndim == 1 and costs.shape[0] == n):
        c = costs.view(1, 1, n)
    else:
        raise ValueError("costs must have shape (N,) or (N,1)")

    turnover = torch.abs(w - w_prev)
    cost = (mask * c * turnover).sum(dim=-1)
    cost = (gamma_cost * cost) / n_t

    return gross - cost


def sharpe_ratio(
    returns: torch.Tensor,
    *,
    annualization_factor: float,
    eps: float,
    dim: int | None = None,
) -> torch.Tensor:
    """Compute a differentiable Sharpe ratio."""
    if dim is None:
        mu = returns.mean()
        var = returns.var(unbiased=False)
    else:
        mu = returns.mean(dim=dim)
        var = returns.var(dim=dim, unbiased=False)

    return (annualization_factor**0.5) * mu / torch.sqrt(var + eps)


def softmin_sharpe(
    window_sharpes: torch.Tensor,
    *,
    tau: float,
) -> torch.Tensor:
    """Soft minimum of window-wise Sharpe ratios (Eq. 33)."""
    return -tau * torch.log(torch.mean(torch.exp(-window_sharpes / tau)))


def robust_sharpe_loss(
    *,
    p: torch.Tensor,
    y_fwd1: torch.Tensor,
    vol_scale: torch.Tensor,
    mask: torch.Tensor,
    costs: torch.Tensor | None,
    burn_in: int,
    gamma_cost: float,
    annualization_factor: float,
    eps: float,
    tau: float,
    lambda_soft: float,
) -> LossOutput:
    """Compute DeePM robust objective loss (Eq. 31).

    L(theta) = - SR_pool(R) - lambda * SoftMin_tau({SR_b}).
    """
    net_r = compute_net_portfolio_returns(
        p=p,
        y_fwd1=y_fwd1,
        vol_scale=vol_scale,
        mask=mask,
        costs=costs,
        gamma_cost=gamma_cost,
    )

    net_r_eff = net_r[:, burn_in:] if burn_in > 0 else net_r

    sr_pool = sharpe_ratio(
        net_r_eff.reshape(-1), annualization_factor=annualization_factor, eps=eps
    )
    sr_windows = sharpe_ratio(net_r_eff, annualization_factor=annualization_factor, eps=eps, dim=1)

    sr_softmin = softmin_sharpe(sr_windows, tau=tau)

    objective = sr_pool + lambda_soft * sr_softmin
    loss = -objective

    return LossOutput(
        loss=loss,
        sharpe_pool=sr_pool.detach(),
        softmin_sharpe=sr_softmin.detach(),
        objective=objective.detach(),
        net_returns=net_r.detach(),
    )
