"""Training utilities for DeePM-style models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .configs import TrainingConfig, validate_training_config
from .dataset import StaticAssetMetadata
from .losses import robust_sharpe_loss
from .utils import set_seed


@dataclass(frozen=True, slots=True)
class TrainHistory:
    steps: list[int]
    train_objective: list[float]
    train_sharpe_pool: list[float]
    val_sharpe_pool: list[float]


class EarlyStopping:
    """Validation-metric early stopping with EMA smoothing."""

    def __init__(
        self,
        *,
        alpha: float,
        min_delta: float,
        patience: int,
        burn_in_iters: int,
    ) -> None:
        self.alpha = float(alpha)
        self.min_delta = float(min_delta)
        self.patience = int(patience)
        self.burn_in_iters = int(burn_in_iters)

        self._ema: float | None = None
        self._best_ema: float = float("-inf")
        self._bad_count: int = 0

    def update(self, value: float, step: int) -> tuple[bool, float, float]:
        """Update state. Returns (stop, ema, best_ema)."""
        if self._ema is None:
            self._ema = float(value)
        else:
            self._ema = self.alpha * float(value) + (1.0 - self.alpha) * self._ema

        if step < self.burn_in_iters:
            return False, self._ema, self._best_ema

        if self._ema >= self._best_ema + self.min_delta:
            self._best_ema = self._ema
            self._bad_count = 0
        else:
            self._bad_count += 1

        return self._bad_count >= self.patience, self._ema, self._best_ema


@torch.no_grad()
def evaluate_pooled_sharpe(
    model: nn.Module,
    loader: DataLoader,
    *,
    static_meta: StaticAssetMetadata,
    cfg: TrainingConfig,
    device: torch.device,
) -> float:
    """Compute pooled Sharpe over the entire loader."""
    model.eval()
    returns: list[torch.Tensor] = []

    for x, y, v, m in loader:
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)
        v = v.to(device=device, dtype=torch.float32)
        m = m.to(device=device, dtype=torch.float32)

        p = model(
            x,
            mask=m,
            asset_ids=static_meta.asset_ids.to(device),
            group_ids=None if static_meta.group_ids is None else static_meta.group_ids.to(device),
            costs=None if static_meta.costs is None else static_meta.costs.to(device),
        )

        out = robust_sharpe_loss(
            p=p,
            y_fwd1=y,
            vol_scale=v,
            mask=m,
            costs=None if static_meta.costs is None else static_meta.costs.to(device),
            burn_in=cfg.burn_in,
            gamma_cost=cfg.gamma_cost,
            annualization_factor=cfg.annualization_factor,
            eps=cfg.sharpe_eps,
            tau=cfg.softmin_tau,
            lambda_soft=cfg.softmin_lambda,
        )
        returns.append(out.net_returns[:, cfg.burn_in :].reshape(-1).cpu())

    if not returns:
        return float("nan")

    r = torch.cat(returns)
    r = r[~torch.isnan(r)]
    if r.numel() < 10:
        return float("nan")

    mu = r.mean()
    var = r.var(unbiased=False)
    sharpe = (cfg.annualization_factor**0.5) * mu / torch.sqrt(var + cfg.sharpe_eps)
    return float(sharpe.item())


def train_model(
    model: nn.Module,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    static_meta: StaticAssetMetadata,
    cfg: TrainingConfig,
) -> tuple[dict[str, torch.Tensor], TrainHistory]:
    """Train a DeePM-style model and return the best state dict."""
    validate_training_config(cfg)
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    stopper = EarlyStopping(
        alpha=cfg.metric_ema_alpha,
        min_delta=cfg.metric_min_delta,
        patience=cfg.early_stopping_patience,
        burn_in_iters=cfg.early_stopping_burn_in_iters,
    )

    history = TrainHistory(steps=[], train_objective=[], train_sharpe_pool=[], val_sharpe_pool=[])
    best_state: dict[str, torch.Tensor] = {}
    best_val_sharpe: float = float("-inf")
    train_iter = iter(train_loader)

    for step in range(1, cfg.max_iters + 1):
        model.train()

        try:
            x, y, v, m = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y, v, m = next(train_iter)

        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)
        v = v.to(device=device, dtype=torch.float32)
        m = m.to(device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)

        p = model(
            x,
            mask=m,
            asset_ids=static_meta.asset_ids.to(device),
            group_ids=None if static_meta.group_ids is None else static_meta.group_ids.to(device),
            costs=None if static_meta.costs is None else static_meta.costs.to(device),
        )

        out = robust_sharpe_loss(
            p=p,
            y_fwd1=y,
            vol_scale=v,
            mask=m,
            costs=None if static_meta.costs is None else static_meta.costs.to(device),
            burn_in=cfg.burn_in,
            gamma_cost=cfg.gamma_cost,
            annualization_factor=cfg.annualization_factor,
            eps=cfg.sharpe_eps,
            tau=cfg.softmin_tau,
            lambda_soft=cfg.softmin_lambda,
        )

        if torch.isnan(out.loss) or torch.isinf(out.loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        out.loss.backward()
        if cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
        optimizer.step()

        if step % cfg.eval_every != 0:
            continue

        val_sharpe = evaluate_pooled_sharpe(
            model, val_loader, static_meta=static_meta, cfg=cfg, device=device
        )

        stop, _ema, _best_ema = stopper.update(val_sharpe, step)

        history.steps.append(step)
        history.train_objective.append(float(out.objective.item()))
        history.train_sharpe_pool.append(float(out.sharpe_pool.item()))
        history.val_sharpe_pool.append(val_sharpe)

        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if stop:
            break

    return best_state, history
