"""PyTorch implementation of the DeePM deep portfolio manager.

Architecture components:
1. Per-asset temporal backbone (shared weights): FiLM conditioning, variable
   selection, LSTM, temporal self-attention.
2. Cross-sectional attention with Directed Delay for causality.
3. Macroeconomic graph prior as adjacency-masked attention.
4. Output: bounded risk weight p_{i,t} in (-1, 1) via tanh.
"""

from __future__ import annotations

import torch
from torch import nn

from .configs import ModelConfig
from .utils import causal_attention_mask


class StaticContextEncoder(nn.Module):
    """Encode per-asset static context (asset id, group id, costs)."""

    def __init__(
        self,
        *,
        n_assets: int,
        n_groups: int | None,
        cfg: ModelConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.asset_emb = nn.Embedding(n_assets, cfg.asset_embedding_dim)

        self.group_emb: nn.Embedding | None = None
        if cfg.use_group_embedding:
            if n_groups is None:
                raise ValueError("n_groups required when use_group_embedding is True")
            self.group_emb = nn.Embedding(n_groups, cfg.group_embedding_dim)

        self.include_cost = cfg.use_cost_in_context

    @property
    def context_dim(self) -> int:
        dim = self.cfg.asset_embedding_dim
        if self.group_emb is not None:
            dim += self.cfg.group_embedding_dim
        if self.include_cost:
            dim += 1
        return dim

    def forward(
        self,
        *,
        asset_ids: torch.Tensor,
        group_ids: torch.Tensor | None,
        costs: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return context embedding of shape (N, C)."""
        emb_list = [self.asset_emb(asset_ids)]

        if self.group_emb is not None:
            if group_ids is None:
                raise ValueError("group_ids required when group_emb is enabled")
            emb_list.append(self.group_emb(group_ids))

        if self.include_cost:
            if costs is None:
                raise ValueError("costs required when use_cost_in_context is True")
            emb_list.append(costs)

        return torch.cat(emb_list, dim=-1)


class FiLM(nn.Module):
    """Feature-wise linear modulation: x -> x * (1 + gamma) + beta."""

    def __init__(self, *, context_dim: int, n_features: int) -> None:
        super().__init__()
        self.proj = nn.Linear(context_dim, 2 * n_features)
        self.n_features = n_features

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gb = self.proj(context)  # (N, 2F)
        gamma, beta = gb[:, : self.n_features], gb[:, self.n_features :]
        gamma = gamma.unsqueeze(0).unsqueeze(0)  # (1, 1, N, F)
        beta = beta.unsqueeze(0).unsqueeze(0)
        return x * (1.0 + gamma) + beta


class VectorizedVariableSelection(nn.Module):
    """Lightweight variable selection network (V-VSN)."""

    def __init__(
        self,
        *,
        n_features: int,
        d_model: int,
        context_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        self.feature_weight = nn.Parameter(torch.empty(n_features, d_model))
        self.feature_bias = nn.Parameter(torch.zeros(n_features, d_model))
        nn.init.xavier_uniform_(self.feature_weight)

        self.selector = nn.Sequential(
            nn.Linear(n_features + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_features),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, t, n, f = x.shape
        context_bt = context.unsqueeze(0).unsqueeze(0).expand(b, t, n, -1)
        logits = self.selector(torch.cat([x, context_bt], dim=-1))
        weights = torch.softmax(logits, dim=-1)

        z = torch.einsum("btnf,fd->btnfd", x, self.feature_weight) + self.feature_bias
        h = (weights.unsqueeze(-1) * z).sum(dim=-2)
        return self.out_norm(h)


class AdapterBlock(nn.Module):
    """FFN adapter with residual connection and LayerNorm."""

    def __init__(self, *, d_model: int, hidden_mult: int, dropout: float) -> None:
        super().__init__()
        d_ff = int(hidden_mult * d_model)
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.ff(self.ln(x)))


class TemporalSelfAttentionBlock(nn.Module):
    """Causal temporal self-attention per asset."""

    def __init__(self, *, d_model: int, n_heads: int, dropout: float, adapter_mult: int) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.adapter = AdapterBlock(d_model=d_model, hidden_mult=adapter_mult, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        attn_mask = causal_attention_mask(t, device=x.device)
        y, _ = self.mha(x, x, x, attn_mask=attn_mask)
        x = self.ln(x + self.dropout(y))
        return self.adapter(x)


class CrossSectionalAttention(nn.Module):
    """Cross-asset attention with Directed Delay (time lag)."""

    def __init__(self, *, d_model: int, n_heads: int, dropout: float, lag: int) -> None:
        super().__init__()
        self.lag = int(lag)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t, n, d = h.shape

        if self.lag > 0:
            pad = torch.zeros((b, self.lag, n, d), device=h.device, dtype=h.dtype)
            h_kv = torch.cat([pad, h[:, : t - self.lag, :, :]], dim=1)
            pad_m = torch.zeros((b, self.lag, n), device=mask.device, dtype=mask.dtype)
            m_kv = torch.cat([pad_m, mask[:, : t - self.lag, :]], dim=1)
        else:
            h_kv = h
            m_kv = mask

        q = h.reshape(b * t, n, d)
        kv = h_kv.reshape(b * t, n, d)
        key_padding_mask = m_kv.reshape(b * t, n) < 0.5

        # When lag > 0, the first `lag` timesteps have all-zero keys and masks.
        # All keys masked → softmax(all -inf) → NaN. Unmask all positions for
        # those rows; attention over zero-valued keys yields zero, so the
        # residual connection passes through h unchanged.
        all_masked = key_padding_mask.all(dim=-1, keepdim=True)
        if all_masked.any():
            key_padding_mask = key_padding_mask & ~all_masked

        out, _ = self.mha(q, kv, kv, key_padding_mask=key_padding_mask)
        out = out.reshape(b, t, n, d)

        return self.ln(h + self.dropout(out))


class MacroGraphAttention(nn.Module):
    """Adjacency-masked cross-asset attention (GAT-like)."""

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        dropout: float,
        adjacency_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("adjacency_mask", adjacency_mask.to(dtype=torch.bool))
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t, n, d = h.shape
        x = h.reshape(b * t, n, d)
        key_padding_mask = mask.reshape(b * t, n) < 0.5

        # Guard: if adjacency_mask + key_padding_mask blocks ALL keys for any
        # query, softmax produces NaN. Unmask everything for those rows.
        combined = self.adjacency_mask.unsqueeze(0) | key_padding_mask.unsqueeze(1)
        all_blocked = combined.all(dim=-1)  # (B*T, N) — True if query i has no valid key
        if all_blocked.any():
            key_padding_mask = key_padding_mask & ~all_blocked

        out, _ = self.mha(x, x, x, attn_mask=self.adjacency_mask, key_padding_mask=key_padding_mask)
        out = out.reshape(b, t, n, d)
        return self.ln(h + self.dropout(out))


class DeepmPolicy(nn.Module):
    """DeePM policy network that outputs risk weights p_{i,t} in (-1, 1)."""

    def __init__(
        self,
        *,
        n_assets: int,
        n_features: int,
        n_groups: int | None,
        adjacency_mask: torch.Tensor | None,
        cfg: ModelConfig,
    ) -> None:
        super().__init__()
        self.n_assets = int(n_assets)
        self.n_features = int(n_features)
        self.cfg = cfg

        self.context_encoder = StaticContextEncoder(n_assets=n_assets, n_groups=n_groups, cfg=cfg)
        context_dim = self.context_encoder.context_dim

        self.film = FiLM(context_dim=context_dim, n_features=n_features)
        self.vvsn = VectorizedVariableSelection(
            n_features=n_features,
            d_model=cfg.d_model,
            context_dim=context_dim,
            hidden_dim=cfg.vvsn_hidden_dim,
            dropout=cfg.dropout,
        )

        self.lstm = nn.LSTM(
            input_size=cfg.d_model,
            hidden_size=cfg.d_model,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
        )
        self.h0_proj = nn.Linear(context_dim, cfg.lstm_layers * cfg.d_model)
        self.c0_proj = nn.Linear(context_dim, cfg.lstm_layers * cfg.d_model)

        self.temporal_blocks = nn.ModuleList(
            [
                TemporalSelfAttentionBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    dropout=cfg.dropout,
                    adapter_mult=cfg.adapter_hidden_mult,
                )
                for _ in range(cfg.temporal_mha_layers)
            ]
        )

        self.cross_attn = CrossSectionalAttention(
            d_model=cfg.d_model,
            n_heads=cfg.cross_attention_heads,
            dropout=cfg.dropout,
            lag=cfg.cross_attention_lag,
        )

        self.macro_graph: MacroGraphAttention | None = None
        if adjacency_mask is not None:
            self.macro_graph = MacroGraphAttention(
                d_model=cfg.d_model,
                n_heads=cfg.macro_gnn_heads,
                dropout=cfg.dropout,
                adjacency_mask=adjacency_mask,
            )

        self.head = nn.Linear(cfg.d_model, 1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor,
        asset_ids: torch.Tensor,
        group_ids: torch.Tensor | None,
        costs: torch.Tensor | None,
    ) -> torch.Tensor:
        """Forward pass: features (B,T,N,F) -> risk weights (B,T,N) in (-1,1)."""
        b, t, n, f = x.shape

        context = self.context_encoder(asset_ids=asset_ids, group_ids=group_ids, costs=costs)
        x_mod = self.film(x, context)
        h = self.vvsn(x_mod, context)  # (B,T,N,D)

        # Per-asset temporal backbone
        h_bn = h.permute(0, 2, 1, 3).reshape(b * n, t, self.cfg.d_model)

        ctx_bn = context.unsqueeze(0).expand(b, n, -1).reshape(b * n, -1)
        h0 = (
            self.h0_proj(ctx_bn)
            .reshape(b * n, self.cfg.lstm_layers, self.cfg.d_model)
            .permute(1, 0, 2)
            .contiguous()
        )
        c0 = (
            self.c0_proj(ctx_bn)
            .reshape(b * n, self.cfg.lstm_layers, self.cfg.d_model)
            .permute(1, 0, 2)
            .contiguous()
        )

        h_bn, _ = self.lstm(h_bn, (h0, c0))

        for block in self.temporal_blocks:
            h_bn = block(h_bn)

        h = h_bn.reshape(b, n, t, self.cfg.d_model).permute(0, 2, 1, 3).contiguous()

        # Cross-sectional blocks
        h = self.cross_attn(h, mask)
        if self.macro_graph is not None:
            h = self.macro_graph(h, mask)

        # Output head -> tanh
        p = torch.tanh(self.head(h).squeeze(-1))
        return p * mask
