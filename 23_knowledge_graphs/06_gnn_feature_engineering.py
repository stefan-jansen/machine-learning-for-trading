# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # GNN Feature Engineering for Hybrid Models
#
# **Chapter 23: Knowledge Graphs for Financial AI**
#
# **Docker image**: `ml4t-gpu`
#
# This notebook tests whether graph-derived embeddings add information to a
# tabular cross-sectional model. It trains a graph-attention autoencoder on a
# pre-target correlation network, then compares tabular and hybrid ridge models
# on the same held-out stocks.
#
# **Learning objectives**
#
# - Build a correlation network without using the target window.
# - Train graph embeddings rather than treating random projections as learned features.
# - Keep universe selection, scaling, and labels on the correct side of the evaluation boundary.
# - Use paired fold results to distinguish an ablation result from a general claim about GNNs.
#
# **Prerequisites**
#
# - Familiarity with correlation matrices, ridge regression, and cross-sectional validation.
# - The frozen NASDAQ Data Link Wiki Prices dataset available through `ML4T_DATA_PATH`.
# - The `ml4t-gpu` Docker service on a CUDA-capable GPU for the production execution.
#
# **Book reference**: Chapter 23, Section 23.5

# %%
"""Train graph embeddings and test their incremental cross-sectional value."""

from __future__ import annotations

from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from scipy import stats
from sklearn.model_selection import KFold
from torch import nn

from data import load_us_equities
from utils.reproducibility import set_global_seeds
from utils.style import COLORS

# %% tags=["parameters"]
N_ASSETS = 200
LOOKBACK_DAYS = 504
TARGET_HORIZON = 21
CORRELATION_THRESHOLD = 0.5
EMBEDDING_DIM = 4
GNN_EPOCHS = 250
SEED = 42

# %%
set_global_seeds(SEED)
if not torch.cuda.is_available():
    raise RuntimeError("This production notebook requires a CUDA-capable GPU")
DEVICE = torch.device("cuda")

print("Configuration:")
print(f"  Assets: {N_ASSETS}")
print(f"  Correlation lookback: {LOOKBACK_DAYS} trading days")
print(f"  Target horizon: {TARGET_HORIZON} trading days")
print(f"  Correlation threshold: {CORRELATION_THRESHOLD}")
print(f"  Graph encoder device: {DEVICE}")

# %% [markdown]
# ## 1. Define the information boundary
#
# The final 21 trading dates form one forward-return window. Liquidity ranking,
# factor construction, and the correlation graph use only observations available
# on or before `feature_as_of`. The evaluation cohort then retains ranked stocks
# observed at both target endpoints. This complete-case filter supports one
# held-out-stock comparison; it is not a point-in-time investable-universe rule
# or a temporal backtest.

# %%
prices = load_us_equities(start_date="2015-01-01")
if prices.schema["timestamp"] == pl.Datetime:
    prices = prices.with_columns(pl.col("timestamp").dt.date())

prices = prices.with_columns((pl.col("adj_close") * pl.col("adj_volume")).alias("dollar_volume"))
trading_dates = prices["timestamp"].unique().sort().to_list()

required_dates = LOOKBACK_DAYS + TARGET_HORIZON + 1
if len(trading_dates) < required_dates:
    raise ValueError(f"Need at least {required_dates} trading dates, found {len(trading_dates)}")

feature_as_of = trading_dates[-TARGET_HORIZON - 1]
evaluation_end = trading_dates[-1]
universe_start = trading_dates[-TARGET_HORIZON - LOOKBACK_DAYS - 1]

print(f"Loaded {prices.height:,} rows for {prices['symbol'].n_unique():,} symbols")
print(f"Feature as-of date: {feature_as_of}")
print(f"Target window: {trading_dates[-TARGET_HORIZON]} to {evaluation_end}")

# %%
universe_history = prices.filter(
    pl.col("timestamp").is_between(universe_start, feature_as_of, closed="both")
)
liquidity_ranking = (
    universe_history.group_by("symbol")
    .agg(
        pl.col("dollar_volume").mean().alias("average_dollar_volume"),
        pl.col("adj_close").count().alias("observations"),
        pl.col("timestamp").max().alias("last_observation"),
    )
    .filter(
        (pl.col("observations") >= LOOKBACK_DAYS) & (pl.col("last_observation") == feature_as_of)
    )
    .sort("average_dollar_volume", descending=True)
)

# %% [markdown]
# Endpoint completeness defines the retrospective evaluation cohort. Sorting
# again after the join preserves the pre-target liquidity ranking because joins
# do not guarantee input order.

# %%
endpoint_complete_symbols = (
    prices.filter(pl.col("timestamp").is_in([feature_as_of, evaluation_end]))
    .group_by("symbol")
    .agg(pl.col("timestamp").n_unique().alias("endpoint_count"))
    .filter(pl.col("endpoint_count") == 2)
    .select("symbol")
)
universe = (
    liquidity_ranking.join(endpoint_complete_symbols, on="symbol", how="inner")
    .sort("average_dollar_volume", descending=True)
    .head(N_ASSETS)
)

selected_symbols = universe["symbol"].to_list()
if len(selected_symbols) != N_ASSETS:
    raise ValueError(
        f"Requested {N_ASSETS} endpoint-complete eligible symbols, found {len(selected_symbols)}"
    )

selected_prices = prices.filter(pl.col("symbol").is_in(selected_symbols))
print(
    f"Selected {len(selected_symbols)} endpoint-complete symbols "
    f"ranked using data through {feature_as_of}"
)
print(f"Leading symbols by pre-target liquidity: {selected_symbols[:10]}")

# %% [markdown]
# ## 2. Build the pre-target correlation graph
#
# Adjacency uses absolute correlation, so strong negative and positive
# co-movement both create an edge. The graph itself is an input available at the
# feature date. It does not use any return from the target window.

# %%
returns = (
    selected_prices.filter(pl.col("timestamp") <= feature_as_of)
    .sort(["symbol", "timestamp"])
    .with_columns(
        (pl.col("adj_close") / pl.col("adj_close").shift(1).over("symbol") - 1).alias("return")
    )
)
returns_wide = (
    returns.pivot(on="symbol", index="timestamp", values="return")
    .sort("timestamp")
    .tail(LOOKBACK_DAYS)
)

symbol_order = [column for column in returns_wide.columns if column != "timestamp"]
returns_matrix = returns_wide.select(symbol_order).to_numpy()
column_means = np.nanmean(returns_matrix, axis=0)
returns_matrix = np.where(np.isnan(returns_matrix), column_means, returns_matrix)

correlation_matrix = np.corrcoef(returns_matrix.T)
adjacency = (np.abs(correlation_matrix) > CORRELATION_THRESHOLD).astype(np.float32)
np.fill_diagonal(adjacency, 0)

n_edges = int(adjacency.sum() // 2)
possible_edges = len(symbol_order) * (len(symbol_order) - 1) / 2
graph_density = n_edges / possible_edges
degrees = adjacency.sum(axis=1)
corr_values = correlation_matrix[np.triu_indices(len(symbol_order), k=1)]

print(f"Returns matrix: {returns_matrix.shape}")
print(f"Graph: {n_edges:,} edges, {graph_density:.2%} density")
print(f"Median degree: {np.median(degrees):.0f}")

# %% [markdown]
# ## 3. Construct node features and the forward target
#
# Each node receives momentum, volatility, mean-reversion, and trend features
# computed through the feature date. The target is the close-to-close return
# from the feature date to the end of the 21-day evaluation window.

# %% [markdown]
# A single helper keeps the eight pre-target transformations together while
# making their information boundary explicit.


# %%
def build_node_feature_row(history: pl.DataFrame, symbol: str) -> list[float]:
    """Compute one stock's features from observations available by formation."""
    closes = history["adj_close"].to_numpy()
    return_values = (
        history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1).alias("return")
        )["return"]
        .drop_nulls()
        .to_numpy()
    )

    if len(closes) < 126 or len(return_values) < 63:
        raise ValueError(f"Insufficient feature history for {symbol}")

    moving_average_20 = np.mean(closes[-20:])
    moving_average_50 = np.mean(closes[-50:])
    return [
        closes[-1] / closes[-21] - 1,
        closes[-1] / closes[-63] - 1,
        closes[-1] / closes[-126] - 1,
        np.std(return_values[-21:]) * np.sqrt(252),
        np.std(return_values[-63:]) * np.sqrt(252),
        (closes[-1] - moving_average_20) / (np.std(closes[-20:]) + 1e-8),
        np.polyfit(np.arange(20), return_values[-20:], 1)[0] * 252,
        moving_average_20 / moving_average_50,
    ]


# %% [markdown]
# The forward target uses exactly the two disclosed endpoints and never enters
# graph representation training.


# %%
def compute_forward_return(
    symbol_prices: pl.DataFrame,
    feature_date: date,
    target_end: date,
) -> float:
    """Compute the close-to-close return across the target horizon."""
    start_close = symbol_prices.filter(pl.col("timestamp") == feature_date)["adj_close"]
    end_close = symbol_prices.filter(pl.col("timestamp") == target_end)["adj_close"]
    if len(start_close) != 1 or len(end_close) != 1:
        raise ValueError("Missing target-window endpoint")
    return float(end_close[0] / start_close[0] - 1)


# %% [markdown]
# Applying the two helpers in symbol order keeps graph rows, tabular features,
# and forward labels aligned.

# %%
feature_rows: list[list[float]] = []
target_values: list[float] = []

for symbol in symbol_order:
    symbol_prices = selected_prices.filter(pl.col("symbol") == symbol).sort("timestamp")
    history = symbol_prices.filter(pl.col("timestamp") <= feature_as_of)
    feature_rows.append(build_node_feature_row(history, symbol))
    target_values.append(compute_forward_return(symbol_prices, feature_as_of, evaluation_end))

node_features = np.nan_to_num(np.asarray(feature_rows), nan=0.0, posinf=0.0, neginf=0.0)
target = np.asarray(target_values)
feature_names = [
    "mom_1m",
    "mom_3m",
    "mom_6m",
    "vol_1m",
    "vol_3m",
    "price_zscore",
    "return_trend",
    "ma_ratio",
]

print(f"Node feature matrix: {node_features.shape}")
print(f"Target range: [{target.min():.3f}, {target.max():.3f}]")

# %% [markdown]
# ## 4. Train graph embeddings inside each fold
#
# A dense graph-attention autoencoder is sufficient for this 200-node teaching
# example and runs directly on CUDA without an additional graph library. For
# each fold:
#
# 1. Feature scaling is fit on training stocks only.
# 2. The encoder is trained on the training-induced subgraph without return labels.
# 3. The trained encoder maps the full pre-target graph for transductive inference.
# 4. Ridge models fit on identical training stocks and predict identical held-out stocks.
#
# Transductive inference may use every stock's pre-target features and edges. It
# never uses a held-out stock's forward-return label during training.

# %% [markdown]
# The scaler exposes its learned moments so the feature and embedding boundaries
# can be tested independently.


# %%
def fit_train_scaler(
    values: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize all rows using moments learned from training rows only."""
    train_mean = values[train_indices].mean(axis=0)
    train_scale = values[train_indices].std(axis=0) + 1e-8
    return (values - train_mean) / train_scale, train_mean, train_scale


# %%
class DenseGATAutoencoder(nn.Module):
    """Single-head graph-attention encoder with feature and edge reconstruction."""

    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, embedding_dim, bias=False)
        self.attention_source = nn.Parameter(torch.empty(embedding_dim))
        self.attention_target = nn.Parameter(torch.empty(embedding_dim))
        self.decoder = nn.Linear(embedding_dim, input_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
        nn.init.normal_(self.attention_source, std=0.1)
        nn.init.normal_(self.attention_target, std=0.1)

    def encode(self, features: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        projected = self.projection(features)
        source_scores = projected @ self.attention_source
        target_scores = projected @ self.attention_target
        attention_logits = self.activation(source_scores[:, None] + target_scores[None, :])
        mask = graph.bool() | torch.eye(graph.shape[0], device=graph.device, dtype=torch.bool)
        attention_logits = attention_logits.masked_fill(
            ~mask, torch.finfo(attention_logits.dtype).min
        )
        attention = torch.softmax(attention_logits, dim=1)
        return attention @ projected

    def forward(
        self, features: torch.Tensor, graph: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encode(features, graph)
        reconstructed_features = self.decoder(embeddings)
        edge_logits = embeddings @ embeddings.T
        return reconstructed_features, edge_logits


# %% [markdown]
# Edge imbalance is derived from the training-induced adjacency matrix. The
# resulting loss objects never inspect held-out nodes or edges.


# %%
def build_graph_losses(
    train_graph: torch.Tensor,
) -> tuple[torch.Tensor, nn.BCEWithLogitsLoss, nn.MSELoss]:
    """Build feature and edge reconstruction losses for one training subgraph."""
    edge_target = train_graph.clone()
    edge_target.fill_diagonal_(1)
    positive_edges = edge_target.sum().clamp_min(1)
    negative_edges = edge_target.numel() - positive_edges
    edge_loss = nn.BCEWithLogitsLoss(pos_weight=negative_edges / positive_edges)
    return edge_target, edge_loss, nn.MSELoss()


# %% [markdown]
# Each fold trains a fresh encoder on its induced subgraph, then applies the
# learned weights to the full pre-target graph for transductive inference.


# %%
def fit_graph_embeddings(
    features: np.ndarray,
    graph: np.ndarray,
    train_indices: np.ndarray,
    fold_seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a fold-local graph autoencoder and return scaled features and embeddings."""
    scaled_features, _, _ = fit_train_scaler(features, train_indices)

    torch.manual_seed(fold_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fold_seed)

    feature_tensor = torch.tensor(scaled_features, dtype=torch.float32, device=DEVICE)
    graph_tensor = torch.tensor(graph, dtype=torch.float32, device=DEVICE)
    train_tensor = torch.tensor(train_indices, dtype=torch.long, device=DEVICE)
    train_features = feature_tensor[train_tensor]
    train_graph = graph_tensor[train_tensor][:, train_tensor]

    model = DenseGATAutoencoder(features.shape[1], EMBEDDING_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    edge_target, edge_loss_fn, feature_loss_fn = build_graph_losses(train_graph)

    final_loss = float("nan")
    model.train()
    for _ in range(GNN_EPOCHS):
        optimizer.zero_grad()
        reconstructed_features, edge_logits = model(train_features, train_graph)
        loss = feature_loss_fn(reconstructed_features, train_features)
        loss = loss + 0.1 * edge_loss_fn(edge_logits, edge_target)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(feature_tensor, graph_tensor).cpu().numpy()

    return scaled_features, embeddings, final_loss


# %% [markdown]
# Spearman IC evaluates rank agreement within each held-out stock fold.


# %%
def compute_ic(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Return cross-sectional Spearman rank correlation."""
    if len(predictions) < 10:
        raise ValueError("Information coefficient requires at least 10 observations")
    result = stats.spearmanr(predictions, actuals)
    return float(result.statistic)


# %% [markdown]
# Ridge fitting keeps the target mean and weights inside the training fold.


# %%
def ridge_predict(
    train_features: np.ndarray,
    train_target: np.ndarray,
    test_features: np.ndarray,
    penalty: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ridge with an unpenalized mean and predict held-out observations."""
    target_mean = train_target.mean()
    gram = train_features.T @ train_features
    weights = np.linalg.solve(
        gram + penalty * np.eye(train_features.shape[1]),
        train_features.T @ (train_target - target_mean),
    )
    return target_mean + test_features @ weights, weights


# %% [markdown]
# One fold evaluation composes the independently testable scaling, graph
# training, ridge, and IC steps without changing their numerical definitions.


# %%
def evaluate_fold(
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, float | int]:
    """Evaluate tabular and hybrid models on one held-out-stock fold."""
    scaled_features, embeddings, graph_loss = fit_graph_embeddings(
        node_features,
        adjacency,
        train_idx,
        SEED + fold,
    )
    scaled_embeddings, _, _ = fit_train_scaler(embeddings, train_idx)
    hybrid_features = np.column_stack([scaled_features, scaled_embeddings])

    tabular_prediction, _ = ridge_predict(
        scaled_features[train_idx], target[train_idx], scaled_features[test_idx]
    )
    hybrid_prediction, _ = ridge_predict(
        hybrid_features[train_idx], target[train_idx], hybrid_features[test_idx]
    )
    tabular_ic = compute_ic(tabular_prediction, target[test_idx])
    hybrid_ic = compute_ic(hybrid_prediction, target[test_idx])
    return {
        "fold": fold,
        "held_out_stocks": len(test_idx),
        "tabular_ic": tabular_ic,
        "hybrid_ic": hybrid_ic,
        "hybrid_minus_tabular": hybrid_ic - tabular_ic,
        "graph_training_loss": graph_loss,
    }


# %% [markdown]
# ## 5. Paired held-out-stock ablation

# %%
splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)
fold_rows: list[dict[str, float | int]] = []

for fold, (train_idx, test_idx) in enumerate(splitter.split(node_features), start=1):
    fold_row = evaluate_fold(fold, train_idx, test_idx)
    fold_rows.append(fold_row)
    print(
        f"Fold {fold}: tabular IC={fold_row['tabular_ic']:+.3f}, "
        f"hybrid IC={fold_row['hybrid_ic']:+.3f}, "
        f"delta={fold_row['hybrid_minus_tabular']:+.3f}, "
        f"graph loss={fold_row['graph_training_loss']:.4f}"
    )

fold_results = pl.DataFrame(fold_rows)
tabular_ic_mean = float(fold_results["tabular_ic"].mean())
hybrid_ic_mean = float(fold_results["hybrid_ic"].mean())
mean_ic_delta = float(fold_results["hybrid_minus_tabular"].mean())

print(f"\nMean tabular IC: {tabular_ic_mean:+.3f}")
print(f"Mean hybrid IC: {hybrid_ic_mean:+.3f}")
print(f"Mean paired delta: {mean_ic_delta:+.3f}")
fold_results

# %% [markdown]
# The five folds partition stocks, not time. Their ICs describe stability across
# held-out subsets of one forward-return window. The paired delta is the narrow
# result: it tests whether these trained graph embeddings helped this model on
# this sample. It does not establish that GNNs generally improve or degrade
# equity forecasts.

# %% [markdown]
# Direct fold labels are spread by a minimum vertical gap. Leader lines preserve
# the exact endpoint association when two hybrid IC values are nearly equal.


# %%
def spread_label_positions(values: np.ndarray, minimum_gap: float) -> np.ndarray:
    """Spread sorted label positions while preserving their vertical order."""
    positions = values.astype(float).copy()
    order = np.argsort(positions)
    for rank in range(1, len(order)):
        lower = order[rank - 1]
        current = order[rank]
        positions[current] = max(positions[current], positions[lower] + minimum_gap)
    return positions


# %% [markdown]
# The first plotting cell draws each paired fold and attaches a collision-safe
# direct label to the hybrid endpoint.

# %%
tabular_fold_ic = fold_results["tabular_ic"].to_numpy()
hybrid_fold_ic = fold_results["hybrid_ic"].to_numpy()
hybrid_label_y = spread_label_positions(hybrid_fold_ic, minimum_gap=0.018)

fig, ax = plt.subplots(figsize=(8.5, 5.2), layout="constrained")
for fold_index, (tabular_ic, hybrid_ic, label_y) in enumerate(
    zip(tabular_fold_ic, hybrid_fold_ic, hybrid_label_y, strict=True), start=1
):
    ax.plot(
        [0, 1],
        [tabular_ic, hybrid_ic],
        color=COLORS["neutral"],
        alpha=0.55,
        linewidth=1.2,
    )
    ax.scatter(
        [0, 1],
        [tabular_ic, hybrid_ic],
        color=[COLORS["blue"], COLORS["amber"]],
        s=45,
        zorder=3,
    )
    ax.annotate(
        f"F{fold_index}",
        xy=(1, hybrid_ic),
        xytext=(1.04, label_y),
        textcoords="data",
        arrowprops={"arrowstyle": "-", "color": COLORS["neutral"], "linewidth": 0.7},
        annotation_clip=False,
        fontsize=8,
        color=COLORS["neutral"],
        va="center",
    )
plt.close(fig)

# %% [markdown]
# The second plotting cell adds the fold means, reference line, result
# annotation, and reader-facing labels.

# %%
ax.scatter(
    [0, 1],
    [tabular_ic_mean, hybrid_ic_mean],
    marker="D",
    s=95,
    color=[COLORS["blue"], COLORS["amber"]],
    edgecolor="white",
    linewidth=0.8,
    zorder=4,
    label="Fold mean",
)
ax.axhline(0, color=COLORS["neutral"], linewidth=0.8)
ax.set_xticks([0, 1], ["Tabular", "Tabular + graph embeddings"])
ax.set_ylabel("Held-out-stock IC (Spearman)")
ax.set_title("Graph features must earn their place in a paired ablation")
ax.text(
    0.02,
    0.02,
    f"Mean paired delta: {mean_ic_delta:+.3f}",
    transform=ax.transAxes,
    color=COLORS["neutral"],
)
ax.legend(frameon=False, loc="upper left")
fig

# %% [markdown]
# ## 6. Inspect the graph input
#
# A distribution view is more legible than a 200-node hairball. The left panel
# shows which correlations cross the edge threshold; the right panel shows how
# unevenly those edges are distributed across stocks.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
axes[0].hist(corr_values, bins=35, color=COLORS["blue"], alpha=0.85)
axes[0].axvline(CORRELATION_THRESHOLD, color=COLORS["amber"], linestyle="--", linewidth=1.5)
axes[0].axvline(-CORRELATION_THRESHOLD, color=COLORS["amber"], linestyle="--", linewidth=1.5)
axes[0].set_xlabel("Pairwise return correlation")
axes[0].set_ylabel("Stock pairs")
axes[0].set_title("Only strong co-movement becomes an edge")

axes[1].hist(degrees, bins=25, color=COLORS["blue"], alpha=0.85)
axes[1].axvline(np.median(degrees), color=COLORS["amber"], linestyle="--", linewidth=1.5)
axes[1].set_xlabel("Node degree")
axes[1].set_ylabel("Stocks")
axes[1].set_title("Connectivity varies across the universe")
fig.suptitle(f"Pre-target correlation graph: {n_edges:,} edges ({graph_density:.1%} density)")
fig.show()

# %% [markdown]
# ## 7. Verification and chapter-impact summary

# %%
assert universe["last_observation"].max() == feature_as_of
assert set(selected_symbols).issubset(set(endpoint_complete_symbols["symbol"].to_list()))
assert returns_wide["timestamp"].max() == feature_as_of
assert np.isfinite(node_features).all()
assert np.isfinite(target).all()
assert fold_results["held_out_stocks"].sum() == len(symbol_order)
assert fold_results["fold"].n_unique() == 5

results = pl.DataFrame(
    {
        "metric": [
            "feature as-of",
            "target end",
            "assets",
            "graph edges",
            "graph density",
            "graph encoder device",
            "mean tabular IC",
            "mean hybrid IC",
            "mean paired IC delta",
        ],
        "value": [
            str(feature_as_of),
            str(evaluation_end),
            str(len(symbol_order)),
            str(n_edges),
            f"{graph_density:.2%}",
            str(DEVICE),
            f"{tabular_ic_mean:+.4f}",
            f"{hybrid_ic_mean:+.4f}",
            f"{mean_ic_delta:+.4f}",
        ],
    }
)
results

# %% [markdown]
# The frozen chapter reports results from an earlier implementation that selected
# the universe with full-sample liquidity, standardized before cross-validation,
# and passed randomly initialized projections off as GNN embeddings. Those
# numbers are not comparable to this corrected experiment and must be treated as
# a documented book-code divergence.
#
# ## Key takeaways
#
# 1. **Train the representation**: Random graph projections are not learned GNN embeddings.
# 2. **Fit preprocessing inside the fold**: Even cross-sectional scaling can leak held-out data.
# 3. **Use a pre-target universe**: Future liquidity cannot decide today's investable set.
# 4. **Interpret the ablation narrowly**: Five stock folds from one target window do not settle
#    whether GNNs help across markets or time.
