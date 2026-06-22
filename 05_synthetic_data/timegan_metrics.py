"""
TimeGAN Evaluation Metrics

Implements the evaluation protocol from:
Yoon, J., Jarrett, D., & van der Schaar, M. (2019).
"Time-series Generative Adversarial Networks." NeurIPS 2019.

Also used for GT-GAN evaluation (same metrics apply to GRU-ODE generators).

Paper Section 5.2 specifies:
1. Predictive Score: Train post-hoc RNN to predict next step
   - Train on synthetic, test on real (TSTR with MAE)
   - Lower MAE = better utility

2. Discriminative Score: Train post-hoc RNN classifier
   - Binary: distinguish real vs synthetic
   - Lower accuracy = better fidelity (harder to tell apart)

Two evaluation modes:

1. **Yoon Mode** (compute_predictive_score_yoon):
   - Matches official TimeGAN repo
   - Predicts ONLY the last feature using other features as input
   - GRU with sigmoid output (expects [0,1] normalized data)
   - Reports raw MAE

2. **Comprehensive Mode** (compute_predictive_score):
   - Predicts ALL features
   - Compares TSTR vs TRTR baseline
   - Reports MAE ratio

Reference: [ref:NCTZ79FF] TimeGAN
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class PostHocLSTMPredictor(nn.Module):
    """
    2-layer LSTM for next-step prediction per TimeGAN paper.

    Architecture (Section 5.2):
    - Input: sequences of shape (batch, seq_len, n_features)
    - 2 LSTM layers with hidden_dim units
    - Dense output layer to n_features
    - Predicts next timestep given previous timesteps

    Used for computing predictive score (TSTR with MAE).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input sequences (batch, seq_len, n_features)

        Returns:
            Predictions for next timestep at each position (batch, seq_len, n_features)
        """
        h, _ = self.lstm(x)
        return self.fc(h)


class PostHocLSTMDiscriminator(nn.Module):
    """
    2-layer LSTM classifier for discriminative score per TimeGAN paper.

    Architecture (Section 5.2):
    - Input: sequences of shape (batch, seq_len, n_features)
    - 2 LSTM layers with hidden_dim units
    - Dense output layer to 1 (sigmoid for binary classification)
    - Classifies sequences as real (1) or synthetic (0)

    Used for computing discriminative score (lower accuracy = better fidelity).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input sequences (batch, seq_len, n_features)

        Returns:
            Logits for real/fake classification (batch, 1)
        """
        h, _ = self.lstm(x)
        # Use last hidden state for classification
        return self.fc(h[:, -1, :])


class GRUPredictor(nn.Module):
    """
    GRU predictor matching official TimeGAN repo.

    Architecture:
    - GRU (not LSTM)
    - hidden_dim = dim // 2
    - Sigmoid output (expects [0, 1] normalized data)
    - Predicts last feature using other features
    """

    def __init__(self, input_dim: int, hidden_dim: int | None = None):
        super().__init__()
        # Paper: hidden_dim = int(dim/2) where dim is total features
        if hidden_dim is None:
            hidden_dim = max(input_dim // 2, 4)
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=1,  # Paper uses single layer
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input sequences (batch, seq_len, n_features-1)
               Note: Last feature already excluded by caller

        Returns:
            Predictions for last feature (batch, seq_len, 1) with sigmoid
        """
        h, _ = self.gru(x)
        return torch.sigmoid(self.fc(h))


class GRUDiscriminator(nn.Module):
    """
    GRU discriminator for TimeGAN evaluation.

    Architecture: GRU with hidden_dim = input_dim // 2, single layer.
    Reports |accuracy - 0.5| where 0.0 = indistinguishable, 0.5 = trivially separable.
    """

    def __init__(self, input_dim: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(input_dim // 2, 4)
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.fc(h[:, -1, :])


def compute_predictive_score_yoon(
    synthetic: np.ndarray,
    real: np.ndarray,
    iterations: int = 5000,
    batch_size: int = 128,
    device: str | None = None,
    quick_test: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Compute predictive score per Yoon et al. (2019).

    Replicates metrics/predictive_metrics.py from official repo.

    Methodology:
    - Predicts ONLY the last feature using other features
    - GRU with sigmoid output (expects [0,1] normalized data)
    - Reports raw MAE (no baseline comparison)

    Args:
        synthetic: Synthetic sequences (n_samples, seq_len, n_features)
        real: Real sequences for evaluation (same shape)
        iterations: Training iterations (default: 5000)
        batch_size: Batch size (default: 128)
        device: 'cuda' or 'cpu' (auto-detected if None)
        quick_test: If True, reduce iterations for testing
        verbose: Print progress

    Returns:
        dict with mae, feature_idx, model
    """
    if quick_test:
        iterations = min(100, iterations)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)
    n_features = synthetic.shape[-1]

    # Check if data appears to be normalized to [0, 1]
    data_min, data_max = synthetic.min(), synthetic.max()
    if data_min < -0.1 or data_max > 1.1:
        if verbose:
            print(f"\nWARNING: Data range [{data_min:.2f}, {data_max:.2f}]")
            print("   Yoon metric expects [0, 1] normalized data.")
            print("   Results may not be comparable to paper.\n")

    if verbose:
        print("Predictive Score (Yoon et al.)")
        print(f"  Features: {n_features - 1} input → 1 output (last feature)")
        print(f"  Iterations: {iterations}, Batch size: {batch_size}")

    hidden_dim = max(n_features // 2, 4)
    model = GRUPredictor(n_features - 1, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.L1Loss()  # MAE loss

    # Prepare training data from synthetic
    # X: all features except last, for all timesteps except last
    # Y: last feature only, shifted by 1 (next-step prediction)
    def prepare_paper_data(data: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare X and Y per paper's methodology."""
        # X: [:-1, :(dim-1)] - all but last timestep, all but last feature
        X = torch.tensor(data[:, :-1, :-1], dtype=torch.float32)
        # Y: [1:, (dim-1)] - shifted by 1, only last feature
        Y = torch.tensor(data[:, 1:, -1:], dtype=torch.float32)
        return X, Y

    X_train, Y_train = prepare_paper_data(synthetic)
    n_train = len(X_train)

    # Training loop (iteration-based, not epoch-based)
    model.train()
    iterator = tqdm(range(iterations), desc="Yoon Predictive", disable=not verbose)
    for _ in iterator:
        # Random mini-batch (paper's approach)
        idx = np.random.permutation(n_train)[:batch_size]
        X_batch = X_train[idx].to(device)
        Y_batch = Y_train[idx].to(device)

        optimizer.zero_grad()
        Y_pred = model(X_batch)
        loss = criterion(Y_pred, Y_batch)
        loss.backward()
        optimizer.step()

        if verbose:
            iterator.set_postfix(loss=loss.item())

    # Evaluate on real data
    model.eval()
    X_real, Y_real = prepare_paper_data(real)
    X_real = X_real.to(device)
    Y_real = Y_real.to(device)

    with torch.no_grad():
        Y_pred = model(X_real)
        mae = torch.abs(Y_pred - Y_real).mean().item()

    if verbose:
        print("\nPredictive Score (Yoon et al.):")
        print(f"  MAE on real data: {mae:.6f}")
        print("  (Paper reports ~0.04 for GOOG stock)")

    return {
        "mae": mae,
        "feature_idx": n_features - 1,
        "model": model,
    }


def compute_discriminative_score_yoon(
    real: np.ndarray,
    synthetic: np.ndarray,
    iterations: int = 2000,
    batch_size: int = 128,
    device: str | None = None,
    quick_test: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Compute discriminative score per Yoon et al. (2019).

    Metric: |accuracy - 0.5| where 0.0 = indistinguishable, 0.5 = trivially separable.
    Data should be MinMax normalized to [0, 1].

    Args:
        real: Real sequences (n_samples, seq_len, n_features)
        synthetic: Synthetic sequences (same shape)
        iterations: Training iterations (default: 2000)
        batch_size: Batch size (default: 128)
        device: 'cuda' or 'cpu' (auto-detected if None)
        quick_test: If True, reduce iterations for testing
        verbose: Print progress

    Returns:
        dict with score, accuracy, model
    """
    if quick_test:
        iterations = min(100, iterations)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)
    n_features = real.shape[-1]

    # Warn if data not normalized
    data_min, data_max = real.min(), real.max()
    if (data_min < -0.1 or data_max > 1.1) and verbose:
        print(f"WARNING: Data range [{data_min:.2f}, {data_max:.2f}] - expected [0, 1]")

    hidden_dim = max(n_features // 2, 4)
    model = GRUDiscriminator(n_features, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    # Balance and combine datasets
    n_samples = min(len(real), len(synthetic))
    X = np.concatenate([real[:n_samples], synthetic[:n_samples]], axis=0)
    y = np.array([1] * n_samples + [0] * n_samples)

    # Shuffle and split 70/30
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    n_train = int(0.7 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # Training (iteration-based)
    model.train()
    iterator = tqdm(range(iterations), desc="Discriminator", disable=not verbose)
    for _ in iterator:
        idx = np.random.permutation(len(X_train_t))[:batch_size]
        X_batch = X_train_t[idx].to(device)
        y_batch = y_train_t[idx].to(device)

        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test_t.to(device))).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)

    accuracy = accuracy_score(y_test, preds)
    score = abs(accuracy - 0.5)

    if verbose:
        print(f"Discriminative: accuracy={accuracy:.1%}, score={score:.4f}")

    return {"score": score, "accuracy": accuracy, "model": model}


def compute_predictive_score(
    synthetic: np.ndarray,
    real_holdout: np.ndarray,
    hidden_dim: int = 128,
    num_layers: int = 2,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str | None = None,
    quick_test: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Compute predictive score per TimeGAN paper (Section 5.2).

    Protocol:
    1. Train LSTM predictor on SYNTHETIC data to predict next timestep
    2. Evaluate on REAL holdout data using MAE
    3. Compare to TRTR baseline (train on real, test on real)

    The ratio TSTR_MAE / TRTR_MAE indicates utility:
    - Ratio ~1.0: Synthetic data preserves predictive patterns
    - Ratio >> 1.0: Synthetic data lost temporal structure

    IMPORTANT: Data is normalized per-feature before evaluation to prevent
    high-variance features (e.g., volume) from dominating the MAE calculation.

    Args:
        synthetic: Synthetic sequences (n_samples, seq_len, n_features)
        real_holdout: Real holdout sequences (never seen by generator)
        hidden_dim: LSTM hidden dimension (paper: 128)
        num_layers: Number of LSTM layers (paper: 2)
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: 'cuda' or 'cpu' (auto-detected if None)
        quick_test: If True, reduce epochs for faster testing
        verbose: Print progress

    Returns:
        dict with:
        - mae_tstr: MAE when trained on synthetic, tested on real (normalized)
        - mae_trtr: MAE when trained on real, tested on real (baseline, normalized)
        - ratio: mae_tstr / mae_trtr (lower is better, 1.0 = perfect)
        - model_tstr: Trained TSTR model
        - model_trtr: Trained TRTR model
    """
    if quick_test:
        epochs = min(10, epochs)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)
    n_features = synthetic.shape[-1]

    # Normalize data per-feature to ensure equal contribution to MAE
    # Compute mean/std from real holdout (the evaluation target)
    holdout_flat = real_holdout.reshape(-1, n_features)
    feat_mean = holdout_flat.mean(axis=0, keepdims=True)
    feat_std = holdout_flat.std(axis=0, keepdims=True) + 1e-8

    if verbose:
        print("\nNormalizing data for fair evaluation...")
        print(f"  Feature means: {feat_mean.flatten()[:3]}... (showing first 3)")
        print(f"  Feature stds:  {feat_std.flatten()[:3]}... (showing first 3)")

    # Normalize all data using holdout statistics
    synthetic_norm = (synthetic - feat_mean) / feat_std
    real_holdout_norm = (real_holdout - feat_mean) / feat_std

    # Split real holdout into train (for TRTR) and test
    n_holdout = len(real_holdout_norm)
    n_train_real = int(0.7 * n_holdout)
    real_train = real_holdout_norm[:n_train_real]
    real_test = real_holdout_norm[n_train_real:]

    def prepare_prediction_data(sequences: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare X (all but last timestep) and Y (shifted by 1)."""
        X = torch.tensor(sequences[:, :-1, :], dtype=torch.float32)
        Y = torch.tensor(sequences[:, 1:, :], dtype=torch.float32)
        return X, Y

    def train_predictor(
        train_data: np.ndarray,
        n_epochs: int,
        desc: str,
    ) -> PostHocLSTMPredictor:
        """Train LSTM predictor on given data."""
        model = PostHocLSTMPredictor(n_features, hidden_dim, num_layers).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.L1Loss()  # MAE loss

        X_train, Y_train = prepare_prediction_data(train_data)
        dataset = TensorDataset(X_train, Y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        iterator = tqdm(range(n_epochs), desc=desc, disable=not verbose)
        for _ in iterator:
            epoch_loss = 0.0
            for X_batch, Y_batch in loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

                optimizer.zero_grad()
                Y_pred = model(X_batch)
                loss = criterion(Y_pred, Y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if verbose:
                iterator.set_postfix(loss=epoch_loss / len(loader))

        return model

    def evaluate_predictor(model: PostHocLSTMPredictor, test_data: np.ndarray) -> float:
        """Evaluate MAE on test data."""
        model.eval()
        X_test, Y_test = prepare_prediction_data(test_data)
        X_test, Y_test = X_test.to(device), Y_test.to(device)

        with torch.no_grad():
            Y_pred = model(X_test)
            mae = torch.abs(Y_pred - Y_test).mean().item()

        return mae

    # TSTR: Train on Synthetic (normalized), Test on Real (normalized)
    model_tstr = train_predictor(synthetic_norm, epochs, "TSTR Predictor")
    mae_tstr = evaluate_predictor(model_tstr, real_test)

    # TRTR: Train on Real, Test on Real (baseline)
    model_trtr = train_predictor(real_train, epochs, "TRTR Predictor")
    mae_trtr = evaluate_predictor(model_trtr, real_test)

    # Compute ratio
    ratio = mae_tstr / mae_trtr if mae_trtr > 0 else float("inf")

    if verbose:
        print("\nPredictive Score (MAE, lower is better):")
        print(f"  TRTR (baseline): {mae_trtr:.6f}")
        print(f"  TSTR (synthetic): {mae_tstr:.6f}")
        print(f"  Ratio (TSTR/TRTR): {ratio:.3f}")
        if ratio <= 1.1:
            print("  Assessment: [OK] Good utility")
        elif ratio <= 1.5:
            print("  Assessment: WARNING: Moderate utility")
        else:
            print("  Assessment: [FAIL] Poor utility")

    return {
        "mae_tstr": mae_tstr,
        "mae_trtr": mae_trtr,
        "ratio": ratio,
        "model_tstr": model_tstr,
        "model_trtr": model_trtr,
    }


def compute_discriminative_score(
    real: np.ndarray,
    synthetic: np.ndarray,
    hidden_dim: int = 128,
    num_layers: int = 2,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str | None = None,
    quick_test: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Compute discriminative score per TimeGAN paper (Section 5.2).

    Protocol:
    1. Train LSTM classifier to distinguish real vs synthetic
    2. Evaluate accuracy on held-out test set
    3. Lower accuracy = better fidelity (harder to tell apart)

    Target: Accuracy ~50% = indistinguishable
    Good: Accuracy < 70%
    Poor: Accuracy > 85%

    Args:
        real: Real sequences (n_samples, seq_len, n_features)
        synthetic: Synthetic sequences (same shape)
        hidden_dim: LSTM hidden dimension (paper: 128)
        num_layers: Number of LSTM layers (paper: 2)
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: 'cuda' or 'cpu' (auto-detected if None)
        quick_test: If True, reduce epochs for faster testing
        verbose: Print progress

    Returns:
        dict with:
        - accuracy: Classification accuracy (lower = better fidelity)
        - auc: ROC AUC score
        - model: Trained discriminator
    """
    if quick_test:
        epochs = min(10, epochs)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)
    n_features = real.shape[-1]

    # Normalize data per-feature (using real data statistics)
    real_flat = real.reshape(-1, n_features)
    feat_mean = real_flat.mean(axis=0, keepdims=True)
    feat_std = real_flat.std(axis=0, keepdims=True) + 1e-8

    real_norm = (real - feat_mean) / feat_std
    synth_norm = (synthetic - feat_mean) / feat_std

    # Balance dataset sizes
    n_samples = min(len(real_norm), len(synth_norm))
    real_subset = real_norm[:n_samples]
    synth_subset = synth_norm[:n_samples]

    # Fixed sequential split (80/20) - matches benchmark methodology
    # This ensures temporal consistency: test set has most recent data from both
    n_train = int(0.8 * n_samples)
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, n_samples)

    # Stack real and synthetic for train/test separately
    X_train = np.concatenate([real_subset[train_idx], synth_subset[train_idx]], axis=0)
    X_test = np.concatenate([real_subset[test_idx], synth_subset[test_idx]], axis=0)

    n_train_samples = len(train_idx)
    n_test_samples = len(test_idx)
    y_train = np.concatenate([np.ones(n_train_samples), np.zeros(n_train_samples)])
    y_test = np.concatenate([np.ones(n_test_samples), np.zeros(n_test_samples)])

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Create data loader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = PostHocLSTMDiscriminator(n_features, hidden_dim, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    model.train()
    iterator = tqdm(range(epochs), desc="Discriminator", disable=not verbose)
    for _ in iterator:
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if verbose:
            iterator.set_postfix(loss=epoch_loss / len(train_loader))

    # Evaluation
    model.eval()
    X_test_t = X_test_t.to(device)

    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)

    accuracy = accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, probs)
    except ValueError:
        auc = 0.5

    if verbose:
        print("\nDiscriminative Score (lower accuracy = better):")
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  ROC AUC: {auc:.4f}")
        if accuracy <= 0.55:
            print("  Assessment: [OK] Excellent fidelity (indistinguishable)")
        elif accuracy <= 0.70:
            print("  Assessment: [OK] Good fidelity")
        elif accuracy <= 0.85:
            print("  Assessment: WARNING: Moderate fidelity")
        else:
            print("  Assessment: [FAIL] Poor fidelity (easily distinguished)")

    return {
        "accuracy": accuracy,
        "auc": auc,
        "model": model,
    }


def run_timegan_evaluation(
    synthetic: np.ndarray,
    real_train: np.ndarray,
    real_holdout: np.ndarray,
    hidden_dim: int = 128,
    num_layers: int = 2,
    epochs_pred: int = 100,
    epochs_disc: int = 50,
    iterations_yoon: int = 5000,
    device: str | None = None,
    quick_test: bool = False,
    verbose: bool = True,
    include_yoon: bool = True,
) -> dict[str, Any]:
    """
    Run complete TimeGAN evaluation suite.

    Combines:
    1. Yoon predictive score (optional): Last-feature prediction
    2. Comprehensive predictive score: All-feature TSTR with baseline
    3. Discriminative score

    Args:
        synthetic: Synthetic sequences from generator
        real_train: Real sequences for discriminative training
        real_holdout: Real sequences for predictive evaluation
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        epochs_pred: Epochs for comprehensive predictive model
        epochs_disc: Epochs for discriminator
        iterations_yoon: Iterations for Yoon metric (default: 5000)
        device: Compute device
        quick_test: Reduce epochs/iterations for testing
        verbose: Print results
        include_yoon: If True, run Yoon metric for comparison

    Returns:
        dict with yoon, predictive, discriminative, summary
    """
    if verbose:
        print("=" * 60)
        print("TimeGAN Evaluation Suite")
        print("=" * 60)
        print("\nReference: Yoon et al., NeurIPS 2019, Section 5.2")

    # Align shapes
    min_samples = min(len(synthetic), len(real_train), len(real_holdout))
    min_seq_len = min(synthetic.shape[1], real_train.shape[1], real_holdout.shape[1])
    min_features = min(synthetic.shape[2], real_train.shape[2], real_holdout.shape[2])

    synthetic = synthetic[:min_samples, :min_seq_len, :min_features]
    real_train = real_train[:min_samples, :min_seq_len, :min_features]
    real_holdout = real_holdout[:min_samples, :min_seq_len, :min_features]

    results = {}

    # 1. Yoon predictive score (optional)
    if include_yoon:
        if verbose:
            print("\n1. YOON PREDICTIVE SCORE")
            print("-" * 40)
            print("(Last-feature prediction per Yoon et al.)")
        yoon_results = compute_predictive_score_yoon(
            synthetic=synthetic,
            real=real_holdout,
            iterations=iterations_yoon,
            device=device,
            quick_test=quick_test,
            verbose=verbose,
        )
        results["yoon"] = yoon_results

    # 2. Comprehensive predictive score
    if verbose:
        section_num = "2" if include_yoon else "1"
        print(f"\n{section_num}. COMPREHENSIVE PREDICTIVE SCORE (Utility)")
        print("-" * 40)
        print("(All-feature TSTR with TRTR baseline)")
    pred_results = compute_predictive_score(
        synthetic=synthetic,
        real_holdout=real_holdout,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        epochs=epochs_pred,
        device=device,
        quick_test=quick_test,
        verbose=verbose,
    )
    results["predictive"] = pred_results

    # 3. Discriminative score
    if verbose:
        section_num = "3" if include_yoon else "2"
        print(f"\n{section_num}. DISCRIMINATIVE SCORE (Fidelity)")
        print("-" * 40)
    disc_results = compute_discriminative_score(
        real=real_train,
        synthetic=synthetic,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        epochs=epochs_disc,
        device=device,
        quick_test=quick_test,
        verbose=verbose,
    )
    results["discriminative"] = disc_results

    # Combined summary
    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        if include_yoon:
            print("\nYoon Metric (last-feature prediction):")
            print(f"  MAE on last feature: {yoon_results['mae']:.6f}")
            print("  (Paper reports ~0.04 for GOOG stock)")

        print("\nComprehensive Assessment:")
        print(f"  Predictive Score (MAE ratio): {pred_results['ratio']:.3f}")
        print(f"  Discriminative Score (accuracy): {disc_results['accuracy']:.1%}")

        # Overall assessment
        pred_good = pred_results["ratio"] <= 1.2
        disc_good = disc_results["accuracy"] <= 0.70

        if pred_good and disc_good:
            print("\n  Overall: [OK] Generator passes both tests")
        elif pred_good or disc_good:
            print("\n  Overall: WARNING: Generator passes one test")
        else:
            print("\n  Overall: [FAIL] Generator fails both tests")

    # Build summary
    summary = {
        "mae_ratio": pred_results["ratio"],
        "disc_accuracy": disc_results["accuracy"],
        "disc_auc": disc_results["auc"],
    }
    if include_yoon:
        summary["yoon_mae"] = yoon_results["mae"]

    results["summary"] = summary
    return results


if __name__ == "__main__":
    # Quick test with random data
    print("TimeGAN Metrics Module - Quick Test")
    print("=" * 40)

    # Generate random test data
    np.random.seed(42)
    n_samples = 100
    seq_len = 24
    n_features = 6

    real_train = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    real_holdout = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    synthetic = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)

    # Run quick evaluation
    results = run_timegan_evaluation(
        synthetic=synthetic,
        real_train=real_train,
        real_holdout=real_holdout,
        quick_test=True,
        verbose=True,
    )

    print("\nTest completed successfully!")
