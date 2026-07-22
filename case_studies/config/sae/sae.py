"""Supervised Autoencoder (Jane Street architecture).

Three-headed network:
1. Decoder: reconstructs input features (regularization)
2. Aux Head: predicts from bottleneck (forces predictive embedding)
3. Main Head: full MLP with skip connection (best predictions)

Features: BatchNorm, Swish activation, GaussianNoise, skip connections.
Supports both classification (sigmoid) and regression (linear) output.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Swish(nn.Module):
    """Swish activation: x * sigmoid(x).

    Also known as SiLU. Implemented explicitly for pedagogical clarity.
    Better than ReLU: smooth, non-monotonic, self-gated, no dead neurons.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class GaussianNoise(nn.Module):
    """Additive Gaussian noise during training (dropout alternative).

    Unlike dropout (multiplicative), this adds continuous noise.
    Better for continuous features where we want uncertainty, not zeroing.

    Args:
        std: Standard deviation of noise to add
    """

    def __init__(self, std: float = 0.1):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


class SupervisedAutoencoder(nn.Module):
    """Jane Street Supervised Autoencoder MLP.

    Three-headed architecture:
    1. Decoder: reconstructs input features (regularization)
    2. Aux Head: predicts from bottleneck (forces predictive embedding)
    3. Main Head: full MLP with skip connection (best predictions)

    Args:
        n_features: Number of input features
        n_labels: Number of output labels
        hidden_units: Hidden layer sizes [encoder, decoder_mlp, mlp1, mlp2, ...]
        dropout_rates: Dropout rates for each position
        noise_std: Standard deviation for input noise
        output_activation: "sigmoid" for classification, "linear" for regression
    """

    # Keras `BatchNormalization()` defaults are eps=1e-3, momentum=0.99 for the
    # running-stat update. PyTorch flips the convention: `momentum` is the weight
    # given to the new batch, so the Keras-equivalent is 1 - 0.99 = 0.01. Using
    # PyTorch's defaults (eps=1e-5, momentum=0.1) makes eval-time stats
    # heavily skew toward the last training cross-section seen in each epoch,
    # producing chronological drift on financial panels.
    BN_EPS = 1e-3
    BN_MOMENTUM = 0.01

    def __init__(
        self,
        n_features: int,
        n_labels: int = 1,
        hidden_units: list[int] | None = None,
        dropout_rates: list[float] | None = None,
        noise_std: float = 0.035,
        output_activation: str = "sigmoid",
    ):
        super().__init__()

        if hidden_units is None:
            hidden_units = [96, 96, 896, 448, 448, 256]
        if dropout_rates is None:
            dropout_rates = [0.035, 0.038, 0.424, 0.104, 0.492, 0.320, 0.272, 0.438]

        if len(hidden_units) != 6:
            raise ValueError(
                f"hidden_units must contain exactly 6 entries (encoder, aux_hidden, "
                f"main_mlp_1..4); got {len(hidden_units)}"
            )
        if len(dropout_rates) != 8:
            raise ValueError(
                f"dropout_rates must contain exactly 8 entries (noise, decoder, aux, "
                f"main_input, main_1..4); got {len(dropout_rates)}"
            )
        if output_activation not in {"sigmoid", "linear", "identity"}:
            raise ValueError(
                f"output_activation must be 'sigmoid' / 'linear' / 'identity'; "
                f"got {output_activation!r}"
            )

        self.n_features = n_features
        self.n_labels = n_labels
        self.output_activation = output_activation

        bn_eps = self.BN_EPS
        bn_mom = self.BN_MOMENTUM

        # Encoder: input -> bottleneck
        self.input_bn = nn.BatchNorm1d(n_features, eps=bn_eps, momentum=bn_mom)
        self.input_noise = GaussianNoise(noise_std)
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_units[0]),
            nn.BatchNorm1d(hidden_units[0], eps=bn_eps, momentum=bn_mom),
            Swish(),
        )

        # Decoder: reconstruct input from bottleneck
        self.decoder_dropout = nn.Dropout(dropout_rates[1])
        self.decoder = nn.Linear(hidden_units[0], n_features)

        # Auxiliary head: predict from decoder output
        def _make_output_act() -> nn.Module:
            return nn.Sigmoid() if output_activation == "sigmoid" else nn.Identity()

        self.aux_head = nn.Sequential(
            nn.Linear(n_features, hidden_units[1]),
            nn.BatchNorm1d(hidden_units[1], eps=bn_eps, momentum=bn_mom),
            Swish(),
            nn.Dropout(dropout_rates[2]),
            nn.Linear(hidden_units[1], n_labels),
            _make_output_act(),
        )

        # Main MLP head with skip connection
        concat_dim = n_features + hidden_units[0]
        self.main_bn = nn.BatchNorm1d(concat_dim, eps=bn_eps, momentum=bn_mom)
        self.main_dropout_input = nn.Dropout(dropout_rates[3])

        mlp_layers: list[nn.Module] = []
        in_dim = concat_dim
        for i, out_dim in enumerate(hidden_units[2:]):
            mlp_layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim, eps=bn_eps, momentum=bn_mom),
                    Swish(),
                    nn.Dropout(dropout_rates[min(i + 4, len(dropout_rates) - 1)]),
                ]
            )
            in_dim = out_dim

        self.main_mlp = nn.Sequential(*mlp_layers)
        self.main_output = nn.Sequential(
            nn.Linear(in_dim, n_labels),
            _make_output_act(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning all three outputs.

        Returns:
            decoded: reconstructed features (for MSE loss)
            aux_pred: auxiliary predictions
            main_pred: main predictions (best quality)
        """
        x_norm = self.input_bn(x)
        x_noisy = self.input_noise(x_norm)

        encoded = self.encoder(x_noisy)

        decoded = self.decoder(self.decoder_dropout(encoded))
        aux_pred = self.aux_head(decoded)

        concat = torch.cat([x_norm, encoded], dim=1)
        concat = self.main_bn(concat)
        concat = self.main_dropout_input(concat)
        mlp_out = self.main_mlp(concat)
        main_pred = self.main_output(mlp_out)

        return decoded, aux_pred, main_pred

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Main predictions only (for inference).

        Forces ``eval`` mode so dropout, Gaussian noise, and BatchNorm batch
        statistics never leak into predictions even if the caller forgot to
        switch the model out of training mode. Restores the prior mode on exit.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                _, _, main_pred = self.forward(x)
            return main_pred
        finally:
            self.train(was_training)
