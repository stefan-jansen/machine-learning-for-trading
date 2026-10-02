"""Aggregation math, calibration, and evaluation functions.

Pure math only — no agent classes. Each agent class is built step-by-step in
its teaching notebook (NB04 ResearchAgent, NB07 DebateAgent, NB08 SupervisorAgent
and AIAForecaster) and mirrored in a helper module for downstream reuse:
``agent_research`` for ResearchAgent and ``agent_specialists`` for DebateAgent
and SupervisorAgent.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from agent_schemas import AggregationResult


def validate_probabilities(values: Sequence[float]) -> None:
    if len(values) == 0:
        raise ValueError("At least one probability is required")
    if any(isinstance(p, bool) or not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("Probabilities must be finite and in [0, 1]")


def _validate_scores(predictions: Sequence[float], outcomes: Sequence[float]) -> None:
    validate_probabilities(predictions)
    if len(predictions) != len(outcomes):
        raise ValueError("Predictions and outcomes must have the same length")
    if any(o not in (0, 1) for o in outcomes):
        raise ValueError("Resolved binary outcomes must be 0 or 1")


def _validate_bootstrap(n_boot: int, alpha: float) -> None:
    if not isinstance(n_boot, int) or n_boot < 1 or not 0 < alpha < 1:
        raise ValueError("n_boot must be positive and alpha must be in (0, 1)")


def _validate_extremization(base: float, d: float | None, correlation: float) -> None:
    validate_probabilities([base])
    if not math.isfinite(correlation) or not 0 <= correlation <= 1:
        raise ValueError("Correlation must be finite and in [0, 1]")
    if d is not None and (not math.isfinite(d) or d <= 0):
        raise ValueError("Extremization factor must be finite and positive")


# ---------------------------------------------------------------------------
# Probability clamping
# ---------------------------------------------------------------------------


def clamp_prob(p: float, eps: float = 1e-6) -> float:
    """Clamp probability to (eps, 1-eps) to avoid log(0)."""
    validate_probabilities([p])
    if not 0 < eps < 0.5:
        raise ValueError("eps must be in (0, 0.5)")
    return max(eps, min(1.0 - eps, p))


# ---------------------------------------------------------------------------
# Platt scaling / extremization
# ---------------------------------------------------------------------------


def platt_scale(p: float, a: float, d: float = 1.0) -> float:
    r"""Platt scaling as described in the AIA Forecaster paper.

    $$p' = \frac{d \cdot p^a}{d \cdot p^a + (1-p)^a}$$

    - $a > 1$ pushes probabilities away from 0.5 (under-confident agents)
    - $a < 1$ pulls toward 0.5 (over-confident agents)
    - $d$ shifts the midpoint asymmetrically
    """
    p = clamp_prob(p)
    num = d * (p**a)
    den = num + ((1.0 - p) ** a)
    return float(num / den)


def logodds_extremize(p: float, a: float) -> float:
    r"""Alternative calibration in log-odds space.

    $$p' = \sigma(a \cdot \text{logit}(p))$$

    Same effect as Platt scaling but operates in log-odds space.
    """
    p = clamp_prob(p)
    logit = math.log(p / (1.0 - p))
    return float(1.0 / (1.0 + math.exp(-a * logit)))


# Illustrative per-model calibration d values — starting points ONLY, not from
# the AIA Forecaster paper. That paper (Alur et al. 2025) recommends a single
# d = sqrt(3) for all forecasters (Neyman & Roughgarden 2022) and deliberately
# avoids per-model tuning to prevent overfitting. Tune these on your own resolved
# forecasts via find_optimal_d; the production path uses find_optimal_d or DEFAULT_D.
MODEL_CALIBRATION_D: dict[str, float] = {
    "claude-sonnet-4-20250514": 1.5,
    "claude-3-5-sonnet": 1.5,
    "claude-3-5-sonnet-20241022": 1.5,
    "claude-3-5-haiku": 1.6,
    "claude-3-opus": 1.4,
    "gpt-4o": 1.8,
    "gpt-4o-2024-08-06": 1.8,
    "gpt-4o-mini": 1.9,
    "gpt-4.1-mini": 1.8,
}

DEFAULT_D = math.sqrt(3)  # ~1.732 (AIA paper recommendation)


def get_model_calibration_d(model: str) -> float:
    """Get recommended calibration d value for a specific LLM model."""
    if model in MODEL_CALIBRATION_D:
        return MODEL_CALIBRATION_D[model]
    for known, d_val in MODEL_CALIBRATION_D.items():
        if model.startswith(known.split("-")[0]):
            return d_val
    return DEFAULT_D


# ---------------------------------------------------------------------------
# Neyman extremization (multi-agent aggregation)
# ---------------------------------------------------------------------------


def neyman_extremize(
    probabilities: Sequence[float],
    base: float = 0.5,
    d: float | None = None,
    correlation: float = 0.5,
) -> AggregationResult:
    r"""Neyman extremization for aggregating multiple forecasts.

    Accounts for correlation between forecasters and pushes the
    aggregate away from the base rate when forecasters agree.

    $$d = \sqrt{\frac{n}{1 + (n-1)\rho}}$$

    $$p_{\text{extreme}} = (p_{\text{mean}} - \text{base}) \cdot d + \text{base}$$

    With $\rho = 0$ (independent): $d = \sqrt{n}$.
    With $\rho = 1$ (identical): $d = 1$ (no extremization).
    """
    validate_probabilities(probabilities)
    _validate_extremization(base, d, correlation)

    n = len(probabilities)
    p_mean = sum(probabilities) / n

    if d is None:
        effective_n = n / (1 + (n - 1) * correlation)
        d = math.sqrt(effective_n)
        d = max(1.0, min(3.0, d))

    p_extreme = (p_mean - base) * d + base
    p_extreme = max(0.01, min(0.99, p_extreme))

    return AggregationResult(
        method="neyman",
        raw_probability=round(p_mean, 4),
        extremized_probability=round(p_extreme, 4),
        extremization_factor=round(d, 4),
        input_probabilities=list(probabilities),
        effective_n=round(d**2, 2),
    )


def neyman_extremize_weighted(
    probabilities: Sequence[float],
    weights: Sequence[float],
    base: float = 0.5,
    d: float | None = None,
    correlation: float = 0.5,
) -> AggregationResult:
    r"""Weighted Neyman extremization using confidence weights.

    Uses the Herfindahl index for effective sample size when
    forecasters have different confidence levels.

    $$\text{HI} = \sum w_i^2, \quad n_{\text{eff}} = \frac{1}{\text{HI}}$$
    """
    validate_probabilities(probabilities)
    _validate_extremization(base, d, correlation)
    if len(probabilities) != len(weights):
        raise ValueError("probabilities and weights must have same length")
    if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("Weights must be finite, nonnegative, and have a positive sum")

    # Normalize weights
    total_w = sum(weights)
    norm_w = [w / total_w for w in weights] if total_w > 0 else [1.0 / len(weights)] * len(weights)

    # Weighted mean
    p_mean = sum(p * w for p, w in zip(probabilities, norm_w, strict=False))

    if d is None:
        herfindahl = sum(w**2 for w in norm_w)
        effective_n = 1.0 / herfindahl if herfindahl > 0 else 1.0
        effective_n = effective_n / (1 + (effective_n - 1) * correlation)
        d = math.sqrt(effective_n)
        d = max(1.0, min(3.0, d))
    else:
        effective_n = d**2

    p_extreme = (p_mean - base) * d + base
    p_extreme = max(0.01, min(0.99, p_extreme))

    return AggregationResult(
        method="neyman_weighted",
        raw_probability=round(p_mean, 4),
        extremized_probability=round(p_extreme, 4),
        extremization_factor=round(d, 4),
        input_probabilities=list(probabilities),
        input_weights=list(weights),
        effective_n=round(effective_n, 2),
    )


# ---------------------------------------------------------------------------
# Calibration optimization
# ---------------------------------------------------------------------------


@dataclass
class CalibrationResult:
    """Result of calibration parameter optimization."""

    optimal_d: float
    brier_before: float
    brier_after: float
    improvement_pct: float


def find_optimal_d(
    forecasts: Sequence[float],
    outcomes: Sequence[int | float],
    d_range: tuple[float, float] = (0.5, 3.0),
    n_steps: int = 50,
) -> CalibrationResult:
    """Find optimal calibration d via grid search on historical data.

    Use this to tune calibration parameters when you have resolved forecasts.
    """
    _validate_scores(forecasts, outcomes)
    if n_steps < 1 or not all(math.isfinite(d) and d > 0 for d in d_range):
        raise ValueError("Calibration requires positive finite factors and n_steps")
    if d_range[0] > d_range[1]:
        raise ValueError("Calibration range must be ordered")

    def _brier(p: float, y: float) -> float:
        return (p - y) ** 2

    baseline = [logodds_extremize(p, 1.0) for p in forecasts]
    brier_before = sum(_brier(p, float(o)) for p, o in zip(baseline, outcomes, strict=False)) / len(
        forecasts
    )

    best_d, best_brier = 1.0, brier_before
    step = (d_range[1] - d_range[0]) / n_steps

    for i in range(n_steps + 1):
        d_val = d_range[0] + i * step
        calibrated = [logodds_extremize(p, d_val) for p in forecasts]
        brier = sum(_brier(p, float(o)) for p, o in zip(calibrated, outcomes, strict=False)) / len(
            forecasts
        )
        if brier < best_brier:
            best_brier = brier
            best_d = d_val

    improvement = (brier_before - best_brier) / brier_before * 100 if brier_before > 0 else 0

    return CalibrationResult(
        optimal_d=round(best_d, 3),
        brier_before=round(brier_before, 4),
        brier_after=round(best_brier, 4),
        improvement_pct=round(improvement, 1),
    )


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def brier_score(predictions: Sequence[float], outcomes: Sequence[float]) -> float:
    """Mean squared error between predicted probabilities and outcomes.

    Perfect score = 0.0. Random (0.5) = 0.25. Worst = 1.0.
    """
    _validate_scores(predictions, outcomes)
    return sum((p - o) ** 2 for p, o in zip(predictions, outcomes, strict=True)) / len(predictions)


def brier_ci(
    predictions: Sequence[float],
    outcomes: Sequence[float],
    n_boot: int = 20000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap confidence interval for a Brier score.

    A Brier score on a handful of resolved questions is a very noisy estimate,
    and a point estimate invites rankings the data cannot support. Resample the
    questions with replacement and report the interval alongside the number.
    """
    _validate_scores(predictions, outcomes)
    _validate_bootstrap(n_boot, alpha)
    rng = random.Random(seed)
    k = len(predictions)
    losses = [(p - o) ** 2 for p, o in zip(predictions, outcomes, strict=True)]
    scores = []
    for _ in range(n_boot):
        idx = [rng.randrange(k) for _ in range(k)]
        scores.append(sum(losses[i] for i in idx) / k)
    scores.sort()
    return (scores[int((alpha / 2) * n_boot)], scores[int((1 - alpha / 2) * n_boot)])


def brier_diff_ci(
    predictions_a: Sequence[float],
    predictions_b: Sequence[float],
    outcomes: Sequence[float],
    n_boot: int = 20000,
    alpha: float = 0.05,
    seed: int = 1,
) -> tuple[float, float]:
    """Bootstrap CI for the *paired* Brier difference (A − B) on one panel.

    Pairing matters: both configurations are scored on the same resampled
    questions, so shared question difficulty cancels. An interval containing
    zero means the panel cannot separate the two configurations.
    """
    _validate_scores(predictions_a, outcomes)
    _validate_scores(predictions_b, outcomes)
    _validate_bootstrap(n_boot, alpha)
    rng = random.Random(seed)
    k = len(outcomes)
    losses = [
        (a - o) ** 2 - (b - o) ** 2
        for a, b, o in zip(predictions_a, predictions_b, outcomes, strict=True)
    ]
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(k) for _ in range(k)]
        diffs.append(sum(losses[i] for i in idx) / k)
    diffs.sort()
    return (diffs[int((alpha / 2) * n_boot)], diffs[int((1 - alpha / 2) * n_boot)])


def log_score(predictions: Sequence[float], outcomes: Sequence[float], eps: float = 1e-12) -> float:
    """Cross-entropy loss. Lower is better. Perfect = 0.0."""
    _validate_scores(predictions, outcomes)
    if not 0 < eps < 0.5:
        raise ValueError("eps must be in (0, 0.5)")
    total = 0.0
    for p, o in zip(predictions, outcomes, strict=False):
        p = max(eps, min(1 - eps, p))
        total += -(o * math.log(p) + (1 - o) * math.log(1 - p))
    return total / len(predictions)


def sharpness(predictions: Sequence[float]) -> float:
    """Average distance from 0.5. Higher = more decisive. Range [0, 0.5]."""
    validate_probabilities(predictions)
    return sum(abs(p - 0.5) for p in predictions) / len(predictions)


def expected_calibration_error(
    predictions: Sequence[float],
    outcomes: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Expected calibration error — weighted |avg_pred - avg_observed| per bin."""
    _validate_scores(predictions, outcomes)
    bins = reliability_bins(list(predictions), list(outcomes), n_bins)
    n = len(predictions)
    return sum(b["count"] / n * abs(b["avg_predicted"] - b["avg_observed"]) for b in bins)


def reliability_bins(
    predictions: list[float],
    outcomes: list[float],
    n_bins: int = 5,
) -> list[dict]:
    """Bin predictions for reliability/calibration curve."""
    _validate_scores(predictions, outcomes)
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")

    bins: list[dict] = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        mask = [(lo <= p < hi) or (i == n_bins - 1 and p == hi) for p in predictions]
        bin_preds = [p for p, m in zip(predictions, mask, strict=False) if m]
        bin_outcomes = [o for o, m in zip(outcomes, mask, strict=False) if m]

        if bin_preds:
            bins.append(
                {
                    "bin": i,
                    "lo": lo,
                    "hi": hi,
                    "count": len(bin_preds),
                    "avg_predicted": sum(bin_preds) / len(bin_preds),
                    "avg_observed": sum(bin_outcomes) / len(bin_outcomes),
                }
            )
    return bins
