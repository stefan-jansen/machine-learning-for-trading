# rl_calibration.py - Market calibration from real data for RL environments
"""
Market Calibration Module for RL Environments

This module fits financial models to REAL market data to produce calibrated
parameters for simulation. RL environments should use these parameters
instead of arbitrary magic numbers.

Models supported:
- GARCH(1,1): Volatility clustering (most common for equities/crypto)
- Regime switching: Markov regime detection for spreads/depth

Data source: Uses actual crypto hourly data from the book's case studies.

Usage:
    from rl_calibration import CryptoMarketCalibrator

    calibrator = CryptoMarketCalibrator()
    params = calibrator.get_execution_env_params()

    # Use in environment:
    env = ExecutionEnv(**params)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

# ML4T configuration
from data import load_crypto_perps


@dataclass
class GARCHParams:
    """GARCH(1,1) parameters fitted to data."""

    omega: float  # Constant term
    alpha: float  # ARCH term (reaction to shocks)
    beta: float  # GARCH term (persistence)
    mu: float  # Mean return
    unconditional_vol: float  # Long-run volatility

    @property
    def persistence(self) -> float:
        """Alpha + beta (should be < 1 for stationarity)."""
        return self.alpha + self.beta


@dataclass
class RegimeParams:
    """Regime switching parameters fitted to data."""

    n_regimes: int
    transition_matrix: np.ndarray  # P(regime_t+1 | regime_t)
    regime_means: dict[str, np.ndarray]  # Mean values per regime
    regime_stds: dict[str, np.ndarray]  # Std values per regime


@dataclass
class ExecutionEnvParams:
    """Calibrated parameters for ExecutionEnv."""

    # GARCH volatility model
    garch: GARCHParams

    # Regime-dependent microstructure
    regimes: RegimeParams

    # Spread parameters (as fraction of price)
    spread_normal: float
    spread_stressed: float

    # Depth parameters (in base units)
    depth_normal: float
    depth_stressed: float

    # Impact parameters
    permanent_impact: float
    temporary_impact: float


class CryptoMarketCalibrator:
    """Calibrate market simulation parameters from real crypto data.

    This class loads actual crypto hourly data and fits models to produce
    realistic simulation parameters for RL training.

    Parameters
    ----------
    symbol : str, default="BTCUSDT"
        Crypto symbol to calibrate from

    Example
    -------
    >>> calibrator = CryptoMarketCalibrator("BTCUSDT")
    >>> params = calibrator.get_execution_env_params()
    >>> print(f"GARCH alpha: {params.garch.alpha:.4f}")
    >>> print(f"GARCH beta: {params.garch.beta:.4f}")
    >>> print(f"Unconditional vol: {params.garch.unconditional_vol:.4f}")
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
    ):
        self.symbol = symbol
        self._df: pl.DataFrame | None = None
        self._returns: np.ndarray | None = None

    def load_data(self) -> pl.DataFrame:
        """Load crypto OHLCV data via canonical loader.

        Returns
        -------
        pl.DataFrame
            Hourly OHLCV data for the symbol
        """
        if self._df is not None:
            return self._df

        df = load_crypto_perps(frequency="1h", symbols=[self.symbol])

        if len(df) == 0:
            all_symbols = (
                load_crypto_perps(frequency="1h").select("symbol").unique().to_series().to_list()
            )
            raise ValueError(f"Symbol {self.symbol} not found. Available: {all_symbols[:10]}")

        # Sort by time
        df = df.sort("timestamp")
        self._df = df

        return df

    def compute_returns(self) -> np.ndarray:
        """Compute log returns from close prices.

        Returns
        -------
        np.ndarray
            Log returns (annualized for hourly data: ~8760 hours/year)
        """
        if self._returns is not None:
            return self._returns

        df = self.load_data()
        close = df["close"].to_numpy()

        # Log returns
        returns = np.diff(np.log(close))
        self._returns = returns

        return returns

    def fit_garch(self) -> GARCHParams:
        """Fit GARCH(1,1) model to returns.

        Uses the arch library for estimation.

        Returns
        -------
        GARCHParams
            Fitted GARCH parameters
        """
        try:
            from arch import arch_model
        except ImportError:
            # Fallback to moment-based estimation
            return self._fit_garch_moments()

        returns = self.compute_returns()

        # Scale returns to percentage for numerical stability
        returns_pct = returns * 100

        # Fit GARCH(1,1)
        model = arch_model(
            returns_pct,
            vol="GARCH",
            p=1,
            q=1,
            mean="Constant",
            rescale=True,
        )

        try:
            result = model.fit(disp="off", show_warning=False)

            # Extract parameters and anchor the unconditional variance to the
            # sample variance so downstream simulators receive stationary,
            # internally consistent GARCH coefficients.
            alpha = result.params["alpha[1]"]
            beta = result.params["beta[1]"]
            mu = result.params["mu"] / 100
            uncond_vol = np.std(returns)

            persistence = alpha + beta
            if not np.isfinite(persistence) or persistence >= 0.98:
                target_persistence = 0.98
                if persistence > 0:
                    scale_factor = target_persistence / persistence
                    alpha *= scale_factor
                    beta *= scale_factor
                else:
                    alpha = 0.05
                    beta = 0.93

            omega = max(uncond_vol**2 * (1 - alpha - beta), 1e-12)

        except Exception:
            # Fallback if optimization fails
            return self._fit_garch_moments()

        return GARCHParams(
            omega=omega,
            alpha=alpha,
            beta=beta,
            mu=mu,
            unconditional_vol=uncond_vol,
        )

    def _fit_garch_moments(self) -> GARCHParams:
        """Moment-based GARCH estimation (fallback).

        Uses autocorrelation of squared returns to estimate parameters.

        Returns
        -------
        GARCHParams
            Estimated parameters
        """
        returns = self.compute_returns()

        # Basic statistics
        mu = np.mean(returns)
        var = np.var(returns)
        vol = np.sqrt(var)

        # Squared returns for GARCH estimation
        eps2 = (returns - mu) ** 2

        # Autocorrelation of squared returns
        # rho_1 = Corr(eps2_t, eps2_{t-1}) ≈ alpha(1 + alpha*beta) / (1 - beta^2 - 2*alpha*beta)
        rho_1 = np.corrcoef(eps2[1:], eps2[:-1])[0, 1]

        # Typical crypto GARCH params (use rho to scale)
        # These are ballpark estimates when formal estimation fails
        alpha = min(0.1, max(0.02, rho_1 * 0.3))
        beta = 0.85
        omega = var * (1 - alpha - beta)

        return GARCHParams(
            omega=omega,
            alpha=alpha,
            beta=beta,
            mu=mu,
            unconditional_vol=vol,
        )

    def fit_regimes(
        self,
        n_regimes: int = 2,
        method: Literal["volatility", "hmm"] = "volatility",
    ) -> RegimeParams:
        """Detect market regimes from data.

        Parameters
        ----------
        n_regimes : int, default=2
            Number of regimes (typically 2: normal/stressed)
        method : str, default="volatility"
            Detection method:
            - "volatility": Simple threshold on rolling volatility
            - "hmm": Hidden Markov Model (requires hmmlearn)

        Returns
        -------
        RegimeParams
            Regime parameters including transition probabilities
        """
        df = self.load_data()
        returns = self.compute_returns()

        # Rolling volatility (24-hour window for hourly data)
        window = 24
        rolling_vol = np.array(
            [np.std(returns[max(0, i - window) : i + 1]) for i in range(len(returns))]
        )

        if method == "volatility":
            # Simple threshold-based regime detection
            vol_threshold = np.percentile(rolling_vol, 75)
            regime = (rolling_vol > vol_threshold).astype(int)

            # Estimate transition probabilities
            transitions = np.zeros((n_regimes, n_regimes))
            for t in range(1, len(regime)):
                transitions[regime[t - 1], regime[t]] += 1

            # Normalize rows
            row_sums = transitions.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            transition_matrix = transitions / row_sums

            # Regime-specific statistics
            regime_means = {
                "volatility": np.array(
                    [
                        np.mean(rolling_vol[regime == r]) if np.sum(regime == r) > 0 else 0
                        for r in range(n_regimes)
                    ]
                ),
                "returns": np.array(
                    [
                        np.mean(returns[regime == r]) if np.sum(regime == r) > 0 else 0
                        for r in range(n_regimes)
                    ]
                ),
            }
            regime_stds = {
                "volatility": np.array(
                    [
                        np.std(rolling_vol[regime == r]) if np.sum(regime == r) > 0 else 0
                        for r in range(n_regimes)
                    ]
                ),
                "returns": np.array(
                    [
                        np.std(returns[regime == r]) if np.sum(regime == r) > 0 else 0
                        for r in range(n_regimes)
                    ]
                ),
            }

        else:
            # HMM-based (requires hmmlearn)
            try:
                from hmmlearn.hmm import GaussianHMM

                X = rolling_vol.reshape(-1, 1)
                model = GaussianHMM(n_components=n_regimes, n_iter=100)
                model.fit(X)
                regime = model.predict(X)
                transition_matrix = model.transmat_

                regime_means = {
                    "volatility": model.means_.flatten(),
                    "returns": np.array([np.mean(returns[regime == r]) for r in range(n_regimes)]),
                }
                regime_stds = {
                    "volatility": np.sqrt(model.covars_.flatten()),
                    "returns": np.array([np.std(returns[regime == r]) for r in range(n_regimes)]),
                }
            except ImportError:
                # Fall back to volatility method
                return self.fit_regimes(n_regimes, method="volatility")

        return RegimeParams(
            n_regimes=n_regimes,
            transition_matrix=transition_matrix,
            regime_means=regime_means,
            regime_stds=regime_stds,
        )

    def estimate_spread_depth(self) -> tuple[float, float, float, float]:
        """Estimate spread and depth parameters.

        For crypto, we estimate from high-low range as proxy for spread,
        and volume as proxy for depth.

        Returns
        -------
        tuple
            (spread_normal, spread_stressed, depth_normal, depth_stressed)
        """
        df = self.load_data()
        returns = self.compute_returns()

        # Use high-low range as spread proxy (skip first to align with returns)
        # Spread ≈ (High - Low) / Close
        spread_proxy = ((df["high"] - df["low"]) / df["close"]).to_numpy()[1:]

        # Volume as depth proxy (align with returns - skip first observation)
        volume = df["volume"].to_numpy()[1:]

        # Regime split (simple: low vol = normal, high vol = stressed)
        window = 24
        rolling_vol = np.array(
            [np.std(returns[max(0, i - window) : i + 1]) for i in range(len(returns))]
        )
        vol_threshold = np.percentile(rolling_vol, 75)
        is_stressed = rolling_vol > vol_threshold

        # All arrays now have same length (len(returns))
        # Clip outliers instead of removing (preserves alignment)
        spread_cap = np.percentile(spread_proxy, 99)
        spread_proxy = np.clip(spread_proxy, 0, spread_cap)

        # Regime-specific statistics
        spread_normal = float(np.median(spread_proxy[~is_stressed]))
        spread_stressed = float(np.median(spread_proxy[is_stressed]))

        # Depth (normalized by typical volume)
        depth_normal = float(np.median(volume[~is_stressed]))
        depth_stressed = float(np.median(volume[is_stressed]))

        return spread_normal, spread_stressed, depth_normal, depth_stressed

    def estimate_impact(self) -> tuple[float, float]:
        """Estimate market impact parameters.

        Uses Amihud illiquidity ratio as a proxy for impact.

        Returns
        -------
        tuple
            (permanent_impact, temporary_impact)
        """
        df = self.load_data()
        returns = self.compute_returns()
        volume = df["volume"].to_numpy()[1:]  # Align with returns

        # Amihud illiquidity: |return| / dollar_volume
        close = df["close"].to_numpy()[1:]
        dollar_volume = volume * close

        # Avoid division by zero
        dollar_volume = np.maximum(dollar_volume, 1)

        illiq = np.abs(returns) / dollar_volume
        median_illiq = np.median(illiq)

        # Scale to realistic impact coefficients
        # These relate move to trade size
        # Permanent impact: fraction of price move per unit traded
        # Temporary impact: additional cost for immediacy
        permanent_impact = float(median_illiq * 1e6)  # Scale for reasonable units
        temporary_impact = float(permanent_impact * 0.1)  # Temporary < permanent

        # Cap at reasonable values
        permanent_impact = min(0.5, max(0.01, permanent_impact))
        temporary_impact = min(0.1, max(0.001, temporary_impact))

        return permanent_impact, temporary_impact

    def get_execution_env_params(self) -> ExecutionEnvParams:
        """Get all calibrated parameters for ExecutionEnv.

        This is the main entry point. Call this to get parameters
        calibrated from real market data.

        Returns
        -------
        ExecutionEnvParams
            Complete set of calibrated parameters

        Example
        -------
        >>> calibrator = CryptoMarketCalibrator("BTCUSDT")
        >>> params = calibrator.get_execution_env_params()
        >>> env = ExecutionEnv(
        ...     permanent_impact=params.permanent_impact,
        ...     temporary_impact=params.temporary_impact,
        ... )
        """
        garch = self.fit_garch()
        regimes = self.fit_regimes()
        spread_n, spread_s, depth_n, depth_s = self.estimate_spread_depth()
        perm_impact, temp_impact = self.estimate_impact()

        return ExecutionEnvParams(
            garch=garch,
            regimes=regimes,
            spread_normal=spread_n,
            spread_stressed=spread_s,
            depth_normal=depth_n,
            depth_stressed=depth_s,
            permanent_impact=perm_impact,
            temporary_impact=temp_impact,
        )

    def summary(self) -> str:
        """Print calibration summary."""
        params = self.get_execution_env_params()

        lines = [
            f"=== Calibration Summary: {self.symbol} ===",
            "",
            "GARCH(1,1) Parameters:",
            f"  omega:              {params.garch.omega:.6f}",
            f"  alpha:              {params.garch.alpha:.4f}",
            f"  beta:               {params.garch.beta:.4f}",
            f"  persistence:        {params.garch.persistence:.4f}",
            f"  unconditional vol:  {params.garch.unconditional_vol:.4f} ({params.garch.unconditional_vol * np.sqrt(8760) * 100:.1f}% annual)",
            "",
            "Regime Parameters:",
            f"  P(stay in normal):  {params.regimes.transition_matrix[0, 0]:.3f}",
            f"  P(stay in stressed):{params.regimes.transition_matrix[1, 1]:.3f}",
            f"  vol (normal):       {params.regimes.regime_means['volatility'][0]:.4f}",
            f"  vol (stressed):     {params.regimes.regime_means['volatility'][1]:.4f}",
            "",
            "Spread/Depth:",
            f"  spread (normal):    {params.spread_normal:.4f} ({params.spread_normal * 100:.2f}%)",
            f"  spread (stressed):  {params.spread_stressed:.4f} ({params.spread_stressed * 100:.2f}%)",
            f"  depth (normal):     {params.depth_normal:,.0f}",
            f"  depth (stressed):   {params.depth_stressed:,.0f}",
            "",
            "Impact:",
            f"  permanent:          {params.permanent_impact:.4f}",
            f"  temporary:          {params.temporary_impact:.4f}",
        ]

        return "\n".join(lines)


# Convenience function for quick calibration
def get_calibrated_params(symbol: str = "BTCUSDT") -> ExecutionEnvParams:
    """Quick function to get calibrated parameters.

    Parameters
    ----------
    symbol : str
        Crypto symbol to calibrate from

    Returns
    -------
    ExecutionEnvParams
        Calibrated parameters for RL environments
    """
    calibrator = CryptoMarketCalibrator(symbol)
    return calibrator.get_execution_env_params()


if __name__ == "__main__":
    # Demo calibration
    print("Calibrating from BTCUSDT...")
    calibrator = CryptoMarketCalibrator("BTCUSDT")
    print(calibrator.summary())
