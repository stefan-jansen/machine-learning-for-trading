"""Tests for the stage-01 helpers in ``case_studies/utils/feasibility.py``.

``panel_acf`` is what every stage-01 notebook draws F4 from, so a defect in it is
a defect in nine figures at once. Three are pinned here: the lag a gap is reported
at, what one constant entity does to the pooled curve, and the reference the band
gives an intraday panel.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from case_studies.utils.feasibility import panel_acf

PHI = 0.6
N_ENTITIES = 40
N_PERIODS = 400
MAX_LAGS = 5


def _ar1(rng: np.random.Generator, n: int, phi: float = PHI) -> np.ndarray:
    values = np.zeros(n)
    for i in range(1, n):
        values[i] = phi * values[i - 1] + rng.normal()
    return values


def _panel(rng: np.random.Generator) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": np.repeat([f"S{i}" for i in range(N_ENTITIES)], N_PERIODS),
            "period": np.tile(np.arange(N_PERIODS), N_ENTITIES),
            "value": np.concatenate([_ar1(rng, N_PERIODS) for _ in range(N_ENTITIES)]),
        }
    )


class TestGapsAreReportedAtTheRightLag:
    """Dropping nulls before lagging closes the gap instead of preserving it."""

    def test_a_gapped_panel_recovers_the_ungapped_autocorrelation(self):
        rng = np.random.default_rng(0)
        panel = _panel(rng)

        dense = panel_acf(
            panel,
            entity_col="symbol",
            value_col="value",
            max_lags=MAX_LAGS,
            period_col="period",
        )
        # Punch out a fifth of the periods at random, independently per entity.
        gapped = panel.filter(pl.Series(rng.random(panel.height) > 0.2))

        exact = panel_acf(
            gapped,
            entity_col="symbol",
            value_col="value",
            max_lags=MAX_LAGS,
            period_col="period",
        )
        compacted = panel_acf(gapped, entity_col="symbol", value_col="value", max_lags=MAX_LAGS)

        truth = dense["acf"].to_numpy()
        # Pairing by period recovers the ungapped curve; pairing by row does not,
        # because a pair straddling a gap is counted one lag too early.
        assert np.abs(exact["acf"].to_numpy()[1] - truth[1]) < 0.03
        assert compacted["acf"].to_numpy()[1] < truth[1] - 0.03

    def test_a_single_gap_is_charged_to_the_lag_it_spans(self):
        """One entity, one missing period, at a lag where the answer is arithmetic."""
        values = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=float)
        periods = np.arange(len(values))
        keep = periods != 3  # the alternation now looks like ... 1, -1, 1, 1, -1 ...

        frame = pl.DataFrame(
            {
                "symbol": ["A"] * int(keep.sum()),
                "period": periods[keep],
                "value": values[keep],
            }
        )

        exact = panel_acf(
            frame,
            entity_col="symbol",
            value_col="value",
            max_lags=2,
            min_obs=5,
            period_col="period",
        )
        compacted = panel_acf(frame, entity_col="symbol", value_col="value", max_lags=2, min_obs=5)

        # Every surviving pair one period apart still alternates, so the pairwise
        # correlation at lag 1 is -1, tapered by the (T - lag) / T that the biased
        # ACF estimator applies over the eight-period span: -7/8.
        assert exact["acf"][1] == pytest.approx(-7 / 8)
        # Compacted, the pair that straddled the gap reads as a lag-1 pair of equal
        # signs, which pulls the coefficient a long way off: -0.607.
        assert compacted["acf"][1] > exact["acf"][1] + 0.25

    def test_period_col_does_not_change_a_panel_with_no_gaps(self):
        rng = np.random.default_rng(1)
        panel = _panel(rng)

        with_periods = panel_acf(
            panel,
            entity_col="symbol",
            value_col="value",
            max_lags=MAX_LAGS,
            period_col="period",
        )
        without = panel_acf(panel, entity_col="symbol", value_col="value", max_lags=MAX_LAGS)

        np.testing.assert_allclose(
            with_periods["acf"].to_numpy(), without["acf"].to_numpy(), rtol=1e-12
        )


class TestEstimatorMatchesTheLibrary:
    """The per-entity estimator is the library's, computed here only to see gaps."""

    def test_agrees_with_compute_acf_on_a_series_with_no_gaps(self):
        from ml4t.diagnostic.evaluation.autocorrelation import compute_acf

        rng = np.random.default_rng(2)
        series = _ar1(rng, 500)
        frame = pl.DataFrame({"symbol": ["A"] * 500, "period": np.arange(500), "value": series})

        ours = panel_acf(
            frame, entity_col="symbol", value_col="value", max_lags=10, period_col="period"
        )["acf"].to_numpy()
        theirs = np.asarray(compute_acf(series, nlags=10).values)[:11]

        np.testing.assert_allclose(ours, theirs, rtol=1e-10, atol=1e-12)


class TestOneConstantEntity:
    """A zero-variance entity used to turn every pooled lag into NaN."""

    def test_a_constant_entity_is_dropped_and_counted(self):
        rng = np.random.default_rng(3)
        panel = _panel(rng)
        constant = pl.DataFrame(
            {
                "symbol": ["FLAT"] * N_PERIODS,
                "period": np.arange(N_PERIODS),
                "value": np.zeros(N_PERIODS),
            }
        )

        result = panel_acf(
            pl.concat([panel, constant]),
            entity_col="symbol",
            value_col="value",
            max_lags=MAX_LAGS,
            period_col="period",
        )

        assert np.all(np.isfinite(result["acf"].to_numpy())), "one flat entity nulled the curve"
        assert result["n_dropped"][0] == 1
        assert result["n_entities"][0] == N_ENTITIES

    def test_every_entity_constant_is_an_error_not_a_blank_figure(self):
        frame = pl.DataFrame(
            {
                "symbol": ["A"] * 100 + ["B"] * 100,
                "period": np.tile(np.arange(100), 2),
                "value": np.zeros(200),
            }
        )

        with pytest.raises(ValueError, match="observations"):
            panel_acf(
                frame,
                entity_col="symbol",
                value_col="value",
                max_lags=MAX_LAGS,
                period_col="period",
            )


class TestTheBandAnIntradayPanelCanUse:
    """`band` is the reference for one entity's curve; `pooled_se` for the mean."""

    def test_pooled_se_is_the_error_of_the_plotted_quantity(self):
        rng = np.random.default_rng(4)
        panel = _panel(rng)

        result = panel_acf(
            panel,
            entity_col="symbol",
            value_col="value",
            max_lags=MAX_LAGS,
            period_col="period",
        )

        # The band is 1.96/sqrt(T) for one entity; the standard error of a mean over
        # 40 entities is far smaller, which is the whole point on short entities.
        assert result["band"][0] == pytest.approx(1.96 / np.sqrt(N_PERIODS))
        assert result["obs_per_entity"][0] == pytest.approx(N_PERIODS)
        assert float(result["pooled_se"][1]) < float(result["band"][1]) / 5

    def test_short_entities_make_the_band_useless_but_not_the_pooled_se(self):
        """The intraday bind: 25 observations per entity, band 0.39, estimate inside it."""
        rng = np.random.default_rng(5)
        per_entity, n_entities = 25, 300
        frame = pl.DataFrame(
            {
                "symbol": np.repeat([f"B{i}" for i in range(n_entities)], per_entity),
                "period": np.tile(np.arange(per_entity), n_entities),
                "value": np.concatenate([_ar1(rng, per_entity) for _ in range(n_entities)]),
            }
        )

        result = panel_acf(
            frame,
            entity_col="symbol",
            value_col="value",
            max_lags=4,
            min_obs=20,
            period_col="period",
        )

        band = float(result["band"][1])
        pooled = float(result["pooled_se"][1])
        assert band > 0.3, "the per-entity band should be wide here, that is the problem"
        assert pooled < band / 5, "the pooled standard error has to stay usable"
        # And it is usable: the AR(1) signal clears it many times over.
        assert float(result["acf"][1]) > 4 * pooled
