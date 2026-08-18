"""A sub-daily label's overlap comes from what the case study declares, not from its name.

The HAC bandwidth on an IC series is set by how many decision bars the label's holding
period covers. Reading that off the label name works for a daily case study and fails for
an intraday one, because ``m`` is the unit letter for both minutes and months:
``fwd_ret_15m`` is fifteen minutes on a 15-minute grid and ``fwd_ret_1m`` is one month on
a monthly one. Resolving the first as months returned 315, so every nasdaq100 IC would
have been published with a 314-lag HAC standard error and a 315-bar bootstrap block.

These tests pin the resolution to ``setup.yaml`` - the buffer and the bar frequency - and
pin the fallback, so the eight daily, weekly and monthly case studies keep the horizon
they already had.
"""

from __future__ import annotations

import pytest

from case_studies.utils.registry.metrics import (
    _horizon_from_declared_buffer,
    _infer_horizon_from_label,
)
from case_studies.utils.registry.registration import (
    _declared_bar_frequency,
    _declared_label_buffer,
)


class TestTheDeclarationDecidesASubDailyHorizon:
    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("fwd_ret_5m", 1),  # shorter than the bar: still one bar of overlap
            ("fwd_ret_15m", 1),
            ("fwd_dir_15m", 1),
            ("fwd_ret_60m", 4),
        ],
    )
    def test_nasdaq_minute_labels_resolve_to_decision_bars(self, label, expected):
        resolved = _horizon_from_declared_buffer(
            _declared_label_buffer("nasdaq100_microstructure", label),
            _declared_bar_frequency("nasdaq100_microstructure"),
        )
        assert resolved == expected

    def test_the_declaration_agrees_with_the_case_study_s_own_rebalance_step(self):
        """setup.yaml states the same overlap twice; they must not disagree."""
        import yaml

        from utils.paths import get_case_study_dir

        setup = yaml.safe_load(
            (get_case_study_dir("nasdaq100_microstructure") / "config" / "setup.yaml").read_text()
        )
        steps = setup["labels"]["rebalance_step"]
        bar = _declared_bar_frequency("nasdaq100_microstructure")

        for label, step in steps.items():
            buffer = _declared_label_buffer("nasdaq100_microstructure", label)
            assert _horizon_from_declared_buffer(buffer, bar) == step, label

    def test_the_name_parse_would_have_read_minutes_as_months(self):
        """The defect this replaces, kept so the fallback is never wired back in."""
        assert _infer_horizon_from_label("fwd_ret_15m") == 315
        assert _horizon_from_declared_buffer("15min", "15_minute") == 1


class TestEveryOtherCaseStudyIsUntouched:
    @pytest.mark.parametrize(
        ("case_study", "label"),
        [
            ("etfs", "fwd_ret_21d"),
            ("etfs", "fwd_ret_5d"),
            ("cme_futures", "fwd_ret_5d"),
            ("crypto_perps_funding", "fwd_ret_8h"),
            ("crypto_perps_funding", "fwd_ret_24h"),
            ("fx_pairs", "fwd_ret_1d"),
            ("sp500_options", "ret_to_expiry"),
            ("us_equities_panel", "fwd_ret_21d"),
            ("us_firm_characteristics", "fwd_ret_1m"),
            ("us_firm_characteristics", "fwd_class_1m"),
        ],
    )
    def test_a_case_study_with_no_declared_bar_frequency_falls_back(self, case_study, label):
        """No bar frequency means no decision grid to count in, so nothing is resolved."""
        resolved = _horizon_from_declared_buffer(
            _declared_label_buffer(case_study, label),
            _declared_bar_frequency(case_study),
        )
        assert resolved is None


class TestDurationParsing:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("15min", 900.0),
            ("60_minute", 3600.0),
            ("8H", 28800.0),
            ("1 hour", 3600.0),
        ],
    )
    def test_sub_daily_durations_resolve(self, text, expected):
        from case_studies.utils.registry.metrics import _duration_seconds

        assert _duration_seconds(text) == expected

    @pytest.mark.parametrize("text", ["21D", "1M", "1W", "", None, "expiry"])
    def test_a_calendar_duration_has_no_fixed_length_and_does_not_resolve(self, text):
        from case_studies.utils.registry.metrics import _duration_seconds

        assert _duration_seconds(text) is None


def test_a_declared_horizon_overrides_the_name_in_the_metric_pass():
    """The wiring, not just the helpers: the metric pass must prefer the declaration."""
    import inspect

    from case_studies.utils.registry import metrics

    source = inspect.getsource(metrics.compute_prediction_fold_metrics)
    declared = source.index("_horizon_from_declared_buffer")
    fallback = source.index("_infer_horizon_from_label")
    assert declared < fallback, "the name parse must only run when the declaration is absent"
