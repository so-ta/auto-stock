"""
Tests for Signal Timeframe Affinity System.

This module validates that:
1. All registered signals have valid timeframe_config() implementations
2. Supported variants are within the declared min/max period bounds
3. All signals are properly classified in INDEPENDENT/RELATIVE/EXTERNAL sets
4. Signal computation produces no parameter range errors
"""

import fnmatch
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import pytest

from src.signals.base import TimeframeAffinity, TimeframeConfig


class TestSignalTimeframeConfig:
    """Validate all signals have proper timeframe_config implementations."""

    def test_all_registered_signals_have_timeframe_config(self):
        """Every registered signal must return a valid TimeframeConfig."""
        from src.signals import SignalRegistry

        errors: List[str] = []

        for signal_name in SignalRegistry.list_all():
            signal_cls = SignalRegistry.get(signal_name)

            try:
                config = signal_cls.timeframe_config()
            except Exception as e:
                errors.append(f"{signal_name}: timeframe_config() raised {e}")
                continue

            # Validate return type
            if not isinstance(config, TimeframeConfig):
                errors.append(
                    f"{signal_name}: timeframe_config() must return TimeframeConfig, "
                    f"got {type(config).__name__}"
                )
                continue

            # Validate min_period
            if config.min_period <= 0:
                errors.append(f"{signal_name}: min_period must be positive, got {config.min_period}")

            # Validate max_period
            if config.max_period < config.min_period:
                errors.append(
                    f"{signal_name}: max_period ({config.max_period}) must be >= "
                    f"min_period ({config.min_period})"
                )

            # Validate supported_variants not empty
            if not config.supported_variants:
                errors.append(f"{signal_name}: must support at least one variant")

            # Validate affinity
            if not isinstance(config.affinity, TimeframeAffinity):
                errors.append(
                    f"{signal_name}: affinity must be TimeframeAffinity, "
                    f"got {type(config.affinity).__name__}"
                )

        assert len(errors) == 0, "TimeframeConfig validation errors:\n" + "\n".join(errors)

    def test_supported_variants_within_range(self):
        """Each supported variant's period must be within min/max bounds."""
        from src.signals import SignalRegistry
        from src.signals.base import DEFAULT_VARIANT_PERIODS

        errors: List[str] = []

        for signal_name in SignalRegistry.list_all():
            signal_cls = SignalRegistry.get(signal_name)
            config = signal_cls.timeframe_config()

            for variant in config.supported_variants:
                period = DEFAULT_VARIANT_PERIODS.get(variant)
                if period is None:
                    errors.append(
                        f"{signal_name}: variant '{variant}' not in DEFAULT_VARIANT_PERIODS"
                    )
                    continue

                if not (config.min_period <= period <= config.max_period):
                    errors.append(
                        f"{signal_name}_{variant}: period {period} outside range "
                        f"[{config.min_period}, {config.max_period}]"
                    )

        assert len(errors) == 0, (
            "Variant period range errors:\n" + "\n".join(errors)
        )


# Signals that have been explicitly configured with timeframe_config
# (not using default implementation)
CONFIGURED_SIGNALS = {
    # mean_reversion.py
    "bollinger_reversion",
    "rsi",
    "zscore_reversion",
    "stochastic_reversion",
    # momentum.py
    "momentum_return",
    "roc",
    "momentum_composite",
    "momentum_acceleration",
    # volatility.py
    "atr",
    "volatility_breakout",
    "volatility_regime",
    # breakout.py
    "donchian_channel",
    "high_low_breakout",
    "range_breakout",
    # volume.py
    "obv_momentum",
    "money_flow_index",
    "vwap_deviation",
    "accumulation_distribution",
    # advanced_technical.py
    "kama",
    "keltner_channel",
    # fifty_two_week_high.py
    "fifty_two_week_high_momentum",
    # lead_lag.py
    "lead_lag",
    # low_vol_premium.py
    "low_vol_premium",
    # yield_curve_signal.py
    "enhanced_yield_curve",
}


class TestSignalClassification:
    """Validate signal classification coverage."""

    def test_configured_signals_are_classified(self):
        """Configured signal variants must match INDEPENDENT/RELATIVE/EXTERNAL patterns."""
        from src.signals import SignalRegistry
        from src.backtest.signal_precompute import (
            INDEPENDENT_SIGNALS,
            RELATIVE_SIGNALS,
            EXTERNAL_SIGNALS,
        )

        all_patterns = INDEPENDENT_SIGNALS | RELATIVE_SIGNALS | EXTERNAL_SIGNALS
        unclassified: Set[str] = set()

        for signal_name in CONFIGURED_SIGNALS:
            try:
                signal_cls = SignalRegistry.get(signal_name)
            except KeyError:
                continue

            config = signal_cls.timeframe_config()

            # Check base name
            matched = any(fnmatch.fnmatch(signal_name, p) for p in all_patterns)
            if not matched:
                unclassified.add(signal_name)

            # Check with supported variants only
            for variant in config.supported_variants:
                full_name = f"{signal_name}_{variant}"
                matched = any(fnmatch.fnmatch(full_name, p) for p in all_patterns)
                if not matched:
                    unclassified.add(full_name)

        assert len(unclassified) == 0, (
            f"Unclassified configured signals: {sorted(unclassified)}"
        )


class TestNoParameterRangeErrors:
    """Ensure no parameter range errors during signal computation."""

    @pytest.fixture
    def sample_prices(self) -> pd.DataFrame:
        """Generate 1 year of sample price data for 3 tickers."""
        np.random.seed(42)

        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        data_rows = []

        for ticker in ["AAPL", "GOOGL", "MSFT"]:
            np.random.seed(hash(ticker) % (2**31))
            prices = 100 * np.cumprod(1 + np.random.randn(len(dates)) * 0.02)

            for i, date in enumerate(dates):
                data_rows.append({
                    "timestamp": date,
                    "ticker": ticker,
                    "close": prices[i],
                    "high": prices[i] * 1.01,
                    "low": prices[i] * 0.99,
                    "volume": 1000000,
                })

        df = pd.DataFrame(data_rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def test_configured_signals_compute_without_parameter_errors(self, sample_prices, caplog):
        """Signals with explicit timeframe_config should compute without errors."""
        from src.signals import SignalRegistry
        from src.signals.base import DEFAULT_VARIANT_PERIODS

        errors: List[str] = []

        for signal_name in CONFIGURED_SIGNALS:
            try:
                signal_cls = SignalRegistry.get(signal_name)
            except KeyError:
                continue  # Signal not registered

            specs = {s.name: s for s in signal_cls.parameter_specs()}
            config = signal_cls.timeframe_config()

            # Find period parameter
            period_param_name = None
            for name in ["period", "lookback", "window", "n_periods", "k_period",
                         "efficiency_period", "ema_period"]:
                if name in specs:
                    period_param_name = name
                    break

            if period_param_name is None:
                # No period param - test with defaults
                try:
                    signal = signal_cls()
                    ticker_data = sample_prices[sample_prices["ticker"] == "AAPL"].copy()
                    ticker_data = ticker_data.set_index("timestamp")
                    signal.compute(ticker_data)
                except ValueError as e:
                    if "out of range" in str(e):
                        errors.append(f"{signal_name}: {e}")
                except Exception:
                    pass  # Other errors are OK for this test
            else:
                # Has period param - test each supported variant
                for variant in config.supported_variants:
                    period = DEFAULT_VARIANT_PERIODS.get(variant)
                    if period is None:
                        continue

                    try:
                        signal = signal_cls(**{period_param_name: period})
                        ticker_data = sample_prices[sample_prices["ticker"] == "AAPL"].copy()
                        ticker_data = ticker_data.set_index("timestamp")
                        signal.compute(ticker_data)
                    except ValueError as e:
                        if "out of range" in str(e):
                            errors.append(f"{signal_name}_{variant}: {e}")
                    except Exception:
                        pass  # Other errors are OK for this test

        assert len(errors) == 0, (
            "Parameter range errors during signal computation:\n" + "\n".join(errors)
        )


class TestEverySignalEveryVariant:
    """Comprehensive test of all signal-variant combinations."""

    def test_configured_declared_variants_are_computable(self):
        """Each configured signal's declared variants must be computable."""
        from src.signals import SignalRegistry
        from src.signals.base import DEFAULT_VARIANT_PERIODS

        errors: List[str] = []

        for signal_name in CONFIGURED_SIGNALS:
            try:
                signal_cls = SignalRegistry.get(signal_name)
            except KeyError:
                continue

            specs = {s.name: s for s in signal_cls.parameter_specs()}
            config = signal_cls.timeframe_config()

            # Find period parameter
            period_param_name = None
            for name in ["period", "lookback", "window", "n_periods", "k_period",
                         "efficiency_period", "ema_period", "atr_period", "high_period"]:
                if name in specs:
                    period_param_name = name
                    break

            if period_param_name is None:
                continue

            spec = specs[period_param_name]

            for variant in config.supported_variants:
                period = DEFAULT_VARIANT_PERIODS.get(variant)
                if period is None:
                    errors.append(
                        f"{signal_name}_{variant}: variant not in DEFAULT_VARIANT_PERIODS"
                    )
                    continue

                # Check against ParameterSpec bounds
                if spec.min_value is not None and period < spec.min_value:
                    errors.append(
                        f"{signal_name}_{variant}: period {period} < spec.min_value {spec.min_value}"
                    )
                if spec.max_value is not None and period > spec.max_value:
                    errors.append(
                        f"{signal_name}_{variant}: period {period} > spec.max_value {spec.max_value}"
                    )

        assert len(errors) == 0, (
            "Signal-variant parameter conflicts:\n" + "\n".join(errors)
        )


class TestTimeframeAffinityCategories:
    """Test that signals are correctly categorized by timeframe."""

    def test_short_term_signals_exclude_long_variants(self):
        """SHORT_TERM signals should not support half_year or yearly variants."""
        from src.signals import SignalRegistry

        violations: List[str] = []

        for signal_name in SignalRegistry.list_all():
            signal_cls = SignalRegistry.get(signal_name)
            config = signal_cls.timeframe_config()

            if config.affinity == TimeframeAffinity.SHORT_TERM:
                long_variants = {"long", "half_year", "yearly"} & set(config.supported_variants)
                if long_variants:
                    violations.append(
                        f"{signal_name}: SHORT_TERM but supports {long_variants}"
                    )

        assert len(violations) == 0, (
            "SHORT_TERM signals with long variants:\n" + "\n".join(violations)
        )

    def test_long_term_only_signals_exclude_short_variants(self):
        """LONG_TERM_ONLY signals should not support short or medium variants."""
        from src.signals import SignalRegistry

        violations: List[str] = []

        for signal_name in SignalRegistry.list_all():
            signal_cls = SignalRegistry.get(signal_name)
            config = signal_cls.timeframe_config()

            if config.affinity == TimeframeAffinity.LONG_TERM_ONLY:
                short_variants = {"short", "medium"} & set(config.supported_variants)
                if short_variants:
                    violations.append(
                        f"{signal_name}: LONG_TERM_ONLY but supports {short_variants}"
                    )

        assert len(violations) == 0, (
            "LONG_TERM_ONLY signals with short variants:\n" + "\n".join(violations)
        )


class TestSignalPrecomputeIntegration:
    """Test integration with SignalPrecomputer."""

    def test_unified_signal_names_respect_timeframe_config(self):
        """_get_unified_signal_names() should respect timeframe affinity."""
        import tempfile
        from src.backtest.signal_precompute import SignalPrecomputer
        from src.utils.storage_backend import StorageBackend, StorageConfig
        from src.signals import SignalRegistry
        from src.signals.base import DEFAULT_VARIANT_PERIODS

        with tempfile.TemporaryDirectory() as tmpdir:
            config = StorageConfig(s3_bucket="test", base_path=tmpdir)
            backend = StorageBackend(config)
            precomputer = SignalPrecomputer(storage_backend=backend)

            signal_names = precomputer._get_unified_signal_names()

            # Check that no unsupported variants are included
            for name in signal_names:
                # Parse signal name and variant
                for variant in ["yearly", "half_year", "long", "medium", "short"]:
                    if name.endswith(f"_{variant}"):
                        base_name = name[: -(len(variant) + 1)]

                        # Skip if base_name is not a registered signal
                        # (might be a signal with underscore in name)
                        try:
                            signal_cls = SignalRegistry.get(base_name)
                        except KeyError:
                            continue

                        tf_config = signal_cls.timeframe_config()

                        assert tf_config.supports_variant(variant), (
                            f"{name}: variant '{variant}' not in "
                            f"supported_variants={tf_config.supported_variants}"
                        )

                        period = DEFAULT_VARIANT_PERIODS.get(variant)
                        assert tf_config.min_period <= period <= tf_config.max_period, (
                            f"{name}: period {period} outside "
                            f"[{tf_config.min_period}, {tf_config.max_period}]"
                        )
                        break
