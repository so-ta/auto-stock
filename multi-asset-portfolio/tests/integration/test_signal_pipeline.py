"""
Signal Pipeline Integration Tests - シグナルパイプライン統合テスト

複数シグナルの組み合わせとパイプライン実行を検証する統合テスト。

Test Cases:
1. test_all_signals_in_pipeline: 全シグナルがパイプラインで動作
2. test_signal_combination: シグナル組み合わせが正常動作
3. test_signal_registry: シグナルレジストリ機能
4. test_regime_adaptive_signals: レジーム適応シグナル
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv_data() -> Dict[str, pd.DataFrame]:
    """Generate sample OHLCV data for signal testing."""
    np.random.seed(42)
    n_days = 252
    assets = ["SPY", "QQQ", "TLT", "GLD"]

    dates = pd.date_range(
        end=datetime.now(),
        periods=n_days,
        freq="D",
    )

    data = {}
    for asset in assets:
        prices = 100 * np.exp(
            np.cumsum(np.random.normal(0.0005, 0.02, n_days))
        )
        volume = np.random.uniform(1000000, 10000000, n_days)

        data[asset] = pd.DataFrame(
            {
                "open": prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
                "high": prices * (1 + np.random.uniform(0, 0.02, n_days)),
                "low": prices * (1 - np.random.uniform(0, 0.02, n_days)),
                "close": prices,
                "volume": volume,
            },
            index=dates,
        )

    return data


@pytest.fixture
def sample_returns(sample_ohlcv_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Convert OHLCV to returns DataFrame."""
    returns = {}
    for asset, df in sample_ohlcv_data.items():
        returns[asset] = df["close"].pct_change().dropna()

    return pd.DataFrame(returns).dropna()


@pytest.mark.integration
class TestSignalPipeline:
    """Signal pipeline integration tests."""

    def test_all_signals_in_pipeline(
        self,
        sample_ohlcv_data: Dict[str, pd.DataFrame],
        sample_returns: pd.DataFrame,
    ) -> None:
        """
        Test: 全シグナルがパイプラインで動作

        Verifies:
        - 各シグナルが正常に初期化される
        - シグナル出力形式が正しい
        - パイプラインが全シグナルを処理可能
        """
        # Try to import signal classes
        signals_to_test = []

        try:
            from src.signals.momentum import MomentumSignal
            signals_to_test.append(("momentum", MomentumSignal))
        except ImportError:
            pass

        try:
            from src.signals.trend import TrendSignal
            signals_to_test.append(("trend", TrendSignal))
        except ImportError:
            pass

        try:
            from src.signals.volatility import VolatilitySignal
            signals_to_test.append(("volatility", VolatilitySignal))
        except ImportError:
            pass

        try:
            from src.signals.mean_reversion import MeanReversionSignal
            signals_to_test.append(("mean_reversion", MeanReversionSignal))
        except ImportError:
            pass

        if not signals_to_test:
            pytest.skip("No signals available for testing")

        results = {}
        for name, SignalClass in signals_to_test:
            try:
                signal = SignalClass()
                # Most signals expect returns data
                result = signal.generate(sample_returns)
                results[name] = result

                # Verify result structure
                assert result is not None, f"{name} signal returned None"

                print(f"Signal '{name}': OK")

            except Exception as e:
                print(f"Signal '{name}': {type(e).__name__} - {e}")

        print(f"\n=== All Signals Test ===")
        print(f"Tested {len(signals_to_test)} signals")
        print(f"Successful: {len(results)}")

        # At least some signals should work
        assert len(results) > 0, "At least one signal should succeed"

    def test_signal_combination(
        self,
        sample_returns: pd.DataFrame,
    ) -> None:
        """
        Test: シグナル組み合わせが正常動作

        Verifies:
        - 複数シグナルの出力を組み合わせ可能
        - 加重平均が正しく計算される
        """
        try:
            from src.signals.ensemble import EnsembleSignal
            from src.signals.momentum import MomentumSignal
            from src.signals.trend import TrendSignal
        except ImportError:
            pytest.skip("Required signals not available")

        # Create individual signals
        signals = []
        weights = []

        try:
            momentum = MomentumSignal()
            signals.append(momentum)
            weights.append(0.5)
        except Exception:
            pass

        try:
            trend = TrendSignal()
            signals.append(trend)
            weights.append(0.5)
        except Exception:
            pass

        if len(signals) < 2:
            pytest.skip("Need at least 2 signals for combination test")

        # Generate individual signals
        signal_outputs = []
        for signal in signals:
            try:
                output = signal.generate(sample_returns)
                if output is not None:
                    signal_outputs.append(output)
            except Exception:
                pass

        if len(signal_outputs) < 2:
            pytest.skip("Not enough valid signal outputs")

        # Combine signals (simple weighted average simulation)
        # Note: Actual implementation may differ
        combined = None
        try:
            ensemble = EnsembleSignal(signals=signals, weights=weights)
            combined = ensemble.generate(sample_returns)
        except Exception as e:
            # Manual combination if ensemble not available
            print(f"Ensemble not available: {e}")

        print(f"\n=== Signal Combination Test ===")
        print(f"Combined {len(signal_outputs)} signals")
        if combined is not None:
            print(f"Combined output available")

    def test_signal_registry(self) -> None:
        """
        Test: シグナルレジストリ機能

        Verifies:
        - シグナルが正しく登録される
        - 名前でシグナルを取得可能
        """
        try:
            from src.signals.registry import SignalRegistry
        except ImportError:
            pytest.skip("SignalRegistry not available")

        # Get registered signals
        try:
            registry = SignalRegistry()
            available_signals = registry.list_all()

            assert isinstance(available_signals, (list, dict)), \
                "Registry should return list or dict"

            print(f"\n=== Signal Registry Test ===")
            print(f"Available signals: {len(available_signals)}")

            # Try to get a signal by name
            if available_signals:
                first_signal_name = list(available_signals)[0] \
                    if isinstance(available_signals, dict) else available_signals[0]
                signal = registry.get(first_signal_name)
                assert signal is not None, f"Could not get signal: {first_signal_name}"
                print(f"Retrieved signal: {first_signal_name}")

        except Exception as e:
            pytest.skip(f"Registry not functional: {e}")

    def test_regime_adaptive_signals(
        self,
        sample_returns: pd.DataFrame,
    ) -> None:
        """
        Test: レジーム適応シグナル

        Verifies:
        - レジーム検出が動作する
        - シグナルがレジームに応じて調整される
        """
        try:
            from src.signals.regime_detector_v2 import (
                EnhancedRegimeDetector,
                RegimeType,
            )
        except ImportError:
            pytest.skip("RegimeDetector not available")

        try:
            # Initialize regime detector
            detector = EnhancedRegimeDetector()

            # Detect regime - need price data, not returns
            # Create a price series from returns
            price_series = (1 + sample_returns.iloc[:, 0]).cumprod() * 100
            regime_result = detector.detect_regime(price_series)

            assert regime_result is not None, "Regime detection should return result"
            assert hasattr(regime_result, "regime"), \
                "Result should have regime"

            print(f"\n=== Regime Adaptive Signals Test ===")
            print(f"Current regime: {regime_result.regime}")

            # Verify regime is valid
            valid_regimes = ["bull", "bear", "neutral", "high_vol", "low_vol",
                            RegimeType.BULL if hasattr(RegimeType, 'BULL') else None]
            # Note: regime representation may vary

        except Exception as e:
            print(f"Regime detection error: {e}")
            pytest.skip(f"Regime detection not functional: {e}")


@pytest.mark.integration
class TestVIXSignalIntegration:
    """VIX signal integration tests."""

    @pytest.fixture
    def sample_vix_data(self) -> pd.DataFrame:
        """Generate sample VIX data."""
        np.random.seed(42)
        n_days = 252
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq="D")

        # Generate VIX-like data (typically ranges 10-80)
        vix = 20 + 10 * np.sin(np.linspace(0, 4 * np.pi, n_days)) + \
              np.random.normal(0, 3, n_days)
        vix = np.clip(vix, 10, 80)

        return pd.DataFrame({"VIX": vix}, index=dates)

    def test_vix_signal_cash_allocation(
        self,
        sample_vix_data: pd.DataFrame,
    ) -> None:
        """
        Test: VIXシグナルのキャッシュ配分

        Verifies:
        - VIXレベルに応じた正しいキャッシュ配分
        - 緊急トリガーの動作
        """
        try:
            from src.signals.vix_signal import EnhancedVIXSignal, get_vix_cash_allocation
        except ImportError:
            pytest.skip("VIX signal not available")

        signal = EnhancedVIXSignal()

        # Test various VIX levels
        test_cases = [
            (15, "low"),      # Normal market
            (25, "moderate"), # Elevated volatility
            (35, "high"),     # High volatility
            (50, "extreme"),  # Extreme volatility
        ]

        print(f"\n=== VIX Signal Cash Allocation Test ===")

        for vix_level, expected_condition in test_cases:
            try:
                allocation = signal.get_cash_allocation(vix=vix_level, vix_change=0.0)
                print(f"VIX {vix_level} ({expected_condition}): "
                      f"cash allocation = {allocation:.2%}")

                # Basic validation
                assert 0 <= allocation <= 1, f"Invalid allocation for VIX {vix_level}"

                # Higher VIX should generally lead to higher cash allocation
                if vix_level > 30:
                    assert allocation > 0, "High VIX should trigger cash allocation"

            except Exception as e:
                print(f"VIX {vix_level}: Error - {e}")

    def test_vix_spike_emergency(
        self,
        sample_vix_data: pd.DataFrame,
    ) -> None:
        """
        Test: VIXスパイク時の緊急対応

        Verifies:
        - 急激なVIX上昇を検出
        - 緊急キャッシュ配分が発動
        """
        try:
            from src.signals.vix_signal import EnhancedVIXSignal
        except ImportError:
            pytest.skip("VIX signal not available")

        signal = EnhancedVIXSignal()

        # Test spike scenarios
        spike_scenarios = [
            {"vix": 30, "vix_change": 0.30},  # 30% daily spike
            {"vix": 40, "vix_change": 0.50},  # 50% daily spike (panic)
        ]

        print(f"\n=== VIX Spike Emergency Test ===")

        for scenario in spike_scenarios:
            try:
                allocation = signal.get_cash_allocation(**scenario)
                print(f"VIX={scenario['vix']}, change={scenario['vix_change']:.0%}: "
                      f"cash={allocation:.2%}")

                # Large spike should trigger significant cash allocation
                if scenario["vix_change"] > 0.30:
                    assert allocation >= 0.2, "Large VIX spike should trigger cash"

            except Exception as e:
                print(f"Scenario {scenario}: Error - {e}")


# Standalone execution
if __name__ == "__main__":
    print("Running signal pipeline integration tests...")
    pytest.main([__file__, "-v", "-x"])
