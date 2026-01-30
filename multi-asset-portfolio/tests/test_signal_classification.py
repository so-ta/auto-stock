"""Tests for signal classification - task_041_5a."""

import tempfile
from datetime import datetime

import polars as pl
import pytest

from src.backtest.signal_precompute import (
    INDEPENDENT_SIGNALS,
    RELATIVE_SIGNALS,
    SignalPrecomputer,
)


class TestSignalClassification:
    """Tests for classify_signal() method."""

    @pytest.fixture
    def precomputer(self):
        """Create SignalPrecomputer instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield SignalPrecomputer(cache_dir=tmpdir)

    def test_momentum_is_independent(self, precomputer):
        """Test: momentum_* signals are independent."""
        assert precomputer.classify_signal("momentum_20") == "independent"
        assert precomputer.classify_signal("momentum_60") == "independent"
        assert precomputer.classify_signal("momentum_252") == "independent"
        print("✅ PASS: momentum_* → independent")

    def test_rsi_is_independent(self, precomputer):
        """Test: rsi_* signals are independent."""
        assert precomputer.classify_signal("rsi_14") == "independent"
        assert precomputer.classify_signal("rsi_7") == "independent"
        print("✅ PASS: rsi_* → independent")

    def test_volatility_is_independent(self, precomputer):
        """Test: volatility_* signals are independent."""
        assert precomputer.classify_signal("volatility_20") == "independent"
        assert precomputer.classify_signal("volatility_60") == "independent"
        print("✅ PASS: volatility_* → independent")

    def test_zscore_is_independent(self, precomputer):
        """Test: zscore_* signals are independent."""
        assert precomputer.classify_signal("zscore_20") == "independent"
        assert precomputer.classify_signal("zscore_60") == "independent"
        print("✅ PASS: zscore_* → independent")

    def test_sharpe_is_independent(self, precomputer):
        """Test: sharpe_* signals are independent."""
        assert precomputer.classify_signal("sharpe_60") == "independent"
        assert precomputer.classify_signal("sharpe_252") == "independent"
        print("✅ PASS: sharpe_* → independent")

    def test_atr_is_independent(self, precomputer):
        """Test: atr_* signals are independent."""
        assert precomputer.classify_signal("atr_14") == "independent"
        assert precomputer.classify_signal("atr_20") == "independent"
        print("✅ PASS: atr_* → independent")

    def test_bollinger_is_independent(self, precomputer):
        """Test: bollinger_* signals are independent."""
        assert precomputer.classify_signal("bollinger_20") == "independent"
        assert precomputer.classify_signal("bollinger_upper") == "independent"
        print("✅ PASS: bollinger_* → independent")

    def test_sector_relative_is_relative(self, precomputer):
        """Test: sector_relative_* signals are relative."""
        assert precomputer.classify_signal("sector_relative_strength") == "relative"
        assert precomputer.classify_signal("sector_relative_momentum") == "relative"
        print("✅ PASS: sector_relative_* → relative")

    def test_cross_asset_is_relative(self, precomputer):
        """Test: cross_asset_* signals are relative."""
        assert precomputer.classify_signal("cross_asset_correlation") == "relative"
        assert precomputer.classify_signal("cross_asset_momentum") == "relative"
        print("✅ PASS: cross_asset_* → relative")

    def test_ranking_is_relative(self, precomputer):
        """Test: ranking_* signals are relative."""
        assert precomputer.classify_signal("ranking_momentum") == "relative"
        assert precomputer.classify_signal("ranking_value") == "relative"
        print("✅ PASS: ranking_* → relative")

    def test_exact_match_relative(self, precomputer):
        """Test: Exact match relative signals."""
        assert precomputer.classify_signal("momentum_factor") == "relative"
        assert precomputer.classify_signal("sector_momentum") == "relative"
        assert precomputer.classify_signal("sector_breadth") == "relative"
        assert precomputer.classify_signal("market_breadth") == "relative"
        print("✅ PASS: Exact match relative signals")

    def test_unknown_signal_is_relative(self, precomputer, caplog):
        """Test: Unknown signals default to relative with warning."""
        import logging
        caplog.set_level(logging.WARNING)

        result = precomputer.classify_signal("unknown_signal_xyz")

        assert result == "relative"
        assert "Unknown signal type" in caplog.text
        print("✅ PASS: unknown_signal → relative (with warning)")

    def test_classify_signals_batch(self, precomputer):
        """Test: classify_signals() batch classification."""
        signals = [
            "momentum_20",
            "momentum_60",
            "rsi_14",
            "sector_relative_strength",
            "ranking_momentum",
        ]

        result = precomputer.classify_signals(signals)

        assert "momentum_20" in result["independent"]
        assert "momentum_60" in result["independent"]
        assert "rsi_14" in result["independent"]
        assert "sector_relative_strength" in result["relative"]
        assert "ranking_momentum" in result["relative"]

        assert len(result["independent"]) == 3
        assert len(result["relative"]) == 2
        print("✅ PASS: classify_signals() batch classification")


def test_all_scenarios():
    """Run all test scenarios and summarize results."""
    print("\n" + "=" * 60)
    print("Signal Classification Test Suite - task_041_5a")
    print("=" * 60 + "\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        precomputer = SignalPrecomputer(cache_dir=tmpdir)

        results = []

        # Test independent signals
        tests_independent = [
            ("momentum_20", "independent"),
            ("momentum_252", "independent"),
            ("rsi_14", "independent"),
            ("volatility_20", "independent"),
            ("zscore_60", "independent"),
            ("sharpe_252", "independent"),
            ("atr_14", "independent"),
            ("bollinger_20", "independent"),
            ("stochastic_14", "independent"),
            ("breakout_20", "independent"),
            ("donchian_20", "independent"),
        ]

        for signal, expected in tests_independent:
            actual = precomputer.classify_signal(signal)
            passed = actual == expected
            results.append((f"{signal} → {expected}", passed))
            print(f"[I] {signal}: {actual} {'✅' if passed else '❌'}")

        # Test relative signals
        tests_relative = [
            ("sector_relative_strength", "relative"),
            ("cross_asset_momentum", "relative"),
            ("momentum_factor", "relative"),
            ("sector_momentum", "relative"),
            ("sector_breadth", "relative"),
            ("market_breadth", "relative"),
            ("ranking_value", "relative"),
        ]

        for signal, expected in tests_relative:
            actual = precomputer.classify_signal(signal)
            passed = actual == expected
            results.append((f"{signal} → {expected}", passed))
            print(f"[R] {signal}: {actual} {'✅' if passed else '❌'}")

        # Test unknown signal (should default to relative)
        unknown = "totally_unknown_signal"
        actual = precomputer.classify_signal(unknown)
        passed = actual == "relative"
        results.append((f"{unknown} → relative (default)", passed))
        print(f"[?] {unknown}: {actual} {'✅' if passed else '❌'}")

        # Test batch classification
        batch = ["momentum_20", "rsi_14", "sector_relative_strength"]
        batch_result = precomputer.classify_signals(batch)
        batch_passed = (
            len(batch_result["independent"]) == 2 and
            len(batch_result["relative"]) == 1
        )
        results.append(("classify_signals() batch", batch_passed))
        print(f"[B] Batch classification: {'✅' if batch_passed else '❌'}")

    print("\n" + "-" * 60)
    print("Summary:")
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            print(f"  {status}: {name}")
            all_passed = False

    if all_passed:
        print("  All tests PASSED! ✅")
    else:
        print("  Some tests FAILED! ❌")

    print("-" * 60)
    print(f"Total: {sum(1 for _, p in results if p)}/{len(results)} passed")
    print("=" * 60 + "\n")

    return all_passed


if __name__ == "__main__":
    test_all_scenarios()
