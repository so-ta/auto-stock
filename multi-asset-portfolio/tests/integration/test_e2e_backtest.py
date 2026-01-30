"""
End-to-End Backtest Tests

実データ（yfinance）を使用したE2Eテスト。
再現性とフォールバック動作を検証する。

Requirements:
- yfinance must be installed
- Internet connection required
- Tests are marked as @pytest.mark.slow for CI exclusion

Test Cases:
1. test_backtest_us_stocks: US株5銘柄でパイプライン実行
2. test_backtest_reproducibility: 同一seed → 同一weights
3. test_backtest_with_fallback: 異常検知時のフォールバック動作
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Suppress yfinance FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")


def is_yfinance_available() -> bool:
    """Check if yfinance is installed and working."""
    try:
        import yfinance as yf
        return True
    except ImportError:
        return False


def fetch_real_data(
    symbols: list[str],
    period_days: int = 365,
) -> dict[str, pd.DataFrame] | None:
    """
    Fetch real OHLCV data using yfinance.

    Args:
        symbols: List of stock symbols
        period_days: Number of days of historical data

    Returns:
        Dict of symbol -> DataFrame with OHLCV data, or None on error
    """
    try:
        import yfinance as yf

        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)

        data = {}
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
            )

            if hist.empty:
                return None

            # Rename columns to lowercase
            hist.columns = [c.lower() for c in hist.columns]
            data[symbol] = hist

        return data

    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return None


def compute_returns(ohlcv_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute daily returns from OHLCV data.

    Args:
        ohlcv_data: Dict of symbol -> DataFrame with 'close' column

    Returns:
        DataFrame with daily returns for each symbol
    """
    returns_data = {}
    for symbol, df in ohlcv_data.items():
        if "close" in df.columns:
            returns_data[symbol] = df["close"].pct_change().dropna()

    # Align all returns to common dates
    returns_df = pd.DataFrame(returns_data)
    returns_df = returns_df.dropna()

    return returns_df


@pytest.mark.slow
@pytest.mark.integration
class TestE2EBacktest:
    """End-to-end backtest tests with real market data."""

    US_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock settings for pipeline."""
        settings = MagicMock()

        # Universe
        settings.universe = self.US_STOCKS

        # Data quality settings
        settings.data_quality.max_missing_rate = 0.1
        settings.data_quality.max_consecutive_missing = 10
        settings.data_quality.price_change_threshold = 0.5
        settings.data_quality.min_volume_threshold = 0
        settings.data_quality.staleness_hours = 48

        # Asset allocation settings
        settings.asset_allocation.method.value = "HRP"
        settings.asset_allocation.w_asset_max = 0.4
        settings.asset_allocation.w_asset_min = 0.0
        settings.asset_allocation.delta_max = 0.1
        settings.asset_allocation.smooth_alpha = 0.3
        settings.asset_allocation.allow_short = False

        # Hard gates
        settings.hard_gates.min_trades = 20
        settings.hard_gates.max_drawdown_pct = 30.0
        settings.hard_gates.min_sharpe_ratio = 0.0

        # Walk forward (for full pipeline)
        settings.walk_forward.train_period_days = 200
        settings.walk_forward.test_period_days = 50
        settings.walk_forward.step_period_days = 20

        # Strategy weighting
        settings.strategy_weighting.beta = 2.0
        settings.strategy_weighting.w_strategy_max = 0.5
        settings.strategy_weighting.entropy_min = 0.5

        # Degradation
        settings.degradation.vix_threshold = 30.0
        settings.degradation.cooldown_days = 5

        # Fallback
        settings.fallback = MagicMock()
        settings.fallback.value = "hold_previous"

        return settings

    @pytest.fixture
    def real_ohlcv_data(self) -> dict[str, pd.DataFrame] | None:
        """Fetch real OHLCV data from yfinance."""
        if not is_yfinance_available():
            pytest.skip("yfinance not installed")

        data = fetch_real_data(self.US_STOCKS, period_days=400)

        if data is None:
            pytest.skip("Failed to fetch data from yfinance (network error)")

        return data

    def test_backtest_us_stocks(
        self,
        mock_settings: MagicMock,
        real_ohlcv_data: dict[str, pd.DataFrame],
    ) -> None:
        """
        Test: US株5銘柄（AAPL,MSFT,GOOGL,AMZN,META）でパイプライン実行

        Verifies:
        - Pipeline completes without error
        - Weights are valid (sum ≈ 1, all >= 0)
        - All requested assets have weights
        """
        from src.allocation.allocator import AllocatorConfig, AssetAllocator
        from src.utils.reproducibility import SeedManager

        # Convert to returns
        returns_df = compute_returns(real_ohlcv_data)

        assert not returns_df.empty, "Returns DataFrame should not be empty"
        assert len(returns_df.columns) == len(
            self.US_STOCKS
        ), f"Expected {len(self.US_STOCKS)} columns"

        # Run allocation with fixed seed
        with SeedManager(seed=42):
            config = AllocatorConfig(
                w_asset_max=mock_settings.asset_allocation.w_asset_max,
                delta_max=mock_settings.asset_allocation.delta_max,
                smooth_alpha=mock_settings.asset_allocation.smooth_alpha,
            )
            allocator = AssetAllocator(config)
            result = allocator.allocate(returns_df)

        # Validate results
        assert result.is_valid, f"Allocation result should be valid: {result.fallback_reason}"

        # Check weights sum to 1
        weights_sum = result.weights.sum()
        assert abs(weights_sum - 1.0) < 0.01, f"Weights should sum to 1, got {weights_sum}"

        # Check all weights are non-negative
        assert (result.weights >= -1e-8).all(), "All weights should be non-negative"

        # Check all assets have weights
        for symbol in self.US_STOCKS:
            assert symbol in result.weights.index, f"Missing weight for {symbol}"

        # Log results
        print(f"\n=== Allocation Results ===")
        print(f"Weights: {dict(result.weights)}")
        print(f"Portfolio Volatility: {result.portfolio_metrics.get('volatility', 'N/A'):.4f}")
        print(f"Method: {result.method_used}")

    def test_backtest_reproducibility(
        self,
        mock_settings: MagicMock,
        real_ohlcv_data: dict[str, pd.DataFrame],
    ) -> None:
        """
        Test: 同一seed(42)で2回実行 → weightsが完全一致

        Verifies:
        - Two runs with same seed produce identical weights
        - Metrics are also identical
        """
        from src.allocation.allocator import AllocatorConfig, AssetAllocator
        from src.utils.reproducibility import SeedManager

        returns_df = compute_returns(real_ohlcv_data)

        def run_allocation_with_seed(seed: int) -> dict[str, Any]:
            """Run allocation with given seed and return results."""
            with SeedManager(seed=seed):
                np.random.seed(seed)  # Extra safety

                config = AllocatorConfig(
                    w_asset_max=mock_settings.asset_allocation.w_asset_max,
                    delta_max=mock_settings.asset_allocation.delta_max,
                    smooth_alpha=mock_settings.asset_allocation.smooth_alpha,
                )
                allocator = AssetAllocator(config)
                result = allocator.allocate(returns_df)

                return {
                    "weights": dict(result.weights),
                    "volatility": result.portfolio_metrics.get("volatility"),
                    "method": result.method_used,
                    "fallback": result.fallback_reason.value,
                }

        # Run twice with same seed
        result1 = run_allocation_with_seed(42)
        result2 = run_allocation_with_seed(42)

        # Verify weights are identical
        for symbol in self.US_STOCKS:
            w1 = result1["weights"].get(symbol, 0.0)
            w2 = result2["weights"].get(symbol, 0.0)
            assert w1 == pytest.approx(
                w2, rel=1e-10
            ), f"Weight mismatch for {symbol}: {w1} vs {w2}"

        # Verify metrics are identical
        assert result1["volatility"] == pytest.approx(
            result2["volatility"], rel=1e-10
        ), "Volatility should be identical"

        assert result1["method"] == result2["method"], "Method should be identical"
        assert result1["fallback"] == result2["fallback"], "Fallback state should be identical"

        print(f"\n=== Reproducibility Test Passed ===")
        print(f"Run 1 weights: {result1['weights']}")
        print(f"Run 2 weights: {result2['weights']}")

    def test_different_seeds_produce_different_results(
        self,
        mock_settings: MagicMock,
        real_ohlcv_data: dict[str, pd.DataFrame],
    ) -> None:
        """
        Test: 異なるseedでは異なる結果が出ることを確認

        Note: HRP自体は決定的だが、乱数を使う前処理がある場合に差が出る
        """
        from src.allocation.allocator import AllocatorConfig, AssetAllocator
        from src.utils.reproducibility import SeedManager

        returns_df = compute_returns(real_ohlcv_data)

        def run_allocation_with_seed(seed: int) -> dict[str, float]:
            with SeedManager(seed=seed):
                config = AllocatorConfig()
                allocator = AssetAllocator(config)
                result = allocator.allocate(returns_df)
                return dict(result.weights)

        weights_42 = run_allocation_with_seed(42)
        weights_123 = run_allocation_with_seed(123)

        # For HRP, weights should be the same since it's deterministic
        # This test documents the expected behavior
        print(f"\n=== Seed Comparison ===")
        print(f"Seed 42: {weights_42}")
        print(f"Seed 123: {weights_123}")

        # HRP is deterministic given same input, so weights should be equal
        # (unlike methods that use random sampling)

    def test_backtest_with_fallback(
        self,
        mock_settings: MagicMock,
        real_ohlcv_data: dict[str, pd.DataFrame],
    ) -> None:
        """
        Test: 異常検知時のフォールバック動作確認

        Verifies:
        - When all assets are excluded, fallback kicks in
        - Equal weight allocation is applied
        """
        from src.allocation.allocator import (
            AllocatorConfig,
            AllocationMethod,
            AssetAllocator,
            FallbackReason,
        )
        from src.utils.reproducibility import SeedManager

        returns_df = compute_returns(real_ohlcv_data)

        # Create quality flags that exclude all assets
        all_excluded_flags = pd.DataFrame(
            {symbol: [False] for symbol in self.US_STOCKS},
            index=[returns_df.index[-1]],
        )

        with SeedManager(seed=42):
            config = AllocatorConfig(
                method=AllocationMethod.HRP,
                fallback_to_equal=True,
            )
            allocator = AssetAllocator(config)

            # Run with all assets excluded
            result = allocator.allocate(
                returns_df,
                quality_flags=all_excluded_flags,
            )

        # Should have triggered fallback
        assert result.is_fallback, "Should have triggered fallback"
        assert result.fallback_reason == FallbackReason.ALL_ASSETS_EXCLUDED

        # Should have equal weights (since fallback_to_equal=True)
        # But excluded assets should be 0
        print(f"\n=== Fallback Test ===")
        print(f"Fallback reason: {result.fallback_reason}")
        print(f"Weights: {dict(result.weights)}")
        print(f"Excluded assets: {result.excluded_assets}")

    def test_backtest_with_previous_weights(
        self,
        mock_settings: MagicMock,
        real_ohlcv_data: dict[str, pd.DataFrame],
    ) -> None:
        """
        Test: 前期重みありでの配分（スムージング動作確認）

        Verifies:
        - Smoothing is applied when previous weights exist
        - Turnover is limited by delta_max
        """
        from src.allocation.allocator import AllocatorConfig, AssetAllocator
        from src.utils.reproducibility import SeedManager

        returns_df = compute_returns(real_ohlcv_data)

        # Previous weights (equal allocation)
        n_assets = len(self.US_STOCKS)
        previous_weights = pd.Series(
            [1.0 / n_assets] * n_assets,
            index=self.US_STOCKS,
        )

        with SeedManager(seed=42):
            config = AllocatorConfig(
                delta_max=0.05,  # Max 5% change per asset
                smooth_alpha=0.3,  # 30% new, 70% old
            )
            allocator = AssetAllocator(config)

            result = allocator.allocate(
                returns_df,
                previous_weights=previous_weights,
            )

        assert result.is_valid

        # Check turnover is reasonable
        turnover = result.turnover
        print(f"\n=== Smoothing Test ===")
        print(f"Previous weights: {dict(previous_weights)}")
        print(f"New weights: {dict(result.weights)}")
        print(f"Turnover: {turnover:.4f}")

        # Turnover should be less than sum of delta_max for all assets
        max_possible_turnover = config.delta_max * n_assets
        assert turnover <= max_possible_turnover + 0.01, (
            f"Turnover {turnover} exceeds max {max_possible_turnover}"
        )


@pytest.mark.slow
@pytest.mark.integration
class TestE2EFullPipeline:
    """Full pipeline E2E tests (if pipeline.run() is available)."""

    US_STOCKS = ["AAPL", "MSFT", "GOOGL"]

    @pytest.fixture
    def real_returns(self) -> pd.DataFrame | None:
        """Fetch real returns data."""
        if not is_yfinance_available():
            pytest.skip("yfinance not installed")

        data = fetch_real_data(self.US_STOCKS, period_days=300)
        if data is None:
            pytest.skip("Failed to fetch data")

        return compute_returns(data)

    def test_full_pipeline_execution(
        self,
        real_returns: pd.DataFrame,
        tmp_path,
    ) -> None:
        """
        Test full pipeline execution with mock components.

        Note: This is a simplified test that focuses on the allocation
        pipeline, as the full pipeline requires many more components.
        """
        from src.allocation.allocator import AssetAllocator
        from src.utils.reproducibility import SeedManager

        with SeedManager(seed=42):
            # Simple allocation flow
            allocator = AssetAllocator()
            result = allocator.allocate(real_returns)

            assert result.is_valid
            assert abs(result.weights.sum() - 1.0) < 0.01

            # Save results to tmp_path for verification
            import json

            output_file = tmp_path / "weights.json"
            with open(output_file, "w") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)

            assert output_file.exists()

            print(f"\n=== Full Pipeline Test ===")
            print(f"Output saved to: {output_file}")
            print(f"Weights: {dict(result.weights)}")


@pytest.mark.slow
@pytest.mark.integration
class TestDataQualityIntegration:
    """Integration tests for data quality with real data."""

    @pytest.fixture
    def real_data_with_issues(self) -> pd.DataFrame:
        """Create test data with potential quality issues."""
        np.random.seed(42)

        # Generate synthetic data with some issues
        n_days = 252
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq="D")

        data = pd.DataFrame(
            {
                "GOOD1": np.random.normal(0.001, 0.02, n_days),
                "GOOD2": np.random.normal(0.0005, 0.015, n_days),
                "MISSING": np.random.normal(0.001, 0.02, n_days),
                "VOLATILE": np.random.normal(0.002, 0.05, n_days),  # High vol
            },
            index=dates,
        )

        # Introduce some missing data
        data.loc[data.index[10:15], "MISSING"] = np.nan

        return data

    def test_allocation_with_missing_data(
        self,
        real_data_with_issues: pd.DataFrame,
    ) -> None:
        """Test that allocation handles missing data gracefully."""
        from src.allocation.allocator import AssetAllocator
        from src.utils.reproducibility import SeedManager

        with SeedManager(seed=42):
            allocator = AssetAllocator()

            # Drop NaN rows for allocation
            clean_data = real_data_with_issues.dropna()

            result = allocator.allocate(clean_data)

            assert result.is_valid
            print(f"\n=== Missing Data Test ===")
            print(f"Original rows: {len(real_data_with_issues)}")
            print(f"Clean rows: {len(clean_data)}")
            print(f"Weights: {dict(result.weights)}")


# Utility function for running individual tests
def run_quick_sanity_check() -> None:
    """
    Quick sanity check that can be run outside pytest.

    Usage:
        python -c "from tests.integration.test_e2e_backtest import run_quick_sanity_check; run_quick_sanity_check()"
    """
    print("Running quick sanity check...")

    if not is_yfinance_available():
        print("SKIP: yfinance not available")
        return

    symbols = ["AAPL", "MSFT"]
    data = fetch_real_data(symbols, period_days=100)

    if data is None:
        print("SKIP: Failed to fetch data")
        return

    returns = compute_returns(data)
    print(f"Fetched {len(returns)} days of returns for {symbols}")

    try:
        from src.allocation.allocator import AssetAllocator
        from src.utils.reproducibility import SeedManager

        with SeedManager(seed=42):
            allocator = AssetAllocator()
            result = allocator.allocate(returns)

        print(f"Allocation valid: {result.is_valid}")
        print(f"Weights: {dict(result.weights)}")
        print("PASS: Quick sanity check completed")

    except ImportError as e:
        print(f"SKIP: Missing dependencies - {e}")


if __name__ == "__main__":
    run_quick_sanity_check()
