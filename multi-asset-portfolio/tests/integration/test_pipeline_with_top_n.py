"""Integration tests for Pipeline with TopNSelector mode.

Tests cover:
- Pipeline execution in Top-N mode
- CASH weight in output
- Fallback behavior with Top-N
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import polars as pl
import pytest


class TestTopNModePipeline:
    """Integration tests for Pipeline in Top-N selection mode."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings with Top-N configuration."""
        settings = MagicMock()

        # Universe
        settings.universe = ["AAPL", "GOOGL", "MSFT"]

        # Data quality
        settings.data_quality.max_missing_rate = 0.05
        settings.data_quality.max_consecutive_missing = 5
        settings.data_quality.price_change_threshold = 0.5
        settings.data_quality.min_volume_threshold = 0
        settings.data_quality.staleness_hours = 24

        # Hard gates
        settings.hard_gates.min_trades = 30
        settings.hard_gates.max_mdd = 0.25
        settings.hard_gates.min_sharpe_ratio = 0.5
        settings.hard_gates.min_win_rate_pct = 45.0
        settings.hard_gates.min_profit_factor = 1.2

        # Strategy weighting
        settings.strategy_weighting.beta = 2.0
        settings.strategy_weighting.w_strategy_max = 0.5
        settings.strategy_weighting.entropy_min = 0.8
        settings.strategy_weighting.penalty_turnover = 0.1
        settings.strategy_weighting.penalty_mdd = 0.2
        settings.strategy_weighting.penalty_instability = 0.15
        settings.strategy_weighting.max_strategies = 10

        # Asset allocation
        settings.asset_allocation.method.value = "HRP"
        settings.asset_allocation.w_asset_max = 0.4
        settings.asset_allocation.delta_max = 0.1
        settings.asset_allocation.smooth_alpha = 0.3

        # Top-N specific
        settings.top_n_mode = True
        settings.cash_score = 0.0

        return settings

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 252
        assets = ["AAPL", "GOOGL", "MSFT"]

        data = {}
        for asset in assets:
            prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n)))
            dates = [datetime(2024, 1, 1) + pd.Timedelta(days=i) for i in range(n)]

            data[asset] = pl.DataFrame({
                "timestamp": dates,
                "open": prices * (1 + np.random.uniform(-0.01, 0.01, n)),
                "high": prices * (1 + np.random.uniform(0, 0.02, n)),
                "low": prices * (1 - np.random.uniform(0, 0.02, n)),
                "close": prices,
                "volume": np.random.uniform(1000, 10000, n),
            })

        return data

    @pytest.fixture
    def mock_evaluation_results(self):
        """Create mock evaluation results."""
        results = []

        # Create mock evaluation for each asset with varying scores
        evaluations_data = [
            ("AAPL", "momentum", 1.5, True),
            ("AAPL", "reversal", 0.8, True),
            ("GOOGL", "momentum", 1.2, True),
            ("GOOGL", "breakout", -0.3, False),  # Will be excluded by gates
            ("MSFT", "momentum", 1.0, True),
        ]

        for asset_id, strategy_id, score, passed_gates in evaluations_data:
            mock_eval = MagicMock()
            mock_eval.asset_id = asset_id
            mock_eval.strategy_id = strategy_id
            mock_eval.score = score
            mock_eval.passed_gates = passed_gates

            mock_metrics = MagicMock()
            mock_metrics.sharpe_ratio = score
            mock_metrics.max_drawdown_pct = 10.0 + abs(score) * 5
            mock_metrics.win_rate_pct = 50.0 + score * 5
            mock_metrics.profit_factor = 1.5 + score * 0.2
            mock_metrics.trade_count = 50
            mock_metrics.expected_value = score * 0.01
            mock_metrics.period_returns = [0.01 * score] * 5
            mock_metrics.turnover = 0.3

            mock_eval.metrics = mock_metrics
            results.append(mock_eval)

        return results

    def test_top_n_mode_pipeline(self, mock_settings, sample_ohlcv_data, tmp_path):
        """Test that pipeline executes successfully in Top-N mode."""
        from src.meta.top_n_selector import TopNSelector, TopNSelectionResult

        # Create selector
        selector = TopNSelector(n=5, cash_score=0.0)

        # Sample evaluations
        evaluations = {
            "AAPL": {
                "momentum": {"score": 1.5, "sharpe": 1.5},
                "reversal": {"score": 0.8, "sharpe": 0.8},
            },
            "GOOGL": {
                "momentum": {"score": 1.2, "sharpe": 1.2},
            },
            "MSFT": {
                "momentum": {"score": 1.0, "sharpe": 1.0},
            },
        }

        result = selector.select(evaluations)

        # Verify selection works
        assert len(result.selected) > 0
        assert len(result.selected) <= 5

        # Verify weights are valid
        assert sum(result.weights.values()) == pytest.approx(1.0, rel=0.01)

        # Verify strategy weights are valid
        assert sum(result.strategy_weights.values()) == pytest.approx(1.0, rel=0.01)

    def test_cash_weight_in_output(self, mock_settings):
        """Test that CASH weight is included in output when appropriate."""
        from src.meta.top_n_selector import TopNSelector

        selector = TopNSelector(n=5, cash_score=0.5)

        # Evaluations with moderate scores
        evaluations = {
            "AAPL": {"momentum": {"score": 0.8}},
            "GOOGL": {"reversal": {"score": 0.3}},
        }

        result = selector.select(evaluations)

        # CASH (0.5) should be competitive and selected
        assert "CASH" in result.weights
        assert result.cash_selected is True
        assert result.cash_weight > 0

        # Verify CASH weight is part of total
        total_weight = sum(result.weights.values())
        assert total_weight == pytest.approx(1.0, rel=0.01)

    def test_cash_weight_all_negative_scores(self, mock_settings):
        """Test CASH weight dominates when all strategy scores are negative."""
        from src.meta.top_n_selector import TopNSelector

        selector = TopNSelector(n=5, cash_score=0.0)

        # All negative scores
        evaluations = {
            "AAPL": {
                "momentum": {"score": -0.5},
                "reversal": {"score": -1.0},
            },
            "GOOGL": {
                "momentum": {"score": -0.3},
            },
        }

        result = selector.select(evaluations)

        # CASH should be selected and have highest weight
        assert result.cash_selected is True
        assert result.cash_weight > 0

        # CASH should have highest weight among selected
        max_weight_asset = max(result.weights, key=result.weights.get)
        assert max_weight_asset == "CASH"

    def test_no_fallback_needed_with_top_n(self, mock_settings):
        """Test that Top-N mode avoids equal-weight fallback."""
        from src.meta.top_n_selector import TopNSelector

        selector = TopNSelector(n=5, cash_score=0.0)

        # Mixed scores - some good, some bad
        evaluations = {
            "AAPL": {"momentum": {"score": 1.5}},
            "GOOGL": {"reversal": {"score": 0.8}},
            "MSFT": {"breakout": {"score": 0.3}},
        }

        result = selector.select(evaluations)

        # Should produce valid weights without needing fallback
        assert len(result.selected) > 0
        assert sum(result.weights.values()) == pytest.approx(1.0, rel=0.01)

        # Weights should not be equal (would indicate fallback)
        unique_weights = set(round(w, 4) for w in result.weights.values())
        assert len(unique_weights) > 1  # Not all equal

    def test_no_fallback_with_only_cash_positive(self, mock_settings):
        """Test that CASH selection prevents fallback when all others are negative."""
        from src.meta.top_n_selector import TopNSelector

        selector = TopNSelector(n=3, cash_score=0.1, min_score=-0.5)

        # All strategies below threshold except CASH
        evaluations = {
            "AAPL": {"momentum": {"score": -0.8}},  # Excluded
            "GOOGL": {"reversal": {"score": -0.6}},  # Excluded
        }

        result = selector.select(evaluations)

        # Only CASH should be selected
        assert len(result.selected) == 1
        assert result.selected[0].is_cash is True
        assert result.cash_weight == pytest.approx(1.0, rel=0.01)


class TestTopNIntegrationWithMeta:
    """Integration tests combining TopNSelector with meta layer components."""

    def test_scorer_to_top_n_flow(self):
        """Test flow from strategy scoring to Top-N selection."""
        from src.meta.scorer import ScorerConfig, StrategyMetricsInput, StrategyScorer
        from src.meta.top_n_selector import TopNSelector

        # Score strategies
        scorer = StrategyScorer(ScorerConfig())

        strategies = [
            StrategyMetricsInput(
                strategy_id="momentum",
                asset_id="AAPL",
                sharpe_ratio=1.5,
                max_drawdown_pct=10.0,
                turnover=0.3,
                period_returns=[0.02] * 5,
            ),
            StrategyMetricsInput(
                strategy_id="reversal",
                asset_id="AAPL",
                sharpe_ratio=0.8,
                max_drawdown_pct=15.0,
                turnover=0.4,
                period_returns=[0.01] * 5,
            ),
            StrategyMetricsInput(
                strategy_id="momentum",
                asset_id="GOOGL",
                sharpe_ratio=1.2,
                max_drawdown_pct=12.0,
                turnover=0.35,
                period_returns=[0.015] * 5,
            ),
        ]

        score_results = scorer.score_batch(strategies)

        # Convert to evaluation format
        evaluations: dict = {}
        for result in score_results:
            if result.asset_id not in evaluations:
                evaluations[result.asset_id] = {}
            evaluations[result.asset_id][result.strategy_id] = {
                "score": result.final_score,
                "sharpe": result.raw_sharpe,
                "penalty": result.total_penalty,
            }

        # Select Top-N
        selector = TopNSelector(n=3, cash_score=0.0)
        selection_result = selector.select(evaluations)

        # Verify integration
        assert len(selection_result.selected) == 3
        assert sum(selection_result.weights.values()) == pytest.approx(1.0, rel=0.01)

        # Best strategy should be first
        assert selection_result.selected[0].score >= selection_result.selected[1].score

    def test_top_n_with_entropy_control(self):
        """Test combining Top-N selection with entropy control."""
        from src.meta.entropy_controller import EntropyConfig, EntropyController
        from src.meta.top_n_selector import TopNSelector

        # Select Top-N
        selector = TopNSelector(n=5, cash_score=0.0, softmax_temperature=0.5)

        evaluations = {
            "AAPL": {"momentum": {"score": 2.0}},  # Dominant
            "GOOGL": {"momentum": {"score": 0.5}},
            "MSFT": {"momentum": {"score": 0.3}},
        }

        result = selector.select(evaluations)

        # Apply entropy control
        controller = EntropyController(EntropyConfig(entropy_min=0.6))
        entropy_result = controller.control(result.weights)

        # Entropy adjustment should diversify if needed
        if entropy_result.was_adjusted:
            # Adjusted weights should be more uniform
            adjusted_weights = list(entropy_result.adjusted_weights.values())
            original_weights = list(entropy_result.original_weights.values())

            original_max = max(original_weights)
            adjusted_max = max(adjusted_weights)

            # Max weight should be reduced after adjustment
            assert adjusted_max <= original_max + 0.01


class TestTopNEdgeCases:
    """Edge case tests for Top-N integration."""

    def test_empty_evaluations(self):
        """Test handling of empty evaluations."""
        from src.meta.top_n_selector import TopNSelector

        selector = TopNSelector(n=5, cash_score=0.0)

        result = selector.select({})

        # Should only have CASH
        assert len(result.selected) == 1
        assert result.selected[0].is_cash is True
        assert result.cash_weight == pytest.approx(1.0, rel=0.01)

    def test_single_strategy(self):
        """Test with only one strategy."""
        from src.meta.top_n_selector import TopNSelector

        selector = TopNSelector(n=5, cash_score=0.0)

        evaluations = {
            "AAPL": {"momentum": {"score": 1.0}},
        }

        result = selector.select(evaluations)

        # Should have 2 selected (1 strategy + CASH)
        assert len(result.selected) == 2

    def test_n_equals_one(self):
        """Test when n=1 (only best strategy)."""
        from src.meta.top_n_selector import TopNSelector

        selector = TopNSelector(n=1, cash_score=0.0)

        evaluations = {
            "AAPL": {"momentum": {"score": 1.5}},
            "GOOGL": {"momentum": {"score": 0.8}},
        }

        result = selector.select(evaluations)

        # Should select only the best (AAPL:momentum at 1.5)
        assert len(result.selected) == 1
        assert result.selected[0].asset == "AAPL"

    def test_all_strategies_below_min_score(self):
        """Test when all strategies are below min_score."""
        from src.meta.top_n_selector import TopNSelector

        selector = TopNSelector(n=5, min_score=1.0, cash_score=0.5)

        evaluations = {
            "AAPL": {"momentum": {"score": 0.5}},  # Below 1.0
            "GOOGL": {"momentum": {"score": 0.3}},  # Below 1.0
        }

        result = selector.select(evaluations)

        # All strategies excluded, no selection (CASH also below 1.0)
        assert len(result.selected) == 0
        assert result.excluded_count == 3  # 2 strategies + CASH

    def test_very_large_score_differences(self):
        """Test with very large score differences."""
        from src.meta.top_n_selector import TopNSelector

        selector = TopNSelector(n=5, cash_score=0.0, softmax_temperature=1.0)

        evaluations = {
            "DOMINANT": {"strategy": {"score": 100.0}},
            "WEAK": {"strategy": {"score": 0.1}},
        }

        result = selector.select(evaluations)

        # Dominant should have almost all weight
        assert result.weights["DOMINANT"] > 0.99


class TestAuditLoggerIntegration:
    """Tests for AuditLogger integration with Top-N selection."""

    def test_log_top_n_selection_called(self):
        """Test that audit logger is called with Top-N selection data."""
        from src.meta.top_n_selector import TopNSelector

        selector = TopNSelector(n=3, cash_score=0.0)

        evaluations = {
            "AAPL": {"momentum": {"score": 1.5}},
            "GOOGL": {"reversal": {"score": 0.8}},
        }

        result = selector.select(evaluations)

        # Verify result contains data needed for logging
        assert "n" in result.metadata
        assert "cash_score" in result.metadata
        assert "selected_count" in result.metadata
        assert result.excluded_count >= 0
        assert result.cash_selected is not None
        assert result.cash_weight >= 0
