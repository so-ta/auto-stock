"""Unit tests for TopNSelector module.

Tests cover:
- Top-N selection from strategy evaluations
- CASH inclusion and positioning
- All-negative score scenarios
- Softmax weight calculation
- min_score filtering
"""

import math

import pytest


class TestTopNSelectorConfig:
    """Tests for TopNSelectorConfig validation."""

    def test_default_config(self):
        """Test default configuration values."""
        from src.meta.top_n_selector import TopNSelectorConfig

        config = TopNSelectorConfig()

        assert config.n == 10
        assert config.cash_score == 0.0
        assert config.min_score == -999.0
        assert config.softmax_temperature == 1.0
        assert config.include_cash is True
        assert config.cash_symbol == "CASH"

    def test_config_validation_n_positive(self):
        """Test that n must be positive."""
        from src.meta.top_n_selector import TopNSelectorConfig

        with pytest.raises(ValueError, match="n must be > 0"):
            TopNSelectorConfig(n=0)

        with pytest.raises(ValueError, match="n must be > 0"):
            TopNSelectorConfig(n=-1)

    def test_config_validation_temperature_positive(self):
        """Test that softmax_temperature must be positive."""
        from src.meta.top_n_selector import TopNSelectorConfig

        with pytest.raises(ValueError, match="softmax_temperature must be > 0"):
            TopNSelectorConfig(softmax_temperature=0.0)

        with pytest.raises(ValueError, match="softmax_temperature must be > 0"):
            TopNSelectorConfig(softmax_temperature=-1.0)

    def test_custom_config(self):
        """Test custom configuration."""
        from src.meta.top_n_selector import TopNSelectorConfig

        config = TopNSelectorConfig(
            n=5,
            cash_score=0.5,
            min_score=-0.5,
            softmax_temperature=2.0,
            include_cash=False,
            cash_symbol="USD",
        )

        assert config.n == 5
        assert config.cash_score == 0.5
        assert config.min_score == -0.5
        assert config.softmax_temperature == 2.0
        assert config.include_cash is False
        assert config.cash_symbol == "USD"


class TestStrategyScore:
    """Tests for StrategyScore dataclass."""

    def test_strategy_key(self):
        """Test strategy_key property."""
        from src.meta.top_n_selector import StrategyScore

        score = StrategyScore(
            asset="AAPL",
            signal_name="momentum",
            score=1.5,
        )

        assert score.strategy_key == "AAPL:momentum"

    def test_to_dict(self):
        """Test serialization to dict."""
        from src.meta.top_n_selector import StrategyScore

        score = StrategyScore(
            asset="GOOGL",
            signal_name="reversal",
            score=0.8,
            metrics={"sharpe": 0.8},
            is_cash=False,
        )

        d = score.to_dict()

        assert d["asset"] == "GOOGL"
        assert d["signal_name"] == "reversal"
        assert d["score"] == 0.8
        assert d["metrics"] == {"sharpe": 0.8}
        assert d["is_cash"] is False
        assert d["strategy_key"] == "GOOGL:reversal"

    def test_cash_strategy(self):
        """Test CASH strategy creation."""
        from src.meta.top_n_selector import StrategyScore

        cash = StrategyScore(
            asset="CASH",
            signal_name="hold",
            score=0.0,
            is_cash=True,
        )

        assert cash.is_cash is True
        assert cash.strategy_key == "CASH:hold"


class TestTopNSelector:
    """Tests for TopNSelector class."""

    @pytest.fixture
    def selector(self):
        """Create selector with default config."""
        from src.meta.top_n_selector import TopNSelector

        return TopNSelector(n=5, cash_score=0.0)

    @pytest.fixture
    def sample_evaluations(self):
        """Create sample evaluations for testing."""
        return {
            "AAPL": {
                "momentum": {"score": 1.5, "sharpe": 1.5},
                "reversal": {"score": 0.8, "sharpe": 0.8},
            },
            "GOOGL": {
                "momentum": {"score": 1.2, "sharpe": 1.2},
                "breakout": {"score": 0.5, "sharpe": 0.5},
            },
            "MSFT": {
                "momentum": {"score": 1.0, "sharpe": 1.0},
            },
        }

    def test_select_top_n(self, selector, sample_evaluations):
        """Test that top N strategies are correctly selected."""
        result = selector.select(sample_evaluations)

        # Should select top 5 (including CASH)
        assert len(result.selected) == 5

        # Verify order (highest score first)
        scores = [s.score for s in result.selected]
        assert scores == sorted(scores, reverse=True)

        # Top score should be AAPL:momentum (1.5)
        assert result.selected[0].asset == "AAPL"
        assert result.selected[0].signal_name == "momentum"

    def test_select_top_n_limited_strategies(self):
        """Test selection when fewer strategies than N are available."""
        from src.meta.top_n_selector import TopNSelector

        selector = TopNSelector(n=10, cash_score=0.0)

        # Only 2 strategies + CASH
        evaluations = {
            "AAPL": {"momentum": {"score": 1.0}},
            "GOOGL": {"momentum": {"score": 0.8}},
        }

        result = selector.select(evaluations)

        # Should select all available (2 strategies + CASH = 3)
        assert len(result.selected) == 3

    def test_cash_included(self, selector, sample_evaluations):
        """Test that CASH is included in candidates."""
        result = selector.select(sample_evaluations)

        # CASH should be in the candidates
        cash_strategies = [s for s in result.selected if s.is_cash]

        # CASH might or might not be selected based on scores
        # But the selector should have considered it
        assert "CASH" in result.weights or result.cash_selected is False

    def test_cash_included_when_score_competitive(self):
        """Test CASH is selected when its score is competitive."""
        from src.meta.top_n_selector import TopNSelector

        selector = TopNSelector(n=3, cash_score=0.5)  # CASH score = 0.5

        # Strategies with low scores
        evaluations = {
            "AAPL": {"momentum": {"score": 0.3}},
            "GOOGL": {"momentum": {"score": 0.4}},
        }

        result = selector.select(evaluations)

        # CASH (0.5) should be highest, so should be selected
        assert result.cash_selected is True
        assert "CASH" in result.weights

    def test_all_negative_scores(self):
        """Test that CASH becomes top choice when all scores are negative."""
        from src.meta.top_n_selector import TopNSelector

        selector = TopNSelector(n=5, cash_score=0.0)

        # All strategies have negative scores
        evaluations = {
            "AAPL": {
                "momentum": {"score": -0.5},
                "reversal": {"score": -1.2},
            },
            "GOOGL": {
                "momentum": {"score": -0.8},
            },
            "MSFT": {
                "breakout": {"score": -0.3},
            },
        }

        result = selector.select(evaluations)

        # CASH (score=0.0) should be first
        assert result.selected[0].is_cash is True
        assert result.selected[0].asset == "CASH"
        assert result.selected[0].score == 0.0

        # CASH should have highest weight
        assert result.cash_selected is True
        assert result.cash_weight > 0

    def test_all_negative_with_cash_negative(self):
        """Test selection when even CASH has negative score."""
        from src.meta.top_n_selector import TopNSelector

        selector = TopNSelector(n=3, cash_score=-0.1)  # CASH also negative

        evaluations = {
            "AAPL": {"momentum": {"score": -0.5}},
            "GOOGL": {"momentum": {"score": -0.3}},
        }

        result = selector.select(evaluations)

        # CASH (-0.1) should still be highest among negatives
        assert result.selected[0].is_cash is True

    def test_calculate_weights(self, selector):
        """Test softmax weight calculation."""
        from src.meta.top_n_selector import StrategyScore

        selected = [
            StrategyScore(asset="AAPL", signal_name="m", score=2.0),
            StrategyScore(asset="GOOGL", signal_name="m", score=1.0),
            StrategyScore(asset="MSFT", signal_name="m", score=0.0),
        ]

        weights = selector.calculate_weights(selected)

        # Weights should sum to 1
        assert sum(weights.values()) == pytest.approx(1.0, rel=0.01)

        # Higher score should have higher weight
        assert weights["AAPL:m"] > weights["GOOGL:m"]
        assert weights["GOOGL:m"] > weights["MSFT:m"]

    def test_calculate_weights_equal_scores(self, selector):
        """Test weight calculation with equal scores."""
        from src.meta.top_n_selector import StrategyScore

        selected = [
            StrategyScore(asset="A", signal_name="x", score=1.0),
            StrategyScore(asset="B", signal_name="x", score=1.0),
            StrategyScore(asset="C", signal_name="x", score=1.0),
        ]

        weights = selector.calculate_weights(selected)

        # Equal scores should give equal weights
        assert weights["A:x"] == pytest.approx(weights["B:x"], rel=0.01)
        assert weights["B:x"] == pytest.approx(weights["C:x"], rel=0.01)
        assert weights["A:x"] == pytest.approx(1.0 / 3, rel=0.01)

    def test_calculate_weights_empty(self, selector):
        """Test weight calculation with empty list."""
        weights = selector.calculate_weights([])

        assert weights == {}

    def test_calculate_weights_temperature_effect(self):
        """Test that temperature affects weight distribution."""
        from src.meta.top_n_selector import StrategyScore, TopNSelector

        selected = [
            StrategyScore(asset="HIGH", signal_name="x", score=2.0),
            StrategyScore(asset="LOW", signal_name="x", score=0.0),
        ]

        # Low temperature (winner takes more)
        low_temp_selector = TopNSelector(n=5, softmax_temperature=0.5)
        low_temp_weights = low_temp_selector.calculate_weights(selected)

        # High temperature (more uniform)
        high_temp_selector = TopNSelector(n=5, softmax_temperature=2.0)
        high_temp_weights = high_temp_selector.calculate_weights(selected)

        # Low temperature should be more concentrated
        assert low_temp_weights["HIGH:x"] > high_temp_weights["HIGH:x"]

    def test_min_score_filter(self):
        """Test that strategies below min_score are filtered."""
        from src.meta.top_n_selector import TopNSelector

        selector = TopNSelector(n=5, min_score=0.0, cash_score=0.1)

        evaluations = {
            "AAPL": {
                "good": {"score": 1.0},
                "bad": {"score": -0.5},  # Below min_score
            },
            "GOOGL": {
                "mediocre": {"score": 0.5},
                "terrible": {"score": -1.0},  # Below min_score
            },
        }

        result = selector.select(evaluations)

        # Check that bad strategies are excluded
        selected_keys = [s.strategy_key for s in result.selected]
        assert "AAPL:bad" not in selected_keys
        assert "GOOGL:terrible" not in selected_keys

        # Good strategies should be included
        assert "AAPL:good" in selected_keys
        assert "GOOGL:mediocre" in selected_keys

        # excluded_count should reflect filtered strategies
        assert result.excluded_count == 2

    def test_min_score_filter_all_excluded(self):
        """Test behavior when all strategies are below min_score."""
        from src.meta.top_n_selector import TopNSelector

        selector = TopNSelector(n=5, min_score=1.0, cash_score=0.5, include_cash=False)

        evaluations = {
            "AAPL": {"low": {"score": 0.5}},
            "GOOGL": {"lower": {"score": 0.3}},
        }

        result = selector.select(evaluations)

        # All strategies excluded
        assert len(result.selected) == 0
        assert result.excluded_count == 2

    def test_aggregate_weights_by_asset(self, selector, sample_evaluations):
        """Test weight aggregation by asset."""
        result = selector.select(sample_evaluations)

        # Weights should be aggregated by asset
        assert isinstance(result.weights, dict)

        # Each asset should appear only once in weights
        for asset in result.weights:
            assert result.weights[asset] >= 0
            assert result.weights[asset] <= 1

        # Sum of asset weights should be 1
        assert sum(result.weights.values()) == pytest.approx(1.0, rel=0.01)

    def test_multiple_strategies_same_asset(self):
        """Test handling of multiple strategies for same asset."""
        from src.meta.top_n_selector import TopNSelector

        selector = TopNSelector(n=5, cash_score=0.0, include_cash=False)

        # Multiple strategies for AAPL
        evaluations = {
            "AAPL": {
                "momentum": {"score": 1.5},
                "reversal": {"score": 1.0},
                "breakout": {"score": 0.8},
            },
        }

        result = selector.select(evaluations)

        # AAPL should get combined weight of its strategies
        assert "AAPL" in result.weights
        assert result.weights["AAPL"] == pytest.approx(1.0, rel=0.01)

    def test_include_cash_false(self):
        """Test selection without CASH."""
        from src.meta.top_n_selector import TopNSelectorConfig, TopNSelector

        config = TopNSelectorConfig(n=3, include_cash=False)
        selector = TopNSelector(config=config)

        evaluations = {
            "AAPL": {"momentum": {"score": 1.0}},
            "GOOGL": {"momentum": {"score": 0.8}},
        }

        result = selector.select(evaluations)

        # CASH should not be in results
        assert result.cash_selected is False
        assert "CASH" not in result.weights
        assert result.cash_weight == 0.0


class TestTopNSelectionResult:
    """Tests for TopNSelectionResult dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        from src.meta.top_n_selector import StrategyScore, TopNSelectionResult

        result = TopNSelectionResult(
            selected=[
                StrategyScore(asset="AAPL", signal_name="m", score=1.0),
            ],
            weights={"AAPL": 1.0},
            strategy_weights={"AAPL:m": 1.0},
            excluded_count=2,
            cash_selected=False,
            cash_weight=0.0,
            metadata={"n": 5},
        )

        d = result.to_dict()

        assert "selected" in d
        assert "weights" in d
        assert "strategy_weights" in d
        assert d["excluded_count"] == 2
        assert d["cash_selected"] is False


class TestSelectFromEvaluationResults:
    """Tests for select_from_evaluation_results method."""

    def test_select_from_evaluation_objects(self):
        """Test selection from StrategyEvaluationResult objects."""
        from unittest.mock import MagicMock

        from src.meta.top_n_selector import TopNSelector

        selector = TopNSelector(n=3, cash_score=0.0)

        # Create mock evaluation results
        eval1 = MagicMock()
        eval1.asset_id = "AAPL"
        eval1.strategy_id = "momentum"
        eval1.score = 1.5
        eval1.metrics = MagicMock()
        eval1.metrics.sharpe_ratio = 1.5
        eval1.metrics.max_drawdown_pct = 10.0
        eval1.metrics.win_rate_pct = 55.0
        eval1.metrics.profit_factor = 1.5
        eval1.metrics.trade_count = 100
        eval1.metrics.expected_value = 0.01

        eval2 = MagicMock()
        eval2.asset_id = "GOOGL"
        eval2.strategy_id = "reversal"
        eval2.score = 0.8
        eval2.metrics = MagicMock()
        eval2.metrics.sharpe_ratio = 0.8
        eval2.metrics.max_drawdown_pct = 15.0
        eval2.metrics.win_rate_pct = 50.0
        eval2.metrics.profit_factor = 1.2
        eval2.metrics.trade_count = 80
        eval2.metrics.expected_value = 0.005

        result = selector.select_from_evaluation_results([eval1, eval2])

        assert len(result.selected) == 3  # 2 strategies + CASH
        assert "AAPL" in result.weights or "AAPL:momentum" in result.strategy_weights


class TestGetTopStrategiesForAsset:
    """Tests for get_top_strategies_for_asset method."""

    @pytest.fixture
    def selector(self):
        """Create selector."""
        from src.meta.top_n_selector import TopNSelector

        return TopNSelector(n=10, min_score=-999.0)

    @pytest.fixture
    def evaluations(self):
        """Create sample evaluations."""
        return {
            "AAPL": {
                "momentum": {"score": 1.5},
                "reversal": {"score": 1.0},
                "breakout": {"score": 0.5},
                "volume": {"score": 0.3},
            },
            "GOOGL": {
                "momentum": {"score": 0.8},
            },
        }

    def test_get_top_k(self, selector, evaluations):
        """Test getting top K strategies for an asset."""
        top = selector.get_top_strategies_for_asset(evaluations, "AAPL", top_k=2)

        assert len(top) == 2
        assert top[0].score == 1.5
        assert top[1].score == 1.0

    def test_get_top_k_missing_asset(self, selector, evaluations):
        """Test getting top K for non-existent asset."""
        top = selector.get_top_strategies_for_asset(evaluations, "MISSING", top_k=3)

        assert len(top) == 0

    def test_get_top_k_less_available(self, selector, evaluations):
        """Test when fewer strategies available than requested."""
        top = selector.get_top_strategies_for_asset(evaluations, "GOOGL", top_k=5)

        assert len(top) == 1  # Only 1 strategy for GOOGL


class TestCreateSelectorFromSettings:
    """Tests for factory function."""

    def test_create_with_default_fallback(self):
        """Test creation falls back to defaults when settings unavailable."""
        from src.meta.top_n_selector import create_selector_from_settings

        # Should not raise, should use defaults
        selector = create_selector_from_settings()

        assert selector.n == 10
        assert selector.cash_score == 0.0
