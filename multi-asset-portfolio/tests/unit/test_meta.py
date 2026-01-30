"""Unit tests for meta layer modules (Scorer, Weighter, EntropyController)."""

import numpy as np
import pytest


class TestScorerConfig:
    """Tests for ScorerConfig."""

    def test_default_config(self):
        """Test default scorer configuration."""
        from src.meta.scorer import ScorerConfig

        config = ScorerConfig()
        assert config.penalty_turnover == 0.1
        assert config.penalty_mdd == 0.2
        assert config.penalty_instability == 0.15

    def test_config_validation_negative_penalty(self):
        """Test that negative penalties raise error."""
        from src.meta.scorer import ScorerConfig

        with pytest.raises(ValueError, match="penalty_turnover must be >= 0"):
            ScorerConfig(penalty_turnover=-0.1)

    def test_config_validation_zero_mdd_normalization(self):
        """Test that zero MDD normalization raises error."""
        from src.meta.scorer import ScorerConfig

        with pytest.raises(ValueError, match="mdd_normalization_pct must be > 0"):
            ScorerConfig(mdd_normalization_pct=0)


class TestStrategyScorer:
    """Tests for StrategyScorer."""

    @pytest.fixture
    def scorer(self):
        """Create scorer with default config."""
        from src.meta.scorer import ScorerConfig, StrategyScorer

        return StrategyScorer(ScorerConfig())

    @pytest.fixture
    def good_metrics(self):
        """Create metrics for a good strategy."""
        from src.meta.scorer import StrategyMetricsInput

        return StrategyMetricsInput(
            strategy_id="momentum",
            asset_id="AAPL",
            sharpe_ratio=1.5,
            max_drawdown_pct=15.0,
            turnover=0.3,
            period_returns=[0.02, 0.01, 0.03, 0.02, 0.01],
        )

    @pytest.fixture
    def poor_metrics(self):
        """Create metrics for a poor strategy."""
        from src.meta.scorer import StrategyMetricsInput

        return StrategyMetricsInput(
            strategy_id="bad_strategy",
            asset_id="AAPL",
            sharpe_ratio=0.3,
            max_drawdown_pct=30.0,
            turnover=0.8,
            period_returns=[-0.02, -0.01, -0.03, 0.01, -0.02],
        )

    def test_score_good_strategy(self, scorer, good_metrics):
        """Test scoring a good strategy."""
        result = scorer.score(good_metrics)

        assert result.final_score > 0
        assert result.raw_sharpe == 1.5
        assert result.strategy_id == "momentum"
        assert result.asset_id == "AAPL"

    def test_score_poor_strategy(self, scorer, poor_metrics):
        """Test scoring a poor strategy."""
        result = scorer.score(poor_metrics)

        # Poor strategy should have lower score
        assert result.final_score < 0.5
        assert result.total_penalty > 0

    def test_penalty_breakdown(self, scorer, good_metrics):
        """Test that penalty breakdown is computed."""
        result = scorer.score(good_metrics)

        assert "turnover" in result.penalty_breakdown
        assert "mdd" in result.penalty_breakdown
        assert "instability" in result.penalty_breakdown

        # Sum of breakdown should equal total penalty
        breakdown_sum = sum(result.penalty_breakdown.values())
        assert breakdown_sum == pytest.approx(result.total_penalty, rel=0.01)

    def test_instability_penalty_calculation(self, scorer):
        """Test instability penalty for different return patterns."""
        from src.meta.scorer import StrategyMetricsInput

        # All positive returns
        stable_metrics = StrategyMetricsInput(
            strategy_id="stable",
            asset_id="TEST",
            sharpe_ratio=1.0,
            max_drawdown_pct=10.0,
            period_returns=[0.01, 0.02, 0.01, 0.03, 0.02],
        )

        # All negative returns
        unstable_metrics = StrategyMetricsInput(
            strategy_id="unstable",
            asset_id="TEST",
            sharpe_ratio=1.0,
            max_drawdown_pct=10.0,
            period_returns=[-0.01, -0.02, -0.01, -0.03, -0.02],
        )

        stable_result = scorer.score(stable_metrics)
        unstable_result = scorer.score(unstable_metrics)

        assert stable_result.penalty_breakdown["instability"] < unstable_result.penalty_breakdown["instability"]

    def test_score_batch(self, scorer, good_metrics, poor_metrics):
        """Test batch scoring."""
        results = scorer.score_batch([good_metrics, poor_metrics])

        assert len(results) == 2
        assert results[0].strategy_id == "momentum"
        assert results[1].strategy_id == "bad_strategy"

    def test_rank_strategies(self, scorer, good_metrics, poor_metrics):
        """Test strategy ranking."""
        results = scorer.score_batch([poor_metrics, good_metrics])
        ranked = scorer.rank_strategies(results)

        # Good strategy should be ranked first
        assert ranked[0].strategy_id == "momentum"
        assert ranked[1].strategy_id == "bad_strategy"

    def test_min_score_clipping(self):
        """Test that scores are clipped to min_score."""
        from src.meta.scorer import ScorerConfig, StrategyMetricsInput, StrategyScorer

        config = ScorerConfig(min_score=0.0)
        scorer = StrategyScorer(config)

        # Very poor metrics that would result in negative score
        metrics = StrategyMetricsInput(
            strategy_id="terrible",
            asset_id="TEST",
            sharpe_ratio=-0.5,
            max_drawdown_pct=50.0,
            turnover=1.0,
            period_returns=[-0.1] * 10,
        )

        result = scorer.score(metrics)
        assert result.final_score >= 0.0

    def test_result_to_dict(self, scorer, good_metrics):
        """Test StrategyScoreResult serialization."""
        result = scorer.score(good_metrics)
        d = result.to_dict()

        assert "strategy_id" in d
        assert "final_score" in d
        assert "penalty_breakdown" in d


class TestStrategyWeighter:
    """Tests for StrategyWeighter module."""

    @pytest.fixture
    def sample_scores(self):
        """Create sample strategy scores."""
        return {
            "strategy_a": 1.5,
            "strategy_b": 1.0,
            "strategy_c": 0.5,
            "strategy_d": 0.1,
        }

    def test_softmax_weighting(self, sample_scores):
        """Test softmax weighting calculation."""
        from src.meta.weighter import StrategyWeighter, WeighterConfig

        config = WeighterConfig(beta=2.0, w_strategy_max=0.5)
        weighter = StrategyWeighter(config)

        weights = weighter.calculate_weights(sample_scores)

        # Weights should sum to 1
        assert sum(weights.values()) == pytest.approx(1.0, rel=0.01)

        # Higher score should get higher weight
        assert weights["strategy_a"] > weights["strategy_b"]
        assert weights["strategy_b"] > weights["strategy_c"]

    def test_weight_cap(self, sample_scores):
        """Test that weights are capped at max."""
        from src.meta.weighter import StrategyWeighter, WeighterConfig

        config = WeighterConfig(beta=10.0, w_strategy_max=0.3)  # High beta, low cap
        weighter = StrategyWeighter(config)

        weights = weighter.calculate_weights(sample_scores)

        # All weights should be <= cap
        assert all(w <= 0.3 + 0.01 for w in weights.values())

    def test_sparsification(self):
        """Test that low scores are clipped to zero."""
        from src.meta.weighter import StrategyWeighter, WeighterConfig

        config = WeighterConfig(min_weight=0.05)
        weighter = StrategyWeighter(config)

        scores = {
            "good": 2.0,
            "mediocre": 0.1,
            "poor": 0.01,
        }

        weights = weighter.calculate_weights(scores)

        # Poor strategy should have zero weight
        assert weights.get("poor", 0) == 0.0 or weights["poor"] < 0.01


class TestEntropyController:
    """Tests for EntropyController module."""

    @pytest.fixture
    def controller(self):
        """Create entropy controller with default config."""
        from src.meta.entropy_controller import EntropyConfig, EntropyController

        return EntropyController(EntropyConfig(entropy_min=0.8))

    def test_calculate_entropy(self, controller):
        """Test entropy calculation."""
        # Uniform distribution (max entropy)
        uniform_weights = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        uniform_entropy = controller.calculate_entropy(uniform_weights)

        # Concentrated distribution (low entropy)
        concentrated_weights = {"a": 0.9, "b": 0.05, "c": 0.03, "d": 0.02}
        concentrated_entropy = controller.calculate_entropy(concentrated_weights)

        assert uniform_entropy > concentrated_entropy
        assert 0 <= uniform_entropy <= 1
        assert 0 <= concentrated_entropy <= 1

    def test_adjust_for_entropy(self, controller):
        """Test entropy adjustment."""
        # Very concentrated weights
        weights = {"a": 0.95, "b": 0.03, "c": 0.02}

        adjusted = controller.adjust_for_entropy(weights)

        # Adjusted should be more uniform
        original_entropy = controller.calculate_entropy(weights)
        adjusted_entropy = controller.calculate_entropy(adjusted)

        assert adjusted_entropy >= original_entropy * 0.9

    def test_entropy_threshold_check(self, controller):
        """Test entropy threshold checking."""
        # High entropy (should pass)
        uniform_weights = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        assert controller.meets_entropy_threshold(uniform_weights) is True

        # Very low entropy (should fail)
        concentrated_weights = {"a": 0.99, "b": 0.01}
        assert controller.meets_entropy_threshold(concentrated_weights) is False


class TestMetaLayerIntegration:
    """Integration tests for meta layer components."""

    def test_scorer_to_weighter_pipeline(self):
        """Test pipeline from scoring to weighting."""
        from src.meta.scorer import ScorerConfig, StrategyMetricsInput, StrategyScorer
        from src.meta.weighter import StrategyWeighter, WeighterConfig

        # Create strategies
        strategies = [
            StrategyMetricsInput(
                strategy_id=f"strategy_{i}",
                asset_id="ASSET",
                sharpe_ratio=1.0 + i * 0.2,
                max_drawdown_pct=10 + i * 2,
                turnover=0.3,
                period_returns=[0.01] * 5,
            )
            for i in range(5)
        ]

        # Score
        scorer = StrategyScorer(ScorerConfig())
        scores = {
            result.strategy_id: result.final_score
            for result in scorer.score_batch(strategies)
        }

        # Weight
        weighter = StrategyWeighter(WeighterConfig(beta=2.0))
        weights = weighter.calculate_weights(scores)

        # Verify
        assert sum(weights.values()) == pytest.approx(1.0, rel=0.01)
        assert all(w >= 0 for w in weights.values())

    def test_full_meta_layer_with_entropy(self):
        """Test full meta layer with entropy control."""
        from src.meta.entropy_controller import EntropyConfig, EntropyController
        from src.meta.scorer import ScorerConfig, StrategyMetricsInput, StrategyScorer
        from src.meta.weighter import StrategyWeighter, WeighterConfig

        # Create strategies with varying quality
        strategies = [
            StrategyMetricsInput(
                strategy_id="excellent",
                asset_id="ASSET",
                sharpe_ratio=2.0,
                max_drawdown_pct=10.0,
                turnover=0.2,
                period_returns=[0.02] * 5,
            ),
            StrategyMetricsInput(
                strategy_id="good",
                asset_id="ASSET",
                sharpe_ratio=1.2,
                max_drawdown_pct=15.0,
                turnover=0.3,
                period_returns=[0.01] * 5,
            ),
            StrategyMetricsInput(
                strategy_id="average",
                asset_id="ASSET",
                sharpe_ratio=0.8,
                max_drawdown_pct=20.0,
                turnover=0.4,
                period_returns=[0.005] * 5,
            ),
        ]

        # Score
        scorer = StrategyScorer(ScorerConfig())
        score_results = scorer.score_batch(strategies)
        scores = {r.strategy_id: r.final_score for r in score_results}

        # Weight
        weighter = StrategyWeighter(WeighterConfig(beta=2.0, w_strategy_max=0.5))
        weights = weighter.calculate_weights(scores)

        # Entropy control
        controller = EntropyController(EntropyConfig(entropy_min=0.5))
        adjusted_weights = controller.adjust_for_entropy(weights)

        # Verify
        assert sum(adjusted_weights.values()) == pytest.approx(1.0, rel=0.01)
        assert controller.meets_entropy_threshold(adjusted_weights)
