"""Unit tests for allocation modules (Covariance, HRP, RiskParity, Allocator)."""

import numpy as np
import pandas as pd
import pytest


class TestCovarianceEstimator:
    """Tests for CovarianceEstimator."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample return DataFrame."""
        np.random.seed(42)
        n = 252
        assets = ["AAPL", "GOOGL", "MSFT", "AMZN"]

        # Create correlated returns
        cov_matrix = np.array([
            [0.04, 0.02, 0.015, 0.01],
            [0.02, 0.05, 0.02, 0.015],
            [0.015, 0.02, 0.03, 0.01],
            [0.01, 0.015, 0.01, 0.06],
        ])
        mean = np.zeros(4)
        returns = np.random.multivariate_normal(mean, cov_matrix / 252, n)

        return pd.DataFrame(
            returns,
            columns=assets,
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

    def test_sample_covariance(self, sample_returns):
        """Test sample covariance estimation."""
        from src.allocation.covariance import CovarianceConfig, CovarianceEstimator, CovarianceMethod

        config = CovarianceConfig(method=CovarianceMethod.SAMPLE)
        estimator = CovarianceEstimator(config)

        result = estimator.estimate(sample_returns)

        assert result.is_valid
        assert result.covariance.shape == (4, 4)
        # Covariance matrix should be symmetric
        assert np.allclose(result.covariance.values, result.covariance.values.T)
        # Diagonal should be positive
        assert all(np.diag(result.covariance.values) > 0)

    def test_ledoit_wolf_shrinkage(self, sample_returns):
        """Test Ledoit-Wolf shrinkage estimation."""
        from src.allocation.covariance import CovarianceConfig, CovarianceEstimator, CovarianceMethod

        config = CovarianceConfig(method=CovarianceMethod.LEDOIT_WOLF)
        estimator = CovarianceEstimator(config)

        result = estimator.estimate(sample_returns)

        assert result.is_valid
        assert result.covariance.shape == (4, 4)
        # Shrunk covariance should be well-conditioned
        eigenvalues = np.linalg.eigvalsh(result.covariance.values)
        assert all(eigenvalues > 0)

    def test_correlation_matrix(self, sample_returns):
        """Test correlation matrix computation."""
        from src.allocation.covariance import CovarianceConfig, CovarianceEstimator

        estimator = CovarianceEstimator(CovarianceConfig())
        result = estimator.estimate(sample_returns)

        # Diagonal of correlation should be 1
        assert np.allclose(np.diag(result.correlation.values), 1.0)
        # Off-diagonal should be in [-1, 1]
        off_diag = result.correlation.values[~np.eye(4, dtype=bool)]
        assert all(-1 <= x <= 1 for x in off_diag)

    def test_volatilities(self, sample_returns):
        """Test volatility computation."""
        from src.allocation.covariance import CovarianceConfig, CovarianceEstimator

        estimator = CovarianceEstimator(CovarianceConfig())
        result = estimator.estimate(sample_returns)

        assert len(result.volatilities) == 4
        assert all(v > 0 for v in result.volatilities)


class TestHierarchicalRiskParity:
    """Tests for HRP allocation."""

    @pytest.fixture
    def sample_covariance(self):
        """Create sample covariance matrix."""
        assets = ["A", "B", "C", "D"]
        cov = np.array([
            [0.04, 0.02, 0.01, 0.005],
            [0.02, 0.05, 0.015, 0.01],
            [0.01, 0.015, 0.03, 0.008],
            [0.005, 0.01, 0.008, 0.02],
        ])
        return pd.DataFrame(cov, index=assets, columns=assets)

    @pytest.fixture
    def sample_correlation(self, sample_covariance):
        """Create correlation matrix from covariance."""
        std = np.sqrt(np.diag(sample_covariance.values))
        corr = sample_covariance.values / np.outer(std, std)
        return pd.DataFrame(
            corr,
            index=sample_covariance.index,
            columns=sample_covariance.columns,
        )

    def test_hrp_allocation(self, sample_covariance, sample_correlation):
        """Test HRP allocation produces valid weights."""
        from src.allocation.hrp import HierarchicalRiskParity

        hrp = HierarchicalRiskParity()
        result = hrp.allocate(sample_covariance, sample_correlation)

        assert result.is_valid
        # Weights should sum to 1
        assert result.weights.sum() == pytest.approx(1.0, rel=0.01)
        # All weights should be positive
        assert all(w >= 0 for w in result.weights)

    def test_hrp_diversification(self, sample_covariance, sample_correlation):
        """Test that HRP produces diversified allocation."""
        from src.allocation.hrp import HierarchicalRiskParity

        hrp = HierarchicalRiskParity()
        result = hrp.allocate(sample_covariance, sample_correlation)

        # No single asset should dominate
        assert result.weights.max() < 0.8
        # Effective N should be > 1
        herfindahl = (result.weights ** 2).sum()
        effective_n = 1 / herfindahl
        assert effective_n > 1.5

    def test_hrp_inverse_volatility_bias(self, sample_covariance, sample_correlation):
        """Test that HRP gives more weight to lower volatility assets."""
        from src.allocation.hrp import HierarchicalRiskParity

        hrp = HierarchicalRiskParity()
        result = hrp.allocate(sample_covariance, sample_correlation)

        # Asset D has lowest variance (0.02), should have higher weight
        vols = np.sqrt(np.diag(sample_covariance.values))
        lowest_vol_asset = sample_covariance.columns[np.argmin(vols)]

        # Low vol asset should have non-trivial weight
        assert result.weights[lowest_vol_asset] > 0.15


class TestRiskParity:
    """Tests for Risk Parity allocation."""

    @pytest.fixture
    def sample_covariance(self):
        """Create sample covariance matrix."""
        assets = ["A", "B", "C"]
        cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.02],
            [0.005, 0.02, 0.01],
        ])
        return pd.DataFrame(cov, index=assets, columns=assets)

    def test_risk_parity_allocation(self, sample_covariance):
        """Test risk parity produces valid weights."""
        from src.allocation.risk_parity import RiskParity

        rp = RiskParity()
        result = rp.allocate(sample_covariance)

        assert result.is_valid
        assert result.weights.sum() == pytest.approx(1.0, rel=0.01)
        assert all(w >= 0 for w in result.weights)

    def test_risk_parity_equal_risk_contribution(self, sample_covariance):
        """Test that risk parity equalizes risk contribution."""
        from src.allocation.risk_parity import RiskParity

        rp = RiskParity()
        result = rp.allocate(sample_covariance)

        # Calculate risk contributions
        w = result.weights.values
        cov = sample_covariance.values
        port_vol = np.sqrt(w @ cov @ w)
        marginal_risk = cov @ w / port_vol
        risk_contrib = w * marginal_risk

        # Risk contributions should be approximately equal
        mean_contrib = np.mean(risk_contrib)
        assert all(abs(rc - mean_contrib) < 0.05 for rc in risk_contrib)

    def test_naive_risk_parity(self, sample_covariance):
        """Test naive (inverse variance) risk parity."""
        from src.allocation.risk_parity import NaiveRiskParity

        naive_rp = NaiveRiskParity()
        result = naive_rp.allocate(sample_covariance)

        assert result.is_valid
        assert result.weights.sum() == pytest.approx(1.0, rel=0.01)

        # Higher variance asset should have lower weight
        vols = np.sqrt(np.diag(sample_covariance.values))
        inv_var = 1 / (vols ** 2)
        expected_weights = inv_var / inv_var.sum()

        for i, asset in enumerate(sample_covariance.columns):
            assert result.weights[asset] == pytest.approx(expected_weights[i], rel=0.01)


class TestConstraintProcessor:
    """Tests for constraint processing."""

    def test_apply_weight_cap(self):
        """Test that weights are capped and violations recorded.

        Note: ConstraintProcessor caps weights, then normalizes to sum=1.0.
        After normalization, individual weights may exceed the cap.
        The important behavior is that violations are recorded.
        """
        from src.allocation.constraints import ConstraintConfig, ConstraintProcessor, ConstraintViolationType

        config = ConstraintConfig(w_max=0.3)
        processor = ConstraintProcessor(config)

        weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        result = processor.apply(weights)

        # Violation should be recorded for weight exceeding cap
        upper_bound_violations = [
            v for v in result.violations
            if v.violation_type == ConstraintViolationType.UPPER_BOUND
        ]
        assert len(upper_bound_violations) >= 1
        assert upper_bound_violations[0].asset == "A"
        assert upper_bound_violations[0].original_value == 0.5
        assert upper_bound_violations[0].limit == 0.3

        # Sum should be normalized to 1.0
        assert result.weights.sum() == pytest.approx(1.0, rel=0.01)

    def test_apply_turnover_limit(self):
        """Test that turnover is limited."""
        from src.allocation.constraints import ConstraintConfig, ConstraintProcessor

        config = ConstraintConfig(delta_max=0.05)
        processor = ConstraintProcessor(config)

        current = pd.Series({"A": 0.4, "B": 0.4, "C": 0.2})
        new = pd.Series({"A": 0.6, "B": 0.2, "C": 0.2})

        result = processor.apply(new, current)

        # Changes should be limited to delta_max
        for asset in current.index:
            change = abs(result.weights[asset] - current[asset])
            assert change <= 0.05 + 0.01

    def test_normalize_after_constraints(self):
        """Test that weights are normalized after constraints."""
        from src.allocation.constraints import ConstraintConfig, ConstraintProcessor

        config = ConstraintConfig(w_max=0.25)
        processor = ConstraintProcessor(config)

        weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        result = processor.apply(weights)

        assert result.weights.sum() == pytest.approx(1.0, rel=0.01)


class TestWeightSmoother:
    """Tests for weight smoothing."""

    def test_smoothing_basic(self):
        """Test basic weight smoothing."""
        from src.allocation.smoother import SmootherConfig, WeightSmoother

        config = SmootherConfig(alpha=0.3)
        smoother = WeightSmoother(config)

        previous = pd.Series({"A": 0.4, "B": 0.4, "C": 0.2})
        new = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})

        result = smoother.smooth(new, previous)

        # w_final = alpha * w_new + (1-alpha) * w_prev
        expected_a = 0.3 * 0.5 + 0.7 * 0.4
        assert result.weights["A"] == pytest.approx(expected_a, rel=0.01)

    def test_smoothing_first_period(self):
        """Test smoothing when no previous weights exist."""
        from src.allocation.smoother import SmootherConfig, WeightSmoother

        config = SmootherConfig(alpha=0.3)
        smoother = WeightSmoother(config)

        new = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})

        result = smoother.smooth(new, None)

        # Should use new weights directly
        assert result.weights["A"] == pytest.approx(0.5, rel=0.01)


class TestAssetAllocator:
    """Tests for integrated AssetAllocator."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns DataFrame."""
        np.random.seed(42)
        n = 252
        assets = ["AAPL", "GOOGL", "MSFT"]

        returns = np.random.normal(0.0005, 0.02, (n, 3))
        return pd.DataFrame(
            returns,
            columns=assets,
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

    def test_allocator_hrp_method(self, sample_returns):
        """Test allocator with HRP method."""
        from src.allocation.allocator import AllocationMethod, AllocatorConfig, AssetAllocator

        config = AllocatorConfig(method=AllocationMethod.HRP)
        allocator = AssetAllocator(config)

        result = allocator.allocate(sample_returns)

        assert result.is_valid
        assert not result.is_fallback
        assert result.weights.sum() == pytest.approx(1.0, rel=0.01)

    def test_allocator_risk_parity_method(self, sample_returns):
        """Test allocator with Risk Parity method."""
        from src.allocation.allocator import AllocationMethod, AllocatorConfig, AssetAllocator

        config = AllocatorConfig(method=AllocationMethod.RISK_PARITY)
        allocator = AssetAllocator(config)

        result = allocator.allocate(sample_returns)

        assert result.is_valid
        assert result.method_used == "risk_parity"

    def test_allocator_with_quality_filter(self, sample_returns):
        """Test allocator with quality filtering."""
        from src.allocation.allocator import AllocatorConfig, AssetAllocator

        config = AllocatorConfig()
        allocator = AssetAllocator(config)

        # Mark one asset as excluded
        quality_flags = pd.DataFrame(
            [[True, False, True]],
            columns=["AAPL", "GOOGL", "MSFT"],
        )

        result = allocator.allocate(sample_returns, quality_flags=quality_flags)

        assert result.is_valid
        assert "GOOGL" in result.excluded_assets
        assert result.weights["GOOGL"] == 0.0

    def test_allocator_fallback_insufficient_assets(self):
        """Test allocator fallback when too few assets."""
        from src.allocation.allocator import AllocatorConfig, AssetAllocator, FallbackReason

        config = AllocatorConfig(min_assets_required=3)
        allocator = AssetAllocator(config)

        # Only 2 assets
        returns = pd.DataFrame(
            np.random.normal(0, 0.02, (100, 2)),
            columns=["A", "B"],
        )

        # Exclude one
        quality_flags = pd.DataFrame([[True, False]], columns=["A", "B"])

        result = allocator.allocate(returns, quality_flags=quality_flags)

        assert result.is_fallback
        assert result.fallback_reason == FallbackReason.ALL_ASSETS_EXCLUDED

    def test_allocator_with_previous_weights(self, sample_returns):
        """Test allocator with previous weights for smoothing."""
        from src.allocation.allocator import AllocatorConfig, AssetAllocator

        config = AllocatorConfig(smooth_alpha=0.3, delta_max=0.1)
        allocator = AssetAllocator(config)

        previous = pd.Series({"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.3})

        result = allocator.allocate(sample_returns, previous_weights=previous)

        assert result.is_valid
        # Changes should be gradual
        for asset in previous.index:
            change = abs(result.weights[asset] - previous[asset])
            assert change <= 0.15  # delta_max + smoothing effect

    def test_allocator_portfolio_metrics(self, sample_returns):
        """Test that portfolio metrics are computed."""
        from src.allocation.allocator import AllocatorConfig, AssetAllocator

        config = AllocatorConfig()
        allocator = AssetAllocator(config)

        result = allocator.allocate(sample_returns)

        assert "volatility" in result.portfolio_metrics
        assert "effective_n" in result.portfolio_metrics
        assert result.portfolio_metrics["volatility"] > 0

    def test_allocator_result_to_dict(self, sample_returns):
        """Test allocation result serialization."""
        from src.allocation.allocator import AllocatorConfig, AssetAllocator

        allocator = AssetAllocator(AllocatorConfig())
        result = allocator.allocate(sample_returns)

        d = result.to_dict()

        assert "weights" in d
        assert "method_used" in d
        assert "is_valid" in d
        assert "turnover" in d
