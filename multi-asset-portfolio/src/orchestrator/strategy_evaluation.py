"""
Strategy Evaluation Module - Evaluates strategies on test data.

Extracted from signal_generation.py for better modularity.
This module handles:
1. Computing signal-based returns
2. Calculating performance metrics
3. Creating evaluation results
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.orchestrator.exceptions import SignalEvaluationError

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    from src.config.settings import Settings
    from src.strategy.gate_checker import StrategyMetrics
    from src.utils.logger import AuditLogger

# Lazy import for DataFrame conversion utilities
_ensure_pandas = None


def _get_ensure_pandas():
    """Lazy import of ensure_pandas to avoid circular imports"""
    global _ensure_pandas
    if _ensure_pandas is None:
        from src.utils.dataframe_utils import ensure_pandas
        _ensure_pandas = ensure_pandas
    return _ensure_pandas


logger = logging.getLogger(__name__)


class StrategyEvaluator:
    """
    Evaluates strategies on test data.

    Responsible for:
    - Computing signal-based returns
    - Calculating performance metrics
    - Creating evaluation results

    Supports two modes:
    - Standard mode: Uses SignalResult objects from SignalGenerator
    - Precomputed mode: Uses cached signal values from SignalPrecomputer
    """

    def __init__(
        self,
        settings: "Settings",
        audit_logger: "AuditLogger | None" = None,
        signal_precomputer: Any | None = None,
    ) -> None:
        """
        Initialize StrategyEvaluator.

        Args:
            settings: Application settings
            audit_logger: Optional audit logger for detailed logging
            signal_precomputer: Optional SignalPrecomputer for cached signal access
        """
        self._settings = settings
        self._audit_logger = audit_logger
        self._logger = logger
        self._signal_precomputer = signal_precomputer

    @property
    def settings(self) -> "Settings":
        """Get settings instance."""
        return self._settings

    def evaluate_strategies(
        self,
        raw_data: dict[str, "pl.DataFrame"],
        signals: dict[str, dict[str, Any]],
    ) -> list[Any]:
        """
        Evaluate strategies on test data.

        This method:
        1. Extracts test period data for each asset
        2. Computes signal-based returns (signal score * next-day return)
        3. Calculates performance metrics using MetricsCalculator
        4. Creates StrategyEvaluationResult for each Asset × Signal

        Args:
            raw_data: Dictionary mapping symbol to DataFrame
            signals: Dictionary mapping symbol -> signal_name -> signal info

        Returns:
            List of StrategyEvaluationResult objects
        """
        import pandas as pd

        from src.strategy.evaluator import EvaluationStatus, StrategyEvaluationResult
        from src.strategy.gate_checker import StrategyMetrics
        from src.strategy.metrics import MetricsCalculator

        evaluations: list[Any] = []

        if not signals:
            self._logger.warning("No signals available for evaluation")
            return evaluations

        # Initialize MetricsCalculator
        risk_free_rate = getattr(self.settings, "risk_free_rate", 0.0)
        metrics_calc = MetricsCalculator(
            annualization_factor=252,
            risk_free_rate=risk_free_rate,
            var_percentile=0.05,
        )

        # Get train/test split configuration
        train_days = self.settings.walk_forward.train_period_days
        test_days = self.settings.walk_forward.test_period_days

        total_evaluated = 0

        for symbol, signal_dict in signals.items():
            # Get raw data for this asset
            df = raw_data.get(symbol)
            if df is None:
                continue

            # Convert polars DataFrame to pandas if needed
            df = _get_ensure_pandas()(df)

            # Ensure index is datetime
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")

            # Split into train/test periods
            total_rows = len(df)
            test_rows = min(test_days, total_rows // 3)
            train_end_idx = total_rows - test_rows

            if train_end_idx < 50 or test_rows < 10:
                self._logger.warning(
                    f"Insufficient data for evaluation: {symbol}"
                )
                continue

            # Extract test period data
            test_data = df.iloc[train_end_idx:].copy()
            test_start = test_data.index[0] if len(test_data) > 0 else None
            test_end = test_data.index[-1] if len(test_data) > 0 else None

            # Calculate test period returns
            test_returns = test_data["close"].pct_change().dropna()

            for signal_name, signal_info in signal_dict.items():
                try:
                    # Check if signal_info is a precomputed float value or a dict
                    is_precomputed = isinstance(signal_info, (int, float))

                    if is_precomputed:
                        # Precomputed mode: Load time series from SignalPrecomputer
                        test_scores = self._get_precomputed_scores(
                            symbol, signal_name, train_end_idx
                        )
                        if test_scores is None:
                            continue
                        signal_params = {}
                    else:
                        # Standard mode: Use SignalResult from SignalGenerator
                        test_scores = self._get_standard_scores(
                            df, signal_name, signal_info, train_end_idx
                        )
                        if test_scores is None:
                            continue
                        signal_params = signal_info.get("params", {})

                    # Align test_scores to test_returns index
                    common_test_idx = test_scores.index.intersection(test_returns.index)
                    if len(common_test_idx) == 0:
                        self._logger.debug(
                            f"No common index for {symbol}/{signal_name}"
                        )
                        continue

                    test_scores_aligned = test_scores.loc[common_test_idx]

                    # Compute signal-based returns
                    signal_returns = self._compute_signal_returns(
                        test_scores_aligned, test_returns
                    )

                    if len(signal_returns) < 10:
                        self._logger.debug(
                            f"Insufficient signal returns for {symbol}/{signal_name}"
                        )
                        continue

                    # Calculate performance metrics
                    perf_metrics = metrics_calc.calculate(
                        returns=signal_returns.values,
                        positions=(
                            test_scores_aligned.values[:-1]
                            if len(test_scores_aligned) > 1
                            else None
                        ),
                        cost_per_trade=self._get_cost_per_trade(),
                    )

                    # Create StrategyMetrics
                    strategy_metrics = StrategyMetrics(
                        strategy_id=signal_name,
                        asset_id=symbol,
                        trade_count=perf_metrics.n_trades,
                        max_drawdown_pct=perf_metrics.max_drawdown * 100,
                        expected_value=perf_metrics.expected_value,
                        sharpe_ratio=perf_metrics.sharpe_ratio,
                        win_rate_pct=perf_metrics.win_rate * 100,
                        profit_factor=perf_metrics.profit_factor,
                        period_returns=[],
                    )

                    # Calculate strategy score for ranking
                    score = self._calculate_strategy_score(strategy_metrics)

                    # Create evaluation result
                    eval_result = StrategyEvaluationResult(
                        strategy_id=signal_name,
                        asset_id=symbol,
                        status=EvaluationStatus.COMPLETED,
                        metrics=strategy_metrics,
                        gate_result=None,
                        walk_forward_summary={
                            "test_start": str(test_start) if test_start else None,
                            "test_end": str(test_end) if test_end else None,
                            "test_samples": len(signal_returns),
                            "signal_params": signal_params,
                            "optimized": signal_info.get("optimized", False) if isinstance(signal_info, dict) else False,
                            "precomputed": is_precomputed,
                        },
                        score=score,
                        error_message=None,
                        evaluated_at=datetime.now(),
                    )

                    evaluations.append(eval_result)
                    total_evaluated += 1

                except Exception as e:
                    # Strict mode: raise exception to stop backtest immediately
                    error_msg = f"Evaluation failed for {symbol}/{signal_name}: {e}"
                    self._logger.error(error_msg)
                    raise SignalEvaluationError(symbol, signal_name, str(e)) from e

        # Log audit info
        self._log_audit(evaluations)

        self._logger.info(
            f"Strategy evaluation completed: {total_evaluated} strategies evaluated"
        )

        return evaluations

    def _get_precomputed_scores(
        self,
        symbol: str,
        signal_name: str,
        train_end_idx: int,
    ) -> "pd.Series | None":
        """Get signal scores from precomputed cache."""
        import pandas as pd

        if self._signal_precomputer is None:
            self._logger.debug(
                f"Skipping {symbol}/{signal_name}: precomputed value but no precomputer"
            )
            return None

        try:
            signal_df = self._signal_precomputer.load_signal(signal_name, symbol)
            if signal_df is None or len(signal_df) == 0:
                self._logger.debug(f"No cached data for {symbol}/{signal_name}")
                return None

            # Convert to pandas Series
            if hasattr(signal_df, "to_pandas"):
                signal_pdf = signal_df.to_pandas()
            else:
                signal_pdf = signal_df

            if "timestamp" in signal_pdf.columns:
                signal_pdf = signal_pdf.set_index("timestamp")

            # Extract test period scores
            return signal_pdf["value"].iloc[train_end_idx:]

        except Exception as e:
            self._logger.debug(
                f"Failed to load precomputed signal {symbol}/{signal_name}: {e}"
            )
            return None

    def _get_standard_scores(
        self,
        df: "pd.DataFrame",
        signal_name: str,
        signal_info: dict,
        train_end_idx: int,
    ) -> "pd.Series | None":
        """Get signal scores using standard signal computation."""
        import pandas as pd

        signal_result = signal_info.get("result")
        if signal_result is None:
            return None

        from src.signals import SignalRegistry

        signal_cls = SignalRegistry.get(signal_name)
        signal_params = signal_info.get("params", {})
        signal = signal_cls(**signal_params)

        # Compute signal using ONLY data up to train_end (no look-ahead)
        lookback_buffer = 300
        buffer_start = max(0, train_end_idx - lookback_buffer)
        train_data = df.iloc[buffer_start:train_end_idx + 1].copy()

        train_result = signal.compute(train_data)

        # Use the last signal value from training period
        last_train_signal = train_result.scores.iloc[-1] if len(train_result.scores) > 0 else 0.0

        # Create constant signal for test period
        test_dates = df.index[train_end_idx:]
        return pd.Series(last_train_signal, index=test_dates)

    def _compute_signal_returns(
        self,
        scores: "pd.Series",
        returns: "pd.Series",
    ) -> "pd.Series":
        """
        Compute strategy returns from signal scores.

        Strategy: Take position proportional to signal score at time t,
        earn return at time t+1.

        Args:
            scores: Signal scores in [-1, +1]
            returns: Price returns

        Returns:
            Strategy returns series
        """
        import pandas as pd

        common_idx = scores.index.intersection(returns.index)

        if len(common_idx) == 0:
            self._logger.debug("No common index between scores and returns")
            return pd.Series(dtype=float)

        aligned_scores = scores.loc[common_idx]
        aligned_returns = returns.loc[common_idx]

        # Shift scores by 1 to avoid lookahead bias
        lagged_scores = aligned_scores.shift(1)

        # Strategy return = position * next return
        strategy_returns = lagged_scores * aligned_returns

        return strategy_returns.dropna()

    def _get_cost_per_trade(self) -> float:
        """Get cost per trade from settings."""
        cost_model = getattr(self.settings, "cost_model", None)
        if cost_model is None:
            return 0.0

        spread_bps = getattr(cost_model, "spread_bps", 10)
        commission_bps = getattr(cost_model, "commission_bps", 5)
        slippage_bps = getattr(cost_model, "slippage_bps", 10)

        total_bps = spread_bps + commission_bps + slippage_bps
        return total_bps / 10000

    def _calculate_strategy_score(self, metrics: "StrategyMetrics") -> float:
        """
        Calculate strategy score for ranking.

        score = Sharpe_adj - penalty
        penalty = lambda1 * turnover + lambda2 * MDD + lambda3 * instability

        Args:
            metrics: Strategy metrics

        Returns:
            Composite score (higher is better)
        """
        import math

        # Handle complex numbers or invalid values in sharpe_ratio
        base_score = metrics.sharpe_ratio
        if isinstance(base_score, complex):
            base_score = base_score.real
        if not math.isfinite(base_score):
            base_score = 0.0

        lambda_mdd = 0.5
        lambda_instability = 0.3

        # Handle complex numbers or invalid values in max_drawdown_pct
        mdd_pct = metrics.max_drawdown_pct
        if isinstance(mdd_pct, complex):
            mdd_pct = abs(mdd_pct.real)
        if not math.isfinite(mdd_pct):
            mdd_pct = 100.0  # Assume worst case
        mdd_penalty = lambda_mdd * (mdd_pct / 25.0)

        instability_penalty = 0.0
        if metrics.period_returns:
            negative_ratio = sum(
                1 for r in metrics.period_returns if r <= 0
            ) / len(metrics.period_returns)
            instability_penalty = lambda_instability * negative_ratio

        score = base_score - mdd_penalty - instability_penalty
        return max(0.0, float(score))

    def _log_audit(self, evaluations: list[Any]) -> None:
        """Log audit info for evaluations."""
        if not self._audit_logger:
            return

        for eval_result in evaluations:
            metrics_dict = {}
            if eval_result.metrics:
                metrics_dict = {
                    "sharpe_ratio": eval_result.metrics.sharpe_ratio,
                    "max_drawdown_pct": eval_result.metrics.max_drawdown_pct,
                    "win_rate_pct": eval_result.metrics.win_rate_pct,
                    "profit_factor": eval_result.metrics.profit_factor,
                    "expected_value": eval_result.metrics.expected_value,
                    "trade_count": eval_result.metrics.trade_count,
                    "score": eval_result.score,
                }

            self._audit_logger.log_strategy_evaluation(
                strategy_name=eval_result.strategy_id,
                symbol=eval_result.asset_id,
                metrics=metrics_dict,
                status=eval_result.status.value,
                reason=eval_result.error_message,
            )
