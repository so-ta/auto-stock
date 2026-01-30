"""
Trading Timing Optimizer and Liquidity Scorer.

Provides tools for optimizing trade execution timing based on:
- Day-of-week effects
- Day-of-month effects
- Volatility conditions
- Liquidity analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class TimingAnalysis:
    """Results of timing effect analysis."""

    dayofweek_effects: pd.Series  # Mean returns by day of week (0=Mon, 4=Fri)
    dayofmonth_effects: pd.Series  # Mean returns by day of month
    best_buy_day: int  # Day of week with lowest returns (best for buying)
    best_sell_day: int  # Day of week with highest returns (best for selling)
    best_buy_monthday: int  # Day of month with lowest returns
    best_sell_monthday: int  # Day of month with highest returns


class TradingTimingOptimizer:
    """
    Analyzes and optimizes trade execution timing.

    Uses historical return patterns to identify optimal execution days
    based on day-of-week and day-of-month effects.
    """

    def __init__(self, analyze_lookback: int = 252):
        """
        Initialize the timing optimizer.

        Args:
            analyze_lookback: Number of days to analyze for timing effects.
                             Default 252 (approximately 1 trading year).
        """
        self.analyze_lookback = analyze_lookback
        self._timing_analysis: Optional[TimingAnalysis] = None

    def analyze_timing_effects(self, returns: pd.DataFrame) -> TimingAnalysis:
        """
        Analyze timing effects in historical returns.

        Computes day-of-week and day-of-month average returns to identify
        systematic patterns that can be exploited for better execution.

        Args:
            returns: DataFrame with datetime index and asset returns.
                    Can be single column or multiple columns (will average).

        Returns:
            TimingAnalysis with computed effects and optimal days.
        """
        if returns.empty:
            raise ValueError("Returns DataFrame is empty")

        # Use recent data based on lookback
        recent_returns = returns.tail(self.analyze_lookback)

        # If multiple columns, compute average return across assets
        if isinstance(recent_returns, pd.DataFrame) and len(recent_returns.columns) > 1:
            avg_returns = recent_returns.mean(axis=1)
        elif isinstance(recent_returns, pd.DataFrame):
            avg_returns = recent_returns.iloc[:, 0]
        else:
            avg_returns = recent_returns

        # Ensure we have a datetime index
        if not isinstance(avg_returns.index, pd.DatetimeIndex):
            try:
                avg_returns.index = pd.to_datetime(avg_returns.index)
            except Exception as e:
                raise ValueError(f"Cannot convert index to datetime: {e}")

        # Day of week effects (0=Monday, 4=Friday)
        dayofweek_effects = avg_returns.groupby(avg_returns.index.dayofweek).mean()
        dayofweek_effects.index = dayofweek_effects.index.astype(int)

        # Day of month effects
        dayofmonth_effects = avg_returns.groupby(avg_returns.index.day).mean()
        dayofmonth_effects.index = dayofmonth_effects.index.astype(int)

        # Find optimal days
        # For buying: want lowest return days (buy low)
        # For selling: want highest return days (sell high)
        best_buy_day = int(dayofweek_effects.idxmin())
        best_sell_day = int(dayofweek_effects.idxmax())
        best_buy_monthday = int(dayofmonth_effects.idxmin())
        best_sell_monthday = int(dayofmonth_effects.idxmax())

        self._timing_analysis = TimingAnalysis(
            dayofweek_effects=dayofweek_effects,
            dayofmonth_effects=dayofmonth_effects,
            best_buy_day=best_buy_day,
            best_sell_day=best_sell_day,
            best_buy_monthday=best_buy_monthday,
            best_sell_monthday=best_sell_monthday,
        )

        return self._timing_analysis

    def get_optimal_execution_day(self, trade_direction: str = "buy") -> int:
        """
        Get the optimal day of week for execution.

        Args:
            trade_direction: "buy" or "sell"

        Returns:
            Day of week (0=Monday, 4=Friday) that is optimal for the trade direction.
            - buy: Returns day with lowest average returns (buy low)
            - sell: Returns day with highest average returns (sell high)

        Raises:
            ValueError: If trade_direction is invalid or analysis not run.
        """
        if self._timing_analysis is None:
            raise ValueError("Must run analyze_timing_effects first")

        trade_direction = trade_direction.lower()
        if trade_direction not in ("buy", "sell"):
            raise ValueError(f"Invalid trade_direction: {trade_direction}. Use 'buy' or 'sell'")

        if trade_direction == "buy":
            return self._timing_analysis.best_buy_day
        else:
            return self._timing_analysis.best_sell_day

    def should_delay_execution(
        self,
        current_vol: float,
        baseline_vol: float,
        vol_threshold: float = 1.5,
    ) -> tuple[bool, str]:
        """
        Determine if execution should be delayed due to high volatility.

        High volatility can lead to worse execution prices and higher slippage.
        This method recommends delaying trades when volatility is elevated.

        Args:
            current_vol: Current volatility measure (e.g., realized vol, VIX).
            baseline_vol: Baseline/normal volatility for comparison.
            vol_threshold: Multiplier threshold above which to recommend delay.
                          Default 1.5 means delay if current > 1.5x baseline.

        Returns:
            Tuple of (should_delay, reason):
            - should_delay: True if execution should be delayed
            - reason: Explanation string
        """
        if baseline_vol <= 0:
            return False, "Invalid baseline volatility"

        vol_ratio = current_vol / baseline_vol

        if vol_ratio > vol_threshold:
            return (
                True,
                f"High volatility: {vol_ratio:.2f}x baseline (threshold: {vol_threshold}x). "
                "Consider delaying execution.",
            )
        elif vol_ratio > vol_threshold * 0.8:
            return (
                False,
                f"Elevated volatility: {vol_ratio:.2f}x baseline. "
                "Monitor closely but execution acceptable.",
            )
        else:
            return (
                False,
                f"Normal volatility: {vol_ratio:.2f}x baseline. "
                "Proceed with execution.",
            )

    def get_day_name(self, day_num: int) -> str:
        """Convert day number to name."""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        if 0 <= day_num < len(days):
            return days[day_num]
        return f"Day {day_num}"


class LiquidityScorer:
    """
    Computes liquidity scores and adjusts portfolio weights for liquidity.

    Low liquidity assets may have higher transaction costs and slippage,
    so their weights should be reduced or capped.
    """

    def __init__(self, volume_lookback: int = 20):
        """
        Initialize the liquidity scorer.

        Args:
            volume_lookback: Number of days to use for volume averaging.
                           Default 20 (approximately 1 trading month).
        """
        self.volume_lookback = volume_lookback

    def compute_liquidity_score(self, volumes: pd.Series) -> float:
        """
        Compute a liquidity score from volume data.

        The score is normalized to 0-1 range where 1 indicates high liquidity.
        Uses a combination of:
        - Average volume level
        - Volume consistency (lower CV = more consistent = better)

        Args:
            volumes: Series of trading volumes.

        Returns:
            Liquidity score between 0 and 1 (1 = high liquidity).
        """
        if volumes.empty or len(volumes) < 2:
            return 0.0

        # Use recent volumes
        recent_volumes = volumes.tail(self.volume_lookback)

        # Filter out zero/negative volumes
        valid_volumes = recent_volumes[recent_volumes > 0]
        if len(valid_volumes) < 2:
            return 0.0

        # Compute statistics
        avg_volume = valid_volumes.mean()
        std_volume = valid_volumes.std()

        # Coefficient of variation (lower is better - more consistent)
        if avg_volume > 0:
            cv = std_volume / avg_volume
        else:
            return 0.0

        # Volume level score: log-transform and normalize
        # Assumes reasonable volume is > 100,000 shares for high score
        volume_level_score = min(1.0, np.log10(max(avg_volume, 1)) / 7)  # 10^7 = 10M shares

        # Consistency score: lower CV = higher score
        consistency_score = max(0.0, 1.0 - min(cv, 1.0))

        # Combined score (weighted average)
        liquidity_score = 0.6 * volume_level_score + 0.4 * consistency_score

        return float(np.clip(liquidity_score, 0.0, 1.0))

    def compute_liquidity_scores_batch(
        self, volumes_dict: dict[str, pd.Series]
    ) -> dict[str, float]:
        """
        Compute liquidity scores for multiple assets.

        Args:
            volumes_dict: Dictionary mapping symbol to volume Series.

        Returns:
            Dictionary mapping symbol to liquidity score.
        """
        return {symbol: self.compute_liquidity_score(vols) for symbol, vols in volumes_dict.items()}

    def adjust_weights_for_liquidity(
        self,
        weights: dict[str, float],
        liquidity_scores: dict[str, float],
        min_liquidity: float = 0.3,
    ) -> dict[str, float]:
        """
        Adjust portfolio weights based on liquidity scores.

        Assets with liquidity below min_liquidity will have their weights
        reduced proportionally, with the excess redistributed to more
        liquid assets.

        Args:
            weights: Dictionary of asset weights (should sum to ~1.0).
            liquidity_scores: Dictionary of liquidity scores per asset (0-1).
            min_liquidity: Minimum acceptable liquidity score.
                          Assets below this will have reduced weights.

        Returns:
            Adjusted weights dictionary, normalized to sum to 1.0.
        """
        if not weights:
            return {}

        adjusted = {}
        total_reduction = 0.0
        liquid_assets_weight = 0.0

        # First pass: identify reductions needed
        for symbol, weight in weights.items():
            liq_score = liquidity_scores.get(symbol, 0.5)  # Default to medium liquidity

            if liq_score < min_liquidity:
                # Reduce weight proportionally to how far below threshold
                reduction_factor = liq_score / min_liquidity
                adjusted_weight = weight * reduction_factor
                total_reduction += weight - adjusted_weight
                adjusted[symbol] = adjusted_weight
            else:
                adjusted[symbol] = weight
                liquid_assets_weight += weight

        # Second pass: redistribute reduced weight to liquid assets
        if total_reduction > 0 and liquid_assets_weight > 0:
            for symbol in adjusted:
                liq_score = liquidity_scores.get(symbol, 0.5)
                if liq_score >= min_liquidity:
                    # Add proportional share of reduced weight
                    original_weight = weights[symbol]
                    share = original_weight / liquid_assets_weight
                    adjusted[symbol] += total_reduction * share

        # Normalize to sum to 1.0
        total_weight = sum(adjusted.values())
        if total_weight > 0:
            adjusted = {k: v / total_weight for k, v in adjusted.items()}

        return adjusted

    def get_liquidity_report(
        self, liquidity_scores: dict[str, float], min_liquidity: float = 0.3
    ) -> dict:
        """
        Generate a report on liquidity conditions.

        Args:
            liquidity_scores: Dictionary of liquidity scores per asset.
            min_liquidity: Threshold for flagging low liquidity.

        Returns:
            Dictionary with liquidity analysis.
        """
        if not liquidity_scores:
            return {"status": "no_data", "assets": {}}

        low_liquidity = {k: v for k, v in liquidity_scores.items() if v < min_liquidity}
        high_liquidity = {k: v for k, v in liquidity_scores.items() if v >= 0.7}
        medium_liquidity = {
            k: v for k, v in liquidity_scores.items() if min_liquidity <= v < 0.7
        }

        avg_score = np.mean(list(liquidity_scores.values()))

        return {
            "status": "healthy" if len(low_liquidity) == 0 else "warning",
            "average_score": float(avg_score),
            "high_liquidity": high_liquidity,
            "medium_liquidity": medium_liquidity,
            "low_liquidity": low_liquidity,
            "low_liquidity_count": len(low_liquidity),
            "recommendation": (
                "All assets have acceptable liquidity"
                if len(low_liquidity) == 0
                else f"Consider reducing exposure to: {list(low_liquidity.keys())}"
            ),
        }
