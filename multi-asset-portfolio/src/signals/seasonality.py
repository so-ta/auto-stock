"""
Seasonality Signals Module.

Provides calendar-based and seasonal signals including:
- Day of Week Effect (Monday/Friday anomalies)
- Month Effect (January effect, etc.)
- Turn of Month Effect

⚠️ OVERFITTING WARNING ⚠️
==========================
Seasonality signals are HIGHLY PRONE TO OVERFITTING.

These patterns:
1. May be data-mined artifacts rather than real market inefficiencies
2. Often disappear after discovery due to market efficiency
3. Vary significantly across different markets and time periods
4. Should be used with EXTREME CAUTION and robust out-of-sample testing

Recommendations:
- Use long lookback periods for statistical significance
- Require high confidence thresholds before acting
- Consider these signals as very weak evidence only
- Always validate on out-of-sample data from different time periods
- Use conservative position sizing when acting on these signals

All signals output normalized values in [-1, +1] range.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.signals.base import ParameterSpec, Signal, SignalResult
from src.signals.registry import SignalRegistry


@SignalRegistry.register(
    "day_of_week",
    category="seasonality",
    description="Day-of-week effect signal (Monday/Friday anomalies)",
    tags=["seasonality", "calendar", "anomaly"],
)
class DayOfWeekSignal(Signal):
    """
    Day of Week Effect Signal.

    Captures day-of-week anomalies such as:
    - Monday Effect: Historically negative returns on Mondays
    - Friday Effect: Historically positive returns on Fridays

    ⚠️ OVERFITTING WARNING: These effects may be spurious or time-varying.
    Academic research shows these anomalies have weakened significantly
    since their initial discovery. Use with extreme caution.

    Output interpretation:
    - Positive: Historically favorable day
    - Negative: Historically unfavorable day
    - The magnitude reflects statistical confidence in historical data
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="lookback",
                default=252,
                searchable=False,
                min_value=126,
                max_value=756,
                description="Historical lookback for calculating day effects",
            ),
            ParameterSpec(
                name="min_samples",
                default=20,
                searchable=False,
                min_value=10,
                max_value=50,
                description="Minimum samples per day for statistical validity",
            ),
            ParameterSpec(
                name="confidence_scale",
                default=0.5,
                searchable=False,
                min_value=0.1,
                max_value=1.0,
                description="Scaling factor for confidence (lower = more conservative)",
            ),
        ]

    def compute(self, data: pd.DataFrame) -> SignalResult:
        """
        Compute day-of-week effect signal.

        Args:
            data: DataFrame with 'close' column and DatetimeIndex

        Returns:
            SignalResult with day-of-week scores
        """
        self.validate_input(data)

        lookback = self._params["lookback"]
        min_samples = self._params["min_samples"]
        confidence_scale = self._params["confidence_scale"]

        close = data["close"]
        returns = close.pct_change()

        # Get day of week (0=Monday, 4=Friday)
        day_of_week = data.index.dayofweek

        scores = pd.Series(index=data.index, dtype=float)

        for i in range(len(data)):
            if i < lookback:
                scores.iloc[i] = 0.0
                continue

            # Get historical returns for each day of week
            historical_window = returns.iloc[max(0, i - lookback) : i]
            historical_days = day_of_week[max(0, i - lookback) : i]

            current_day = day_of_week[i]

            # Calculate mean return for current day of week
            day_returns = historical_window[historical_days == current_day]

            if len(day_returns) < min_samples:
                scores.iloc[i] = 0.0
                continue

            day_mean = day_returns.mean()
            day_std = day_returns.std()

            # Calculate overall mean for comparison
            overall_mean = historical_window.mean()
            overall_std = historical_window.std()

            if overall_std == 0 or day_std == 0:
                scores.iloc[i] = 0.0
                continue

            # Calculate z-score of day's average return vs overall
            # and apply confidence scaling
            excess_return = day_mean - overall_mean
            z_score = excess_return / (day_std / np.sqrt(len(day_returns)))

            # Apply confidence scale and tanh normalization
            raw_score = z_score * confidence_scale
            scores.iloc[i] = np.tanh(raw_score * 0.5)

        return SignalResult(
            scores=scores,
            metadata={
                "signal_type": "day_of_week",
                "lookback": lookback,
                "min_samples": min_samples,
                "confidence_scale": confidence_scale,
                "warning": "OVERFITTING RISK: Day-of-week effects may be spurious",
            },
        )


@SignalRegistry.register(
    "month_effect",
    category="seasonality",
    description="Month-of-year effect signal (January effect, etc.)",
    tags=["seasonality", "calendar", "anomaly"],
)
class MonthEffectSignal(Signal):
    """
    Month Effect Signal.

    Captures month-of-year anomalies such as:
    - January Effect: Historically higher returns in January
    - Sell in May: Historically weaker summer months
    - Santa Claus Rally: Strong late December performance

    ⚠️ OVERFITTING WARNING: Monthly effects are especially prone to
    data mining bias. The January effect has largely disappeared in
    recent decades. Use with extreme caution.

    Output interpretation:
    - Positive: Historically favorable month
    - Negative: Historically unfavorable month
    - The magnitude reflects statistical confidence
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="lookback_years",
                default=5,
                searchable=False,
                min_value=3,
                max_value=20,
                description="Historical lookback in years",
            ),
            ParameterSpec(
                name="min_years",
                default=3,
                searchable=False,
                min_value=2,
                max_value=10,
                description="Minimum years of data per month for validity",
            ),
            ParameterSpec(
                name="confidence_scale",
                default=0.3,
                searchable=False,
                min_value=0.1,
                max_value=1.0,
                description="Scaling factor for confidence (lower = more conservative)",
            ),
        ]

    def compute(self, data: pd.DataFrame) -> SignalResult:
        """
        Compute month effect signal.

        Args:
            data: DataFrame with 'close' column and DatetimeIndex

        Returns:
            SignalResult with month effect scores
        """
        self.validate_input(data)

        lookback_years = self._params["lookback_years"]
        min_years = self._params["min_years"]
        confidence_scale = self._params["confidence_scale"]

        close = data["close"]
        returns = close.pct_change()

        # Get month (1=Jan, 12=Dec)
        months = data.index.month
        years = data.index.year

        lookback_days = lookback_years * 252  # Approximate trading days per year

        scores = pd.Series(index=data.index, dtype=float)

        for i in range(len(data)):
            if i < lookback_days:
                scores.iloc[i] = 0.0
                continue

            # Get historical returns for each month
            historical_window = returns.iloc[max(0, i - lookback_days) : i]
            historical_months = months[max(0, i - lookback_days) : i]
            historical_years = years[max(0, i - lookback_days) : i]

            current_month = months[i]

            # Get returns for current month across different years
            month_mask = historical_months == current_month
            month_returns = historical_window[month_mask]

            # Count unique years
            unique_years = historical_years[month_mask].nunique()

            if unique_years < min_years or len(month_returns) < 10:
                scores.iloc[i] = 0.0
                continue

            month_mean = month_returns.mean()
            month_std = month_returns.std()

            # Calculate overall statistics
            overall_mean = historical_window.mean()
            overall_std = historical_window.std()

            if overall_std == 0 or month_std == 0:
                scores.iloc[i] = 0.0
                continue

            # Calculate excess return and statistical significance
            excess_return = month_mean - overall_mean
            se = month_std / np.sqrt(len(month_returns))
            t_stat = excess_return / se if se > 0 else 0

            # Apply conservative confidence scaling
            raw_score = t_stat * confidence_scale
            scores.iloc[i] = np.tanh(raw_score * 0.3)

        return SignalResult(
            scores=scores,
            metadata={
                "signal_type": "month_effect",
                "lookback_years": lookback_years,
                "min_years": min_years,
                "confidence_scale": confidence_scale,
                "warning": "OVERFITTING RISK: Month effects are highly unreliable",
            },
        )


@SignalRegistry.register(
    "turn_of_month",
    category="seasonality",
    description="Turn-of-month effect signal (month-end/month-start anomaly)",
    tags=["seasonality", "calendar", "anomaly"],
)
class TurnOfMonthSignal(Signal):
    """
    Turn of Month Effect Signal.

    Captures the turn-of-month anomaly:
    - Last few days and first few days of each month tend to
      have higher returns than mid-month periods
    - Often attributed to institutional fund flows and paycheck effects

    ⚠️ OVERFITTING WARNING: While more robust than other calendar effects,
    this anomaly still requires careful validation. Transaction costs and
    market impact may eliminate any practical edge.

    Output interpretation:
    - Positive: Near month turn (last/first days)
    - Negative: Mid-month (historically weaker period)
    - Magnitude reflects distance from month turn
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="end_days",
                default=3,
                searchable=True,
                min_value=1,
                max_value=5,
                step=1,
                description="Number of days at month end to consider",
            ),
            ParameterSpec(
                name="start_days",
                default=3,
                searchable=True,
                min_value=1,
                max_value=5,
                step=1,
                description="Number of days at month start to consider",
            ),
            ParameterSpec(
                name="lookback",
                default=252,
                searchable=False,
                min_value=126,
                max_value=504,
                description="Historical lookback for validation",
            ),
            ParameterSpec(
                name="confidence_scale",
                default=0.6,
                searchable=False,
                min_value=0.1,
                max_value=1.0,
                description="Scaling factor for confidence",
            ),
        ]

    def compute(self, data: pd.DataFrame) -> SignalResult:
        """
        Compute turn-of-month effect signal.

        Args:
            data: DataFrame with 'close' column and DatetimeIndex

        Returns:
            SignalResult with turn-of-month scores
        """
        self.validate_input(data)

        end_days = self._params["end_days"]
        start_days = self._params["start_days"]
        lookback = self._params["lookback"]
        confidence_scale = self._params["confidence_scale"]

        close = data["close"]
        returns = close.pct_change()

        # Calculate day of month and days until month end
        # Convert to Series for consistent iloc/slicing access
        day_of_month = pd.Series(data.index.day, index=data.index)

        # Calculate days until end of month (as Series)
        days_in_month = pd.Series(data.index.days_in_month, index=data.index)
        days_until_end = days_in_month - day_of_month

        scores = pd.Series(index=data.index, dtype=float)

        for i in range(len(data)):
            current_dom = day_of_month.iloc[i]
            current_due = days_until_end.iloc[i]

            # Determine if we're in the turn-of-month window
            is_month_start = current_dom <= start_days
            is_month_end = current_due < end_days

            # Base score based on position
            if is_month_start:
                # Score higher for first day, decreasing
                base_score = 1.0 - (current_dom - 1) / start_days * 0.5
            elif is_month_end:
                # Score higher closer to month end
                base_score = 1.0 - current_due / end_days * 0.5
            else:
                # Mid-month: slight negative bias
                mid_point = (days_in_month.iloc[i] - start_days - end_days) / 2
                dist_from_mid = abs(current_dom - start_days - mid_point) / mid_point
                base_score = -0.3 * (1 - dist_from_mid)

            # Validate with historical data if available
            if i >= lookback:
                historical_window = returns.iloc[max(0, i - lookback) : i]
                historical_dom = day_of_month.iloc[max(0, i - lookback) : i]
                historical_due = days_until_end.iloc[max(0, i - lookback) : i]

                # Calculate average returns for turn vs mid-month
                turn_mask = (historical_dom <= start_days) | (historical_due < end_days)
                turn_returns = historical_window[turn_mask]
                mid_returns = historical_window[~turn_mask]

                if len(turn_returns) > 10 and len(mid_returns) > 10:
                    turn_mean = turn_returns.mean()
                    mid_mean = mid_returns.mean()

                    # Adjust base score by historical evidence
                    historical_diff = turn_mean - mid_mean
                    hist_adjustment = np.tanh(historical_diff * 100)  # Scale small differences

                    # Blend base score with historical evidence
                    base_score = 0.6 * base_score + 0.4 * hist_adjustment

            # Apply confidence scaling
            scores.iloc[i] = base_score * confidence_scale

        # Final normalization to ensure [-1, +1] bounds
        scores = scores.clip(-1, 1)

        return SignalResult(
            scores=scores,
            metadata={
                "signal_type": "turn_of_month",
                "end_days": end_days,
                "start_days": start_days,
                "lookback": lookback,
                "confidence_scale": confidence_scale,
                "warning": "OVERFITTING RISK: Validate with out-of-sample data",
            },
        )
