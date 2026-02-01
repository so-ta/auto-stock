"""
Lot Size Adjuster - 購入単位（ロットサイズ）調整

リバランス時に重みを実際の購入可能な株数に変換し、
ロット単位で丸め処理を行う。

アルゴリズム（最適化アプローチ）:
1. ETF（fractional_allowed=True）は端数のまま
2. 株式は floor で切り捨て
3. 余りキャッシュで乖離が大きい順に+1ロット追加

使用例:
    from src.allocation.lot_adjuster import LotSizeAdjuster, LotAdjustmentResult

    adjuster = LotSizeAdjuster()
    result = adjuster.adjust_to_lot_size(
        target_weights={"AAPL": 0.3, "7203.T": 0.4, "SPY": 0.3},
        prices={"AAPL": 150.0, "7203.T": 2500.0, "SPY": 450.0},
        portfolio_value=1000000.0,
        lot_sizes={"AAPL": 1, "7203.T": 100, "SPY": 1},
        fractional_allowed={"AAPL": False, "7203.T": False, "SPY": True},
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LotAdjustmentResult:
    """ロット調整結果

    Attributes:
        target_shares: 各銘柄の目標株数（整数）
        fractional_shares: ETF用の端数株数
        adjusted_weights: 丸め後の実際の重み
        cash_remainder: 残りキャッシュ（丸め誤差）
        weight_deviation: 目標重みとの乖離（RMSE）
        total_invested: 投資総額
        adjustments_made: 調整内容のログ
    """

    target_shares: dict[str, int] = field(default_factory=dict)
    fractional_shares: dict[str, float] = field(default_factory=dict)
    adjusted_weights: dict[str, float] = field(default_factory=dict)
    cash_remainder: float = 0.0
    weight_deviation: float = 0.0
    total_invested: float = 0.0
    adjustments_made: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "target_shares": self.target_shares,
            "fractional_shares": self.fractional_shares,
            "adjusted_weights": self.adjusted_weights,
            "cash_remainder": self.cash_remainder,
            "weight_deviation": self.weight_deviation,
            "total_invested": self.total_invested,
            "adjustments_made": self.adjustments_made,
        }

    def get_shares(self, symbol: str) -> int | float:
        """銘柄の株数を取得（整数 or 端数）"""
        if symbol in self.fractional_shares:
            return self.fractional_shares[symbol]
        return self.target_shares.get(symbol, 0)


class LotSizeAdjuster:
    """ロットサイズ調整クラス

    重みを実際の購入可能な株数に変換し、ロット単位で丸め処理を行う。
    最適化アプローチ: floor後に余りキャッシュで乖離が大きい銘柄に+1ロット。

    Example:
        >>> adjuster = LotSizeAdjuster()
        >>> result = adjuster.adjust_to_lot_size(
        ...     target_weights={"AAPL": 0.5, "7203.T": 0.5},
        ...     prices={"AAPL": 150.0, "7203.T": 2500.0},
        ...     portfolio_value=1000000.0,
        ...     lot_sizes={"AAPL": 1, "7203.T": 100},
        ...     fractional_allowed={"AAPL": False, "7203.T": False},
        ... )
        >>> print(result.target_shares)
        {'AAPL': 3333, '7203.T': 200}
    """

    def __init__(self, min_position_value: float = 0.0) -> None:
        """初期化

        Args:
            min_position_value: 最小ポジション金額（これ以下は0株に）
        """
        self.min_position_value = min_position_value

    def adjust_to_lot_size(
        self,
        target_weights: dict[str, float],
        prices: dict[str, float],
        portfolio_value: float,
        lot_sizes: dict[str, int] | None = None,
        fractional_allowed: dict[str, bool] | None = None,
    ) -> LotAdjustmentResult:
        """重みを購入可能な株数に変換し、調整後の重みを返す

        アルゴリズム:
        1. ETF（fractional_allowed=True）は端数のまま計算
        2. 株式は floor で切り捨て
        3. 余りキャッシュで乖離が大きい順に+1ロット追加

        Args:
            target_weights: 目標重み（symbol -> weight, 合計1.0）
            prices: 各銘柄の価格（symbol -> price）
            portfolio_value: ポートフォリオ総額
            lot_sizes: 各銘柄の最小購入単位（Noneの場合は全て1）
            fractional_allowed: 各銘柄の端数購入可否（Noneの場合は全てFalse）

        Returns:
            LotAdjustmentResult: 調整結果
        """
        if lot_sizes is None:
            lot_sizes = {symbol: 1 for symbol in target_weights}
        if fractional_allowed is None:
            fractional_allowed = {symbol: False for symbol in target_weights}

        result = LotAdjustmentResult()
        result.adjustments_made = []

        # Step 1: 各銘柄の目標金額と理論株数を計算
        target_amounts: dict[str, float] = {}
        theoretical_shares: dict[str, float] = {}

        for symbol, weight in target_weights.items():
            if symbol not in prices or prices[symbol] <= 0:
                result.adjustments_made.append(f"{symbol}: 価格が無効、スキップ")
                continue

            target_amount = portfolio_value * weight
            target_amounts[symbol] = target_amount
            theoretical_shares[symbol] = target_amount / prices[symbol]

        # Step 2: 端数購入可能な銘柄と整数株のみの銘柄を分離
        fractional_symbols: list[str] = []
        integer_symbols: list[str] = []

        for symbol in theoretical_shares:
            if fractional_allowed.get(symbol, False):
                fractional_symbols.append(symbol)
            else:
                integer_symbols.append(symbol)

        # Step 3: 端数購入可能な銘柄はそのまま
        for symbol in fractional_symbols:
            shares = theoretical_shares[symbol]
            if self.min_position_value > 0:
                position_value = shares * prices[symbol]
                if position_value < self.min_position_value:
                    result.adjustments_made.append(
                        f"{symbol}: 最小金額未満、0株に"
                    )
                    shares = 0.0
            result.fractional_shares[symbol] = shares

        # Step 4: 整数株のみの銘柄は floor で切り捨て
        floor_shares: dict[str, int] = {}
        floor_remainders: dict[str, float] = {}  # 切り捨てで失った金額

        for symbol in integer_symbols:
            lot = lot_sizes.get(symbol, 1)
            theoretical = theoretical_shares[symbol]

            # ロット単位で切り捨て
            lots = math.floor(theoretical / lot)
            shares = lots * lot

            # 最小金額チェック
            if self.min_position_value > 0:
                position_value = shares * prices[symbol]
                if position_value < self.min_position_value:
                    result.adjustments_made.append(
                        f"{symbol}: 最小金額未満、0株に"
                    )
                    shares = 0

            floor_shares[symbol] = shares

            # 切り捨てで失った金額を記録
            actual_amount = shares * prices[symbol]
            target_amount = target_amounts[symbol]
            floor_remainders[symbol] = target_amount - actual_amount

        # Step 5: 余りキャッシュを計算
        total_used = sum(
            floor_shares[s] * prices[s] for s in integer_symbols
        ) + sum(
            result.fractional_shares[s] * prices[s] for s in fractional_symbols
        )
        remaining_cash = portfolio_value - total_used

        # Step 6: 余りキャッシュで乖離が大きい銘柄に+1ロット追加
        # 乖離が大きい順（切り捨てで失った金額が大きい順）にソート
        sorted_symbols = sorted(
            integer_symbols,
            key=lambda s: floor_remainders.get(s, 0),
            reverse=True,
        )

        for symbol in sorted_symbols:
            lot = lot_sizes.get(symbol, 1)
            cost_per_lot = lot * prices[symbol]

            if remaining_cash >= cost_per_lot:
                floor_shares[symbol] += lot
                remaining_cash -= cost_per_lot
                result.adjustments_made.append(
                    f"{symbol}: +{lot}株追加（乖離最小化）"
                )

        # Step 7: 結果を格納
        result.target_shares = floor_shares
        result.cash_remainder = remaining_cash

        # Step 8: 調整後の重みを計算
        total_invested = 0.0
        for symbol, shares in result.target_shares.items():
            amount = shares * prices[symbol]
            total_invested += amount
            result.adjusted_weights[symbol] = amount / portfolio_value

        for symbol, shares in result.fractional_shares.items():
            amount = shares * prices[symbol]
            total_invested += amount
            result.adjusted_weights[symbol] = amount / portfolio_value

        result.total_invested = total_invested

        # Step 9: 目標重みとの乖離（RMSE）を計算
        squared_errors = []
        for symbol, target_weight in target_weights.items():
            actual_weight = result.adjusted_weights.get(symbol, 0.0)
            squared_errors.append((target_weight - actual_weight) ** 2)

        if squared_errors:
            result.weight_deviation = math.sqrt(sum(squared_errors) / len(squared_errors))

        return result

    def calculate_required_capital(
        self,
        target_weights: dict[str, float],
        prices: dict[str, float],
        lot_sizes: dict[str, int] | None = None,
        fractional_allowed: dict[str, bool] | None = None,
    ) -> dict[str, float]:
        """各銘柄の最小必要資金を計算

        1ロット購入するのに必要な金額を返す。

        Args:
            target_weights: 目標重み
            prices: 各銘柄の価格
            lot_sizes: 各銘柄の最小購入単位
            fractional_allowed: 各銘柄の端数購入可否

        Returns:
            symbol -> 最小必要資金の辞書
        """
        if lot_sizes is None:
            lot_sizes = {symbol: 1 for symbol in target_weights}
        if fractional_allowed is None:
            fractional_allowed = {symbol: False for symbol in target_weights}

        required = {}
        for symbol in target_weights:
            if symbol not in prices:
                continue

            if fractional_allowed.get(symbol, False):
                # 端数購入可能なら最小金額は設定次第
                required[symbol] = self.min_position_value if self.min_position_value > 0 else 0.01
            else:
                # 整数株のみなら1ロット分
                lot = lot_sizes.get(symbol, 1)
                required[symbol] = lot * prices[symbol]

        return required
