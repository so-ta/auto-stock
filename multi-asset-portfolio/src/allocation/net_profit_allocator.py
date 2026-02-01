"""
Net Profit Allocator - ネット期待利益ベースの資産配分

期待リターンから取引コストを差し引いた「ネット期待利益」に基づき、
シンプルかつ透明性の高い資産配分を行う。

設計思想:
- 複数の重複するコスト抑制メカニズムを統合
- ネット期待利益 = 期待リターン - リバランスコスト
- ネット期待利益 > 0 の銘柄のみ保有
- 取引回数の抑制は不要（コストは既に織り込み済み）

カテゴリ別コスト対応:
- 米国株式: $2（固定額）
- 日本株式: 0.1%（パーセンテージ）
- ETF: 0.2%（パーセンテージ）

使用例:
    from src.allocation.net_profit_allocator import NetProfitAllocator
    from src.allocation.transaction_cost import DEFAULT_COST_SCHEDULE
    from src.data.asset_master import load_asset_master

    # カテゴリ別コストを使用
    allocator = NetProfitAllocator(
        cost_schedule=DEFAULT_COST_SCHEDULE,
        asset_master=load_asset_master(),
        max_weight=0.20,
    )

    result = allocator.allocate(
        expected_returns={"AAPL": 0.12, "7203.T": 0.08, "SPY": 0.05},
        current_weights={"AAPL": 0.3},
        portfolio_value=100000,  # 固定額コストの計算に必要
    )

    # 後方互換性: 従来のbps指定も可能
    allocator = NetProfitAllocator(
        transaction_cost_bps=20.0,
        max_weight=0.20,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .transaction_cost import (
    TransactionCostConfig,
    TransactionCostSchedule,
)

if TYPE_CHECKING:
    from src.data.asset_master import AssetMaster


@dataclass
class NetProfitAllocationResult:
    """ネット期待利益ベースの配分結果

    Attributes:
        weights: 最終ウェイト配分（銘柄 -> ウェイト）
        expected_returns: 各銘柄の期待リターン（リバランス期間換算後）
        net_profits: 各銘柄のネット期待利益（期待リターン - コスト）
        transaction_costs: 各銘柄の予測取引コスト
        turnover: 片道ターンオーバー
        expected_portfolio_return: ポートフォリオ期待リターン
        metadata: 追加メタデータ
    """

    weights: dict[str, float]
    expected_returns: dict[str, float]
    net_profits: dict[str, float]
    transaction_costs: dict[str, float]
    turnover: float
    expected_portfolio_return: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "weights": self.weights,
            "expected_returns": self.expected_returns,
            "net_profits": self.net_profits,
            "transaction_costs": self.transaction_costs,
            "turnover": self.turnover,
            "expected_portfolio_return": self.expected_portfolio_return,
            "metadata": self.metadata,
        }


class NetProfitAllocator:
    """ネット期待利益に基づく資産配分

    期待リターンから取引コストを差し引いたネット期待利益を計算し、
    その比率または均等配分で資産配分を決定する。

    カテゴリ別コストに対応:
    - cost_schedule を指定すると、カテゴリごとに異なるコストを適用
    - asset_master を指定すると、銘柄からカテゴリを自動判定
    - 後方互換性: transaction_cost_bps のみ指定も可能

    Args:
        transaction_cost_bps: 取引コスト（bps）。後方互換性のため残す。デフォルト20bps。
        cost_schedule: カテゴリ別取引コストスケジュール。指定時はこちらを優先。
        asset_master: 銘柄マスタ。カテゴリ自動判定に使用。
        max_weight: 単一銘柄の最大ウェイト。デフォルト0.20（20%）。
        min_net_profit: ネット期待利益の閾値。これ以下は現金。デフォルト0.0。
        allocation_method: 配分方法。"proportional"（比例）または"equal"（均等）。
        periods_per_year: 年間リバランス回数。月次=12, 週次=52, 日次=252。

    Example:
        >>> # カテゴリ別コストを使用
        >>> from src.allocation.transaction_cost import DEFAULT_COST_SCHEDULE
        >>> allocator = NetProfitAllocator(
        ...     cost_schedule=DEFAULT_COST_SCHEDULE,
        ...     max_weight=0.20,
        ... )
        >>> result = allocator.allocate(
        ...     expected_returns={"AAPL": 0.12, "GOOG": 0.08},
        ...     current_weights={"AAPL": 0.5},
        ...     portfolio_value=100000,
        ... )
        >>> print(result.weights)
    """

    def __init__(
        self,
        transaction_cost_bps: float = 20.0,
        cost_schedule: TransactionCostSchedule | None = None,
        asset_master: "AssetMaster | None" = None,
        max_weight: float = 0.20,
        min_net_profit: float = 0.0,
        allocation_method: str = "proportional",
        periods_per_year: int = 12,
    ) -> None:
        if transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps must be non-negative")
        if not 0 < max_weight <= 1.0:
            raise ValueError("max_weight must be between 0 and 1")
        if allocation_method not in ("proportional", "equal"):
            raise ValueError("allocation_method must be 'proportional' or 'equal'")
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")

        self.transaction_cost_bps = transaction_cost_bps
        self.cost_rate = transaction_cost_bps / 10000
        self.max_weight = max_weight
        self.min_net_profit = min_net_profit
        self.allocation_method = allocation_method
        self.periods_per_year = periods_per_year

        # カテゴリ別コスト設定
        if cost_schedule is not None:
            self.cost_schedule = cost_schedule
            self._use_category_costs = True
        else:
            # 後方互換性: bpsからデフォルトスケジュールを作成
            self.cost_schedule = TransactionCostSchedule(
                default=TransactionCostConfig(cost_type="bps", value=transaction_cost_bps)
            )
            self._use_category_costs = False

        self.asset_master = asset_master

    def allocate(
        self,
        expected_returns: pd.Series | dict[str, float],
        current_weights: dict[str, float] | None = None,
        portfolio_value: float | None = None,
        categories: dict[str, str] | None = None,
    ) -> NetProfitAllocationResult:
        """ネット期待利益に基づいて資産配分を決定

        1. 各銘柄の期待リターンを取得（年率 → リバランス期間に換算）
        2. リバランスコスト = |目標ウェイト - 現ウェイト| × コスト率
           （カテゴリ別コストを使用する場合は銘柄ごとに異なるコスト率）
        3. ネット期待利益 = 期待リターン - リバランスコスト
        4. ネット期待利益 > 閾値 の銘柄のみ保有
        5. ネット期待利益の比率（または均等）で配分

        Args:
            expected_returns: 各銘柄の期待リターン（年率）
            current_weights: 現在のウェイト配分。Noneの場合は全て0。
            portfolio_value: ポートフォリオ総額。固定額コストの計算に使用。
            categories: 銘柄→カテゴリのマッピング。指定時はasset_masterより優先。

        Returns:
            NetProfitAllocationResult: 配分結果
        """
        if current_weights is None:
            current_weights = {}

        if isinstance(expected_returns, dict):
            expected_returns = pd.Series(expected_returns)

        if expected_returns.empty:
            return NetProfitAllocationResult(
                weights={"CASH": 1.0},
                expected_returns={},
                net_profits={},
                transaction_costs={},
                turnover=sum(current_weights.values()),
                expected_portfolio_return=0.0,
                metadata={"reason": "empty_expected_returns"},
            )

        # 年率 → リバランス期間に換算
        period_returns = expected_returns / self.periods_per_year

        # 全銘柄のネット期待利益を計算（第1パス: 仮の目標ウェイトで計算）
        net_profits: dict[str, float] = {}
        transaction_costs: dict[str, float] = {}

        for symbol in period_returns.index:
            exp_ret = period_returns[symbol]
            curr_weight = current_weights.get(symbol, 0.0)

            # 初回推定: 期待リターン正なら保有、負なら売却と仮定
            tentative_target = self.max_weight if exp_ret > 0 else 0.0

            # 取引コスト = ウェイト変化 × コスト率
            weight_change = abs(tentative_target - curr_weight)

            # カテゴリを取得
            category = self._get_category(symbol, categories)

            # カテゴリ別コスト計算
            cost = self.cost_schedule.calculate_cost(
                symbol=symbol,
                weight_change=weight_change,
                portfolio_value=portfolio_value,
                category=category,
            )

            # ネット期待利益
            net_profit = exp_ret - cost
            net_profits[symbol] = net_profit
            transaction_costs[symbol] = cost

        # ネット期待利益 > 閾値 の銘柄のみ選択
        selected = {s: p for s, p in net_profits.items() if p > self.min_net_profit}

        if not selected:
            # 全銘柄が閾値以下なら現金100%
            all_symbols = set(period_returns.index) | set(current_weights.keys())
            total_turnover = sum(current_weights.values()) / 2

            return NetProfitAllocationResult(
                weights={"CASH": 1.0},
                expected_returns=period_returns.to_dict(),
                net_profits=net_profits,
                transaction_costs=transaction_costs,
                turnover=total_turnover,
                expected_portfolio_return=0.0,
                metadata={"reason": "all_below_threshold"},
            )

        # 配分方法に応じてウェイトを決定
        if self.allocation_method == "equal":
            # 均等配分
            raw_weight = 1.0 / len(selected)
            weights = {s: min(raw_weight, self.max_weight) for s in selected}
        else:
            # 比例配分（ネット期待利益に比例）
            total_profit = sum(selected.values())
            if total_profit <= 0:
                # 全てが非正の場合は均等配分にフォールバック
                raw_weight = 1.0 / len(selected)
                weights = {s: min(raw_weight, self.max_weight) for s in selected}
            else:
                weights = {s: p / total_profit for s, p in selected.items()}
                # max_weight制約を適用
                weights = {s: min(w, self.max_weight) for s, w in weights.items()}

        # 正規化（max_weight制約後の合計が1未満なら残りはCASH）
        total = sum(weights.values())
        if total < 1.0:
            weights["CASH"] = 1.0 - total
        elif total > 1.0:
            # max_weight制約で合計が1を超える場合は正規化
            weights = {s: w / total for s, w in weights.items()}

        # 取引コストの再計算（最終ウェイトベース）
        final_transaction_costs: dict[str, float] = {}
        for symbol in period_returns.index:
            curr_weight = current_weights.get(symbol, 0.0)
            new_weight = weights.get(symbol, 0.0)
            weight_change = abs(new_weight - curr_weight)

            # カテゴリ別コスト計算
            category = self._get_category(symbol, categories)
            cost = self.cost_schedule.calculate_cost(
                symbol=symbol,
                weight_change=weight_change,
                portfolio_value=portfolio_value,
                category=category,
            )
            final_transaction_costs[symbol] = cost

        # ターンオーバー計算
        all_symbols = set(weights.keys()) | set(current_weights.keys())
        all_symbols.discard("CASH")
        turnover = (
            sum(
                abs(weights.get(s, 0.0) - current_weights.get(s, 0.0))
                for s in all_symbols
            )
            / 2
        )

        # 期待ポートフォリオリターン（ネット）
        exp_port_ret = sum(
            period_returns.get(s, 0.0) * w for s, w in weights.items() if s != "CASH"
        )

        return NetProfitAllocationResult(
            weights=weights,
            expected_returns=period_returns.to_dict(),
            net_profits=net_profits,
            transaction_costs=final_transaction_costs,
            turnover=turnover,
            expected_portfolio_return=exp_port_ret,
        )

    def _get_category(
        self,
        symbol: str,
        categories: dict[str, str] | None = None,
    ) -> str | None:
        """銘柄のカテゴリを取得

        Args:
            symbol: ティッカーシンボル
            categories: 銘柄→カテゴリのマッピング（優先）

        Returns:
            カテゴリ名（不明な場合はNone）
        """
        # 明示的に指定されたカテゴリを優先
        if categories and symbol in categories:
            return categories[symbol]

        # asset_masterから取得
        if self.asset_master is not None:
            info = self.asset_master.get(symbol)
            if info.category:
                return info.category

        return None

    def calculate_net_profit(
        self,
        expected_return: float,
        current_weight: float,
        target_weight: float,
        symbol: str | None = None,
        category: str | None = None,
        portfolio_value: float | None = None,
    ) -> tuple[float, float]:
        """単一銘柄のネット期待利益を計算

        Args:
            expected_return: 期待リターン（リバランス期間ベース）
            current_weight: 現在のウェイト
            target_weight: 目標ウェイト
            symbol: ティッカーシンボル（カテゴリ別コスト用）
            category: カテゴリ（明示指定、symbolより優先）
            portfolio_value: ポートフォリオ総額（固定額コスト用）

        Returns:
            tuple[float, float]: (ネット期待利益, 取引コスト)
        """
        weight_change = abs(target_weight - current_weight)

        if symbol is not None or category is not None:
            # カテゴリ別コスト計算
            cost = self.cost_schedule.calculate_cost(
                symbol=symbol or "",
                weight_change=weight_change,
                portfolio_value=portfolio_value,
                category=category,
            )
        else:
            # 後方互換性: デフォルトコストレートを使用
            cost = weight_change * self.cost_rate

        net_profit = expected_return - cost
        return net_profit, cost


def create_net_profit_allocator(
    transaction_cost_bps: float = 20.0,
    cost_schedule: TransactionCostSchedule | None = None,
    asset_master: "AssetMaster | None" = None,
    max_weight: float = 0.20,
    min_net_profit: float = 0.0,
    allocation_method: str = "proportional",
    periods_per_year: int = 12,
) -> NetProfitAllocator:
    """NetProfitAllocatorのファクトリ関数

    Args:
        transaction_cost_bps: 取引コスト（bps）。後方互換性用。
        cost_schedule: カテゴリ別取引コストスケジュール。
        asset_master: 銘柄マスタ。カテゴリ自動判定に使用。
        max_weight: 単一銘柄の最大ウェイト
        min_net_profit: ネット期待利益の閾値
        allocation_method: 配分方法 ("proportional" or "equal")
        periods_per_year: 年間リバランス回数

    Returns:
        NetProfitAllocator: 設定済みのアロケータ
    """
    return NetProfitAllocator(
        transaction_cost_bps=transaction_cost_bps,
        cost_schedule=cost_schedule,
        asset_master=asset_master,
        max_weight=max_weight,
        min_net_profit=min_net_profit,
        allocation_method=allocation_method,
        periods_per_year=periods_per_year,
    )
