"""
Portfolio Manager - ポートフォリオ管理

ポートフォリオの設定読み込み、保有資産管理、リバランス計算を統合。

使用例:
    from src.portfolio.manager import PortfolioManager

    manager = PortfolioManager()
    portfolios = manager.list_portfolios()
    holdings = manager.load_holdings("japan_stocks")
    result = manager.calculate_rebalance("japan_stocks", target_weights)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.portfolio.config import PortfolioConfig
from src.portfolio.holdings import Holdings
from src.portfolio.history import PortfolioHistory

logger = logging.getLogger(__name__)


@dataclass
class RebalanceResult:
    """リバランス結果

    Attributes:
        portfolio_id: ポートフォリオID
        timestamp: 実行日時
        needs_rebalance: リバランスが必要か
        target_weights: 目標重み
        current_weights: 現在重み
        orders: 発注リスト（OrderSummary）
        adjustment: ロット調整結果（LotAdjustmentResult）
        message: メッセージ
    """

    portfolio_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    needs_rebalance: bool = False
    target_weights: dict[str, float] = field(default_factory=dict)
    current_weights: dict[str, float] = field(default_factory=dict)
    orders: Any = None  # OrderSummary
    adjustment: Any = None  # LotAdjustmentResult
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "portfolio_id": self.portfolio_id,
            "timestamp": self.timestamp.isoformat(),
            "needs_rebalance": self.needs_rebalance,
            "target_weights": self.target_weights,
            "current_weights": self.current_weights,
            "orders": self.orders.to_dict() if self.orders else None,
            "adjustment": self.adjustment.to_dict() if self.adjustment else None,
            "message": self.message,
        }


class PortfolioManager:
    """ポートフォリオ管理クラス

    ポートフォリオの設定読み込み、保有資産管理、リバランス計算を統合。

    使用例:
        manager = PortfolioManager()
        portfolios = manager.list_portfolios()
        holdings = manager.load_holdings("japan_stocks")
    """

    def __init__(
        self,
        config_dir: str | Path = "config/portfolios",
        state_dir: str | Path = "data/portfolio_state",
    ):
        """初期化

        Args:
            config_dir: ポートフォリオ設定ディレクトリ
            state_dir: 保有資産状態ディレクトリ
        """
        self.config_dir = Path(config_dir)
        self.state_dir = Path(state_dir)

        # ディレクトリを作成
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self._config_cache: dict[str, PortfolioConfig] = {}

    def list_portfolios(self) -> list[PortfolioConfig]:
        """ポートフォリオ一覧を取得

        Returns:
            PortfolioConfigのリスト
        """
        portfolios = []

        for yaml_file in self.config_dir.glob("*.yaml"):
            try:
                config = PortfolioConfig.from_yaml(yaml_file)
                self._config_cache[config.id] = config
                portfolios.append(config)
            except Exception as e:
                logger.warning(f"Failed to load portfolio config {yaml_file}: {e}")
                continue

        return sorted(portfolios, key=lambda p: p.name)

    def get_portfolio_config(self, portfolio_id: str) -> PortfolioConfig | None:
        """ポートフォリオ設定を取得

        Args:
            portfolio_id: ポートフォリオID

        Returns:
            PortfolioConfig or None
        """
        # キャッシュを確認
        if portfolio_id in self._config_cache:
            return self._config_cache[portfolio_id]

        # ファイルから読み込み
        yaml_path = self.config_dir / f"{portfolio_id}.yaml"
        if not yaml_path.exists():
            return None

        try:
            config = PortfolioConfig.from_yaml(yaml_path)
            self._config_cache[portfolio_id] = config
            return config
        except Exception as e:
            logger.error(f"Failed to load portfolio config: {e}")
            return None

    def save_portfolio_config(self, config: PortfolioConfig) -> bool:
        """ポートフォリオ設定を保存

        Args:
            config: PortfolioConfig

        Returns:
            成功したらTrue
        """
        try:
            yaml_path = self.config_dir / f"{config.id}.yaml"
            config.to_yaml(yaml_path)
            self._config_cache[config.id] = config
            logger.info(f"Portfolio config saved: {config.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save portfolio config: {e}")
            return False

    def delete_portfolio(self, portfolio_id: str) -> bool:
        """ポートフォリオを削除

        Args:
            portfolio_id: ポートフォリオID

        Returns:
            成功したらTrue
        """
        # 設定ファイルを削除
        yaml_path = self.config_dir / f"{portfolio_id}.yaml"
        if yaml_path.exists():
            try:
                yaml_path.unlink()
            except OSError as e:
                logger.error(f"Failed to delete config file: {e}")
                return False

        # キャッシュから削除
        self._config_cache.pop(portfolio_id, None)

        logger.info(f"Portfolio deleted: {portfolio_id}")
        return True

    def load_holdings(self, portfolio_id: str) -> Holdings | None:
        """保有資産を読み込み

        Args:
            portfolio_id: ポートフォリオID

        Returns:
            Holdings or None
        """
        return Holdings.load(portfolio_id, self.state_dir)

    def save_holdings(self, holdings: Holdings) -> bool:
        """保有資産を保存

        Args:
            holdings: Holdings

        Returns:
            成功したらTrue
        """
        return holdings.save(self.state_dir)

    def get_or_create_holdings(self, portfolio_id: str) -> Holdings:
        """保有資産を取得（なければ作成）

        Args:
            portfolio_id: ポートフォリオID

        Returns:
            Holdings
        """
        holdings = self.load_holdings(portfolio_id)

        if holdings is None:
            config = self.get_portfolio_config(portfolio_id)
            initial_capital = config.initial_capital if config else 0.0
            currency = config.currency if config else "JPY"
            holdings = Holdings.create_empty(portfolio_id, initial_capital, currency)

        return holdings

    def update_prices(
        self,
        portfolio_id: str,
        prices: dict[str, float],
    ) -> Holdings | None:
        """保有資産の価格を更新

        Args:
            portfolio_id: ポートフォリオID
            prices: symbol -> price の辞書

        Returns:
            更新されたHoldings or None
        """
        holdings = self.load_holdings(portfolio_id)
        if holdings is None:
            return None

        holdings.update_prices(prices)
        self.save_holdings(holdings)

        return holdings

    def calculate_rebalance(
        self,
        portfolio_id: str,
        target_weights: dict[str, float],
        prices: dict[str, float] | None = None,
        threshold: float = 0.01,
    ) -> RebalanceResult:
        """リバランスを計算

        Args:
            portfolio_id: ポートフォリオID
            target_weights: 目標重み
            prices: 価格（未指定時は保有資産の現在価格を使用）
            threshold: リバランス閾値（重み乖離がこれ以上で実行）

        Returns:
            RebalanceResult
        """
        from src.allocation.lot_adjuster import LotSizeAdjuster
        from src.allocation.order_generator import OrderGenerator

        # 設定と保有資産を読み込み
        config = self.get_portfolio_config(portfolio_id)
        holdings = self.get_or_create_holdings(portfolio_id)

        if config is None:
            return RebalanceResult(
                portfolio_id=portfolio_id,
                needs_rebalance=False,
                message="Portfolio config not found",
            )

        # 価格が未指定なら保有資産から取得
        if prices is None:
            prices = {
                symbol: pos.current_price
                for symbol, pos in holdings.positions.items()
            }

        # ロットサイズと端数許可の辞書を作成
        lot_sizes = {}
        fractional_allowed = {}
        for symbol in target_weights:
            lot_sizes[symbol] = config.get_lot_size(symbol)
            fractional_allowed[symbol] = config.is_fractional_allowed(symbol)

        # 現在の重みを取得
        current_weights = holdings.get_current_weights()

        # 乖離をチェック
        max_deviation = 0.0
        for symbol, target in target_weights.items():
            current = current_weights.get(symbol, 0.0)
            deviation = abs(target - current)
            if deviation > max_deviation:
                max_deviation = deviation

        # 閾値未満ならリバランス不要
        if max_deviation < threshold:
            return RebalanceResult(
                portfolio_id=portfolio_id,
                needs_rebalance=False,
                target_weights=target_weights,
                current_weights=current_weights,
                message=f"Max deviation {max_deviation:.2%} < threshold {threshold:.2%}",
            )

        # ロット調整
        adjuster = LotSizeAdjuster(min_position_value=0.0)
        adjustment = adjuster.adjust_to_lot_size(
            target_weights=target_weights,
            prices=prices,
            portfolio_value=holdings.total_value,
            lot_sizes=lot_sizes,
            fractional_allowed=fractional_allowed,
        )

        # 発注リスト生成
        generator = OrderGenerator(min_order_amount=0, min_order_shares=0)
        orders = generator.generate_orders(
            current_positions=holdings.get_current_shares(),
            target_result=adjustment,
            prices=prices,
            lot_sizes=lot_sizes,
        )

        return RebalanceResult(
            portfolio_id=portfolio_id,
            needs_rebalance=True,
            target_weights=target_weights,
            current_weights=current_weights,
            orders=orders,
            adjustment=adjustment,
            message=f"Rebalance needed: {orders.buy_count} buys, {orders.sell_count} sells",
        )

    def apply_rebalance(
        self,
        portfolio_id: str,
        result: RebalanceResult,
    ) -> Holdings | None:
        """リバランス結果を保有資産に適用

        注意: これは約定を前提とした仮更新。実際の約定後に手動で調整が必要。

        Args:
            portfolio_id: ポートフォリオID
            result: RebalanceResult

        Returns:
            更新されたHoldings or None
        """
        if not result.needs_rebalance or result.adjustment is None:
            return None

        holdings = self.get_or_create_holdings(portfolio_id)
        config = self.get_portfolio_config(portfolio_id)

        # 目標株数を適用
        for symbol, shares in result.adjustment.target_shares.items():
            if symbol in holdings.positions:
                # 既存ポジションを更新
                holdings.positions[symbol].shares = shares
                holdings.positions[symbol].market_value = (
                    shares * holdings.positions[symbol].current_price
                )
            elif shares > 0:
                # 新規ポジションを追加
                price = result.adjustment.adjusted_weights.get(symbol, 0) * holdings.total_value / shares if shares > 0 else 0
                lot_size = config.get_lot_size(symbol) if config else 1
                holdings.add_position(
                    symbol=symbol,
                    shares=shares,
                    avg_cost=price,
                    current_price=price,
                    lot_size=lot_size,
                )

        # 端数株も適用
        for symbol, shares in result.adjustment.fractional_shares.items():
            if symbol in holdings.positions:
                holdings.positions[symbol].shares = shares
                holdings.positions[symbol].market_value = (
                    shares * holdings.positions[symbol].current_price
                )
                holdings.positions[symbol].is_fractional = True
            elif shares > 0:
                price = result.adjustment.adjusted_weights.get(symbol, 0) * holdings.total_value / shares if shares > 0 else 0
                holdings.add_position(
                    symbol=symbol,
                    shares=shares,
                    avg_cost=price,
                    current_price=price,
                    lot_size=1,
                    is_fractional=True,
                )

        # 0株になったポジションを削除
        symbols_to_remove = [
            symbol for symbol, pos in holdings.positions.items()
            if pos.shares <= 0
        ]
        for symbol in symbols_to_remove:
            holdings.remove_position(symbol)

        # 残りキャッシュを設定
        holdings.cash = result.adjustment.cash_remainder

        # メタデータを更新
        holdings.metadata["last_rebalance"] = datetime.now().isoformat()
        holdings.metadata["rebalance_count"] = holdings.metadata.get("rebalance_count", 0) + 1

        # 保存
        self.save_holdings(holdings)

        return holdings

    def get_history(self, portfolio_id: str) -> PortfolioHistory:
        """履歴管理オブジェクトを取得

        Args:
            portfolio_id: ポートフォリオID

        Returns:
            PortfolioHistory
        """
        return PortfolioHistory(portfolio_id, self.state_dir)

    def record_snapshot(
        self,
        portfolio_id: str,
        is_rebalance: bool = False,
    ) -> None:
        """日次スナップショットを記録

        Args:
            portfolio_id: ポートフォリオID
            is_rebalance: リバランス実行日かどうか
        """
        holdings = self.load_holdings(portfolio_id)
        if holdings is None:
            logger.warning(f"Cannot record snapshot: no holdings for {portfolio_id}")
            return

        config = self.get_portfolio_config(portfolio_id)
        initial_capital = config.initial_capital if config else None

        history = self.get_history(portfolio_id)
        history.record_snapshot(holdings, is_rebalance, initial_capital)
