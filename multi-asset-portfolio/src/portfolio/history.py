"""
Portfolio History - パフォーマンス履歴追跡

毎日のアセット配分・評価額を記録し、推移を追跡する。

使用例:
    from src.portfolio.history import PortfolioHistory

    history = PortfolioHistory("japan_stocks")
    history.record_snapshot(holdings, is_rebalance=True)
    snapshots = history.get_history(start_date, end_date)
    metrics = history.get_performance_metrics()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Parquetはオプショナル（なければJSONにフォールバック）
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class PositionSnapshot:
    """ポジションスナップショット

    Attributes:
        symbol: ティッカーシンボル
        shares: 保有株数
        price: 価格
        market_value: 時価
        weight: ポートフォリオ内比率
    """

    symbol: str
    shares: int | float
    price: float
    market_value: float
    weight: float

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "symbol": self.symbol,
            "shares": self.shares,
            "price": self.price,
            "market_value": self.market_value,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PositionSnapshot":
        """辞書から作成"""
        return cls(
            symbol=data["symbol"],
            shares=data["shares"],
            price=data["price"],
            market_value=data["market_value"],
            weight=data["weight"],
        )


@dataclass
class DailySnapshot:
    """日次スナップショット

    Attributes:
        date: 日付
        portfolio_id: ポートフォリオID
        total_value: 総資産価値
        cash: 現金残高
        positions: ポジションスナップショット辞書
        daily_return: 前日比リターン
        cumulative_return: 累積リターン（初期資本比）
        is_rebalance_day: リバランス実行日かどうか
        is_dynamic: 動的計算されたスナップショットかどうか
    """

    date: date
    portfolio_id: str
    total_value: float
    cash: float
    positions: dict[str, PositionSnapshot] = field(default_factory=dict)
    daily_return: float | None = None
    cumulative_return: float = 0.0
    is_rebalance_day: bool = False
    is_dynamic: bool = False

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "date": self.date.isoformat(),
            "portfolio_id": self.portfolio_id,
            "total_value": self.total_value,
            "cash": self.cash,
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "daily_return": self.daily_return,
            "cumulative_return": self.cumulative_return,
            "is_rebalance_day": self.is_rebalance_day,
            "is_dynamic": self.is_dynamic,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DailySnapshot":
        """辞書から作成"""
        date_val = data["date"]
        if isinstance(date_val, str):
            date_val = date.fromisoformat(date_val)

        positions = {}
        for symbol, pos_data in data.get("positions", {}).items():
            positions[symbol] = PositionSnapshot.from_dict(pos_data)

        return cls(
            date=date_val,
            portfolio_id=data["portfolio_id"],
            total_value=data["total_value"],
            cash=data["cash"],
            positions=positions,
            daily_return=data.get("daily_return"),
            cumulative_return=data.get("cumulative_return", 0.0),
            is_rebalance_day=data.get("is_rebalance_day", False),
            is_dynamic=data.get("is_dynamic", False),
        )


@dataclass
class PerformanceMetrics:
    """パフォーマンス指標

    Attributes:
        total_return: 総リターン
        annualized_return: 年率リターン
        volatility: ボラティリティ（年率）
        sharpe_ratio: シャープレシオ
        max_drawdown: 最大ドローダウン
        win_rate: 勝率（日次ベース）
        rebalance_count: リバランス回数
        start_date: 開始日
        end_date: 終了日
    """

    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    rebalance_count: int = 0
    start_date: date | None = None
    end_date: date | None = None

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "rebalance_count": self.rebalance_count,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
        }


class PortfolioHistory:
    """ポートフォリオ履歴管理

    日次スナップショットの記録・取得、パフォーマンス指標の計算を行う。

    使用例:
        history = PortfolioHistory("japan_stocks")
        history.record_snapshot(holdings, is_rebalance=True)
        snapshots = history.get_history(start_date, end_date)
    """

    def __init__(
        self,
        portfolio_id: str,
        base_dir: str | Path = "data/portfolio_state",
    ):
        """初期化

        Args:
            portfolio_id: ポートフォリオID
            base_dir: 保存先ベースディレクトリ
        """
        self.portfolio_id = portfolio_id
        self.base_dir = Path(base_dir)
        self.history_dir = self.base_dir / portfolio_id / "history"
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def record_snapshot(
        self,
        holdings: Any,  # Holdings型を遅延インポートで使用
        is_rebalance: bool = False,
        initial_capital: float | None = None,
    ) -> DailySnapshot:
        """日次スナップショットを記録

        Args:
            holdings: Holdings オブジェクト
            is_rebalance: リバランス実行日かどうか
            initial_capital: 初期資本（累積リターン計算用）

        Returns:
            記録されたDailySnapshot
        """
        today = date.today()

        # ポジションスナップショットを作成
        positions = {}
        for symbol, pos in holdings.positions.items():
            positions[symbol] = PositionSnapshot(
                symbol=symbol,
                shares=pos.shares,
                price=pos.current_price,
                market_value=pos.market_value,
                weight=pos.weight,
            )

        # 前日のスナップショットを取得
        previous = self._get_previous_snapshot(today)

        # 日次リターンを計算
        daily_return = None
        if previous and previous.total_value > 0:
            daily_return = (holdings.total_value / previous.total_value) - 1

        # 累積リターンを計算
        cumulative_return = 0.0
        if initial_capital and initial_capital > 0:
            cumulative_return = (holdings.total_value / initial_capital) - 1
        elif previous:
            # 初期資本が不明な場合は前日の累積リターンから計算
            if daily_return is not None:
                cumulative_return = (1 + previous.cumulative_return) * (1 + daily_return) - 1
            else:
                cumulative_return = previous.cumulative_return

        snapshot = DailySnapshot(
            date=today,
            portfolio_id=self.portfolio_id,
            total_value=holdings.total_value,
            cash=holdings.cash,
            positions=positions,
            daily_return=daily_return,
            cumulative_return=cumulative_return,
            is_rebalance_day=is_rebalance,
        )

        self._save_snapshot(snapshot)
        return snapshot

    def get_history(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[DailySnapshot]:
        """期間の履歴を取得

        Args:
            start_date: 開始日（None=全期間）
            end_date: 終了日（None=全期間）

        Returns:
            DailySnapshotのリスト（日付昇順）
        """
        snapshots = []

        # 月別ファイルをスキャン
        for json_file in sorted(self.history_dir.glob("*.json")):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)

                for record in data:
                    snapshot = DailySnapshot.from_dict(record)

                    # 期間フィルタ
                    if start_date and snapshot.date < start_date:
                        continue
                    if end_date and snapshot.date > end_date:
                        continue

                    snapshots.append(snapshot)

            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load history file {json_file}: {e}")
                continue

        return sorted(snapshots, key=lambda s: s.date)

    def get_performance_metrics(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        risk_free_rate: float = 0.0,
    ) -> PerformanceMetrics:
        """パフォーマンス指標を計算

        Args:
            start_date: 開始日
            end_date: 終了日
            risk_free_rate: 無リスク金利（年率）

        Returns:
            PerformanceMetrics
        """
        snapshots = self.get_history(start_date, end_date)

        if len(snapshots) < 2:
            return PerformanceMetrics()

        # 日次リターンを収集
        daily_returns = [s.daily_return for s in snapshots if s.daily_return is not None]

        if not daily_returns:
            return PerformanceMetrics(
                start_date=snapshots[0].date,
                end_date=snapshots[-1].date,
            )

        # 総リターン
        total_return = snapshots[-1].cumulative_return

        # 年率リターン
        days = (snapshots[-1].date - snapshots[0].date).days
        if days > 0:
            annualized_return = (1 + total_return) ** (365 / days) - 1
        else:
            annualized_return = 0.0

        # ボラティリティ（年率）
        import math
        mean_return = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_return) ** 2 for r in daily_returns) / len(daily_returns)
        daily_volatility = math.sqrt(variance)
        volatility = daily_volatility * math.sqrt(252)  # 年率換算

        # シャープレシオ
        if volatility > 0:
            excess_return = annualized_return - risk_free_rate
            sharpe_ratio = excess_return / volatility
        else:
            sharpe_ratio = 0.0

        # 最大ドローダウン
        max_value = snapshots[0].total_value
        max_drawdown = 0.0
        for snapshot in snapshots:
            if snapshot.total_value > max_value:
                max_value = snapshot.total_value
            drawdown = (max_value - snapshot.total_value) / max_value
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # 勝率
        positive_days = sum(1 for r in daily_returns if r > 0)
        win_rate = positive_days / len(daily_returns) if daily_returns else 0.0

        # リバランス回数
        rebalance_count = sum(1 for s in snapshots if s.is_rebalance_day)

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            rebalance_count=rebalance_count,
            start_date=snapshots[0].date,
            end_date=snapshots[-1].date,
        )

    def get_latest_snapshot(self) -> DailySnapshot | None:
        """最新のスナップショットを取得

        Returns:
            DailySnapshot or None
        """
        snapshots = self.get_history()
        return snapshots[-1] if snapshots else None

    def export_to_csv(self, output_path: str | Path) -> bool:
        """履歴をCSVにエクスポート

        Args:
            output_path: 出力先パス

        Returns:
            成功したらTrue
        """
        snapshots = self.get_history()

        if not snapshots:
            logger.warning("No history to export")
            return False

        try:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)

            with open(output, "w", encoding="utf-8") as f:
                # ヘッダー
                f.write("date,total_value,cash,daily_return,cumulative_return,is_rebalance_day\n")

                for snapshot in snapshots:
                    daily_ret = f"{snapshot.daily_return:.6f}" if snapshot.daily_return else ""
                    f.write(
                        f"{snapshot.date},"
                        f"{snapshot.total_value:.2f},"
                        f"{snapshot.cash:.2f},"
                        f"{daily_ret},"
                        f"{snapshot.cumulative_return:.6f},"
                        f"{snapshot.is_rebalance_day}\n"
                    )

            logger.info(f"History exported to {output}")
            return True

        except OSError as e:
            logger.error(f"Failed to export history: {e}")
            return False

    def _get_previous_snapshot(self, current_date: date) -> DailySnapshot | None:
        """前日のスナップショットを取得（内部用）"""
        snapshots = self.get_history(end_date=current_date)

        # 当日を除く最新を返す
        for snapshot in reversed(snapshots):
            if snapshot.date < current_date:
                return snapshot

        return None

    def _save_snapshot(self, snapshot: DailySnapshot) -> None:
        """スナップショットを保存（内部用）

        月別ファイルに追記する形式で保存。
        """
        month_str = snapshot.date.strftime("%Y-%m")
        month_file = self.history_dir / f"{month_str}.json"

        # 既存データを読み込み
        existing = []
        if month_file.exists():
            try:
                with open(month_file, encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = []

        # 同じ日付のデータがあれば上書き
        new_data = [
            record for record in existing
            if record.get("date") != snapshot.date.isoformat()
        ]
        new_data.append(snapshot.to_dict())

        # 日付でソート
        new_data.sort(key=lambda x: x["date"])

        # 保存
        try:
            with open(month_file, "w", encoding="utf-8") as f:
                json.dump(new_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Snapshot saved: {snapshot.date}")
        except OSError as e:
            logger.error(f"Failed to save snapshot: {e}")

    # =========================================================================
    # スナップショット編集機能
    # =========================================================================

    def get_snapshot_by_date(self, target_date: date) -> DailySnapshot | None:
        """特定日付のスナップショットを取得

        Args:
            target_date: 取得する日付

        Returns:
            DailySnapshot or None
        """
        snapshots = self.get_history(start_date=target_date, end_date=target_date)
        return snapshots[0] if snapshots else None

    def update_snapshot(
        self,
        target_date: date,
        cash: float,
        positions: dict[str, dict],
        is_rebalance: bool | None = None,
    ) -> DailySnapshot | None:
        """既存スナップショットを更新

        Args:
            target_date: 更新対象の日付
            cash: 更新後の現金残高
            positions: 更新後のポジション情報（symbol -> {shares, price} の辞書）
            is_rebalance: リバランス日かどうか（Noneの場合は既存値を維持）

        Returns:
            更新後のDailySnapshot or None
        """
        existing = self.get_snapshot_by_date(target_date)

        # 新しいポジションスナップショットを作成
        position_snapshots = {}
        total_position_value = 0.0

        for symbol, pos_data in positions.items():
            shares = pos_data.get("shares", 0)
            price = pos_data.get("price", 0)
            market_value = shares * price
            total_position_value += market_value

            position_snapshots[symbol] = PositionSnapshot(
                symbol=symbol,
                shares=shares,
                price=price,
                market_value=market_value,
                weight=0,  # 後で計算
            )

        total_value = total_position_value + cash

        # 重みを計算
        if total_value > 0:
            for pos in position_snapshots.values():
                pos.weight = pos.market_value / total_value

        # 前日のスナップショットを取得して日次リターンを計算
        previous = self._get_previous_snapshot(target_date)
        daily_return = None
        if previous and previous.total_value > 0:
            daily_return = (total_value / previous.total_value) - 1

        # 累積リターンを計算
        cumulative_return = 0.0
        if previous:
            if daily_return is not None:
                cumulative_return = (1 + previous.cumulative_return) * (1 + daily_return) - 1
            else:
                cumulative_return = previous.cumulative_return

        # リバランスフラグの決定
        if is_rebalance is None:
            is_rebalance = existing.is_rebalance_day if existing else False

        snapshot = DailySnapshot(
            date=target_date,
            portfolio_id=self.portfolio_id,
            total_value=total_value,
            cash=cash,
            positions=position_snapshots,
            daily_return=daily_return,
            cumulative_return=cumulative_return,
            is_rebalance_day=is_rebalance,
            is_dynamic=False,
        )

        self._save_snapshot(snapshot)

        # 後続のスナップショットの累積リターンを再計算
        self._recalculate_cumulative_returns(target_date)

        logger.info(f"Snapshot updated: {target_date}")
        return snapshot

    def delete_snapshot(self, target_date: date) -> bool:
        """スナップショットを削除

        Args:
            target_date: 削除対象の日付

        Returns:
            成功したらTrue
        """
        month_str = target_date.strftime("%Y-%m")
        month_file = self.history_dir / f"{month_str}.json"

        if not month_file.exists():
            logger.warning(f"No history file for {month_str}")
            return False

        try:
            with open(month_file, encoding="utf-8") as f:
                existing = json.load(f)

            # 対象日付のデータを削除
            new_data = [
                record for record in existing
                if record.get("date") != target_date.isoformat()
            ]

            if len(new_data) == len(existing):
                logger.warning(f"Snapshot not found for {target_date}")
                return False

            # 保存
            with open(month_file, "w", encoding="utf-8") as f:
                json.dump(new_data, f, indent=2, ensure_ascii=False)

            # 後続のスナップショットの累積リターンを再計算
            self._recalculate_cumulative_returns(target_date)

            logger.info(f"Snapshot deleted: {target_date}")
            return True

        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to delete snapshot: {e}")
            return False

    def _recalculate_cumulative_returns(self, from_date: date) -> None:
        """指定日以降の累積リターンを再計算（内部用）

        Args:
            from_date: 再計算開始日
        """
        snapshots = self.get_history(start_date=from_date)
        if not snapshots:
            return

        # 前日のスナップショットを取得
        previous = self._get_previous_snapshot(from_date)

        for i, snapshot in enumerate(snapshots):
            if i == 0:
                if previous and previous.total_value > 0:
                    daily_return = (snapshot.total_value / previous.total_value) - 1
                    cumulative_return = (1 + previous.cumulative_return) * (1 + daily_return) - 1
                else:
                    daily_return = None
                    cumulative_return = 0.0
            else:
                prev = snapshots[i - 1]
                if prev.total_value > 0:
                    daily_return = (snapshot.total_value / prev.total_value) - 1
                    cumulative_return = (1 + prev.cumulative_return) * (1 + daily_return) - 1
                else:
                    daily_return = None
                    cumulative_return = prev.cumulative_return

            snapshot.daily_return = daily_return
            snapshot.cumulative_return = cumulative_return
            self._save_snapshot(snapshot)

    # =========================================================================
    # スナップショット補完機能
    # =========================================================================

    def get_filled_history(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        fill_missing: bool = True,
    ) -> list[DailySnapshot]:
        """日付範囲の履歴を取得し、欠けている日を動的計算で補完する

        Args:
            start_date: 開始日
            end_date: 終了日
            fill_missing: True = 欠けている日を補完

        Returns:
            連続した日次スナップショットのリスト（補完含む）
        """
        snapshots = self.get_history(start_date, end_date)

        if not fill_missing or not snapshots:
            return snapshots

        # 開始日と終了日を決定
        # 実スナップショットの範囲と今日の日付を考慮
        if snapshots:
            snapshot_dates = [s.date for s in snapshots]
            actual_start = start_date or min(snapshot_dates)
            actual_end = end_date or max(date.today(), max(snapshot_dates))
        else:
            actual_start = start_date or date.today()
            actual_end = end_date or date.today()

        return self._fill_missing_snapshots(snapshots, actual_start, actual_end)

    def _fill_missing_snapshots(
        self,
        snapshots: list[DailySnapshot],
        start_date: date,
        end_date: date,
    ) -> list[DailySnapshot]:
        """欠けている日を前日ポジション+当日価格で補完（内部用）

        Args:
            snapshots: 既存のスナップショット
            start_date: 開始日
            end_date: 終了日

        Returns:
            補完済みスナップショットリスト
        """
        from datetime import timedelta

        filled = []
        snapshot_map = {s.date: s for s in snapshots}

        current = start_date
        prev_snapshot = self._get_previous_snapshot(start_date)

        while current <= end_date:
            if current in snapshot_map:
                # スナップショットあり → そのまま使用
                snapshot = snapshot_map[current]
                filled.append(snapshot)
                prev_snapshot = snapshot
            elif prev_snapshot:
                # スナップショットなし → 動的計算
                dynamic = self._calculate_dynamic_snapshot(prev_snapshot, current)
                filled.append(dynamic)
                # 動的計算されたスナップショットは prev_snapshot として使用しない
                # 価格更新なしで次の日も同じ基準を使う
            current += timedelta(days=1)

        return filled

    def _calculate_dynamic_snapshot(
        self,
        prev_snapshot: DailySnapshot,
        target_date: date,
    ) -> DailySnapshot:
        """前日のポジションと当日の価格から動的にスナップショットを計算（内部用）

        Args:
            prev_snapshot: 前日のスナップショット
            target_date: 計算対象日

        Returns:
            動的計算されたDailySnapshot
        """
        # 当日の価格を取得
        prices = self._fetch_prices_for_date(target_date, list(prev_snapshot.positions.keys()))

        # 前日のポジション × 当日価格で時価を計算
        positions = {}
        total_position_value = 0.0

        for symbol, pos in prev_snapshot.positions.items():
            # 価格取得失敗時は前日価格を使用
            price = prices.get(symbol, pos.price)
            market_value = pos.shares * price
            positions[symbol] = PositionSnapshot(
                symbol=symbol,
                shares=pos.shares,
                price=price,
                market_value=market_value,
                weight=0,  # 後で計算
            )
            total_position_value += market_value

        total_value = total_position_value + prev_snapshot.cash

        # 重みを計算
        if total_value > 0:
            for pos in positions.values():
                pos.weight = pos.market_value / total_value

        # 日次リターンを計算
        daily_return = None
        if prev_snapshot.total_value > 0:
            daily_return = (total_value / prev_snapshot.total_value) - 1

        # 累積リターンを計算
        cumulative_return = prev_snapshot.cumulative_return
        if daily_return is not None:
            cumulative_return = (1 + prev_snapshot.cumulative_return) * (1 + daily_return) - 1

        return DailySnapshot(
            date=target_date,
            portfolio_id=prev_snapshot.portfolio_id,
            total_value=total_value,
            cash=prev_snapshot.cash,
            positions=positions,
            daily_return=daily_return,
            cumulative_return=cumulative_return,
            is_rebalance_day=False,
            is_dynamic=True,
        )

    def _fetch_prices_for_date(
        self,
        target_date: date,
        symbols: list[str],
    ) -> dict[str, float]:
        """指定日付の価格を取得（内部用）

        Args:
            target_date: 対象日付
            symbols: 銘柄シンボルのリスト

        Returns:
            symbol -> price の辞書
        """
        if not symbols:
            return {}

        try:
            from src.data.batch_fetcher import BatchDataFetcher
            from datetime import timedelta
            import asyncio

            fetcher = BatchDataFetcher(
                max_concurrent=5,
                rate_limit_per_sec=2.0,
                cache_max_age_days=1,
            )

            # 対象日付を含む期間を取得（数日間のバッファを持たせる）
            end_date_str = target_date.strftime("%Y-%m-%d")
            start_date_str = (target_date - timedelta(days=5)).strftime("%Y-%m-%d")

            # 非同期処理をsyncで実行
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # すでにイベントループが動いている場合はfetch_all_syncを使用
                    batch_result = fetcher.fetch_all_sync(symbols, start_date_str, end_date_str)
                else:
                    batch_result = loop.run_until_complete(
                        fetcher.fetch_all(symbols, start_date_str, end_date_str)
                    )
            except RuntimeError:
                # イベントループがない場合
                batch_result = asyncio.run(
                    fetcher.fetch_all(symbols, start_date_str, end_date_str)
                )

            prices = {}

            # BatchResult型の場合
            if hasattr(batch_result, 'results'):
                for symbol, result in batch_result.results.items():
                    if result.data is not None and len(result.data) > 0:
                        df = result.data
                        # 対象日付以前の最新価格を取得
                        df_filtered = df[df.index <= target_date.strftime("%Y-%m-%d")]
                        if len(df_filtered) > 0:
                            close_col = None
                            for col in df_filtered.columns:
                                if 'close' in str(col).lower():
                                    close_col = col
                                    break
                            if close_col:
                                prices[symbol] = float(df_filtered[close_col].iloc[-1])
            else:
                # dict型の場合
                for symbol, df in batch_result.items():
                    if df is not None and len(df) > 0:
                        df_filtered = df[df.index <= target_date.strftime("%Y-%m-%d")]
                        if len(df_filtered) > 0:
                            close_col = None
                            for col in df_filtered.columns:
                                if 'close' in str(col).lower():
                                    close_col = col
                                    break
                            if close_col:
                                prices[symbol] = float(df_filtered[close_col].iloc[-1])

            return prices

        except Exception as e:
            logger.warning(f"Failed to fetch prices for {target_date}: {e}")
            return {}
