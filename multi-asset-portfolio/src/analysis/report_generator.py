"""
Report Generator - パフォーマンスレポート生成

ポートフォリオのパフォーマンスをHTMLまたはテキスト形式でレポート生成。
Jinja2テンプレートを使用したプロフェッショナルなHTML出力をサポート。

Usage:
    from src.analysis.report_generator import ReportGenerator, ComparisonResult

    # 比較結果を作成
    comparison = ComparisonResult(
        portfolio_metrics=portfolio_metrics,
        benchmark_metrics={"SPY": spy_metrics, "QQQ": qqq_metrics},
    )

    # レポート生成
    generator = ReportGenerator()
    html = generator.generate_html_report(
        comparison,
        portfolio_name="My Portfolio",
        start_date="2010-01-01",
        end_date="2025-01-01",
        output_path="report.html",
    )

    # ターミナル表示用テキスト
    text = generator.generate_text_report(comparison, "My Portfolio")
    print(text)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

# Jinja2 import with fallback
try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False
    Environment = None
    FileSystemLoader = None

if TYPE_CHECKING:
    from src.backtest.rebalance_tracker import RebalanceTracker, ForecastMetrics
    from src.data.asset_master import AssetMaster, AssetInfo
    from src.utils.storage_backend import StorageBackend


@dataclass
class PortfolioMetrics:
    """ポートフォリオメトリクス"""

    # リターン系
    total_return: float = 0.0
    annual_return: float = 0.0
    monthly_return: float = 0.0

    # リスク系
    volatility: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0  # Value at Risk (95%)
    cvar_95: float = 0.0  # Conditional VaR (95%)

    # リスク調整リターン
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0

    # その他
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # メタ情報
    n_trades: int = 0
    n_periods: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "monthly_return": self.monthly_return,
            "volatility": self.volatility,
            "max_drawdown": self.max_drawdown,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "information_ratio": self.information_ratio,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "n_trades": self.n_trades,
            "n_periods": self.n_periods,
        }


@dataclass
class ComparisonResult:
    """ポートフォリオとベンチマークの比較結果"""

    portfolio_metrics: PortfolioMetrics
    benchmark_metrics: Dict[str, PortfolioMetrics] = field(default_factory=dict)

    # オプション: 追加情報
    portfolio_name: str = "Portfolio"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    notes: str = ""

    @property
    def benchmarks(self) -> List[str]:
        """ベンチマーク名リスト"""
        return list(self.benchmark_metrics.keys())

    def get_benchmark(self, name: str) -> Optional[PortfolioMetrics]:
        """指定ベンチマークのメトリクスを取得"""
        return self.benchmark_metrics.get(name)

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "portfolio": self.portfolio_metrics.to_dict(),
            "benchmarks": {
                name: metrics.to_dict()
                for name, metrics in self.benchmark_metrics.items()
            },
            "portfolio_name": self.portfolio_name,
            "start_date": self.start_date,
            "end_date": self.end_date,
        }


class ReportGenerator:
    """
    パフォーマンスレポート生成

    HTMLレポート（Jinja2テンプレート使用）およびテキストレポートを生成。
    """

    # デフォルトテンプレートディレクトリ
    DEFAULT_TEMPLATE_DIR = Path(__file__).parent / "templates"

    def __init__(self, template_dir: Optional[Union[str, Path]] = None):
        """
        初期化

        Args:
            template_dir: Jinja2テンプレートディレクトリ（Noneでデフォルト使用）
        """
        if template_dir is None:
            self.template_dir = self.DEFAULT_TEMPLATE_DIR
        else:
            self.template_dir = Path(template_dir)

        self._jinja_env: Optional[Any] = None

    def _get_jinja_env(self) -> Any:
        """Jinja2環境を取得（遅延初期化）"""
        if not HAS_JINJA2:
            raise ImportError(
                "Jinja2 is required for HTML report generation. "
                "Install with: pip install jinja2"
            )

        if self._jinja_env is None:
            self._jinja_env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                autoescape=select_autoescape(["html", "xml"]),
            )
            # カスタムフィルター追加
            self._jinja_env.filters["format_pct"] = self._format_percentage
            self._jinja_env.filters["format_ratio"] = self._format_ratio

        return self._jinja_env

    def generate_html_report(
        self,
        comparison: ComparisonResult,
        portfolio_name: str,
        start_date: str,
        end_date: str,
        output_path: str,
        storage_backend: Optional["StorageBackend"] = None,
    ) -> str:
        """
        HTMLレポートを生成

        Args:
            comparison: 比較結果
            portfolio_name: ポートフォリオ名
            start_date: 開始日
            end_date: 終了日
            output_path: 出力ファイルパス
            storage_backend: オプショナルなStorageBackend（S3サポート用）

        Returns:
            str: 生成されたHTMLコンテンツ
        """
        # テンプレートデータ作成
        context = {
            "portfolio_name": portfolio_name,
            "start_date": start_date,
            "end_date": end_date,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": self._create_summary_section(comparison),
            "returns_table": self._create_returns_table(comparison),
            "risk_table": self._create_risk_table(comparison),
            "ratio_table": self._create_ratio_table(comparison),
            "comparison": comparison,
        }

        # テンプレートレンダリング
        html_content = None

        if HAS_JINJA2:
            try:
                env = self._get_jinja_env()
                template = env.get_template("performance_report.html")
                html_content = template.render(**context)
            except Exception:
                # テンプレートがない場合は内蔵テンプレート使用
                pass

        if html_content is None:
            # Jinja2なし or テンプレートエラー時は内蔵テンプレート使用
            html_content = self._generate_builtin_html(context)

        # ファイル出力
        if storage_backend:
            storage_backend.write_text(html_content, output_path)
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

        return html_content

    def generate_text_report(
        self,
        comparison: ComparisonResult,
        portfolio_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> str:
        """
        テキストレポートを生成（ターミナル表示用）

        Args:
            comparison: 比較結果
            portfolio_name: ポートフォリオ名
            start_date: 開始日（省略時はcomparison から取得）
            end_date: 終了日（省略時はcomparison から取得）

        Returns:
            str: テキストレポート
        """
        start = start_date or comparison.start_date or "N/A"
        end = end_date or comparison.end_date or "N/A"

        lines = []
        separator = "=" * 60

        # ヘッダー
        lines.append(separator)
        lines.append(f"パフォーマンスレポート: {portfolio_name}")
        lines.append(f"期間: {start} 〜 {end}")
        lines.append(separator)
        lines.append("")

        # サマリーセクション
        lines.append("【サマリー】")
        lines.append("")

        # テーブルヘッダー
        benchmarks = comparison.benchmarks
        header = f"{'':24s}  {'ポートフォリオ':>14s}"
        for bm in benchmarks:
            header += f"  {bm:>10s}"
        lines.append(header)
        lines.append("-" * len(header))

        # メトリクス行
        pm = comparison.portfolio_metrics

        # 年率リターン
        row = f"{'年率リターン':24s}  {self._format_percentage(pm.annual_return):>14s}"
        for bm in benchmarks:
            bm_metrics = comparison.get_benchmark(bm)
            if bm_metrics:
                row += f"  {self._format_percentage(bm_metrics.annual_return):>10s}"
        lines.append(row)

        # シャープレシオ
        row = f"{'シャープレシオ':24s}  {self._format_ratio(pm.sharpe_ratio):>14s}"
        for bm in benchmarks:
            bm_metrics = comparison.get_benchmark(bm)
            if bm_metrics:
                row += f"  {self._format_ratio(bm_metrics.sharpe_ratio):>10s}"
        lines.append(row)

        # ソルティノレシオ
        row = f"{'ソルティノレシオ':24s}  {self._format_ratio(pm.sortino_ratio):>14s}"
        for bm in benchmarks:
            bm_metrics = comparison.get_benchmark(bm)
            if bm_metrics:
                row += f"  {self._format_ratio(bm_metrics.sortino_ratio):>10s}"
        lines.append(row)

        # 最大ドローダウン
        row = f"{'最大ドローダウン':24s}  {self._format_percentage(pm.max_drawdown):>14s}"
        for bm in benchmarks:
            bm_metrics = comparison.get_benchmark(bm)
            if bm_metrics:
                row += f"  {self._format_percentage(bm_metrics.max_drawdown):>10s}"
        lines.append(row)

        # ボラティリティ
        row = f"{'ボラティリティ':24s}  {self._format_percentage(pm.volatility):>14s}"
        for bm in benchmarks:
            bm_metrics = comparison.get_benchmark(bm)
            if bm_metrics:
                row += f"  {self._format_percentage(bm_metrics.volatility):>10s}"
        lines.append(row)

        # カルマーレシオ
        row = f"{'カルマーレシオ':24s}  {self._format_ratio(pm.calmar_ratio):>14s}"
        for bm in benchmarks:
            bm_metrics = comparison.get_benchmark(bm)
            if bm_metrics:
                row += f"  {self._format_ratio(bm_metrics.calmar_ratio):>10s}"
        lines.append(row)

        lines.append("")
        lines.append(separator)

        return "\n".join(lines)

    def generate_forecast_text_section(
        self,
        tracker: "RebalanceTracker",
    ) -> str:
        """
        予測 vs 実績リターンのテキストセクションを生成

        Args:
            tracker: RebalanceTracker インスタンス

        Returns:
            str: テキストセクション
        """
        metrics = tracker.get_forecast_metrics()
        if metrics is None:
            return ""

        records_df = tracker.to_dataframe()
        if records_df.empty:
            return ""

        lines = [
            "",
            "## 予測 vs 実績リターン",
            "",
            "| リバランス日 | 予測リターン | 実績リターン | 予測誤差 | 取引コスト |",
            "|-------------|------------|------------|---------|----------|",
        ]

        for _, row in records_df.iterrows():
            expected = row.get("expected_return")
            actual = row.get("actual_return")
            error = row.get("forecast_error")
            cost = row.get("total_cost", 0)
            date_str = str(row.get("date", ""))[:10]

            exp_str = f"{expected*100:.2f}%" if expected is not None else "N/A"
            act_str = f"{actual*100:.2f}%" if actual is not None else "N/A"
            err_str = f"{error*100:+.2f}%" if error is not None else "N/A"
            cost_str = f"{cost*100:.2f}%" if cost is not None else "N/A"

            lines.append(
                f"| {date_str} | {exp_str} | {act_str} | {err_str} | {cost_str} |"
            )

        lines.extend([
            "",
            "### サマリー",
            f"- 平均予測リターン: {metrics.mean_expected*100:.2f}%",
            f"- 平均実績リターン: {metrics.mean_actual*100:.2f}%",
            f"- 平均予測誤差: {metrics.mean_error*100:+.2f}%",
            f"- 予測誤差の標準偏差: {metrics.std_error*100:.2f}%",
            f"- 予測の相関係数: {metrics.correlation:.3f}",
            f"- 総取引コスト: {metrics.total_cost*100:.2f}%",
            f"- 累計ターンオーバー: {metrics.total_turnover*100:.2f}%",
            f"- リバランス回数: {metrics.n_rebalances}",
        ])

        return "\n".join(lines)

    def generate_holdings_text_section(
        self,
        weights: Dict[str, float],
        asset_master: Optional["AssetMaster"] = None,
        title: str = "保有銘柄一覧",
    ) -> str:
        """
        保有銘柄一覧のテキストセクションを生成

        Args:
            weights: 銘柄→ウェイトの辞書
            asset_master: AssetMaster インスタンス（Noneの場合は自動読み込み）
            title: セクションタイトル

        Returns:
            str: テキストセクション
        """
        if not weights:
            return ""

        # AssetMaster読み込み
        if asset_master is None:
            try:
                from src.data.asset_master import load_asset_master
                asset_master = load_asset_master()
            except ImportError:
                asset_master = None

        # CASHを除外してソート
        sorted_weights = sorted(
            [(s, w) for s, w in weights.items() if s != "CASH" and w > 0],
            key=lambda x: -x[1]
        )

        if not sorted_weights:
            return ""

        lines = [
            "",
            f"## {title}",
            "",
            "| シンボル | 銘柄名 | 区分 | ウェイト |",
            "|---------|-------|------|---------|",
        ]

        for symbol, weight in sorted_weights:
            if asset_master:
                info = asset_master.get(symbol)
                name = info.display_name
                category = info.category_display
            else:
                name = symbol
                category = "-"

            lines.append(f"| {symbol} | {name} | {category} | {weight*100:.2f}% |")

        # CASH表示
        cash_weight = weights.get("CASH", 0)
        if cash_weight > 0:
            lines.append(f"| CASH | 現金 | - | {cash_weight*100:.2f}% |")

        # サマリー
        n_assets = len(sorted_weights)
        total_weight = sum(w for _, w in sorted_weights)
        lines.extend([
            "",
            f"- 保有銘柄数: {n_assets}",
            f"- 投資比率: {total_weight*100:.1f}%",
            f"- 現金比率: {cash_weight*100:.1f}%",
        ])

        return "\n".join(lines)

    def generate_holdings_by_category_text(
        self,
        weights: Dict[str, float],
        asset_master: Optional["AssetMaster"] = None,
    ) -> str:
        """
        カテゴリ別保有銘柄のテキストセクションを生成

        Args:
            weights: 銘柄→ウェイトの辞書
            asset_master: AssetMaster インスタンス

        Returns:
            str: テキストセクション
        """
        if not weights:
            return ""

        # AssetMaster読み込み
        if asset_master is None:
            try:
                from src.data.asset_master import load_asset_master
                asset_master = load_asset_master()
            except ImportError:
                return ""

        # カテゴリ別に集計
        category_weights: Dict[str, Dict[str, float]] = {}
        for symbol, weight in weights.items():
            if symbol == "CASH" or weight <= 0:
                continue
            info = asset_master.get(symbol)
            cat = info.category_display or "その他"
            if cat not in category_weights:
                category_weights[cat] = {}
            category_weights[cat][symbol] = weight

        if not category_weights:
            return ""

        lines = [
            "",
            "## カテゴリ別保有状況",
            "",
            "| カテゴリ | 銘柄数 | 合計ウェイト |",
            "|---------|-------|------------|",
        ]

        for cat in sorted(category_weights.keys()):
            cat_data = category_weights[cat]
            n = len(cat_data)
            total = sum(cat_data.values())
            lines.append(f"| {cat} | {n} | {total*100:.2f}% |")

        cash_weight = weights.get("CASH", 0)
        if cash_weight > 0:
            lines.append(f"| 現金 | - | {cash_weight*100:.2f}% |")

        return "\n".join(lines)

    def _generate_holdings_html_section(
        self,
        weights: Dict[str, float],
        asset_master: Optional["AssetMaster"] = None,
    ) -> str:
        """保有銘柄一覧のHTMLセクションを生成"""
        if not weights:
            return ""

        # AssetMaster読み込み
        if asset_master is None:
            try:
                from src.data.asset_master import load_asset_master
                asset_master = load_asset_master()
            except ImportError:
                asset_master = None

        # CASHを除外してソート
        sorted_weights = sorted(
            [(s, w) for s, w in weights.items() if s != "CASH" and w > 0],
            key=lambda x: -x[1]
        )

        if not sorted_weights:
            return ""

        rows_html = ""
        for symbol, weight in sorted_weights:
            if asset_master:
                info = asset_master.get(symbol)
                name = info.display_name
                category = info.category_display
            else:
                name = symbol
                category = "-"

            cat_display = f"{category}"

            rows_html += f"""
                <tr>
                    <td><strong>{symbol}</strong></td>
                    <td>{name}</td>
                    <td>{cat_display}</td>
                    <td>{weight*100:.2f}%</td>
                </tr>"""

        # CASH
        cash_weight = weights.get("CASH", 0)
        if cash_weight > 0:
            rows_html += f"""
                <tr class="highlight">
                    <td><strong>CASH</strong></td>
                    <td>現金</td>
                    <td>-</td>
                    <td>{cash_weight*100:.2f}%</td>
                </tr>"""

        n_assets = len(sorted_weights)
        total_weight = sum(w for _, w in sorted_weights)

        return f"""
        <h2>保有銘柄一覧</h2>
        <p>保有銘柄数: <strong>{n_assets}</strong> | 投資比率: <strong>{total_weight*100:.1f}%</strong> | 現金比率: <strong>{cash_weight*100:.1f}%</strong></p>
        <table>
            <thead>
                <tr>
                    <th>シンボル</th>
                    <th>銘柄名</th>
                    <th>区分</th>
                    <th>ウェイト</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        """

    def generate_html_report_with_forecast(
        self,
        comparison: ComparisonResult,
        portfolio_name: str,
        start_date: str,
        end_date: str,
        output_path: str,
        tracker: Optional["RebalanceTracker"] = None,
        final_weights: Optional[Dict[str, float]] = None,
        asset_master: Optional["AssetMaster"] = None,
    ) -> str:
        """
        予測vs実績セクションを含むHTMLレポートを生成

        Args:
            comparison: 比較結果
            portfolio_name: ポートフォリオ名
            start_date: 開始日
            end_date: 終了日
            output_path: 出力ファイルパス
            tracker: RebalanceTracker インスタンス（オプション）
            final_weights: 最終保有ウェイト（オプション）
            asset_master: AssetMaster インスタンス（オプション、自動読み込み可）

        Returns:
            str: 生成されたHTMLコンテンツ
        """
        # AssetMaster読み込み
        if asset_master is None:
            try:
                from src.data.asset_master import load_asset_master
                asset_master = load_asset_master()
            except ImportError:
                asset_master = None

        # テンプレートデータ作成
        context = {
            "portfolio_name": portfolio_name,
            "start_date": start_date,
            "end_date": end_date,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": self._create_summary_section(comparison),
            "returns_table": self._create_returns_table(comparison),
            "risk_table": self._create_risk_table(comparison),
            "ratio_table": self._create_ratio_table(comparison),
            "comparison": comparison,
            "forecast_data": None,
            "forecast_metrics": None,
            "final_weights": final_weights,
            "asset_master": asset_master,
            "stock_returns": None,
            "monthly_performance": None,
        }

        # 予測データを追加
        if tracker is not None:
            metrics = tracker.get_forecast_metrics()
            if metrics is not None:
                records_df = tracker.to_dataframe()
                context["forecast_data"] = records_df.to_dict("records")
                context["forecast_metrics"] = metrics.to_dict()

            # 最終ウェイトがない場合、最後のリバランス記録から取得
            records = tracker.get_records()
            if final_weights is None and records:
                context["final_weights"] = records[-1].weights_after

            # 銘柄別累積リターンを計算
            context["stock_returns"] = self._calculate_stock_returns(records)

            # 月次パフォーマンスを計算
            context["monthly_performance"] = self._calculate_monthly_performance(records)

        # テンプレートレンダリング
        html_content = None

        if HAS_JINJA2:
            try:
                env = self._get_jinja_env()
                template = env.get_template("performance_report.html")
                html_content = template.render(**context)
            except Exception:
                # テンプレートがない場合は内蔵テンプレート使用
                pass

        if html_content is None:
            # Jinja2なし or テンプレートエラー時は内蔵テンプレート使用
            html_content = self._generate_builtin_html_with_forecast(context)

        # ファイル出力
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return html_content

    def _generate_forecast_html_section(self, context: Dict[str, Any]) -> str:
        """予測vs実績のHTMLセクションを生成"""
        forecast_data = context.get("forecast_data")
        forecast_metrics = context.get("forecast_metrics")

        if not forecast_data or not forecast_metrics:
            return ""

        # テーブル行生成
        rows_html = ""
        for row in forecast_data:
            expected = row.get("expected_return")
            actual = row.get("actual_return")
            error = row.get("forecast_error")
            cost = row.get("total_cost", 0)
            date_str = str(row.get("date", ""))[:10]

            exp_str = f"{expected*100:.2f}%" if expected is not None else "N/A"
            act_str = f"{actual*100:.2f}%" if actual is not None else "N/A"
            err_class = "positive" if error and error > 0 else "negative" if error else ""
            err_str = f"{error*100:+.2f}%" if error is not None else "N/A"
            cost_str = f"{cost*100:.2f}%" if cost is not None else "N/A"

            rows_html += f"""
                <tr>
                    <td>{date_str}</td>
                    <td>{exp_str}</td>
                    <td>{act_str}</td>
                    <td class="{err_class}">{err_str}</td>
                    <td>{cost_str}</td>
                </tr>"""

        return f"""
        <h2>予測 vs 実績リターン</h2>
        <table>
            <thead>
                <tr>
                    <th>リバランス日</th>
                    <th>予測リターン</th>
                    <th>実績リターン</th>
                    <th>予測誤差</th>
                    <th>取引コスト</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>

        <h3>予測精度サマリー</h3>
        <table>
            <tr><td>平均予測リターン</td><td>{forecast_metrics['mean_expected']*100:.2f}%</td></tr>
            <tr><td>平均実績リターン</td><td>{forecast_metrics['mean_actual']*100:.2f}%</td></tr>
            <tr><td>平均予測誤差</td><td>{forecast_metrics['mean_error']*100:+.2f}%</td></tr>
            <tr><td>予測誤差の標準偏差</td><td>{forecast_metrics['std_error']*100:.2f}%</td></tr>
            <tr><td>予測の相関係数</td><td>{forecast_metrics['correlation']:.3f}</td></tr>
            <tr><td>総取引コスト</td><td>{forecast_metrics['total_cost']*100:.2f}%</td></tr>
            <tr><td>累計ターンオーバー</td><td>{forecast_metrics['total_turnover']*100:.2f}%</td></tr>
            <tr><td>リバランス回数</td><td>{forecast_metrics['n_rebalances']}</td></tr>
        </table>
        """

    def _generate_builtin_html_with_forecast(self, context: Dict[str, Any]) -> str:
        """予測セクションを含む内蔵HTMLテンプレート"""
        summary = context["summary"]
        portfolio_name = context["portfolio_name"]
        start_date = context["start_date"]
        end_date = context["end_date"]
        generated_at = context["generated_at"]

        # ベンチマーク列生成
        benchmarks = list(summary["benchmarks"].keys())
        bm_headers = "".join(f"<th>{bm}</th>" for bm in benchmarks)

        def get_bm_cells(metric: str, is_pct: bool = True) -> str:
            cells = ""
            for bm in benchmarks:
                val = summary["benchmarks"][bm].get(metric, 0)
                if is_pct:
                    cells += f"<td>{val*100:.2f}%</td>"
                else:
                    cells += f"<td>{val:.2f}</td>"
            return cells

        # 予測セクション
        forecast_section = self._generate_forecast_html_section(context)

        # 保有銘柄セクション
        holdings_section = ""
        final_weights = context.get("final_weights")
        asset_master = context.get("asset_master")
        if final_weights:
            holdings_section = self._generate_holdings_html_section(final_weights, asset_master)

        html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>パフォーマンスレポート - {portfolio_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        h1 {{
            color: #1a1a2e;
            border-bottom: 3px solid #4a69bd;
            padding-bottom: 15px;
            margin-bottom: 25px;
        }}
        .meta {{
            color: #666;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #4a69bd;
            margin: 25px 0 15px;
            padding-bottom: 8px;
            border-bottom: 1px solid #ddd;
        }}
        h3 {{
            color: #666;
            margin: 20px 0 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 25px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: right;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background-color: #4a69bd;
            color: white;
            font-weight: 600;
        }}
        th:first-child, td:first-child {{
            text-align: left;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .positive {{
            color: #27ae60;
        }}
        .negative {{
            color: #e74c3c;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #999;
            font-size: 0.9em;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>パフォーマンスレポート</h1>
        <div class="meta">
            <p><strong>ポートフォリオ:</strong> {portfolio_name}</p>
            <p><strong>期間:</strong> {start_date} 〜 {end_date}</p>
            <p><strong>生成日時:</strong> {generated_at}</p>
        </div>

        <h2>サマリー</h2>
        <table>
            <thead>
                <tr>
                    <th>メトリクス</th>
                    <th>ポートフォリオ</th>
                    {bm_headers}
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>年率リターン</td>
                    <td class="{'positive' if summary['portfolio']['annual_return'] > 0 else 'negative'}">{summary['portfolio']['annual_return']*100:.2f}%</td>
                    {get_bm_cells('annual_return')}
                </tr>
                <tr>
                    <td>シャープレシオ</td>
                    <td>{summary['portfolio']['sharpe_ratio']:.2f}</td>
                    {get_bm_cells('sharpe_ratio', False)}
                </tr>
                <tr>
                    <td>最大ドローダウン</td>
                    <td class="negative">{summary['portfolio']['max_drawdown']*100:.2f}%</td>
                    {get_bm_cells('max_drawdown')}
                </tr>
                <tr>
                    <td>ボラティリティ</td>
                    <td>{summary['portfolio']['volatility']*100:.2f}%</td>
                    {get_bm_cells('volatility')}
                </tr>
            </tbody>
        </table>

        {forecast_section}

        {holdings_section}

        <div class="footer">
            <p>Generated by ReportGenerator | multi-asset-portfolio</p>
        </div>
    </div>
</body>
</html>"""

        return html

    def _create_summary_section(self, comparison: ComparisonResult) -> Dict[str, Any]:
        """サマリーセクションデータ作成"""
        pm = comparison.portfolio_metrics

        summary = {
            "portfolio": {
                "annual_return": pm.annual_return,
                "sharpe_ratio": pm.sharpe_ratio,
                "max_drawdown": pm.max_drawdown,
                "volatility": pm.volatility,
            },
            "benchmarks": {},
        }

        for name, metrics in comparison.benchmark_metrics.items():
            summary["benchmarks"][name] = {
                "annual_return": metrics.annual_return,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown,
                "volatility": metrics.volatility,
            }

        return summary

    def _create_returns_table(self, comparison: ComparisonResult) -> List[Dict[str, Any]]:
        """リターン比較表データ作成"""
        pm = comparison.portfolio_metrics

        rows = [
            {
                "metric": "年率リターン",
                "portfolio": pm.annual_return,
                "benchmarks": {
                    name: m.annual_return
                    for name, m in comparison.benchmark_metrics.items()
                },
            },
            {
                "metric": "トータルリターン",
                "portfolio": pm.total_return,
                "benchmarks": {
                    name: m.total_return
                    for name, m in comparison.benchmark_metrics.items()
                },
            },
            {
                "metric": "月次リターン",
                "portfolio": pm.monthly_return,
                "benchmarks": {
                    name: m.monthly_return
                    for name, m in comparison.benchmark_metrics.items()
                },
            },
        ]

        return rows

    def _create_risk_table(self, comparison: ComparisonResult) -> List[Dict[str, Any]]:
        """リスク比較表データ作成"""
        pm = comparison.portfolio_metrics

        rows = [
            {
                "metric": "ボラティリティ",
                "portfolio": pm.volatility,
                "benchmarks": {
                    name: m.volatility
                    for name, m in comparison.benchmark_metrics.items()
                },
            },
            {
                "metric": "最大ドローダウン",
                "portfolio": pm.max_drawdown,
                "benchmarks": {
                    name: m.max_drawdown
                    for name, m in comparison.benchmark_metrics.items()
                },
            },
            {
                "metric": "VaR (95%)",
                "portfolio": pm.var_95,
                "benchmarks": {
                    name: m.var_95
                    for name, m in comparison.benchmark_metrics.items()
                },
            },
        ]

        return rows

    def _create_ratio_table(self, comparison: ComparisonResult) -> List[Dict[str, Any]]:
        """レシオ比較表データ作成"""
        pm = comparison.portfolio_metrics

        rows = [
            {
                "metric": "シャープレシオ",
                "portfolio": pm.sharpe_ratio,
                "benchmarks": {
                    name: m.sharpe_ratio
                    for name, m in comparison.benchmark_metrics.items()
                },
            },
            {
                "metric": "ソルティノレシオ",
                "portfolio": pm.sortino_ratio,
                "benchmarks": {
                    name: m.sortino_ratio
                    for name, m in comparison.benchmark_metrics.items()
                },
            },
            {
                "metric": "カルマーレシオ",
                "portfolio": pm.calmar_ratio,
                "benchmarks": {
                    name: m.calmar_ratio
                    for name, m in comparison.benchmark_metrics.items()
                },
            },
        ]

        return rows

    def _calculate_stock_returns(
        self,
        records: List[Any],
    ) -> Dict[str, float]:
        """
        銘柄別の累積リターンを計算

        Args:
            records: RebalanceRecord のリスト

        Returns:
            銘柄 -> 累積リターンの辞書
        """
        if not records:
            return {}

        # 各銘柄の累積リターンを計算
        stock_returns: Dict[str, float] = {}

        for record in records:
            if not hasattr(record, 'actual_returns_per_asset'):
                continue

            for symbol, ret in record.actual_returns_per_asset.items():
                if symbol == 'CASH':
                    continue
                if ret is None:
                    continue

                # 累積リターン（複利）
                if symbol not in stock_returns:
                    stock_returns[symbol] = 1.0
                stock_returns[symbol] *= (1.0 + ret)

        # 1を引いてリターン率に変換
        for symbol in stock_returns:
            stock_returns[symbol] = stock_returns[symbol] - 1.0

        return stock_returns

    def _calculate_monthly_performance(
        self,
        records: List[Any],
    ) -> List[Dict[str, Any]]:
        """
        月次パフォーマンスを計算

        Args:
            records: RebalanceRecord のリスト

        Returns:
            月次パフォーマンスのリスト
        """
        if not records:
            return []

        from collections import defaultdict

        # 月ごとにリターンを集計
        monthly_data: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"returns": [], "count": 0}
        )

        for record in records:
            if not hasattr(record, 'date') or record.actual_return is None:
                continue

            # 月を取得 (YYYY-MM形式)
            if hasattr(record.date, 'strftime'):
                month = record.date.strftime("%Y-%m")
            else:
                month = str(record.date)[:7]

            monthly_data[month]["returns"].append(record.actual_return)
            monthly_data[month]["count"] += 1

        if not monthly_data:
            return []

        # 月次リターンと累積リターンを計算
        result = []
        cumulative = 1.0

        for month in sorted(monthly_data.keys()):
            data = monthly_data[month]
            # 月次リターン（複利）
            monthly_return = 1.0
            for ret in data["returns"]:
                monthly_return *= (1.0 + ret)
            monthly_return -= 1.0

            # 累積リターン
            cumulative *= (1.0 + monthly_return)

            result.append({
                "month": month,
                "return": monthly_return,
                "cumulative": cumulative - 1.0,
                "rebalance_count": data["count"],
            })

        return result

    def _format_percentage(self, value: float) -> str:
        """パーセント表示フォーマット"""
        if value is None:
            return "N/A"
        return f"{value * 100:.2f}%"

    def _format_ratio(self, value: float) -> str:
        """レシオ表示フォーマット"""
        if value is None:
            return "N/A"
        return f"{value:.2f}"

    def _generate_builtin_html(self, context: Dict[str, Any]) -> str:
        """内蔵HTMLテンプレート（フォールバック用）"""
        summary = context["summary"]
        portfolio_name = context["portfolio_name"]
        start_date = context["start_date"]
        end_date = context["end_date"]
        generated_at = context["generated_at"]

        # ベンチマーク列生成
        benchmarks = list(summary["benchmarks"].keys())
        bm_headers = "".join(f"<th>{bm}</th>" for bm in benchmarks)

        def get_bm_cells(metric: str, is_pct: bool = True) -> str:
            cells = ""
            for bm in benchmarks:
                val = summary["benchmarks"][bm].get(metric, 0)
                if is_pct:
                    cells += f"<td>{val*100:.2f}%</td>"
                else:
                    cells += f"<td>{val:.2f}</td>"
            return cells

        html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>パフォーマンスレポート - {portfolio_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        h1 {{
            color: #1a1a2e;
            border-bottom: 3px solid #4a69bd;
            padding-bottom: 15px;
            margin-bottom: 25px;
        }}
        .meta {{
            color: #666;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #4a69bd;
            margin: 25px 0 15px;
            padding-bottom: 8px;
            border-bottom: 1px solid #ddd;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 25px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: right;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background-color: #4a69bd;
            color: white;
            font-weight: 600;
        }}
        th:first-child, td:first-child {{
            text-align: left;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .positive {{
            color: #27ae60;
        }}
        .negative {{
            color: #e74c3c;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #999;
            font-size: 0.9em;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>パフォーマンスレポート</h1>
        <div class="meta">
            <p><strong>ポートフォリオ:</strong> {portfolio_name}</p>
            <p><strong>期間:</strong> {start_date} 〜 {end_date}</p>
            <p><strong>生成日時:</strong> {generated_at}</p>
        </div>

        <h2>サマリー</h2>
        <table>
            <thead>
                <tr>
                    <th>メトリクス</th>
                    <th>ポートフォリオ</th>
                    {bm_headers}
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>年率リターン</td>
                    <td class="{'positive' if summary['portfolio']['annual_return'] > 0 else 'negative'}">{summary['portfolio']['annual_return']*100:.2f}%</td>
                    {get_bm_cells('annual_return')}
                </tr>
                <tr>
                    <td>シャープレシオ</td>
                    <td>{summary['portfolio']['sharpe_ratio']:.2f}</td>
                    {get_bm_cells('sharpe_ratio', False)}
                </tr>
                <tr>
                    <td>最大ドローダウン</td>
                    <td class="negative">{summary['portfolio']['max_drawdown']*100:.2f}%</td>
                    {get_bm_cells('max_drawdown')}
                </tr>
                <tr>
                    <td>ボラティリティ</td>
                    <td>{summary['portfolio']['volatility']*100:.2f}%</td>
                    {get_bm_cells('volatility')}
                </tr>
            </tbody>
        </table>

        <div class="footer">
            <p>Generated by ReportGenerator | multi-asset-portfolio</p>
        </div>
    </div>
</body>
</html>"""

        return html

    def generate_comparison_html(
        self,
        comparison: "BacktestComparisonResult",
        timeseries_list: List[tuple],
        output_path: str,
    ) -> str:
        """
        複数バックテスト結果の比較HTMLレポートを生成

        Args:
            comparison: BacktestComparator.compare()の結果
            timeseries_list: [(name, timeseries_df), ...] 時系列データリスト
            output_path: 出力ファイルパス

        Returns:
            str: 生成されたHTMLコンテンツ
        """
        archives = comparison.archives
        if not archives:
            raise ValueError("No archives to compare")

        # グラフ用データ準備（Base64エンコード）
        chart_data = self._generate_comparison_charts(timeseries_list)

        # HTML生成
        html = self._generate_comparison_builtin_html(
            archives=archives,
            chart_data=chart_data,
            config_diffs=comparison.config_diffs,
        )

        # ファイル出力
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        return html

    def _generate_comparison_charts(
        self,
        timeseries_list: List[tuple],
    ) -> Dict[str, str]:
        """比較用チャートを生成（Base64エンコード）"""
        chart_data = {}

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import io
            import base64

            # 累積リターン曲線
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))

            # 1. ポートフォリオ価値の比較
            ax1 = axes[0]
            for name, ts in timeseries_list:
                if "portfolio_value" in ts.columns:
                    normalized = ts["portfolio_value"] / ts["portfolio_value"].iloc[0]
                    ax1.plot(ts.index, normalized, label=name, linewidth=1.5)
            ax1.set_title("Portfolio Value Comparison (Normalized)")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Normalized Value")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. ドローダウン比較
            ax2 = axes[1]
            for name, ts in timeseries_list:
                if "drawdown" in ts.columns:
                    ax2.fill_between(
                        ts.index,
                        ts["drawdown"] * 100,
                        0,
                        alpha=0.3,
                        label=name,
                    )
            ax2.set_title("Drawdown Comparison")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Drawdown (%)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Base64エンコード
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            chart_data["comparison_chart"] = base64.b64encode(buf.read()).decode()
            plt.close(fig)

        except ImportError:
            # matplotlibがない場合はスキップ
            pass
        except Exception:
            pass

        return chart_data

    def _generate_comparison_builtin_html(
        self,
        archives: List,
        chart_data: Dict[str, str],
        config_diffs: Dict[str, List],
    ) -> str:
        """比較用の内蔵HTMLテンプレート"""
        # メトリクス比較テーブル生成
        metrics_rows = []
        metrics_to_show = [
            ("total_return", "Total Return", "{:+.2%}"),
            ("annual_return", "Annual Return", "{:+.2%}"),
            ("sharpe_ratio", "Sharpe Ratio", "{:.3f}"),
            ("sortino_ratio", "Sortino Ratio", "{:.3f}"),
            ("max_drawdown", "Max Drawdown", "{:.2%}"),
            ("volatility", "Volatility", "{:.2%}"),
            ("calmar_ratio", "Calmar Ratio", "{:.3f}"),
            ("win_rate", "Win Rate", "{:.2%}"),
        ]

        for key, label, fmt in metrics_to_show:
            row_html = f"<tr><td><strong>{label}</strong></td>"
            for archive in archives:
                val = archive.metrics.get(key)
                if val is not None:
                    row_html += f"<td>{fmt.format(val)}</td>"
                else:
                    row_html += "<td>-</td>"
            row_html += "</tr>"
            metrics_rows.append(row_html)

        # アーカイブヘッダー
        archive_headers = "".join(
            f"<th>{a.name or a.archive_id[-12:]}</th>" for a in archives
        )

        # 設定差分テーブル
        config_diff_rows = []
        for key, values in config_diffs.items():
            row_html = f"<tr><td><strong>{key}</strong></td>"
            for val in values:
                row_html += f"<td>{val}</td>"
            row_html += "</tr>"
            config_diff_rows.append(row_html)

        # チャート埋め込み
        chart_html = ""
        if "comparison_chart" in chart_data:
            chart_html = f'''
            <div class="chart-section">
                <h2>Performance Charts</h2>
                <img src="data:image/png;base64,{chart_data["comparison_chart"]}"
                     alt="Comparison Chart" style="max-width: 100%;">
            </div>
            '''

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Comparison Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: right;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: 500;
        }}
        td:first-child {{
            text-align: left;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        tr:hover {{
            background-color: #e8f4fd;
        }}
        .positive {{
            color: #27ae60;
        }}
        .negative {{
            color: #e74c3c;
        }}
        .chart-section {{
            margin: 30px 0;
            text-align: center;
        }}
        .archive-info {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
            text-align: center;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Backtest Comparison Report</h1>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <div class="archive-info">
            <strong>Archives Compared:</strong>
            <ul>
                {"".join(f'<li>{a.archive_id} - {a.name or "Unnamed"}</li>' for a in archives)}
            </ul>
        </div>

        <h2>Performance Metrics</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    {archive_headers}
                </tr>
            </thead>
            <tbody>
                {"".join(metrics_rows)}
            </tbody>
        </table>

        {chart_html}

        {"<h2>Configuration Differences</h2><table><thead><tr><th>Parameter</th>" + archive_headers + "</tr></thead><tbody>" + "".join(config_diff_rows) + "</tbody></table>" if config_diff_rows else ""}

        <div class="footer">
            <p>Generated by multi-asset-portfolio ReportGenerator</p>
        </div>
    </div>
</body>
</html>'''

        return html
