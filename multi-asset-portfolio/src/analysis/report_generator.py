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
from typing import Any, Dict, List, Optional, Union

# Jinja2 import with fallback
try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False
    Environment = None
    FileSystemLoader = None


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
    ) -> str:
        """
        HTMLレポートを生成

        Args:
            comparison: 比較結果
            portfolio_name: ポートフォリオ名
            start_date: 開始日
            end_date: 終了日
            output_path: 出力ファイルパス

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
