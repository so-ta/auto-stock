# パフォーマンスレポート機能ガイド

バックテスト結果を主要指数と比較し、プロフェッショナルなレポートを生成する機能のガイド。

## 概要

このモジュールは以下の機能を提供します：

- **ベンチマーク比較**: SPY, QQQ, DIA等の主要指数とパフォーマンスを比較
- **レポート生成**: HTML/PDF/テキスト形式でレポート出力
- **グラフ生成**: 累積リターン、ドローダウン、ヒートマップ等の可視化
- **ダッシュボード**: インタラクティブなWeb UIで結果を確認

## インストール

### 必須パッケージ

```bash
uv add numpy pandas yfinance
```

### オプションパッケージ

```bash
# HTMLレポート（Jinja2テンプレート）
uv add jinja2

# PDF生成
uv add weasyprint

# 静的グラフ（matplotlib）
uv add matplotlib japanize-matplotlib

# インタラクティブグラフ（Plotly）
uv add plotly kaleido

# ダッシュボード
uv add dash
```

## クイックスタート

### CLIでレポート生成

```bash
# 基本使用
uv run python scripts/generate_performance_report.py \
    --start 2010-01-01 \
    --end 2025-01-01 \
    --output reports/

# ベンチマーク指定
uv run python scripts/generate_performance_report.py \
    --benchmarks SPY,QQQ,DIA \
    --format html

# 既存バックテスト結果から
uv run python scripts/generate_performance_report.py \
    --backtest-result results/backtest_monthly.json \
    --format both

# グラフ付き
uv run python scripts/generate_performance_report.py \
    --charts \
    --output reports/
```

### Pythonコードでレポート生成

```python
from src.analysis import (
    ReportGenerator,
    PortfolioMetrics,
    ComparisonResult,
)

# メトリクスを作成
portfolio = PortfolioMetrics(
    annual_return=0.125,
    sharpe_ratio=0.85,
    max_drawdown=-0.182,
    volatility=0.15,
)

spy = PortfolioMetrics(
    annual_return=0.102,
    sharpe_ratio=0.72,
    max_drawdown=-0.339,
)

# 比較結果を作成
comparison = ComparisonResult(
    portfolio_metrics=portfolio,
    benchmark_metrics={"SPY": spy},
)

# レポート生成
generator = ReportGenerator()

# HTMLレポート
generator.generate_html_report(
    comparison,
    portfolio_name="My Portfolio",
    start_date="2010-01-01",
    end_date="2025-01-01",
    output_path="reports/performance.html",
)

# テキストレポート（ターミナル表示）
text = generator.generate_text_report(comparison, "My Portfolio")
print(text)
```

### グラフ生成

```python
from src.analysis import StaticChartGenerator
import pandas as pd

# サンプルデータ
portfolio_returns = pd.Series(...)  # 日次リターン
benchmark_returns = pd.DataFrame({
    "SPY": spy_returns,
    "QQQ": qqq_returns,
})

# グラフ生成器
generator = StaticChartGenerator(figsize=(12, 6), dpi=150)

# 個別チャート
fig = generator.plot_equity_comparison(portfolio_returns, benchmark_returns)
fig.savefig("equity.png")

fig = generator.plot_monthly_heatmap(portfolio_returns)
fig.savefig("heatmap.png")

# 全チャート一括保存
files = generator.save_all_charts(
    portfolio_returns,
    benchmark_returns,
    output_dir="charts/",
    format="png",
)
```

## APIリファレンス

### PortfolioMetrics

パフォーマンスメトリクスを保持するデータクラス。

```python
@dataclass
class PortfolioMetrics:
    # リターン系
    total_return: float = 0.0
    annual_return: float = 0.0
    monthly_return: float = 0.0

    # リスク系
    volatility: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0

    # リスク調整リターン
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
```

### ComparisonResult

ポートフォリオとベンチマークの比較結果。

```python
@dataclass
class ComparisonResult:
    portfolio_metrics: PortfolioMetrics
    benchmark_metrics: Dict[str, PortfolioMetrics]

    @property
    def benchmarks(self) -> List[str]:
        """ベンチマーク名リスト"""

    def get_benchmark(self, name: str) -> Optional[PortfolioMetrics]:
        """指定ベンチマークのメトリクス取得"""

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
```

### ReportGenerator

レポート生成クラス。

```python
class ReportGenerator:
    def __init__(self, template_dir: Optional[str] = None):
        """Jinja2テンプレートディレクトリ指定"""

    def generate_html_report(
        self,
        comparison: ComparisonResult,
        portfolio_name: str,
        start_date: str,
        end_date: str,
        output_path: str,
    ) -> str:
        """HTMLレポート生成"""

    def generate_text_report(
        self,
        comparison: ComparisonResult,
        portfolio_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> str:
        """テキストレポート生成（ターミナル表示用）"""
```

### StaticChartGenerator

静的グラフ生成クラス（matplotlib）。

```python
class StaticChartGenerator:
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 6),
        style: str = "seaborn-v0_8-whitegrid",
        dpi: int = 150,
    ):
        """初期化"""

    def plot_equity_comparison(
        self,
        portfolio: pd.Series,
        benchmarks: pd.DataFrame,
        title: str = "資産推移比較",
    ) -> Figure:
        """累積リターン比較グラフ"""

    def plot_drawdown_comparison(
        self,
        portfolio_dd: pd.Series,
        benchmark_dds: pd.DataFrame,
    ) -> Figure:
        """ドローダウン比較グラフ"""

    def plot_monthly_heatmap(
        self,
        returns: pd.Series,
        title: str = "月次リターン",
    ) -> Figure:
        """月次リターンヒートマップ"""

    def plot_rolling_sharpe(
        self,
        returns: pd.Series,
        window: int = 252,
    ) -> Figure:
        """ローリングシャープレシオ"""

    def save_all_charts(
        self,
        portfolio: pd.Series,
        benchmarks: pd.DataFrame,
        output_dir: str,
        format: str = "png",
    ) -> List[str]:
        """全グラフを保存"""
```

## 出力形式

### HTML（インタラクティブ）

- プロフェッショナルなデザイン
- レスポンシブ対応
- サマリーカード表示
- 比較テーブル

```bash
python scripts/generate_performance_report.py --format html
```

### PDF（印刷用）

- weasyprint使用
- 高解像度出力
- 印刷最適化レイアウト

```bash
python scripts/generate_performance_report.py --format pdf
```

**注意**: PDF生成には `weasyprint` が必要です。

```bash
uv add weasyprint
```

### テキスト（ターミナル表示）

```
============================================================
パフォーマンスレポート: My Portfolio
期間: 2010-01-01 〜 2025-01-01
============================================================

【サマリー】
                          ポートフォリオ       SPY       QQQ
------------------------------------------------------------------
年率リターン                      12.50%    10.20%    14.80%
シャープレシオ                      0.85      0.72      0.82
最大ドローダウン                  -18.20%   -33.90%   -35.10%
```

### グラフ（PNG/PDF/SVG）

| グラフ | 説明 |
|--------|------|
| equity_comparison.png | 累積リターン比較 |
| drawdown_comparison.png | ドローダウン比較 |
| monthly_heatmap.png | 月次リターンヒートマップ |
| rolling_sharpe.png | ローリングシャープレシオ |
| returns_distribution.png | リターン分布 |

## CLIオプション一覧

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--backtest-result` | 既存結果JSONファイル | なし |
| `--start` | 開始日 | 2010-01-01 |
| `--end` | 終了日 | 今日 |
| `--frequency` | リバランス頻度 | monthly |
| `--benchmarks` | 比較ベンチマーク | SPY,QQQ,DIA,IWM,VT,EWJ |
| `--output` | 出力ディレクトリ | reports/ |
| `--format` | 出力形式 | html |
| `--portfolio-name` | ポートフォリオ名 | Multi-Asset Portfolio |
| `--charts` | グラフ生成 | False |
| `--verbose` | 詳細ログ | False |

## トラブルシューティング

### yfinanceエラー

**症状**: `No data found for ticker XXX`

**対処**:
1. ティッカーシンボルが正しいか確認
2. 期間が妥当か確認（上場前の日付指定など）
3. ネットワーク接続を確認
4. yfinanceをアップデート: `uv add yfinance --upgrade`

### PDF生成エラー（weasyprint）

**症状**: `ImportError: No module named 'weasyprint'`

**対処**:
```bash
uv add weasyprint
```

**症状**: システムフォントエラー

**対処**（macOS）:
```bash
brew install pango
```

**対処**（Ubuntu）:
```bash
sudo apt-get install libpango-1.0-0 libpangoft2-1.0-0
```

### グラフ生成エラー（matplotlib）

**症状**: `ImportError: No module named 'matplotlib'`

**対処**:
```bash
uv add matplotlib
```

**症状**: 日本語フォントが表示されない

**対処**:
```bash
uv add japanize-matplotlib
```

### Plotlyグラフエラー

**症状**: 静的画像が保存できない

**対処**:
```bash
uv add kaleido
```

## 関連ドキュメント

- [バックテスト高速化オプション](backtest_acceleration_options.md)
- [S3キャッシュガイド](s3_cache_guide.md)
- [システム概要](system_overview_for_management.md)
