# バックテスト統一規格 v1.0

> **Version**: 1.0.0
> **Last Updated**: 2026-01-29
> **Task Reference**: STD-001 (cmd_029)

## 概要

本ドキュメントは、全バックテストが準拠すべき統一規格を定義する。
cmd_028で発生した条件不統一問題を解決し、公平な比較を可能にする。

### 背景：cmd_028の問題点

| 頻度 | エンジン | 銘柄数 | 初期資本 | 取引コスト | slippage |
|------|---------|--------|----------|------------|----------|
| 日次 | VectorBT | 490 | $1M | 10bps | 5bps |
| 週次 | 独自実装 | 828 | $1M | 不明 | なし |
| 月次 | Streaming | 889 | $1M | 10bps | なし |

**問題**: 銘柄数、エンジン、コスト設定が異なり、比較不可能。

---

## 必須パラメータ

全バックテストは以下のパラメータを**厳密に**遵守すること。

| パラメータ | 値 | 理由 |
|-----------|-----|------|
| `initial_capital` | $1,000,000 | 機関投資家規模、現実的なポジションサイズ |
| `transaction_cost_bps` | 10 | 市場平均コスト |
| `slippage_bps` | 5 | 流動性影響、市場インパクト |
| `total_cost_bps` | 15 | 取引コスト + slippage |
| `universe` | 全銘柄統一 | 同一データセットを使用 |
| `period_start` | 2010-01-01 | 15年バックテスト開始 |
| `period_end` | 2024-12-31 | 15年バックテスト終了 |
| `engine` | VectorBTStyleEngine | 高速・高精度・統一インターフェース |
| `risk_free_rate` | 0.02 (2%) | Sharpe計算用 |
| `allow_short` | false | ロングオンリー |
| `max_weight` | 1.0 | 最大ウェイト |
| `min_weight` | 0.0 | 最小ウェイト |

### ユニバース統一規格

```yaml
# config/universe_standard.yaml
universe:
  source: "config/universe_full.yaml"
  filter:
    min_data_days: 252  # 最低1年のデータ
    min_avg_volume: 100000  # 流動性確保
  expected_count: 500-600  # データ品質フィルタ後
```

**重要**: yfinanceでデータ取得失敗した銘柄は自動除外。
実際の銘柄数は報告書に明記すること。

---

## 実行コマンド

### 標準バックテスト実行スクリプト

```bash
# 日次リバランス
python scripts/run_standard_backtest.py \
  --frequency daily \
  --output results/backtest_daily_standard.json

# 週次リバランス
python scripts/run_standard_backtest.py \
  --frequency weekly \
  --output results/backtest_weekly_standard.json

# 月次リバランス
python scripts/run_standard_backtest.py \
  --frequency monthly \
  --output results/backtest_monthly_standard.json
```

### Pythonコード例

```python
from datetime import datetime
from src.backtest.factory import BacktestEngineFactory
from src.backtest.base import UnifiedBacktestConfig

# 統一設定
config = UnifiedBacktestConfig(
    start_date=datetime(2010, 1, 1),
    end_date=datetime(2024, 12, 31),
    initial_capital=1_000_000.0,
    rebalance_frequency="monthly",  # daily / weekly / monthly
    transaction_cost_bps=10.0,
    slippage_bps=5.0,
    allow_short=False,
    risk_free_rate=0.02,
)

# エンジン生成（VectorBT固定）
engine = BacktestEngineFactory.create(
    mode="vectorbt",
    config=config,
)

# バックテスト実行
result = engine.run(universe, prices, config, None)
```

---

## 結果フォーマット

### 必須指標

全バックテスト結果は以下の指標を含むこと。

| 指標 | 説明 | 単位 |
|------|------|------|
| `total_return` | 累積リターン | 比率 (1.0 = 100%) |
| `annual_return` | 年率リターン | 比率 |
| `sharpe_ratio` | シャープレシオ（年率化） | - |
| `sortino_ratio` | ソルティノレシオ（年率化） | - |
| `max_drawdown` | 最大ドローダウン | 比率（負の値） |
| `volatility` | 年率ボラティリティ | 比率 |
| `calmar_ratio` | カルマーレシオ | - |
| `win_rate` | 勝率（日次） | 比率 |
| `n_days` | 取引日数 | 日 |

### JSON出力形式

```json
{
  "metrics": {
    "total_return": 10.08,
    "annual_return": 0.1743,
    "sharpe_ratio": 1.06,
    "sortino_ratio": 0.99,
    "max_drawdown": -0.357,
    "volatility": 0.164,
    "calmar_ratio": 0.49,
    "win_rate": 0.566
  },
  "trading_stats": {
    "n_days": 3773,
    "n_rebalances": 180
  },
  "values": {
    "initial_value": 1000000.0,
    "final_value": 11083956.0
  },
  "config": {
    "start_date": "2010-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 1000000.0,
    "rebalance_frequency": "monthly",
    "transaction_cost_bps": 10.0,
    "slippage_bps": 5.0
  },
  "metadata": {
    "engine": "VectorBTStyleEngine",
    "universe_size": 490,
    "execution_time_seconds": 127.4,
    "timestamp": "2026-01-29T08:00:00"
  }
}
```

---

## チェックリスト

バックテスト実行前に以下を確認すること。

### 設定チェック
- [ ] 初期資本が$1,000,000であること
- [ ] 取引コストが10bpsであること
- [ ] slippageが5bpsであること
- [ ] 期間が2010-01-01 〜 2024-12-31であること
- [ ] エンジンがVectorBTStyleEngineであること
- [ ] risk_free_rateが0.02であること

### データチェック
- [ ] ユニバースが統一されていること
- [ ] 銘柄数が報告書に記載されていること
- [ ] データ品質フィルタが適用されていること

### 結果チェック
- [ ] 必須指標が全て含まれていること
- [ ] メタデータにengineとuniverse_sizeが記載されていること
- [ ] JSONファイルが正しく保存されていること

---

## リバランス頻度別の期待値

過去の実行結果に基づく参考値（統一条件での再実行が必要）。

| 頻度 | 年率リターン | Sharpe | MDD | 備考 |
|------|-------------|--------|-----|------|
| 日次 | 15-20% | 0.8-1.2 | -30〜-40% | 高頻度、高コスト |
| 週次 | 10-15% | 0.5-0.8 | -25〜-35% | バランス型 |
| 月次 | 10-15% | 0.8-1.0 | -25〜-35% | 低コスト |

**注意**: これらは参考値であり、統一条件での再実行が必要。

---

## ベンチマーク比較

全バックテストは以下のベンチマークと比較すること。

| ベンチマーク | シンボル | 期待リターン | Sharpe |
|--------------|---------|-------------|--------|
| S&P 500 | SPY | 13.7% | 0.80 |
| NASDAQ 100 | QQQ | 18.5% | 0.91 |
| 60/40 Portfolio | - | 10.1% | 1.00 |
| US Aggregate Bond | AGG | 2.3% | 0.49 |

---

## 関連ファイル

- `scripts/run_standard_backtest.py` - 標準バックテスト実行スクリプト
- `scripts/validate_backtest.py` - バックテスト検証スクリプト
- `config/universe_standard.yaml` - 統一ユニバース設定
- `src/backtest/factory.py` - エンジンファクトリ
- `src/backtest/base.py` - 統一インターフェース
- `.github/workflows/backtest-validation.yml` - CI検証ワークフロー

---

## CI/CD 検証

### GitHub Actions ワークフロー

バックテスト品質を継続的に監視するため、GitHub Actionsで自動検証を実行する。

**ワークフロー:** `.github/workflows/backtest-validation.yml`

**トリガー:**
- 毎週日曜日 00:00 UTC（定期実行）
- `main`/`develop`へのpush時
- PRでの変更時（`src/**`変更時）
- 手動トリガー

### 検証項目

| 項目 | 基準 | 重大度 |
|------|------|--------|
| n_rebalances | 期待値±5%以内 | Warning |
| Sharpe Ratio | >= -1.0 | Error |
| Sharpe低下率 | 前回比-10%以内 | Error |
| Max Drawdown | > -60% | Error |
| Max Drawdown | > -40% | Warning |

### 検証スクリプト

```bash
# 単一ファイル検証
python scripts/validate_backtest.py results/backtest_daily_standard.json

# ベースラインとの比較
python scripts/validate_backtest.py results/backtest_daily_standard.json \
    --baseline results/baseline.json

# JSON出力
python scripts/validate_backtest.py results/backtest_daily_standard.json --json
```

---

## 変更履歴

| 日付 | バージョン | 変更内容 |
|------|-----------|----------|
| 2026-01-29 | 1.1.0 | CI/CD検証追加（task_029_8） |
| 2026-01-29 | 1.0.0 | 初版作成（STD-001） |
