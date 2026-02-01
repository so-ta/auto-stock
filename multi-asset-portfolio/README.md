# Multi-Asset Portfolio

> 動的配分、ウォークフォワード検証、マルチ戦略アンサンブルを備えた自動マルチアセットポートフォリオ管理システム

[![CI](https://github.com/so-ta/auto-stock/actions/workflows/ci.yml/badge.svg)](https://github.com/so-ta/auto-stock/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 概要

Multi-Asset Portfolioは以下を目的とした定量ポートフォリオ管理システムです：

- **マルチアセット配分** - 株式、ETF、FX、コモディティを横断した配分
- **ウォークフォワード検証** - オーバーフィッティング防止
- **動的リバランス** - 市場レジームに基づく調整
- **リスク管理** - ドローダウン保護とフォールバックモード

---

## 機能一覧

| カテゴリ | 機能 | 説明 |
|---------|------|------|
| **配分** | HRP配分 | 階層的リスクパリティによる堅牢な分散投資 |
| | リスクパリティ | アセット間の均等リスク寄与 |
| **検証** | ウォークフォワード | ローリング訓練/テスト検証 |
| **戦略** | マルチ戦略 | モメンタム、平均回帰、マクロシグナルのアンサンブル |
| | レジーム検出 | 市場状況に基づく適応パラメータ |
| **高速化** | Numba JIT | 5-10倍の計算高速化 |
| | GPU加速 | CuPy/CUDAによる高速化 |
| **リスク** | ドローダウン保護 | 段階的リスク削減 |
| | フォールバック | システム耐障害性 |

---

## クイックスタート

### インストール

```bash
# リポジトリをクローン
git clone git@github.com:so-ta/auto-stock.git
cd auto-stock/multi-asset-portfolio

# 仮想環境を作成
python3 -m venv .venv
source .venv/bin/activate

# インストール
pip install -e ".[dev]"
```

詳細なインストール手順: [docs/INSTALLATION.md](docs/INSTALLATION.md)

### バックテスト実行

```bash
# デフォルトユニバースでバックテスト
uv run python -m src.main --backtest --start 2020-01-01 --end 2024-12-31

# 特定アセットで実行
uv run python -m src.main --backtest --universe SPY,QQQ,TLT,GLD,BTC-USD
```

### Python API

```python
from src.backtest.fast_engine import FastBacktestEngine, FastBacktestConfig
from datetime import datetime

config = FastBacktestConfig(
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2024, 12, 31),
    initial_capital=100000.0,
    rebalance_frequency="monthly",
)

engine = FastBacktestEngine(config)
result = engine.run(prices_df)

print(f"シャープレシオ: {result.sharpe_ratio:.2f}")
print(f"最大ドローダウン: {result.max_drawdown:.2%}")
```

---

## ドキュメントマップ

### 使い始める

| 目的 | ドキュメント |
|------|-------------|
| インストールしたい | [INSTALLATION.md](docs/INSTALLATION.md) |
| 本番環境にデプロイしたい | [DEPLOYMENT.md](docs/DEPLOYMENT.md) |
| 問題を解決したい | [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) |

### 機能を使う

| 目的 | ドキュメント |
|------|-------------|
| S3キャッシュを設定したい | [s3_cache_guide.md](docs/s3_cache_guide.md) |
| 高速化オプションを知りたい | [backtest_acceleration_options.md](docs/backtest_acceleration_options.md) |
| レポートを生成したい | [performance_report_guide.md](docs/performance_report_guide.md) |

### 開発・拡張する

| 目的 | ドキュメント |
|------|-------------|
| アーキテクチャを理解したい | [ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| キャッシュシステムを理解したい | [CACHE_SYSTEM.md](docs/CACHE_SYSTEM.md) |
| 新規シグナルを追加したい | [SIGNAL_PIPELINE.md](docs/SIGNAL_PIPELINE.md) |
| バックテストエンジンを追加したい | [ENGINE_INTEGRATION.md](docs/ENGINE_INTEGRATION.md) |
| パイプラインを変更したい | [ORCHESTRATOR_RESPONSIBILITIES.md](docs/ORCHESTRATOR_RESPONSIBILITIES.md) |

### リファレンス

| 目的 | ドキュメント |
|------|-------------|
| バックテスト規格を確認したい | [BACKTEST_STANDARD.md](docs/BACKTEST_STANDARD.md) |
| ログ規約を確認したい | [LOGGING_GUIDELINES.md](docs/LOGGING_GUIDELINES.md) |
| PRチェックリストを確認したい | [DESIGN_REVIEW_CHECKLIST.md](docs/DESIGN_REVIEW_CHECKLIST.md) |
| システム概要（非技術者向け） | [system_overview_for_management.md](docs/system_overview_for_management.md) |

---

## 設定

設定は `config/` 内のYAMLファイルで管理：

```
config/
├── default.yaml      # デフォルト設定（マスター）
├── local.yaml        # ローカルオーバーライド（gitignore対象）
└── universe.yaml     # アセットユニバース定義
```

主要設定セクション:

```yaml
rebalance:
  frequency: "monthly"        # weekly | monthly | quarterly

walk_forward:
  train_period_days: 504      # 約2年の訓練期間
  test_period_days: 126       # 約6ヶ月のテスト期間

hard_gates:
  min_sharpe_ratio: 0.5
  max_drawdown_pct: 25.0

asset_allocation:
  method: "HRP"               # HRP | risk_parity | mean_variance
```

詳細は `config/default.yaml` を参照。

---

## テスト

```bash
# 全テスト実行
make test

# カバレッジ付き
make test-cov

# 特定テスト
uv run pytest tests/unit/test_fast_engine.py -v
```

---

## コントリビュート

1. リポジトリをフォーク
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. テストを実行 (`make test`)
4. リンターを実行 (`make lint`)
5. 変更をコミット
6. プルリクエストを作成

### コードスタイル

- Python 3.11+ 型ヒント必須
- Ruffでリント/フォーマット
- MyPyで型チェック

---

## ライセンス

このプロジェクトはMITライセンスの下でライセンスされています - 詳細は [LICENSE](LICENSE) ファイルを参照。

---

**Version**: 1.5.0 | **Last Updated**: 2026-02-01
