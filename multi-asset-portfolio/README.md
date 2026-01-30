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

## 機能

### コア機能

| 機能 | 説明 |
|------|------|
| **HRP配分** | 階層的リスクパリティによる堅牢な分散投資 |
| **リスクパリティ** | アセット間の均等リスク寄与 |
| **ウォークフォワード** | ローリング訓練/テスト検証 |
| **マルチ戦略** | モメンタム、平均回帰、マクロシグナルのアンサンブル |
| **レジーム検出** | 市場状況に基づく適応パラメータ |

### 高度な機能

- **Numba JIT高速化** - 5-10倍の計算高速化
- **複数バックテストエンジン** - Fast、Streaming、VectorBTスタイル
- **取引コスト最適化** - ターンオーバー制約付き
- **ドローダウン保護** - 段階的リスク削減
- **フォールバックモード** - システム耐障害性

## インストール

### 前提条件

- Python 3.11以上
- pip または uv パッケージマネージャ

### クイックインストール

```bash
# リポジトリをクローン
git clone git@github.com:so-ta/auto-stock.git
cd auto-stock/multi-asset-portfolio

# 仮想環境を作成（推奨）
python3 -m venv .venv
source .venv/bin/activate

# pipでインストール
pip install -e ".[dev]"

# または uv（より高速）
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### オプション依存関係

```bash
# S3キャッシュサポート
pip install -e ".[s3]"
# または: uv pip install -e ".[s3]"

# GPU加速（NVIDIA CUDA）
pip install -e ".[gpu]"

# Ray分散処理
pip install -e ".[distributed]"

# 全てインストール
pip install -e ".[dev,s3]"
```

### インストール確認

```bash
# テスト実行
make test

# リントチェック
make lint
```

## クイックスタート

### 1. 基本バックテスト

```bash
# デフォルトユニバースでバックテスト実行
uv run python -m src.main --backtest --start 2020-01-01 --end 2024-12-31

# 特定アセットで実行
uv run python -m src.main --backtest --universe SPY,QQQ,TLT,GLD,BTC-USD
```

### 2. カスタム設定の使用

```bash
# デフォルト設定を使用
uv run python -m src.main --config config/default.yaml

# ローカルオーバーライドを使用
cp config/default.yaml config/local.yaml
# config/local.yaml を編集
uv run python -m src.main --config config/local.yaml
```

### 3. Python API

```python
from src.backtest.fast_engine import FastBacktestEngine, FastBacktestConfig
from datetime import datetime

# 設定
config = FastBacktestConfig(
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2024, 12, 31),
    initial_capital=100000.0,
    rebalance_frequency="monthly",
)

# バックテスト実行
engine = FastBacktestEngine(config)
result = engine.run(prices_df)

# 結果表示
print(f"シャープレシオ: {result.sharpe_ratio:.2f}")
print(f"最大ドローダウン: {result.max_drawdown:.2%}")
print(f"トータルリターン: {result.total_return:.2%}")
```

## CLI使用方法

```bash
# ヘルプ表示
uv run python -m src.main --help

# バックテストモード
uv run python -m src.main --backtest [オプション]

# ライブモード（ペーパートレード）
uv run python -m src.main --live [オプション]

# オプション:
#   --config PATH       YAML設定ファイルパス
#   --universe ASSETS   カンマ区切りアセットリスト
#   --start DATE        開始日 (YYYY-MM-DD)
#   --end DATE          終了日 (YYYY-MM-DD)
#   --capital AMOUNT    初期資本
#   --output PATH       結果出力ディレクトリ
```

## 設定

設定は `config/` 内のYAMLファイルで管理：

```
config/
├── default.yaml      # デフォルト設定（マスター）
├── local.yaml        # ローカルオーバーライド（gitignore対象）
├── universe.yaml     # アセットユニバース定義
└── universe_full.yaml # フルユニバース（800+アセット）
```

### 主要設定セクション

```yaml
# リバランス
rebalance:
  frequency: "monthly"        # weekly | monthly | quarterly
  min_trade_threshold: 0.02   # 2%未満の取引はスキップ

# ウォークフォワード検証
walk_forward:
  train_period_days: 504      # 約2年の訓練期間
  test_period_days: 126       # 約6ヶ月のテスト期間
  purge_gap_days: 5           # データリーク防止ギャップ

# リスク管理
hard_gates:
  min_sharpe_ratio: 0.5
  max_drawdown_pct: 25.0
  min_win_rate_pct: 45.0

# アセット配分
asset_allocation:
  method: "HRP"               # HRP | risk_parity | mean_variance
  w_asset_max: 0.2            # 1アセット最大20%
```

詳細は `config/default.yaml` を参照。

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                    パイプラインオーケストレータ              │
│                   (src/orchestrator/pipeline.py)             │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│    データ     │    │   シグナル    │    │    配分       │
│  (src/data/)  │    │ (src/signals/)│    │(src/allocation│
│               │    │               │    │               │
│ - フェッチャー│    │ - モメンタム  │    │ - HRP         │
│ - 品質チェック│    │ - 平均回帰    │    │ - リスクパリティ│
│ - キャッシュ  │    │ - レジーム    │    │ - CVaR        │
└───────────────┘    └───────────────┘    └───────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
            ┌───────────────┐    ┌───────────────┐
            │  バックテスト │    │    リスク     │
            │(src/backtest/)│    │  (src/risk/)  │
            │               │    │               │
            │ - Fastエンジン│    │ - ドローダウン│
            │ - Streaming   │    │ - VaR/CVaR   │
            │ - VectorBT    │    │ - ストレステスト│
            └───────────────┘    └───────────────┘
```

### モジュール概要

| モジュール | 目的 |
|-----------|------|
| `src/data/` | データ取得、キャッシュ、品質チェック |
| `src/signals/` | シグナル生成（モメンタム、平均回帰、マクロ） |
| `src/allocation/` | ポートフォリオ配分アルゴリズム |
| `src/backtest/` | バックテストエンジン |
| `src/orchestrator/` | パイプライン統合 |
| `src/risk/` | リスク指標と管理 |
| `src/strategy/` | 戦略評価とゲート |
| `src/config/` | 設定管理 |

## テスト

```bash
# 全テスト実行
make test

# カバレッジ付き実行
make test-cov

# 特定テストファイル実行
uv run pytest tests/unit/test_fast_engine.py -v

# 統合テスト実行
uv run pytest tests/integration/ -v
```

### テスト構造

```
tests/
├── unit/              # 個別モジュールのユニットテスト
├── integration/       # エンジン互換性の統合テスト
└── conftest.py        # 共有フィクスチャ
```

## トラブルシューティング

### よくある問題

#### 1. インポートエラー

```bash
# 開発モードでパッケージがインストールされていることを確認
pip install -e ".[dev]"
```

#### 2. Numbaスレッディングエラー

```
ValueError: No threading layer could be loaded
```

解決策：
```bash
pip install tbb  # または intel-openmp
```

#### 3. データ取得失敗

- インターネット接続を確認
- APIレート制限を確認
- `config/universe.yaml` のティッカーが有効か確認

#### 4. 大規模ユニバースでのメモリ問題

```yaml
# config/default.yaml で以下を減少:
universe:
  max_assets: 100  # 500から減少
```

### デバッグモード

```bash
# デバッグログを有効化
export PORTFOLIO_LOG_LEVEL=DEBUG
uv run python -m src.main --backtest
```

## パフォーマンスのヒント

1. **Numbaを使用** - 設定で `use_numba: true` を有効化で5-10倍高速化
2. **データキャッシュ** - データキャッシュを有効化してAPI呼び出しを削減
3. **ユニバースを縮小** - 開発時はアセット数を制限して高速化
4. **FastBacktestEngine** - ほとんどのケースで最速のエンジン

## 専用サーバーセットアップ

専用サーバーでフルリソースを活用して実行する場合：

### 1. クイックスタート（専用サーバー）

```bash
# クローンとセットアップ
git clone git@github.com:so-ta/auto-stock.git
cd auto-stock/multi-asset-portfolio

# uvをインストール（未インストールの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 仮想環境を作成して依存関係をインストール
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev,s3]"

# S3を使用する場合：認証情報を設定
cat > .env << 'EOF'
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=ap-northeast-1
EOF
source .env

# リソース設定初期化（CPU/RAM/GPUを自動検出）
python -c "from src.config import print_resource_summary; print_resource_summary()"

# 15年バックテスト実行（全頻度）
python scripts/run_all_backtests.py

# または特定頻度で実行
python scripts/run_standard_backtest.py --start 2010-01-01 --end 2025-01-01 --frequency monthly
python scripts/run_standard_backtest.py --start 2010-01-01 --end 2025-01-01 --frequency weekly
python scripts/run_standard_backtest.py --start 2010-01-01 --end 2025-01-01 --frequency daily
```

### 2. バックテスト実行オプション

#### 標準バックテスト（推奨）
```bash
# 月次リバランス（最速、約1-2分）
uv run python scripts/run_standard_backtest.py \
  --start 2010-01-01 \
  --end 2025-01-01 \
  --frequency monthly

# 週次リバランス（約5-10分）
uv run python scripts/run_standard_backtest.py \
  --start 2010-01-01 \
  --end 2025-01-01 \
  --frequency weekly

# 日次リバランス（最適化なしで約30-60分）
uv run python scripts/run_standard_backtest.py \
  --start 2010-01-01 \
  --end 2025-01-01 \
  --frequency daily
```

#### 高性能バックテスト（Numba並列）
```bash
# Numba並列最適化で実行（1137倍高速化）
uv run python -c "
from src.config import init_resource_config
from src.orchestrator.unified_executor import UnifiedExecutor

# フルリソース使用で初期化
config = init_resource_config(dedicated_server=True)
print(f'{config.max_workers}ワーカー、{config.cache_max_memory_mb}MBキャッシュを使用')

# バックテスト実行
executor = UnifiedExecutor()
result = executor.run_backtest_with_checkpoint(
    start_date='2010-01-01',
    end_date='2025-01-01',
    frequency='daily',
    checkpoint_interval=100,
)
print(f'シャープ: {result.sharpe_ratio:.3f}, リターン: {result.total_return:.2%}')
"
```

#### 再開可能バックテスト（チェックポイントサポート）
```bash
# チェックポイント付きで開始（進捗を自動保存）
uv run python scripts/run_all_backtests.py --checkpoint-interval 50

# 中断した場合はチェックポイントから再開
uv run python scripts/run_all_backtests.py --resume
```

### 3. 期待される結果

| 頻度 | リバランス回数 | 所要時間（最適化後） | シャープ（目標） |
|------|--------------|-------------------|----------------|
| 月次 | 約180 | 1-2分 | > 0.7 |
| 週次 | 約780 | 5-10分 | > 0.6 |
| 日次 | 約3,900 | 10-30分 | > 0.5 |

### 4. 動的リソース設定

システムは利用可能なハードウェアリソースを自動検出して使用：

```python
from src.config import get_current_resource_config, print_resource_summary

# 検出されたリソースと計算された設定を表示
print_resource_summary()

# プログラムから設定にアクセス
config = get_current_resource_config()
print(f"CPUワーカー: {config.max_workers}")
print(f"キャッシュメモリ: {config.cache_max_memory_mb} MB")
print(f"GPU利用可能: {config.use_gpu}")
```

### 5. リソース設定オプション

| 設定 | デフォルト（共有） | 専用サーバー |
|-----|------------------|-------------|
| `max_workers` | CPUコア数 - 1 | 全CPUコア |
| `cache_max_memory_mb` | 利用可能の25% | 利用可能の70% |
| `cache_max_entries` | 5,000 | メモリベース（自動） |
| `disable_chunking` | False | True（RAM >= 16GBの場合） |
| `cache_max_disk_mb` | 空き容量の10% | 無制限 |

### 6. S3キャッシュセットアップ（クラウド環境）

インスタンス間で共有キャッシュを使用するクラウドデプロイ用：

```bash
# S3サポートをインストール
uv pip install -e ".[dev,s3]"

# AWS認証情報を.envファイルで管理（推奨）
cat > .env << 'EOF'
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=ap-northeast-1
EOF

# 環境変数を読み込み
source .env

# または直接エクスポート
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
```

```yaml
# config/local.yaml でS3バックエンドを設定
storage:
  backend: "s3"  # "local" または "s3"
  s3_bucket: "your-bucket-name"
  s3_prefix: ".cache"
  local_cache_enabled: true
  local_cache_path: "/tmp/.backtest_cache"
  local_cache_ttl_hours: 24
```

> **Note**: `.env` ファイルは `.gitignore` に含まれており、リポジトリにコミットされません。

### 7. 既存キャッシュのS3移行

```bash
# ドライラン（アップロードファイルをプレビュー）
uv run python scripts/migrate_cache_to_s3.py --dry-run

# 移行実行
uv run python scripts/migrate_cache_to_s3.py --bucket your-bucket-name
```

### 8. GPUアクセラレーション（オプション）

NVIDIA GPUサポート用：

```bash
# CuPyをインストール（CUDA 11.x）
pip install cupy-cuda11x

# またはCUDA 12.x用
pip install cupy-cuda12x

# インストール確認
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

システムは利用可能な場合、自動的にGPUを検出して使用します。

### 9. Ray分散処理（オプション）

複数コア/マシンでの分散処理用：

```bash
# Rayをインストール
pip install ray

# Rayバックエンドで実行
uv run python -m src.main --backtest --engine ray
```

### 10. 推奨サーバースペック

| ワークロード | CPU | RAM | ストレージ | GPU |
|------------|-----|-----|-----------|-----|
| 開発 | 4+コア | 8GB | SSD 50GB | - |
| 本番（月次） | 8+コア | 16GB | SSD 100GB | オプション |
| 本番（日次） | 16+コア | 32GB+ | SSD 200GB | 推奨 |
| 高性能 | 32+コア | 64GB+ | NVMe 500GB | 必須 |

## コントリビュート

1. リポジトリをフォーク
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. テストを実行 (`make test`)
4. リンターを実行 (`make lint`)
5. 変更をコミット (`git commit -m 'Add amazing feature'`)
6. ブランチにプッシュ (`git push origin feature/amazing-feature`)
7. プルリクエストを作成

### コードスタイル

- Python 3.11+ 型ヒント必須
- Ruffでリント
- MyPyで型チェック
- Black（ruff経由）でフォーマット

## ライセンス

このプロジェクトはMITライセンスの下でライセンスされています - 詳細は [LICENSE](LICENSE) ファイルを参照。

## 謝辞

- VectorBT - アーキテクチャのインスピレーション
- PyPortfolioOpt - 配分アルゴリズム
- Numbaチーム - JITコンパイルサポート

---

**Version**: 1.3.1 | **Last Updated**: 2026-01-30
