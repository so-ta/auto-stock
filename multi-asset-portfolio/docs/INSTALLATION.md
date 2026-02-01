# インストールガイド

> **Last Updated**: 2026-02-01

---

## 目次

1. [前提条件](#前提条件)
2. [基本インストール](#基本インストール)
3. [オプション依存関係](#オプション依存関係)
4. [環境変数設定](#環境変数設定)
5. [インストール確認](#インストール確認)
6. [開発環境セットアップ](#開発環境セットアップ)

---

## 前提条件

### 必須要件

| 要件 | バージョン | 確認コマンド |
|------|-----------|-------------|
| Python | 3.11以上 | `python3 --version` |
| pip | 最新推奨 | `pip --version` |
| Git | 2.x | `git --version` |

### 推奨要件

| 要件 | 用途 |
|------|------|
| uv | 高速パッケージマネージャ |
| NVIDIA GPU | GPU計算高速化 |
| 16GB+ RAM | 大規模バックテスト |

---

## 基本インストール

### 方法1: pip（標準）

```bash
# リポジトリをクローン
git clone git@github.com:so-ta/auto-stock.git
cd auto-stock/multi-asset-portfolio

# 仮想環境を作成
python3 -m venv .venv
source .venv/bin/activate

# 開発モードでインストール
pip install -e ".[dev]"
```

### 方法2: uv（推奨・高速）

```bash
# uvをインストール（未インストールの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# リポジトリをクローン
git clone git@github.com:so-ta/auto-stock.git
cd auto-stock/multi-asset-portfolio

# 仮想環境を作成してインストール
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

---

## オプション依存関係

### S3キャッシュサポート

チーム共有キャッシュやクラウド環境向け：

```bash
pip install -e ".[s3]"
# または
uv pip install -e ".[s3]"
```

必要なパッケージ: `fsspec`, `s3fs`, `boto3`

詳細設定: [S3キャッシュガイド](s3_cache_guide.md)

### GPU加速（NVIDIA CUDA）

NVIDIA GPU使用時の計算高速化：

```bash
pip install -e ".[gpu]"
```

**CuPyのインストール**（CUDAバージョンに合わせて選択）:

```bash
# CUDA 11.x
pip install cupy-cuda11x

# CUDA 12.x
pip install cupy-cuda12x
```

**インストール確認**:

```bash
python -c "import cupy; print(f'GPU: {cupy.cuda.runtime.getDeviceCount()} devices')"
```

### Ray分散処理

マルチコア/マルチマシン並列処理：

```bash
pip install -e ".[distributed]"
# または
pip install ray
```

### 全依存関係インストール

```bash
pip install -e ".[dev,s3,gpu,distributed]"
```

---

## 環境変数設定

### AWS S3認証情報（S3キャッシュ使用時）

#### 方法1: .envファイル（推奨）

```bash
cat > .env << 'EOF'
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=ap-northeast-1
EOF

source .env
```

> **Note**: `.env`は`.gitignore`に含まれており、コミットされません。

#### 方法2: AWS CLIプロファイル

```bash
aws configure
```

#### 方法3: IAMロール（EC2/ECS）

EC2インスタンスやECSタスクでは、IAMロールをアタッチすることで認証情報の設定が不要になります。

### デバッグモード

```bash
# デバッグログを有効化
export PORTFOLIO_LOG_LEVEL=DEBUG
```

### カスタムキャッシュ設定

```bash
# S3バケット名をオーバーライド
export BACKTEST_S3_BUCKET="my-custom-bucket"
export BACKTEST_S3_PREFIX="backtest-cache"
```

---

## インストール確認

### 基本テスト

```bash
# ユニットテスト実行
make test

# または直接pytest
uv run pytest tests/unit/ -v
```

### リントチェック

```bash
make lint
```

### システムリソース確認

```bash
python -c "from src.config import print_resource_summary; print_resource_summary()"
```

出力例:
```
============================================================
  RESOURCE CONFIGURATION
============================================================
  CPU Workers: 8
  Cache Memory: 4096 MB
  Cache Entries: 10000
  GPU Available: True
  Dedicated Server: False
============================================================
```

### バックテスト動作確認

```bash
# 短いテストバックテスト
uv run python scripts/run_backtest.py --test -f monthly
```

---

## 開発環境セットアップ

### pre-commitフック

```bash
pip install pre-commit
pre-commit install
```

### 型チェック

```bash
mypy src/
```

### フォーマット

```bash
# Ruffでフォーマット
ruff format src/ tests/

# リント修正
ruff check --fix src/ tests/
```

### テストカバレッジ

```bash
make test-cov
```

---

## 次のステップ

- [クイックスタート](../README.md#クイックスタート) - 基本的な使い方
- [デプロイガイド](DEPLOYMENT.md) - 本番環境セットアップ
- [アーキテクチャ](ARCHITECTURE.md) - システム設計の理解
- [トラブルシューティング](TROUBLESHOOTING.md) - 問題解決

---

## 関連ドキュメント

| ドキュメント | 概要 |
|-------------|------|
| [README](../README.md) | プロジェクト概要 |
| [DEPLOYMENT.md](DEPLOYMENT.md) | 本番環境デプロイ |
| [s3_cache_guide.md](s3_cache_guide.md) | S3キャッシュ詳細設定 |
| [backtest_acceleration_options.md](backtest_acceleration_options.md) | 高速化オプション |
