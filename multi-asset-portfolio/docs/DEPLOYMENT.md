# デプロイガイド

> **Last Updated**: 2026-02-01

---

## 目次

1. [推奨サーバースペック](#推奨サーバースペック)
2. [本番環境セットアップ](#本番環境セットアップ)
3. [リソース設定](#リソース設定)
4. [S3設定](#s3設定)
5. [GPU設定](#gpu設定)
6. [運用チェックリスト](#運用チェックリスト)

---

## 推奨サーバースペック

| ワークロード | CPU | RAM | ストレージ | GPU | 用途 |
|------------|-----|-----|-----------|-----|------|
| 開発 | 4+コア | 8GB | SSD 50GB | - | ローカル開発 |
| 本番（月次） | 8+コア | 16GB | SSD 100GB | オプション | 月次リバランス |
| 本番（日次） | 16+コア | 32GB+ | SSD 200GB | 推奨 | 日次リバランス |
| 高性能 | 32+コア | 64GB+ | NVMe 500GB | 必須 | 大規模バックテスト |

### AWS推奨インスタンス

| インスタンス | スペック | 料金（東京） | ユースケース |
|-------------|---------|-------------|-------------|
| c6i.2xlarge | 8vCPU, 16GB | $0.40/時 | 月次バックテスト |
| c6i.4xlarge | 16vCPU, 32GB | $0.80/時 | 週次バックテスト |
| g4dn.xlarge | 4vCPU, T4 GPU | $0.71/時 | GPU高速化 |
| g4dn.2xlarge | 8vCPU, T4 GPU | $1.06/時 | 本格GPU利用 |

---

## 本番環境セットアップ

### クイックスタート

```bash
# クローンとセットアップ
git clone git@github.com:so-ta/auto-stock.git
cd auto-stock/multi-asset-portfolio

# uvをインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# 仮想環境と依存関係
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev,s3]"

# S3認証情報（使用する場合）
cat > .env << 'EOF'
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=ap-northeast-1
EOF
source .env

# リソース設定確認
python -c "from src.config import print_resource_summary; print_resource_summary()"
```

### バックテスト実行

```bash
# 月次リバランス（最速、約1-2分）
python scripts/run_backtest.py --start 2010-01-01 --end 2025-01-01 -f monthly

# 週次リバランス（約5-10分）
python scripts/run_backtest.py --start 2010-01-01 --end 2025-01-01 -f weekly

# 日次リバランス（最適化後約10-30分）
python scripts/run_backtest.py --start 2010-01-01 --end 2025-01-01 -f daily
```

### チェックポイント付き実行

```bash
# 進捗を自動保存（中断からの再開が可能）
python scripts/run_backtest.py -f daily --checkpoint-interval 50

# 結果確認
python scripts/backtest_results.py list
```

---

## リソース設定

### 自動リソース検出

システムは利用可能なハードウェアを自動検出して最適な設定を適用します：

```python
from src.config import get_current_resource_config, print_resource_summary

# 現在の設定を表示
print_resource_summary()

# プログラムからアクセス
config = get_current_resource_config()
print(f"CPUワーカー: {config.max_workers}")
print(f"キャッシュメモリ: {config.cache_max_memory_mb} MB")
print(f"GPU利用可能: {config.use_gpu}")
```

### 設定パラメータ

| 設定 | デフォルト（共有） | 専用サーバー | 説明 |
|-----|------------------|-------------|------|
| `max_workers` | CPUコア - 1 | 全CPUコア | 並列ワーカー数 |
| `cache_max_memory_mb` | 利用可能の25% | 利用可能の70% | キャッシュメモリ上限 |
| `cache_max_entries` | 5,000 | 自動計算 | キャッシュエントリ数 |
| `disable_chunking` | False | True（RAM≥16GB） | データチャンキング無効化 |
| `cache_max_disk_mb` | 空き容量の10% | 無制限 | ディスクキャッシュ上限 |

### 専用サーバーモード

フルリソースを使用する場合：

```python
from src.config import init_resource_config
from src.orchestrator.unified_executor import UnifiedExecutor

# 専用サーバーモードで初期化
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
```

---

## S3設定

### 設定ファイル

```yaml
# config/local.yaml
storage:
  backend: "s3"                     # "local" または "s3"
  s3_bucket: "your-bucket-name"     # S3バケット名
  s3_prefix: ".cache"               # オブジェクトプレフィックス
  s3_region: "ap-northeast-1"       # AWSリージョン
  local_cache_enabled: true         # ローカルキャッシュ併用
  local_cache_path: "/tmp/.backtest_cache"
  local_cache_ttl_hours: 24
```

### 認証設定

1. **環境変数**（推奨）
   ```bash
   export AWS_ACCESS_KEY_ID="your_key"
   export AWS_SECRET_ACCESS_KEY="your_secret"
   export AWS_DEFAULT_REGION="ap-northeast-1"
   ```

2. **IAMロール**（EC2/ECS）
   - インスタンスにIAMロールをアタッチ
   - 必要な権限: `s3:GetObject`, `s3:PutObject`, `s3:DeleteObject`, `s3:ListBucket`

詳細設定とトラブルシューティング: [S3キャッシュガイド](s3_cache_guide.md)

---

## GPU設定

### NVIDIA CUDAセットアップ

```bash
# CuPyをインストール
pip install cupy-cuda11x  # CUDA 11.x
# または
pip install cupy-cuda12x  # CUDA 12.x

# 確認
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

### GPU有効化

システムはGPUを自動検出しますが、明示的に有効化する場合：

```python
from src.backtest.fast_engine import FastBacktestConfig

config = FastBacktestConfig(
    use_gpu=True,
    use_numba=True,
    numba_parallel=True,
)
```

### 期待される高速化

| 設定 | 高速化倍率 | 備考 |
|------|-----------|------|
| Numba JIT | 8-12倍 | デフォルト有効 |
| Numba並列 | 追加4-8倍 | `numba_parallel=True` |
| GPU (CuPy) | 10-50倍 | 共分散計算等 |
| 合計 | 40-150倍 | GPU有効時 |

詳細オプション: [バックテスト高速化オプション](backtest_acceleration_options.md)

---

## 運用チェックリスト

### デプロイ前

- [ ] Python 3.11+がインストールされている
- [ ] 仮想環境がアクティベートされている
- [ ] 依存関係がインストールされている (`pip install -e ".[dev]"`)
- [ ] ユニットテストが通る (`make test`)
- [ ] リソース設定が確認済み

### S3使用時

- [ ] AWS認証情報が設定されている
- [ ] S3バケットにアクセスできる (`aws s3 ls s3://bucket-name/`)
- [ ] `config/local.yaml`でS3バックエンドが設定されている
- [ ] ローカルキャッシュディレクトリのディスク容量が十分

### GPU使用時

- [ ] CUDAドライバがインストールされている
- [ ] CuPyがインストールされている
- [ ] GPUが検出される (`python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"`)

### 本番運用

- [ ] ログレベルが適切（INFO推奨、DEBUGは開発時のみ）
- [ ] チェックポイント間隔が設定されている
- [ ] ディスク容量が十分（最低50GB空き）
- [ ] メモリ使用量の監視体制がある

---

## 期待されるパフォーマンス

| 頻度 | リバランス回数 | 所要時間（最適化後） | シャープ（目標） |
|------|--------------|-------------------|----------------|
| 月次 | 約180 | 1-2分 | > 0.7 |
| 週次 | 約780 | 5-10分 | > 0.6 |
| 日次 | 約3,900 | 10-30分 | > 0.5 |

---

## 関連ドキュメント

| ドキュメント | 概要 |
|-------------|------|
| [INSTALLATION.md](INSTALLATION.md) | インストール手順 |
| [s3_cache_guide.md](s3_cache_guide.md) | S3キャッシュ詳細設定 |
| [backtest_acceleration_options.md](backtest_acceleration_options.md) | 高速化オプション |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | 問題解決 |
