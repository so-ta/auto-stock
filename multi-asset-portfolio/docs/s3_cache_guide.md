# S3キャッシュ利用ガイド

> **Version**: 1.0.0
> **Last Updated**: 2026-01-30
> **Task Reference**: task_045_13 (cmd_045)

## 概要

本ドキュメントは、バックテストキャッシュをAWS S3に保存・共有するための設定・使用方法を説明する。

### S3キャッシュの利点

| 項目 | ローカルキャッシュ | S3キャッシュ |
|------|------------------|-------------|
| チーム共有 | 不可 | 可能 |
| 永続性 | マシン依存 | 高耐久性 |
| ストレージ容量 | ディスク制限 | 実質無制限 |
| 複数マシン | 再計算必要 | 共有可能 |

### アーキテクチャ

```
┌─────────────────┐     ┌──────────────────┐
│  Application    │────▶│  StorageBackend  │
└─────────────────┘     └────────┬─────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
            ┌──────────────┐         ┌──────────────┐
            │ Local Cache  │         │    AWS S3    │
            │ (TTL: 24h)   │         │   (Backend)  │
            └──────────────┘         └──────────────┘
```

---

## 1. セットアップ手順

### 1.1 AWS認証情報の設定

#### 方法1: 環境変数（推奨）

```bash
# ~/.bashrc または ~/.zshrc に追加
export AWS_ACCESS_KEY_ID="AKIAIOSFODNN7EXAMPLE"
export AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
export AWS_DEFAULT_REGION="ap-northeast-1"
```

設定後、シェルを再起動または `source ~/.bashrc` を実行。

#### 方法2: AWS CLIプロファイル

```bash
# AWS CLIをインストール（未インストールの場合）
pip install awscli

# 認証情報を設定
aws configure
```

プロンプトで以下を入力：
```
AWS Access Key ID: AKIAIOSFODNN7EXAMPLE
AWS Secret Access Key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
Default region name: ap-northeast-1
Default output format: json
```

#### 方法3: IAMロール（EC2/ECS環境）

EC2インスタンスやECSタスクでは、IAMロールを使用することで認証情報を環境変数で管理する必要がなくなる。
インスタンス/タスクに適切なS3アクセス権限を持つIAMロールをアタッチすること。

### 1.2 AWS CLIでの接続確認

```bash
# 認証情報の確認
aws sts get-caller-identity

# S3バケットへのアクセス確認
aws s3 ls s3://your-bucket-name/

# テストファイルの書き込み
aws s3 cp /dev/null s3://your-bucket-name/test.txt
aws s3 rm s3://your-bucket-name/test.txt
```

### 1.3 必要なPythonパッケージ

```bash
# S3バックエンドに必要なパッケージ
pip install fsspec s3fs boto3
```

---

## 2. 設定ファイルの記述方法

### 2.1 config/default.yaml のstorageセクション

```yaml
# config/default.yaml

storage:
  # バックエンドタイプ: "local" または "s3"
  backend: "s3"

  # ローカルモード時のベースパス（backend: "local" の場合）
  base_path: ".cache"

  # S3設定（backend: "s3" の場合）
  s3_bucket: "your-bucket-name"       # S3バケット名
  s3_prefix: ".cache"                 # オブジェクトプレフィックス
  s3_region: "ap-northeast-1"         # AWSリージョン

  # ローカルキャッシュ設定（S3モード時）
  local_cache_enabled: true           # ローカルキャッシュを有効化
  local_cache_path: "/tmp/.backtest_cache"  # ローカルキャッシュパス
  local_cache_ttl_hours: 24           # キャッシュ有効期限（時間）
```

### 2.2 Pythonコードでの使用方法

#### Settings経由での使用（推奨）

```python
from src.config.settings import Settings

# Settingsを読み込み（config/default.yaml を自動読み込み）
settings = Settings()

# StorageConfigを取得
storage_config = settings.storage.to_storage_config()

# FastBacktestEngineで使用
from src.backtest.fast_engine import FastBacktestConfig, FastBacktestEngine

config = FastBacktestConfig(
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2024, 12, 31),
    storage_config=storage_config,
)
engine = FastBacktestEngine(config)
```

#### 直接StorageConfigを作成

```python
from src.utils.storage_backend import StorageConfig, get_storage_backend

# S3モードで作成
storage_config = StorageConfig(
    backend="s3",
    s3_bucket="your-bucket-name",
    s3_prefix=".cache",
    s3_region="ap-northeast-1",
    local_cache_enabled=True,
    local_cache_path="/tmp/.backtest_cache",
    local_cache_ttl_hours=24,
)

# StorageBackendを取得
backend = get_storage_backend(storage_config)

# 透過的にファイル操作
backend.write_parquet(df, "signals/momentum_20.parquet")
df = backend.read_parquet("signals/momentum_20.parquet")
```

---

## 3. CLI使用例

### 3.1 基本的な使用方法

```bash
# S3キャッシュモードでバックテスト実行（config/default.yamlでbackend: "s3"を設定）
python scripts/run_backtest.py -f monthly

# 特定の頻度で実行
python scripts/run_backtest.py -f daily

# テストモード（5銘柄×短期間）
python scripts/run_backtest.py --test -f monthly
```

### 3.2 環境変数でのカスタマイズ

```bash
# バケット名を環境変数で指定
export BACKTEST_S3_BUCKET="my-custom-bucket"
export BACKTEST_S3_PREFIX="backtest-cache"

python scripts/run_backtest.py -f monthly
```

### 3.3 AWS認証情報を指定して実行

```bash
# 一時的に認証情報を指定
AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=xxx \
python scripts/run_backtest.py -f monthly
```

### 3.4 ローカルキャッシュのクリア

```bash
# ローカルキャッシュを削除（S3キャッシュは保持）
rm -rf /tmp/.backtest_cache
```

---

## 4. トラブルシューティング

### 4.1 認証エラー

#### エラーメッセージ
```
botocore.exceptions.NoCredentialsError: Unable to locate credentials
```

#### 解決方法
1. 環境変数が正しく設定されているか確認
   ```bash
   echo $AWS_ACCESS_KEY_ID
   echo $AWS_SECRET_ACCESS_KEY
   ```

2. AWS CLIで認証確認
   ```bash
   aws sts get-caller-identity
   ```

3. 認証情報の再設定
   ```bash
   aws configure
   ```

### 4.2 バケットアクセスエラー

#### エラーメッセージ
```
botocore.exceptions.ClientError: An error occurred (AccessDenied)
```

#### 解決方法
1. バケット名が正しいか確認
   ```bash
   aws s3 ls s3://your-bucket-name/
   ```

2. IAMポリシーで以下の権限を確認：
   - `s3:GetObject`
   - `s3:PutObject`
   - `s3:DeleteObject`
   - `s3:ListBucket`

3. 必要な最小限のIAMポリシー例：
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "s3:GetObject",
           "s3:PutObject",
           "s3:DeleteObject",
           "s3:ListBucket"
         ],
         "Resource": [
           "arn:aws:s3:::your-bucket-name",
           "arn:aws:s3:::your-bucket-name/*"
         ]
       }
     ]
   }
   ```

### 4.3 ネットワークエラー

#### エラーメッセージ
```
botocore.exceptions.EndpointConnectionError: Could not connect to the endpoint URL
```

#### 解決方法
1. インターネット接続を確認
   ```bash
   ping s3.ap-northeast-1.amazonaws.com
   ```

2. プロキシ設定が必要な場合
   ```bash
   export HTTP_PROXY="http://proxy.example.com:8080"
   export HTTPS_PROXY="http://proxy.example.com:8080"
   ```

3. VPCエンドポイントの確認（VPC内の場合）
   - S3 VPCエンドポイントが設定されているか確認

### 4.4 ローカルキャッシュの問題

#### 症状
- 古いデータが返される
- キャッシュが期限切れにならない

#### 解決方法
```bash
# ローカルキャッシュを完全削除
rm -rf /tmp/.backtest_cache

# キャッシュメタデータも削除
rm -f /tmp/.backtest_cache/.cache_metadata.json
```

---

## 5. コスト見積もり

### 5.1 S3料金の概要（東京リージョン）

| 項目 | 料金 |
|------|------|
| ストレージ | $0.025/GB/月 |
| PUT/COPY/POST/LIST | $0.0047/1,000リクエスト |
| GET/SELECT | $0.00037/1,000リクエスト |
| データ転送（インターネット） | $0.114/GB（最初の10TB） |

※ 2026年1月時点の料金。最新は[AWS料金ページ](https://aws.amazon.com/s3/pricing/)を参照。

### 5.2 キャッシュサイズの目安

| キャッシュ種別 | 1銘柄あたり | 100銘柄 | 1000銘柄 |
|--------------|-----------|---------|---------|
| シグナルキャッシュ | ~5MB | ~500MB | ~5GB |
| 共分散キャッシュ | ~10MB | ~1GB | ~10GB |
| データキャッシュ | ~2MB | ~200MB | ~2GB |
| **合計（概算）** | ~17MB | ~1.7GB | ~17GB |

### 5.3 月額コスト見積もり

#### 小規模利用（100銘柄、1人）
| 項目 | 数量 | 料金 |
|------|------|------|
| ストレージ（2GB） | 2GB | $0.05 |
| リクエスト（1,000回/月） | 1K | $0.005 |
| **月額合計** | | **約$0.06** |

#### 中規模利用（500銘柄、チーム5人）
| 項目 | 数量 | 料金 |
|------|------|------|
| ストレージ（10GB） | 10GB | $0.25 |
| リクエスト（10,000回/月） | 10K | $0.05 |
| データ転送（50GB/月） | 50GB | $5.70 |
| **月額合計** | | **約$6** |

#### 大規模利用（1000銘柄、チーム10人）
| 項目 | 数量 | 料金 |
|------|------|------|
| ストレージ（20GB） | 20GB | $0.50 |
| リクエスト（50,000回/月） | 50K | $0.24 |
| データ転送（200GB/月） | 200GB | $22.80 |
| **月額合計** | | **約$24** |

### 5.4 コスト削減のヒント

1. **ローカルキャッシュを活用**
   - `local_cache_enabled: true` でS3アクセス回数を削減
   - TTLを適切に設定（デフォルト24時間）

2. **S3 Intelligent-Tieringの活用**
   - アクセス頻度に応じて自動的にストレージクラスを最適化
   - 設定方法: バケットのライフサイクルポリシーで設定

3. **不要なキャッシュの定期削除**
   - S3ライフサイクルポリシーで古いオブジェクトを自動削除
   ```bash
   aws s3api put-bucket-lifecycle-configuration \
     --bucket your-bucket-name \
     --lifecycle-configuration file://lifecycle.json
   ```

---

## 6. 参考資料

### 関連ドキュメント

| ドキュメント | 概要 |
|-------------|------|
| [CACHE_SYSTEM.md](CACHE_SYSTEM.md) | キャッシュシステム全体の解説 |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | S3/キャッシュ問題の解決方法 |
| [DEPLOYMENT.md](DEPLOYMENT.md) | 本番環境でのS3設定 |

### 実装・設定ファイル

- [StorageBackend実装](../src/utils/storage_backend.py)
- [設定ファイル](../config/default.yaml)
- [バックテスト実行スクリプト](../scripts/run_backtest.py)

### 外部ドキュメント

- [AWS S3ドキュメント](https://docs.aws.amazon.com/s3/)
- [fsspec公式ドキュメント](https://filesystem-spec.readthedocs.io/)
