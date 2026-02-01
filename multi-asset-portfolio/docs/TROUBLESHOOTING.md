# トラブルシューティングガイド

> **Last Updated**: 2026-02-01

---

## 目次

1. [問題カテゴリ別インデックス](#問題カテゴリ別インデックス)
2. [インストール・環境問題](#インストール環境問題)
3. [データ取得問題](#データ取得問題)
4. [バックテスト問題](#バックテスト問題)
5. [パフォーマンス問題](#パフォーマンス問題)
6. [S3/キャッシュ問題](#s3キャッシュ問題)
7. [ログ・信頼性問題](#ログ信頼性問題)
8. [デバッグモード](#デバッグモード)
9. [よくある質問](#よくある質問)

---

## 問題カテゴリ別インデックス

| 症状 | 参照セクション |
|------|---------------|
| インポートエラー | [インストール・環境問題](#インストール環境問題) |
| Numbaスレッディングエラー | [インストール・環境問題](#インストール環境問題) |
| データ取得失敗 | [データ取得問題](#データ取得問題) |
| yfinanceエラー | [データ取得問題](#データ取得問題) |
| メモリ不足 | [パフォーマンス問題](#パフォーマンス問題) |
| 実行が遅い | [パフォーマンス問題](#パフォーマンス問題) |
| S3認証エラー | [S3/キャッシュ問題](#s3キャッシュ問題) |
| キャッシュが古い | [S3/キャッシュ問題](#s3キャッシュ問題) |
| GPU検出されない | [バックテスト問題](#バックテスト問題) |
| 信頼性スコアが低い | [ログ・信頼性問題](#ログ信頼性問題) |
| ログが表示されない | [ログ・信頼性問題](#ログ信頼性問題) |
| Viewerでログタブが空 | [ログ・信頼性問題](#ログ信頼性問題) |

---

## インストール・環境問題

### インポートエラー

**症状**:
```
ModuleNotFoundError: No module named 'src'
```

**原因**: パッケージが開発モードでインストールされていない

**解決方法**:
```bash
# 開発モードでインストール
pip install -e ".[dev]"
```

### Numbaスレッディングエラー

**症状**:
```
ValueError: No threading layer could be loaded
```

**原因**: Numbaのスレッディングバックエンドが見つからない

**解決方法**:
```bash
# TBBをインストール（推奨）
pip install tbb

# または Intel OpenMP
pip install intel-openmp
```

### Python バージョンエラー

**症状**:
```
ERROR: Package requires Python >=3.11
```

**解決方法**:
```bash
# Pythonバージョン確認
python3 --version

# pyenvでPython 3.11+をインストール
pyenv install 3.11.0
pyenv local 3.11.0
```

### 仮想環境のアクティベーション

**症状**: コマンドが認識されない、依存関係が見つからない

**解決方法**:
```bash
# 仮想環境をアクティベート
source .venv/bin/activate

# アクティベート確認
which python  # .venv/bin/python が表示されるはず
```

---

## データ取得問題

### yfinance データ取得失敗

**症状**:
```
No data found for ticker XXX
```

**原因**: ティッカーが無効、上場廃止、または期間外

**解決方法**:
1. ティッカーシンボルが正しいか確認
2. Yahoo Financeで直接確認: `https://finance.yahoo.com/quote/XXX`
3. 期間が妥当か確認（上場前の日付指定など）
4. yfinanceをアップデート:
   ```bash
   pip install yfinance --upgrade
   ```

### APIレート制限

**症状**:
```
429 Too Many Requests
```

**解決方法**:
1. リクエスト間隔を空ける
2. ユニバースを分割して取得
3. キャッシュを活用してAPI呼び出しを削減

### ネットワーク接続エラー

**症状**:
```
ConnectionError: Failed to establish a new connection
```

**解決方法**:
1. インターネット接続を確認
2. プロキシ設定が必要な場合:
   ```bash
   export HTTP_PROXY="http://proxy.example.com:8080"
   export HTTPS_PROXY="http://proxy.example.com:8080"
   ```

---

## バックテスト問題

### NaN値エラー

**症状**:
```
ValueError: Input contains NaN
```

**原因**: 価格データに欠損がある

**解決方法**:
1. 品質チェックを有効化:
   ```yaml
   # config/default.yaml
   data:
     quality_check:
       enabled: true
       min_data_points: 252
       max_missing_pct: 5.0
   ```

2. 問題のあるティッカーを特定:
   ```python
   import pandas as pd
   df = pd.read_parquet(".cache/data/prices.parquet")
   print(df.isna().sum())
   ```

### GPU検出されない

**症状**:
```
CuPy not available, falling back to NumPy
```

**解決方法**:
1. CUDAドライバ確認:
   ```bash
   nvidia-smi
   ```

2. CuPyを正しいCUDAバージョンでインストール:
   ```bash
   # CUDA 11.x
   pip install cupy-cuda11x
   # CUDA 12.x
   pip install cupy-cuda12x
   ```

3. インストール確認:
   ```bash
   python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
   ```

### チェックポイント読み込みエラー

**症状**:
```
CheckpointLoadError: Incompatible checkpoint version
```

**原因**: ライブラリバージョンが変更された

**解決方法**:
```bash
# 古いチェックポイントを削除
rm -rf checkpoints/

# 最初から実行
python scripts/run_backtest.py -f daily
```

---

## パフォーマンス問題

### メモリ不足（OOM）

**症状**:
```
MemoryError: Unable to allocate
```

**解決方法**:

1. **ユニバースを縮小**:
   ```yaml
   # config/default.yaml
   universe:
     max_assets: 100  # 500から縮小
   ```

2. **キャッシュメモリを制限**:
   ```python
   from src.config import init_resource_config
   config = init_resource_config(
       cache_max_memory_mb=512,  # 制限
   )
   ```

3. **チャンキングを有効化**:
   ```yaml
   # config/default.yaml
   optimization:
     disable_chunking: false
   ```

### 実行が遅い

**原因と解決方法**:

| 原因 | 確認方法 | 解決方法 |
|------|---------|---------|
| Numba無効 | `use_numba: false` | `use_numba: true` に変更 |
| キャッシュミス | ログでキャッシュヒット率確認 | キャッシュディレクトリを確認 |
| I/Oボトルネック | ディスク使用率確認 | SSDを使用、S3キャッシュ |
| CPU制限 | `top`でCPU使用率確認 | ワーカー数を調整 |

**推奨高速化設定**:
```python
config = FastBacktestConfig(
    use_numba=True,
    numba_parallel=True,  # マルチコア活用
    use_gpu=True,         # GPU利用可能な場合
    precompute_signals=True,
    use_incremental_cov=True,
)
```

### 日次バックテストが遅い

**期待所要時間**:

| 設定 | 所要時間 |
|------|---------|
| 最適化なし | 30-60分 |
| Numba有効 | 5-10分 |
| Numba + GPU | 1-3分 |
| 全最適化 | 30秒-1分 |

**高速化チェックリスト**:
- [ ] `use_numba: true`
- [ ] `numba_parallel: true`
- [ ] `precompute_signals: true`
- [ ] `use_incremental_cov: true`
- [ ] シグナルキャッシュが有効

---

## S3/キャッシュ問題

### S3認証エラー

**症状**:
```
botocore.exceptions.NoCredentialsError: Unable to locate credentials
```

**解決方法**:
1. 環境変数を確認:
   ```bash
   echo $AWS_ACCESS_KEY_ID
   echo $AWS_SECRET_ACCESS_KEY
   ```

2. AWS CLIで認証確認:
   ```bash
   aws sts get-caller-identity
   ```

3. 認証情報を再設定:
   ```bash
   aws configure
   ```

### S3バケットアクセスエラー

**症状**:
```
botocore.exceptions.ClientError: An error occurred (AccessDenied)
```

**解決方法**:
1. バケット名を確認:
   ```bash
   aws s3 ls s3://your-bucket-name/
   ```

2. IAMポリシーで以下の権限を確認:
   - `s3:GetObject`
   - `s3:PutObject`
   - `s3:DeleteObject`
   - `s3:ListBucket`

3. 最小IAMポリシー例:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [{
       "Effect": "Allow",
       "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
       "Resource": [
         "arn:aws:s3:::your-bucket-name",
         "arn:aws:s3:::your-bucket-name/*"
       ]
     }]
   }
   ```

### キャッシュが古い・不整合

**症状**:
- 古いデータが返される
- キャッシュが期限切れにならない
- 設定変更が反映されない

**解決方法**:
```bash
# ローカルキャッシュを完全削除
rm -rf /tmp/.backtest_cache
rm -rf .cache/

# キャッシュメタデータも削除
rm -f /tmp/.backtest_cache/.cache_metadata.json

# S3キャッシュを削除（必要な場合）
aws s3 rm s3://your-bucket/.cache/ --recursive
```

### S3接続タイムアウト

**症状**:
```
botocore.exceptions.EndpointConnectionError: Could not connect to the endpoint URL
```

**解決方法**:
1. ネットワーク接続確認:
   ```bash
   ping s3.ap-northeast-1.amazonaws.com
   ```

2. VPCエンドポイント確認（VPC内の場合）

3. プロキシ設定:
   ```bash
   export HTTP_PROXY="http://proxy.example.com:8080"
   export HTTPS_PROXY="http://proxy.example.com:8080"
   ```

---

## ログ・信頼性問題

### 信頼性スコアが低い

**症状**:
- Viewer で「信頼性: 低」または「信頼性なし」と表示される
- `result.reliability.score` が 0.7 未満

**原因の特定**:
```python
# 信頼性レポートを確認
print(result.reliability)
# {
#   "score": 0.55,
#   "level": "low",
#   "reasons": [
#     "Errors occurred during execution (3 times)",
#     "High number of warnings (15)"
#   ]
# }
```

**解決方法**:

1. **ログを確認**:
   ```bash
   # アーカイブのログファイルを確認
   cat results/{archive_id}/logs.jsonl | jq 'select(.level == "ERROR")'
   ```

2. **よくある原因と対処**:

   | 原因 | 対処 |
   |------|------|
   | データ取得失敗 | ユニバースのティッカーを確認、ネットワーク接続確認 |
   | 共分散計算失敗 | データ期間を長くする、銘柄数を減らす |
   | シグナル計算失敗 | 個別シグナルのログを確認 |
   | 警告多発 | データ品質チェック設定を調整 |

3. **信頼性スコアの基準**:
   - ≥ 0.9: high（問題なし）
   - 0.7-0.9: medium（許容範囲）
   - 0.4-0.7: low（結果を慎重に解釈）
   - < 0.4: unreliable（結果を信頼しない）

### ログが表示されない

**症状**:
- Viewer のログタブが空
- `logs.jsonl` が生成されない

**解決方法**:

1. **ログコレクターの確認**:
   ```python
   from src.utils.logger import get_log_collector

   collector = get_log_collector()
   if collector is None:
       print("Log collector not initialized")
   else:
       print(f"Logs collected: {collector.get_stats()}")
   ```

2. **Pipeline経由で実行しているか確認**:
   ログ収集は `Pipeline.run()` 経由でのみ有効。
   直接 `FastBacktestEngine` を使用するとログは収集されない。

3. **アーカイブ保存時に log_collector を渡しているか確認**:
   ```python
   store.save(
       result=result,
       log_collector=executor.log_collector,  # 必須
   )
   ```

### Viewerでログタブが空

**症状**:
- ログタブをクリックしても何も表示されない
- 「ログを読み込んでいます...」のまま

**解決方法**:

1. **logs.jsonl の存在確認**:
   ```bash
   ls -la results/{archive_id}/logs.jsonl
   ```

2. **APIレスポンス確認**:
   ```bash
   curl http://localhost:8000/backtest/api/logs/{archive_id}
   ```

3. **ブラウザコンソールでエラー確認**:
   Developer Tools (F12) → Console タブ

### ログの確認方法

**CLIでログを確認**:
```bash
# 全ログ
cat results/{archive_id}/logs.jsonl | jq .

# エラーのみ
cat results/{archive_id}/logs.jsonl | jq 'select(.level == "ERROR")'

# 警告のみ
cat results/{archive_id}/logs.jsonl | jq 'select(.level == "WARNING")'

# 特定コンポーネント
cat results/{archive_id}/logs.jsonl | jq 'select(.component == "pipeline")'

# 統計サマリー
cat results/{archive_id}/logs.jsonl | jq -s 'group_by(.level) | map({level: .[0].level, count: length})'
```

**Pythonでログを確認**:
```python
from src.utils.pipeline_log_collector import PipelineLogCollector

logs = PipelineLogCollector.load_from_file("results/{archive_id}/logs.jsonl")
errors = [l for l in logs if l.get("level") == "ERROR"]
print(f"Errors: {len(errors)}")
for e in errors:
    print(f"  {e['timestamp']}: {e['event']}")
```

**Viewerでログを確認**:
1. バックテスト結果ページ (`/backtest/{id}`) を開く
2. 「ログ」タブをクリック
3. レベルフィルタで絞り込み（ERROR, WARNING, INFO, DEBUG）
4. 検索ボックスでキーワード検索

---

## デバッグモード

### ログレベル設定

```bash
# DEBUGレベルを有効化
export PORTFOLIO_LOG_LEVEL=DEBUG
python scripts/run_backtest.py -f monthly
```

### ログレベル一覧

| レベル | 用途 |
|--------|------|
| DEBUG | 詳細デバッグ情報 |
| INFO | 通常運用（推奨） |
| WARNING | 警告のみ |
| ERROR | エラーのみ |

### Pythonデバッガ

```bash
# pdbを使用してデバッグ
python -m pdb scripts/run_backtest.py -f monthly
```

### プロファイリング

```bash
# cProfileで実行時間計測
python -m cProfile -s cumulative scripts/run_backtest.py -f monthly 2>&1 | head -50
```

---

## よくある質問

### Q: どのリバランス頻度を選べばいい？

**A**: 目的に応じて選択:
- **月次**: 最速、取引コスト最小、長期投資向け
- **週次**: バランス、中期投資向け
- **日次**: 最高精度、短期戦略向け、計算コスト高

### Q: キャッシュはどのくらいディスクを使う？

**A**: ユニバースサイズに依存:

| 銘柄数 | シグナルキャッシュ | 共分散キャッシュ | 合計 |
|--------|------------------|----------------|------|
| 100 | ~500MB | ~1GB | ~1.7GB |
| 500 | ~2.5GB | ~5GB | ~8.5GB |
| 1000 | ~5GB | ~10GB | ~17GB |

### Q: GPUは必須？

**A**: 必須ではありませんが、日次バックテストでは大幅な高速化が期待できます。月次・週次ではCPUのみでも十分実用的な速度です。

### Q: S3とローカルキャッシュどちらを使うべき？

**A**:
- **単一マシン開発**: ローカルキャッシュ
- **チーム共有/クラウド**: S3キャッシュ
- **両方**: S3バックエンド + ローカルキャッシュ併用（推奨）

### Q: 信頼性スコアが低いとき結果は使えない？

**A**:
- **high (≥0.9)**: 問題なし、そのまま使用可能
- **medium (0.7-0.9)**: 軽微な問題あり、結果は参考になる
- **low (0.4-0.7)**: 注意が必要、reasons を確認して問題を理解した上で使用
- **unreliable (<0.4)**: 結果を信頼すべきでない、原因を修正して再実行

### Q: ログはどのくらいディスクを使う？

**A**: 実行の複雑さに依存しますが、通常 1-10MB 程度です。リングバッファ（デフォルト1000エントリ）でメモリ使用量を制限しています。

---

## サポート

問題が解決しない場合:

1. [GitHub Issues](https://github.com/so-ta/auto-stock/issues) で報告
2. 以下の情報を含めてください:
   - エラーメッセージ全文
   - Python/ライブラリバージョン
   - OS情報
   - 再現手順

---

## 関連ドキュメント

| ドキュメント | 概要 |
|-------------|------|
| [INSTALLATION.md](INSTALLATION.md) | インストール手順 |
| [DEPLOYMENT.md](DEPLOYMENT.md) | 本番環境デプロイ |
| [s3_cache_guide.md](s3_cache_guide.md) | S3キャッシュ詳細 |
| [backtest_acceleration_options.md](backtest_acceleration_options.md) | 高速化オプション |
| [LOGGING_GUIDELINES.md](LOGGING_GUIDELINES.md) | ログ設定詳細 |
