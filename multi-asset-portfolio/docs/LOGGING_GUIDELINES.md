# Logging Guidelines

Multi-Asset Portfolio System のログレベル統一ガイドライン。

## Log Levels

### ERROR

**使用場面**: システム継続不可能なエラー。即座に対応が必要。

```python
# 例: データベース接続失敗
logger.error("Failed to connect to database: %s", e)

# 例: 必須ファイルが見つからない
logger.error("Required config file not found: %s", path)

# 例: クリティカルな計算エラー
logger.error("Portfolio optimization failed: division by zero")
```

**基準**:
- 処理を続行できないエラー
- データ整合性に影響するエラー
- ユーザーに通知が必要なエラー
- オンコール対応が必要な状況

### WARNING

**使用場面**: 予期しない状態だが継続可能。フォールバック発動時。

```python
# 例: フォールバックモード発動
logger.warning("Anomaly detected, switching to fallback mode: %s", mode)

# 例: データ品質問題（処理は継続）
logger.warning("Missing data for %s, using interpolation", symbol)

# 例: 非推奨機能の使用
logger.warning("Deprecated function called: use new_func() instead")

# 例: パフォーマンス閾値超過
logger.warning("Backtest took %.1fs (threshold: %.1fs)", elapsed, threshold)
```

**基準**:
- 処理は継続できるが、予期しない状態
- フォールバック/リカバリーが発動した
- 将来問題になりうる状態
- 監視・アラート対象

### INFO

**使用場面**: 重要な処理の開始/完了。設定変更。正常な業務イベント。

```python
# 例: 処理開始/完了
logger.info("Starting backtest for %d assets", len(universe))
logger.info("Backtest completed: Sharpe=%.2f, Return=%.1f%%", sharpe, ret)

# 例: 設定変更
logger.info("Rebalance frequency changed: %s -> %s", old, new)

# 例: 重要なイベント
logger.info("Signal registered: %s (category=%s)", name, category)
logger.info("Cache cleared: %d entries removed", count)
```

**基準**:
- 処理のマイルストーン（開始/完了）
- 設定・状態の変更
- 運用上重要なイベント
- 本番環境で常時有効

### DEBUG

**使用場面**: 詳細なデバッグ情報。開発・トラブルシューティング用。

```python
# 例: 計算の中間結果
logger.debug("Covariance matrix shape: %s", cov.shape)
logger.debug("Signal scores: mean=%.4f, std=%.4f", mean, std)

# 例: ループ内の詳細
logger.debug("Processing asset %d/%d: %s", i, n, symbol)

# 例: 内部状態
logger.debug("Cache hit: key=%s", key)
logger.debug("Weight constraints applied: min=%.2f, max=%.2f", min_w, max_w)
```

**基準**:
- 開発・デバッグ時のみ必要
- 高頻度で出力される情報
- 内部状態の詳細
- 本番では通常無効化

## Best Practices

### 1. 構造化ログフォーマット

```python
# Good: パラメータ化
logger.info("Processed %d assets in %.2fs", count, elapsed)

# Bad: 文字列連結
logger.info("Processed " + str(count) + " assets in " + str(elapsed) + "s")
```

### 2. 適切なコンテキスト

```python
# Good: 識別可能な情報を含める
logger.error("Failed to fetch data for %s: %s", symbol, error)

# Bad: コンテキスト不足
logger.error("Data fetch failed")
```

### 3. 例外ログ

```python
# Good: exc_info=True で完全なトレースバック
try:
    result = compute()
except Exception as e:
    logger.error("Computation failed: %s", e, exc_info=True)

# Bad: スタックトレースなし
except Exception as e:
    logger.error("Error: %s", e)
```

### 4. ログレベルの条件チェック

```python
# Good: 高コストな処理は条件付き
if logger.isEnabledFor(logging.DEBUG):
    logger.debug("Matrix details: %s", expensive_matrix_repr())

# Bad: 常に評価
logger.debug("Matrix details: %s", expensive_matrix_repr())
```

### 5. センシティブ情報

```python
# Good: マスキング
logger.info("API key configured: %s...%s", key[:4], key[-4:])

# Bad: 平文
logger.info("API key: %s", key)
```

## Module-Specific Guidelines

### backtest/

| イベント | レベル | 例 |
|---------|--------|-----|
| バックテスト開始 | INFO | `Starting backtest: 2020-01-01 to 2024-12-31` |
| バックテスト完了 | INFO | `Backtest completed: Sharpe=1.5, MaxDD=-15%` |
| リバランス実行 | DEBUG | `Rebalance at 2024-01-15: turnover=12%` |
| 計算エラー | ERROR | `Portfolio optimization failed` |
| データ欠損（継続可） | WARNING | `Missing price for AAPL on 2024-01-02` |

### signals/

| イベント | レベル | 例 |
|---------|--------|-----|
| シグナル登録 | INFO | `Signal registered: momentum (category=trend)` |
| シグナル計算完了 | DEBUG | `Signal computed: mean=0.15, std=0.32` |
| データ取得失敗（フォールバック） | WARNING | `FRED API failed, using mock data` |
| 無効なパラメータ | ERROR | `Invalid lookback: must be positive` |

### orchestrator/

| イベント | レベル | 例 |
|---------|--------|-----|
| パイプライン開始 | INFO | `Pipeline started: universe=50 assets` |
| 異常検出 | WARNING | `Anomaly detected: VIX spike` |
| フォールバック発動 | WARNING | `Fallback mode: HOLD_PREVIOUS` |
| パイプライン失敗 | ERROR | `Pipeline failed: data validation error` |

### data/

| イベント | レベル | 例 |
|---------|--------|-----|
| データ取得開始 | INFO | `Fetching data for 50 symbols` |
| キャッシュヒット | DEBUG | `Cache hit: prices_2024-01-01` |
| API制限到達 | WARNING | `Rate limit reached, waiting 60s` |
| データ品質問題 | WARNING | `Data quality issue: 5% missing values` |

## Configuration

### 開発環境

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### 本番環境

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 特定モジュールのみDEBUG
logging.getLogger('src.backtest').setLevel(logging.DEBUG)
```

### structlog (推奨)

```python
import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()
logger.info("backtest_completed", sharpe=1.5, max_dd=-0.15)
```

## Migration Checklist

既存コードをガイドラインに準拠させる際のチェックリスト：

- [ ] `print()` を適切な `logger.*()` に置換
- [ ] ERROR: 本当に処理続行不可能か確認
- [ ] WARNING: フォールバック/リカバリーがあるか確認
- [ ] INFO: 本番で常時出力して問題ないか確認
- [ ] DEBUG: 高頻度ログは条件付きにする
- [ ] 例外ログに `exc_info=True` を追加
- [ ] センシティブ情報のマスキング

## References

- [Python Logging HOWTO](https://docs.python.org/3/howto/logging.html)
- [structlog Documentation](https://www.structlog.org/)
- [12 Factor App: Logs](https://12factor.net/logs)
