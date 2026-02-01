# キャッシュシステムガイド

> **Last Updated**: 2026-02-01

---

## 目次

1. [キャッシュアーキテクチャ概要](#キャッシュアーキテクチャ概要)
2. [UnifiedCacheManagerの使い方](#unifiedcachemanagerの使い方)
3. [CacheTypeの種類と用途](#cachetypeの種類と用途)
4. [インクリメンタル共分散キャッシュ](#インクリメンタル共分散キャッシュ)
5. [シグナル事前計算キャッシュ](#シグナル事前計算キャッシュ)
6. [キャッシュ無効化とバージョニング](#キャッシュ無効化とバージョニング)

---

## キャッシュアーキテクチャ概要

### レイヤー構造

```
┌─────────────────────────────────────────────────────────┐
│                  Application Layer                       │
│   (FastBacktestEngine, Pipeline, UnifiedExecutor)        │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              UnifiedCacheManager                         │
│  - 統一API提供                                           │
│  - グローバルメモリ制限                                   │
│  - キャッシュポリシー管理                                 │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  SignalCache  │   │DataFrameCache │   │   LRUCache    │
│  (シグナル)   │   │  (DataFrame)  │   │    (汎用)     │
└───────────────┘   └───────────────┘   └───────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   StorageBackend                         │
│  - Local Mode: ファイルシステム                          │
│  - S3 Mode: AWS S3                                       │
│  - Hybrid Mode: ローカル + S3                            │
└─────────────────────────────────────────────────────────┘
```

### 設計原則

1. **統一インターフェース**: 全キャッシュは `CacheInterface` を実装
2. **S3互換性**: `StorageBackend` 経由で透過的にS3対応
3. **メモリ制御**: グローバルメモリ制限で OOM 防止
4. **遅延初期化**: 必要時にのみキャッシュを生成

---

## UnifiedCacheManagerの使い方

### 基本使用

```python
from src.utils.cache_manager import UnifiedCacheManager, CacheType

# マネージャー取得（シングルトン）
from src.utils.cache_manager import unified_cache_manager

# または新規作成
manager = UnifiedCacheManager(
    global_memory_limit_mb=1024,
    cache_base_dir="./cache",
)

# キャッシュ取得
signal_cache = manager.get_cache(CacheType.SIGNAL)
data_cache = manager.get_cache(CacheType.DATA)
```

### ショートカット関数

```python
from src.utils.cache_manager import (
    get_signal_cache,
    get_data_cache,
    get_dataframe_cache,
    clear_all_caches,
    get_cache_summary,
)

# シグナルキャッシュ取得
signal_cache = get_signal_cache()

# 全キャッシュクリア
clear_all_caches()

# 統計サマリ表示
print(get_cache_summary())
```

### S3バックエンド統合

```python
from src.utils.cache_manager import UnifiedCacheManager
from src.utils.storage_backend import StorageConfig

# S3設定
storage_config = StorageConfig(
    backend="s3",
    s3_bucket="my-bucket",
    s3_prefix=".cache",
    local_cache_enabled=True,
)

# S3対応マネージャー
manager = UnifiedCacheManager(
    storage_config=storage_config,
)
```

### 統計情報取得

```python
# 個別キャッシュの統計
stats = manager.get_stats("signal")
print(f"ヒット率: {stats.hit_rate:.2%}")
print(f"メモリ使用量: {stats.memory_bytes / 1024 / 1024:.1f} MB")

# 全キャッシュの統計
all_stats = manager.get_all_stats()
for name, stats in all_stats.items():
    print(f"{name}: {stats.hit_rate:.2%}")

# サマリ表示
print(manager.summary())
```

---

## CacheTypeの種類と用途

### SIGNAL - シグナル計算キャッシュ

```python
from src.utils.cache_manager import CacheType

cache = manager.get_cache(CacheType.SIGNAL)
```

**用途**: モメンタム、RSI等のシグナル計算結果
**キー形式**: `{signal_name}_{params}_{ticker}`
**データ形式**: numpy.ndarray / pandas.Series

**特徴**:
- メモリ + ディスク2階層
- バージョン管理による自動無効化
- 増分更新対応

### DATAFRAME - DataFrame汎用キャッシュ

```python
cache = manager.get_cache(CacheType.DATAFRAME)
```

**用途**: 中間計算結果のDataFrame
**キー形式**: 任意の文字列
**データ形式**: pandas.DataFrame / polars.DataFrame

**特徴**:
- Parquet形式で永続化
- 圧縮対応
- メモリエントリ数制限（大きいため）

### DATA - OHLCVデータキャッシュ

```python
cache = manager.get_cache(CacheType.DATA)
```

**用途**: 株価等の時系列データ
**キー形式**: `{ticker}_{start}_{end}`
**データ形式**: pandas.DataFrame

**特徴**:
- API呼び出し削減
- 有効期限管理
- S3バックエンド対応

### LRU - 汎用LRUキャッシュ

```python
cache = manager.get_cache(CacheType.LRU)
```

**用途**: 小さな計算結果の一時保存
**キー形式**: 任意の文字列
**データ形式**: 任意

**特徴**:
- メモリのみ
- LRU（Least Recently Used）eviction
- 高速アクセス

### INCREMENTAL - インクリメンタル計算キャッシュ

```python
cache = manager.get_cache(CacheType.INCREMENTAL)
```

**用途**: 増分計算が可能な統計量
**キー形式**: `{計算種別}_{params}`
**データ形式**: 計算状態オブジェクト

**特徴**:
- 中間状態の保存
- 増分更新
- 共分散行列等に使用

---

## インクリメンタル共分散キャッシュ

### 概要

共分散行列の計算を高速化するためのインクリメンタルキャッシュ。

```python
from src.backtest.covariance_cache import CovarianceCache, CovarianceConfig

# 設定
config = CovarianceConfig(
    lookback=252,
    min_periods=60,
    shrinkage_method="ledoit_wolf",
)

# キャッシュ作成
cov_cache = CovarianceCache(config)

# 共分散取得（自動的にキャッシュ）
cov_matrix = cov_cache.get_covariance(returns_df, as_of_date)

# 増分更新（新しいデータ追加時）
cov_cache.update_incremental(new_returns)
```

### 高速化効果

| シナリオ | 初回計算 | 2回目以降 |
|---------|---------|----------|
| 100銘柄 × 252日 | 0.5秒 | 0.01秒 |
| 500銘柄 × 252日 | 5秒 | 0.05秒 |
| 1000銘柄 × 252日 | 20秒 | 0.1秒 |

---

## シグナル事前計算キャッシュ

### 概要

バックテスト高速化のため、シグナルを事前計算してキャッシュ。

```python
from src.backtest.signal_precompute import SignalPrecomputer

# 事前計算器作成
precomputer = SignalPrecomputer(
    cache_dir=".cache/signals",
    storage_backend=backend,  # オプション：S3対応
)

# 全シグナル事前計算
precomputer.precompute_all(prices_df, config)

# バックテスト中の取得
signal_value = precomputer.get_signal_at_date(
    signal_name="momentum_20",
    ticker="SPY",
    date=rebalance_date,
)
```

### キャッシュ無効化条件

```python
from src.backtest.signal_precompute import PrecomputeMetadata

# メタデータでキャッシュ有効性をチェック
metadata = PrecomputeMetadata(
    signal_registry_hash="abc123",  # シグナル定義のハッシュ
    signal_config_hash="def456",    # パラメータのハッシュ
    universe_hash="ghi789",         # ユニバースのハッシュ
    version="1.2.0",                # ライブラリバージョン
)
```

キャッシュが無効化される条件:
1. **バージョン変更**: ライブラリバージョンが変わった
2. **シグナル定義変更**: シグナルの追加/削除
3. **パラメータ変更**: シグナルパラメータの変更
4. **ユニバース変更**: 銘柄リストの変更

---

## キャッシュ無効化とバージョニング

### 自動無効化

```python
# バージョン管理されたキャッシュ
PRECOMPUTE_VERSION = "1.2.0"

# メタデータで検証
is_valid, reason = cached_metadata.is_cache_valid(current_metadata)
if not is_valid:
    logger.info(f"キャッシュ無効化: {reason}")
    # 再計算
```

### 手動無効化

```python
# 特定キャッシュをクリア
manager.clear("signal")

# タイプ別クリア
manager.clear_by_type(CacheType.SIGNAL)

# 全クリア
manager.clear_all()

# 期限切れのみ削除
manager.cleanup_all()
```

### ローカルキャッシュクリア（CLI）

```bash
# シグナルキャッシュ
rm -rf .cache/signals/

# 全キャッシュ
rm -rf .cache/

# ローカルキャッシュ（S3モード時）
rm -rf /tmp/.backtest_cache/
```

---

## カスタムキャッシュの追加

### アダプターパターン

```python
from src.utils.cache_manager import CacheInterface, CacheType, UnifiedCacheStats

class MyCustomCacheAdapter(CacheInterface[Any]):
    """カスタムキャッシュのアダプター"""

    def __init__(self, name: str = "my_cache"):
        self._name = name
        self._data: dict[str, Any] = {}
        self._hits = 0
        self._misses = 0

    @property
    def cache_type(self) -> CacheType:
        return CacheType.LRU  # または適切なタイプ

    @property
    def name(self) -> str:
        return self._name

    def get(self, key: str) -> Any | None:
        if key in self._data:
            self._hits += 1
            return self._data[key]
        self._misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        self._data[key] = value

    def clear(self) -> None:
        self._data.clear()

    def get_stats(self) -> UnifiedCacheStats:
        return UnifiedCacheStats(
            name=self._name,
            cache_type=self.cache_type,
            hits=self._hits,
            misses=self._misses,
            size=len(self._data),
        )

# 登録
manager.register_cache("my_cache", MyCustomCacheAdapter())
```

---

## 関連ドキュメント

| ドキュメント | 概要 |
|-------------|------|
| [s3_cache_guide.md](s3_cache_guide.md) | S3キャッシュ設定 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | システムアーキテクチャ |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | キャッシュ問題の解決 |
