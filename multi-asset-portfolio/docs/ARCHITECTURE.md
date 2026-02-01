# システムアーキテクチャ

> **Last Updated**: 2026-02-01

---

## 目次

1. [全体アーキテクチャ](#全体アーキテクチャ)
2. [レイヤー構造](#レイヤー構造)
3. [パイプラインフロー](#パイプラインフロー)
4. [モジュール間依存関係](#モジュール間依存関係)
5. [設計原則](#設計原則)

---

## 全体アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                      CLI / Entry Points                      │
│              (src/main.py, scripts/run_backtest.py)          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator Layer                        │
├─────────────────────────────────────────────────────────────┤
│  UnifiedExecutor │ Pipeline │ DataPreparation │ CMD016      │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│    Signal     │    │   Strategy    │    │  Allocation   │
│    Layer      │    │    Layer      │    │    Layer      │
├───────────────┤    ├───────────────┤    ├───────────────┤
│ - momentum    │    │ - evaluator   │    │ - HRP         │
│ - mean_revert │    │ - gates       │    │ - risk_parity │
│ - regime      │    │ - ensemble    │    │ - NCO         │
│ - technical   │    │ - smoothing   │    │ - CVaR        │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Backtest Layer                          │
├─────────────────────────────────────────────────────────────┤
│  FastEngine │ StreamingEngine │ SignalPrecompute │ Cov      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Cache Layer                            │
├─────────────────────────────────────────────────────────────┤
│  UnifiedCacheManager │ SignalCache │ DataFrameCache │ LRU   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Storage Layer                           │
├─────────────────────────────────────────────────────────────┤
│       StorageBackend (Local / S3 / Hybrid)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Data Layer                             │
├─────────────────────────────────────────────────────────────┤
│  DataFetcher │ UniverseLoader │ DataCache │ QualityCheck    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Logging Layer                           │
├─────────────────────────────────────────────────────────────┤
│  PipelineLogCollector │ ReliabilityCalculator │ ProgressTracker │
└─────────────────────────────────────────────────────────────┘
```

---

## レイヤー構造

### 1. CLI / Entry Points

| ファイル | 目的 |
|---------|------|
| `src/main.py` | メインエントリポイント |
| `scripts/run_backtest.py` | バックテスト実行スクリプト |
| `scripts/backtest_results.py` | 結果確認スクリプト |

### 2. Orchestrator Layer

パイプライン全体の制御を担当。

| モジュール | 責務 |
|-----------|------|
| `UnifiedExecutor` | バックテスト統一実行、チェックポイント管理 |
| `Pipeline` | 処理ステップの定義と実行順序制御 |
| `DataPreparation` | データ前処理、品質チェック |
| `CMD016Integrator` | 高度な機能（VIX、ドローダウン保護等）の統合 |

### 3. Signal Layer

シグナル生成を担当。

| モジュール | 責務 |
|-----------|------|
| `momentum.py` | モメンタムシグナル |
| `mean_reversion.py` | 平均回帰シグナル |
| `regime_detector.py` | 市場レジーム検出 |
| `technical.py` | テクニカル指標 |
| `registry.py` | シグナルレジストリ |

### 4. Strategy Layer

戦略評価とアンサンブルを担当。

| モジュール | 責務 |
|-----------|------|
| `evaluator.py` | 戦略パフォーマンス評価 |
| `gates.py` | ハードゲートによるフィルタリング |
| `ensemble.py` | 戦略アンサンブル |
| `smoothing.py` | シグナル平滑化 |

### 5. Allocation Layer

ポートフォリオ配分を担当。

| モジュール | 責務 |
|-----------|------|
| `hrp.py` | 階層的リスクパリティ |
| `risk_parity.py` | リスクパリティ |
| `nco.py` | Nested Clustered Optimization |
| `cvar.py` | CVaR最適化 |

### 6. Backtest Layer

バックテスト実行を担当。

| モジュール | 責務 |
|-----------|------|
| `fast_engine.py` | 高速バックテストエンジン |
| `streaming_engine.py` | ストリーミングエンジン |
| `signal_precompute.py` | シグナル事前計算 |
| `covariance_cache.py` | 共分散キャッシュ |

### 7. Cache Layer

キャッシュ管理を担当。

| モジュール | 責務 |
|-----------|------|
| `cache_manager.py` | 統一キャッシュ管理 |
| `cache.py` | 基本キャッシュ実装 |

### 8. Storage Layer

永続化を担当。

| モジュール | 責務 |
|-----------|------|
| `storage_backend.py` | S3/ローカル抽象化 |

### 9. Data Layer

データ取得を担当。

| モジュール | 責務 |
|-----------|------|
| `adapters/stock.py` | 株式データアダプター |
| `universe_loader.py` | ユニバース読み込み |
| `cache.py` | データキャッシュ |

### 10. Logging Layer

ログ収集・信頼性評価を担当。

| モジュール | 責務 |
|-----------|------|
| `pipeline_log_collector.py` | パイプライン実行ログの統一収集 |
| `reliability.py` | 信頼性スコア計算 |
| `progress_tracker.py` | 進捗追跡、Viewer連携 |
| `logger.py` | structlog設定、ログブリッジ |

**ログフロー**:
```
structlog
    │
    ├──> Console (CLI)
    │
    └──> PipelineLogCollector
              │
              ├──> ProgressTracker (SSE → Viewer)
              │
              └──> logs.jsonl (永続化)
                       │
                       └──> ReliabilityCalculator
                                 │
                                 └──> result.reliability
```

---

## パイプラインフロー

### 17ステップ処理フロー

```
1. INITIALIZE
   └─ 設定読み込み、キャッシュ初期化

2. DATA_FETCH
   └─ 価格データ取得（キャッシュ優先）

3. QUALITY_CHECK
   └─ データ品質検証、欠損処理

4. REGIME_DETECTION
   └─ 市場レジーム判定

5. SIGNAL_GENERATION
   └─ 各種シグナル計算

6. STRATEGY_EVALUATION
   └─ 戦略パフォーマンス評価

7. GATE_CHECK
   └─ ハードゲートによるフィルタリング

8. ENSEMBLE_COMBINE
   └─ 戦略アンサンブル

9. STRATEGY_WEIGHTING
   └─ 戦略重み計算

10. RISK_ESTIMATION
    └─ リスク指標計算

11. ASSET_ALLOCATION
    └─ HRP/リスクパリティ配分

12. DYNAMIC_WEIGHTING
    └─ 動的重み調整

13. CMD016_INTEGRATION
    └─ VIX/ドローダウン保護等

14. CMD017_INTEGRATION
    └─ 追加機能統合

15. SMOOTHING
    └─ シグナル平滑化

16. ANOMALY_DETECTION
    └─ 異常値検出・対処

17. OUTPUT_GENERATION
    └─ 結果出力
```

### 処理フロー図

```
価格データ
    │
    ▼
┌─────────────┐
│ 品質チェック │
└─────────────┘
    │
    ▼
┌─────────────┐     ┌─────────────┐
│レジーム検出 │────▶│適応パラメータ│
└─────────────┘     └─────────────┘
    │                     │
    ▼                     ▼
┌─────────────┐     ┌─────────────┐
│シグナル生成 │◀────│パラメータ注入│
└─────────────┘     └─────────────┘
    │
    ▼
┌─────────────┐
│ ゲートチェック│
└─────────────┘
    │
    ▼
┌─────────────┐
│アンサンブル │
└─────────────┘
    │
    ▼
┌─────────────┐     ┌─────────────┐
│ アセット配分 │────▶│リスク調整   │
└─────────────┘     └─────────────┘
    │
    ▼
┌─────────────┐
│  最終重み   │
└─────────────┘
```

---

## モジュール間依存関係

### 依存関係図

```
src/main.py
    │
    ├─▶ src/orchestrator/unified_executor.py
    │       │
    │       ├─▶ src/orchestrator/pipeline.py
    │       ├─▶ src/backtest/fast_engine.py
    │       └─▶ src/utils/cache_manager.py
    │
    └─▶ src/config/settings.py
            │
            └─▶ config/default.yaml
```

### 主要な依存方向

1. **上位 → 下位**: Orchestrator → Backtest → Cache → Storage
2. **横方向禁止**: Signal ↛ Strategy（直接参照禁止）
3. **設定は最下層**: 全モジュールが `settings.py` に依存可能

---

## 設計原則

### 1. 依存性注入（DI）

```python
# ✅ 良い例：依存性注入
class FastBacktestEngine:
    def __init__(self, config: FastBacktestConfig, cache: CacheInterface):
        self._config = config
        self._cache = cache

# ❌ 悪い例：直接生成
class FastBacktestEngine:
    def __init__(self):
        self._cache = LRUCache()  # 直接生成
```

### 2. インターフェース分離

```python
# 統一インターフェース
class CacheInterface(ABC):
    @abstractmethod
    def get(self, key: str) -> Any: ...
    @abstractmethod
    def put(self, key: str, value: Any) -> None: ...

# 具象実装
class SignalCache(CacheInterface): ...
class DataFrameCache(CacheInterface): ...
```

### 3. 単一責任

| モジュール | 単一責任 |
|-----------|---------|
| `SignalPrecomputer` | シグナル事前計算のみ |
| `StorageBackend` | ファイルI/Oのみ |
| `UnifiedExecutor` | 実行制御のみ |

### 4. 設定の外部化

```python
# ✅ 良い例：設定から読み込み
from src.config.settings import Settings
settings = Settings()
max_workers = settings.optimization.max_workers

# ❌ 悪い例：ハードコード
max_workers = 8
```

---

## 関連ドキュメント

| ドキュメント | 概要 |
|-------------|------|
| [CACHE_SYSTEM.md](CACHE_SYSTEM.md) | キャッシュシステム詳細 |
| [ORCHESTRATOR_RESPONSIBILITIES.md](ORCHESTRATOR_RESPONSIBILITIES.md) | オーケストレータ責務 |
| [ENGINE_INTEGRATION.md](ENGINE_INTEGRATION.md) | エンジン追加ガイド |
| [SIGNAL_PIPELINE.md](SIGNAL_PIPELINE.md) | シグナルパイプライン |
| [LOGGING_GUIDELINES.md](LOGGING_GUIDELINES.md) | ログ基盤・信頼性評価 |
