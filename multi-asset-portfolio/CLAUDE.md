# multi-asset-portfolio アーキテクチャガイド

> **Last Updated**: 2026-02-01
> **対象**: Claude Code / AIエージェント

---

## クイックナビゲーション

**タスク別ドキュメントマップ**: [docs/AGENT_INDEX.md](docs/AGENT_INDEX.md)

| 目的 | 参照ドキュメント |
|------|-----------------|
| データソース確認 | [DATA_SOURCES.md](docs/DATA_SOURCES.md) |
| キャッシュ実装 | [CACHE_SYSTEM.md](docs/CACHE_SYSTEM.md) |
| シグナル追加 | [SIGNAL_PIPELINE.md](docs/SIGNAL_PIPELINE.md) |
| パイプライン変更 | [ORCHESTRATOR_RESPONSIBILITIES.md](docs/ORCHESTRATOR_RESPONSIBILITIES.md) |
| エンジン追加 | [ENGINE_INTEGRATION.md](docs/ENGINE_INTEGRATION.md) |
| リスク管理 | [CMD016_INTEGRATION.md](docs/CMD016_INTEGRATION.md) |
| ML機能 | [ML_COMPONENTS.md](docs/ML_COMPONENTS.md) |
| トラブルシューティング | [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) |

---

## 実装前の必須確認事項

新機能を実装する前に、以下を**必ず確認**せよ。独自実装を避け、既存の抽象化を活用すること。

### 1. キャッシュ実装時

**確認すべきファイル:**
```
src/utils/cache_manager.py    # UnifiedCacheManager（統一インターフェース）
src/utils/storage_backend.py  # StorageBackend（S3/ローカル抽象化）
```

**既存のキャッシュタイプ:**
```python
from src.utils.cache_manager import CacheType

CacheType.SIGNAL      # シグナル計算キャッシュ
CacheType.DATAFRAME   # DataFrame汎用キャッシュ
CacheType.DATA        # OHLCVデータキャッシュ
CacheType.LRU         # 汎用LRUキャッシュ
CacheType.INCREMENTAL # インクリメンタル計算キャッシュ
```

**正しいパターン:**
```python
# ❌ 悪い例：独自キャッシュ作成
class MyCache:
    def __init__(self):
        self._cache_dir = Path(".cache/my_cache")  # ハードコード

# ✅ 良い例：既存の抽象化を使用
from src.utils.cache_manager import UnifiedCacheManager, CacheType

cache_mgr = UnifiedCacheManager(storage_backend=backend)
cache = cache_mgr.get_cache(CacheType.SIGNAL)
```

### 2. ストレージ実装時

**確認すべきファイル:**
```
src/utils/storage_backend.py  # StorageBackend
src/config/settings.py        # StorageSettings
```

**正しいパターン:**
```python
# ❌ 悪い例：直接ファイルI/O
with open(".cache/data.pkl", "wb") as f:
    pickle.dump(data, f)

# ✅ 良い例：StorageBackend経由
from src.utils.storage_backend import get_storage_backend

backend = get_storage_backend(storage_config)
backend.write_pickle(data, "data.pkl")
```

### 3. パイプライン処理時

**確認すべきファイル:**
```
src/orchestrator/pipeline.py          # Pipeline（処理フロー）
src/orchestrator/unified_executor.py  # UnifiedExecutor（統一実行）
src/orchestrator/data_preparation.py  # DataPreparation
```

**既存のパイプラインステップ:**
```
INITIALIZE → DATA_FETCH → QUALITY_CHECK → REGIME_DETECTION →
SIGNAL_GENERATION → STRATEGY_EVALUATION → GATE_CHECK →
ENSEMBLE_COMBINE → STRATEGY_WEIGHTING → RISK_ESTIMATION →
ASSET_ALLOCATION → DYNAMIC_WEIGHTING → CMD016_INTEGRATION →
CMD017_INTEGRATION → SMOOTHING → ANOMALY_DETECTION → OUTPUT_GENERATION
```

### 4. 設定管理時

**確認すべきファイル:**
```
src/config/settings.py    # Settings（Pydantic設定）
config/default.yaml       # デフォルト設定
```

**正しいパターン:**
```python
# ❌ 悪い例：ハードコード設定
cache_path = ".cache/signals"

# ✅ 良い例：Settings経由
from src.config.settings import Settings
settings = Settings()
cache_path = settings.storage.local_cache_path
```

---

## アーキテクチャ概要

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
├─────────────────────────────────────────────────────────┤
│  UnifiedExecutor  │  Pipeline  │  DataPreparation       │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              UnifiedCacheManager                         │
├─────────────────────────────────────────────────────────┤
│  SignalCache │ DataFrameCache │ CovarianceCache │ etc.  │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   StorageBackend                         │
├─────────────────────────────────────────────────────────┤
│     Local Mode   │   S3 Mode   │   Hybrid Mode          │
└─────────────────────────────────────────────────────────┘
```

詳細: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## 禁止事項

1. **キャッシュディレクトリのハードコード禁止**
   - 新しい `.cache/xxx` パスを直接書かない
   - 既存の設定やStorageBackendを使用

2. **独自キャッシュクラスの作成禁止**
   - `UnifiedCacheManager` のアダプターとして実装
   - `CacheInterface` を継承

3. **直接ファイルI/O禁止**
   - `StorageBackend` 経由でS3互換性を維持

4. **設定のハードコード禁止**
   - `Settings` クラスまたは `config/default.yaml` を使用

---

## 実装チェックリスト

新機能実装時は以下を確認:

- [ ] `grep -r "class.*Cache" src/` で既存キャッシュを確認したか
- [ ] `grep -r "StorageBackend" src/` で既存ストレージ抽象化を確認したか
- [ ] `src/utils/cache_manager.py` を読んだか
- [ ] `src/config/settings.py` で設定項目を確認したか
- [ ] 既存のパターン（アダプター等）に従っているか

---

## 主要モジュール一覧

| モジュール | 責務 | 確認タイミング |
|-----------|------|---------------|
| `cache_manager.py` | キャッシュ統一管理 | キャッシュ実装時 |
| `storage_backend.py` | S3/ローカル抽象化 | ファイルI/O時 |
| `settings.py` | 設定管理 | 設定追加時 |
| `pipeline.py` | 処理フロー | 新ステップ追加時 |
| `unified_executor.py` | バックテスト実行 | 実行ロジック変更時 |
| `signal_precompute.py` | シグナル事前計算 | シグナル追加時 |
| `covariance_cache.py` | 共分散キャッシュ | リスク計算変更時 |

---

## ドキュメント構造

```
docs/
├── AGENT_INDEX.md              # エージェント高速ナビ
│
├── # アーキテクチャ・設計
├── ARCHITECTURE.md             # システム全体構造
├── ORCHESTRATOR_RESPONSIBILITIES.md
│
├── # セットアップ・デプロイ
├── INSTALLATION.md             # インストール手順
├── DEPLOYMENT.md               # 本番環境デプロイ
│
├── # バックテスト
├── BACKTEST_STANDARD.md        # バックテスト統一規格
├── ENGINE_INTEGRATION.md       # エンジン追加ガイド
├── backtest_acceleration_options.md
│
├── # キャッシュ・ストレージ
├── CACHE_SYSTEM.md             # キャッシュシステム詳細
├── s3_cache_guide.md           # S3キャッシュ設定
│
├── # シグナル・パイプライン
├── SIGNAL_PIPELINE.md          # シグナル追加手順
├── CMD016_INTEGRATION.md       # CMD016機能統合
│
├── # MLコンポーネント
├── ML_COMPONENTS.md            # ML機能ガイド
│
├── # 運用・トラブルシューティング
├── TROUBLESHOOTING.md          # 問題解決ガイド
├── performance_report_guide.md
├── LOGGING_GUIDELINES.md
│
├── # レビュー・プロセス
├── DESIGN_REVIEW_CHECKLIST.md
│
└── # 非技術者向け
    └── system_overview_for_management.md
```
