# エージェント向けドキュメントインデックス

> **対象**: Claude Code / AIエージェント
> **Last Updated**: 2026-02-01

---

## 高速ナビゲーション

### 実装タスク別ガイド

#### キャッシュ機能を実装したい

```
1. [必読] CLAUDE.md - 禁止事項確認
2. [必読] CACHE_SYSTEM.md - 既存キャッシュパターン
3. [参考] s3_cache_guide.md - S3統合
```

**キーファイル**:
- `src/utils/cache_manager.py` - UnifiedCacheManager
- `src/utils/storage_backend.py` - StorageBackend
- `src/backtest/cache.py` - LRUCache, SignalCache

#### データソースを確認したい

```
1. [必読] DATA_SOURCES.md - シグナル・銘柄のデータソース一覧
2. [参考] CACHE_SYSTEM.md - キャッシュシステム
```

**キーファイル**:
- `src/data/adapters/stock.py` - StockAdapter (Yahoo Finance)
- `config/asset_master.yaml` - 銘柄ユニバース定義
- `src/signals/` - 各シグナル実装

#### 新規シグナルを追加したい

```
1. [必読] SIGNAL_PIPELINE.md - シグナル追加手順
2. [必読] BACKTEST_STANDARD.md - 評価基準
3. [参考] DATA_SOURCES.md - 利用可能データソース
4. [参考] ARCHITECTURE.md - パイプラインフロー
```

**キーファイル**:
- `src/signals/base.py` - BaseSignal
- `src/signals/registry.py` - シグナルレジストリ
- `src/backtest/signal_precompute.py` - 事前計算

#### パイプライン処理を変更したい

```
1. [必読] ORCHESTRATOR_RESPONSIBILITIES.md - 責務定義
2. [必読] ARCHITECTURE.md - 処理フロー
3. [参考] BACKTEST_STANDARD.md - 入出力規格
```

**キーファイル**:
- `src/orchestrator/pipeline.py` - Pipeline
- `src/orchestrator/unified_executor.py` - UnifiedExecutor
- `src/orchestrator/data_preparation.py` - DataPreparation

#### バックテストエンジンを追加したい

```
1. [必読] ENGINE_INTEGRATION.md - 統合ガイド
2. [必読] BACKTEST_STANDARD.md - 必須パラメータ
3. [参考] ARCHITECTURE.md - エンジン配置
```

**キーファイル**:
- `src/backtest/fast_engine.py` - FastBacktestEngine
- `src/backtest/base.py` - BaseEngine
- `src/backtest/streaming_engine.py` - StreamingEngine

#### リスク管理機能を実装したい

```
1. [必読] CMD016_INTEGRATION.md - 統合順序
2. [必読] ARCHITECTURE.md - リスク層の位置
3. [参考] SIGNAL_PIPELINE.md - シグナル連携
```

**キーファイル**:
- `src/orchestrator/cmd016_integrator.py` - CMD016Integrator
- `src/risk/` - リスク管理モジュール群

#### ML機能を実装したい

```
1. [必読] ML_COMPONENTS.md - 利用可能モジュール
2. [必読] ARCHITECTURE.md - ML層の位置
3. [参考] SIGNAL_PIPELINE.md - シグナル生成連携
```

**キーファイル**:
- `src/ml/__init__.py` - 公開API一覧
- `src/ml/return_predictor.py` - リターン予測
- `src/ml/xgboost_stacking.py` - スタッキング

#### ログ・信頼性機能を実装したい

```
1. [必読] LOGGING_GUIDELINES.md - ログ基盤・信頼性評価
2. [必読] ARCHITECTURE.md - ログレイヤーの位置
3. [参考] TROUBLESHOOTING.md - ログ確認方法
```

**キーファイル**:
- `src/utils/pipeline_log_collector.py` - PipelineLogCollector
- `src/backtest/reliability.py` - ReliabilityCalculator
- `src/utils/progress_tracker.py` - ProgressTracker
- `src/utils/logger.py` - structlog設定、ブリッジ

---

## キーファイル直接アクセス

### カテゴリ別キーファイル

| カテゴリ | 確認すべきファイル | 目的 |
|---------|-------------------|------|
| **キャッシュ** | `src/utils/cache_manager.py` | 統一キャッシュAPI |
| | `src/utils/storage_backend.py` | S3/ローカル抽象化 |
| **設定** | `src/config/settings.py` | 設定管理 |
| | `config/default.yaml` | デフォルト設定値 |
| **パイプライン** | `src/orchestrator/pipeline.py` | 処理フロー |
| | `src/orchestrator/unified_executor.py` | 統一実行 |
| **バックテスト** | `src/backtest/fast_engine.py` | メインエンジン |
| | `src/backtest/signal_precompute.py` | シグナル事前計算 |
| **シグナル** | `src/signals/base.py` | シグナル基底クラス |
| | `src/signals/registry.py` | シグナルレジストリ |
| **ML** | `src/ml/__init__.py` | ML公開API |
| **ログ** | `src/utils/pipeline_log_collector.py` | パイプラインログ収集 |
| | `src/backtest/reliability.py` | 信頼性評価 |
| | `src/utils/progress_tracker.py` | 進捗追跡、Viewer連携 |

### 重要なクラス・関数

```python
# キャッシュ
from src.utils.cache_manager import UnifiedCacheManager, CacheType
from src.utils.storage_backend import get_storage_backend, StorageConfig

# 設定
from src.config.settings import Settings

# パイプライン
from src.orchestrator.pipeline import Pipeline
from src.orchestrator.unified_executor import UnifiedExecutor

# バックテスト
from src.backtest.fast_engine import FastBacktestEngine, FastBacktestConfig

# シグナル
from src.signals.base import BaseSignal
from src.signals.registry import SignalRegistry

# ML
from src.ml import ReturnPredictor, XGBoostStacker

# ログ・信頼性
from src.utils.pipeline_log_collector import PipelineLogCollector
from src.backtest.reliability import ReliabilityCalculator, ReliabilityAssessment
from src.utils.logger import set_log_collector, get_log_collector
```

---

## ドキュメント優先度マトリクス

### 新規実装時の読み順

| 優先度 | ドキュメント | 理由 |
|--------|-------------|------|
| **1** | CLAUDE.md | 禁止事項・必須確認事項 |
| **2** | ARCHITECTURE.md | 全体構造の理解 |
| **3** | 該当機能のドキュメント | 詳細な実装ガイド |
| **4** | BACKTEST_STANDARD.md | 入出力規格の確認 |

### バグ修正時の読み順

| 優先度 | ドキュメント | 理由 |
|--------|-------------|------|
| **1** | TROUBLESHOOTING.md | 既知の問題確認 |
| **2** | LOGGING_GUIDELINES.md | ログ出力の理解 |
| **3** | 該当モジュールのdocstring | 実装詳細 |

### レビュー時の読み順

| 優先度 | ドキュメント | 理由 |
|--------|-------------|------|
| **1** | DESIGN_REVIEW_CHECKLIST.md | レビュー項目 |
| **2** | CLAUDE.md | 禁止事項の違反確認 |
| **3** | BACKTEST_STANDARD.md | 規格準拠確認 |

---

## よく使うgrepパターン

```bash
# キャッシュ関連の実装を探す
grep -r "UnifiedCacheManager\|CacheType\|StorageBackend" src/

# 設定の使用箇所を探す
grep -r "Settings\(\)\|get_cache_path" src/

# パイプラインステップを探す
grep -r "PipelineStep\|@step" src/orchestrator/

# シグナル実装を探す
grep -r "class.*Signal.*BaseSignal" src/signals/
```

---

## ドキュメント相互参照

```
CLAUDE.md (エージェントルール)
    ↓ 参照
AGENT_INDEX.md (本ドキュメント)
    ↓ 誘導
┌─────────────────────────────────────────────┐
│ 機能別ドキュメント                           │
├─────────────────────────────────────────────┤
│ ARCHITECTURE.md     ← 全体構造              │
│ DATA_SOURCES.md     ← データソース一覧      │
│ CACHE_SYSTEM.md     ← キャッシュ詳細        │
│ SIGNAL_PIPELINE.md  ← シグナル追加          │
│ CMD016_INTEGRATION.md ← リスク管理統合      │
│ ML_COMPONENTS.md    ← ML機能                │
│ ENGINE_INTEGRATION.md ← エンジン追加        │
└─────────────────────────────────────────────┘
    ↓ 規格参照
BACKTEST_STANDARD.md (バックテスト規格)
```

---

## 禁止事項クイックリファレンス

```
❌ キャッシュディレクトリのハードコード
❌ 独自キャッシュクラスの作成
❌ 直接ファイルI/O（StorageBackend経由で）
❌ 設定のハードコード
```

詳細は [CLAUDE.md](../CLAUDE.md) を参照。

---

## 関連ドキュメント

| ドキュメント | 概要 | 参照タイミング |
|-------------|------|---------------|
| [CLAUDE.md](../CLAUDE.md) | エージェント必須ルール | 常に最初に確認 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | システムアーキテクチャ | 実装前 |
| [DATA_SOURCES.md](DATA_SOURCES.md) | データソース・シグナル一覧 | データ確認時 |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | 問題解決 | エラー発生時 |
