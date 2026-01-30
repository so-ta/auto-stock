# 500行超ファイルリファクタリング計画

**作成日:** 2026-01-29
**作成者:** ashigaru4
**対象:** multi-asset-portfolio/src/

## 1. 概要

### 現状
- 500行超ファイル: **96ファイル（全体の64%）**
- 平均行数: 約850行
- 最大: 2,175行（fast_engine.py）

### 目標
- 各モジュール500行以下
- 単一責任原則の適用
- テスタビリティの向上

---

## 2. 優先度リスト（上位20）

| Rank | Priority | Lines | Imports | File |
|------|----------|-------|---------|------|
| 1 | 22.4 | 862 | 26 | src/config/settings.py |
| 2 | 8.7 | 2,175 | 4 | src/backtest/fast_engine.py |
| 3 | 7.6 | 1,888 | 4 | src/orchestrator/pipeline.py |
| 4 | 7.4 | 919 | 8 | src/backtest/base.py |
| 5 | 5.3 | 658 | 8 | src/utils/logger.py |
| 6 | 3.9 | 1,945 | 2 | src/backtest/engine.py |
| 7 | 3.7 | 1,234 | 3 | src/backtest/vectorbt_engine.py |
| 8 | 3.6 | 715 | 5 | src/allocation/allocator.py |
| 9 | 2.3 | 1,136 | 2 | src/signals/trend.py |
| 10 | 2.2 | 739 | 3 | src/signals/mean_reversion.py |

**Priority Score = (Imports × Lines) / 1000**

---

## 3. 上位10ファイルの分割案

### 3.1 src/config/settings.py (862行, 24クラス)

**現状分析:**
- 24個のPydanticモデル/Enumが1ファイルに集中
- 設定カテゴリが混在（Allocation, Backtest, Signals, Risk等）

**分割案:**
```
src/config/
├── settings.py          # メインSettings（~200行）
├── enums.py             # Enum定義（~100行）
├── allocation.py        # 配分関連設定（~150行）
├── backtest.py          # バックテスト設定（~150行）
├── signals.py           # シグナル設定（~100行）
└── risk.py              # リスク管理設定（~100行）
```

**影響範囲:** 26ファイル（最高優先度）
**工数見積:** 2時間

---

### 3.2 src/backtest/fast_engine.py (2,175行, 6クラス)

**現状分析:**
- FastBacktestEngine（メインクラス）が巨大
- Numba JIT関数が混在
- シミュレーションロジックが複雑

**分割案:**
```
src/backtest/
├── fast_engine.py       # FastBacktestEngine（~500行）
├── fast_config.py       # FastBacktestConfig（~150行）
├── fast_simulation.py   # シミュレーションロジック（~600行）
├── fast_numba.py        # Numba JIT関数（~400行）
├── fast_unified.py      # 統一インターフェース（~300行）
└── fast_result.py       # SimulationResult（~200行）
```

**影響範囲:** 4ファイル
**工数見積:** 4時間

---

### 3.3 src/orchestrator/pipeline.py (1,888行, 6クラス)

**現状分析:**
- Pipelineクラスが47メソッドを持つ
- CMD016/CMD017統合ロジックが複雑
- 各ステップが独立可能

**分割案:**
```
src/orchestrator/
├── pipeline.py          # Pipeline基盤（~400行）
├── pipeline_steps.py    # 各ステップ実装（~500行）
├── pipeline_cmd016.py   # CMD016統合（~300行）
├── pipeline_cmd017.py   # CMD017統合（~400行）
└── pipeline_result.py   # PipelineResult（~200行）
```

**影響範囲:** 4ファイル
**工数見積:** 3時間

---

### 3.4 src/backtest/base.py (919行, 7クラス)

**現状分析:**
- 共通インターフェース定義
- 結果クラスが複数
- ユーティリティ関数が混在

**分割案:**
```
src/backtest/
├── base.py              # 基底クラス・インターフェース（~300行）
├── unified_config.py    # UnifiedBacktestConfig（~200行）
├── unified_result.py    # UnifiedBacktestResult（~250行）
└── rebalance_utils.py   # リバランスユーティリティ（~150行）
```

**影響範囲:** 8ファイル
**工数見積:** 2時間

---

### 3.5 src/utils/logger.py (658行, 4クラス)

**現状分析:**
- AuditLoggerが多くのメソッドを持つ
- レンダラーとユーティリティが混在

**分割案:**
```
src/utils/
├── logger.py            # 基本ロガー設定（~200行）
├── audit_logger.py      # AuditLogger（~300行）
└── log_utils.py         # ユーティリティ（~150行）
```

**影響範囲:** 8ファイル
**工数見積:** 1.5時間

---

### 3.6 src/backtest/engine.py (1,945行, 5クラス)

**現状分析:**
- BacktestEngineが53メソッドを持つ
- Walk-Forwardロジックが複雑

**分割案:**
```
src/backtest/
├── engine.py            # BacktestEngine基盤（~500行）
├── engine_walkforward.py # Walk-Forward実装（~500行）
├── engine_simulation.py  # シミュレーション（~500行）
└── engine_result.py      # BacktestResult（~300行）
```

**影響範囲:** 2ファイル
**工数見積:** 4時間

---

### 3.7 src/backtest/vectorbt_engine.py (1,234行, 3クラス)

**現状分析:**
- VectorBT統合ロジック
- トップレベル関数が8個

**分割案:**
```
src/backtest/
├── vectorbt_engine.py   # VectorBTEngine（~400行）
├── vectorbt_utils.py    # ユーティリティ関数（~400行）
└── vectorbt_result.py   # VectorBTResult（~300行）
```

**影響範囲:** 3ファイル
**工数見積:** 2.5時間

---

### 3.8 src/allocation/allocator.py (715行, 5クラス)

**現状分析:**
- AssetAllocatorが17メソッド
- 配分メソッドが混在

**分割案:**
```
src/allocation/
├── allocator.py         # AssetAllocator基盤（~300行）
├── allocator_hrp.py     # HRP実装（~200行）
└── allocator_result.py  # AllocationResult（~150行）
```

**影響範囲:** 5ファイル
**工数見積:** 1.5時間

---

### 3.9 src/signals/trend.py (1,136行, 6クラス)

**現状分析:**
- 6つのトレンドシグナルクラス
- 各クラスが独立

**分割案:**
```
src/signals/
├── trend.py             # 基本トレンドシグナル（~300行）
├── dual_momentum.py     # DualMomentum（~200行）
├── adaptive_trend.py    # AdaptiveTrend（~200行）
├── cross_asset.py       # CrossAssetMomentum（~200行）
└── multi_timeframe.py   # MultiTimeframe（~200行）
```

**影響範囲:** 2ファイル
**工数見積:** 2時間

---

### 3.10 src/signals/mean_reversion.py (739行, 5クラス)

**現状分析:**
- 5つの平均回帰シグナルクラス
- RollingCacheMixinが共通

**分割案:**
```
src/signals/
├── mean_reversion.py    # 基本MR + Mixin（~300行）
├── bollinger.py         # BollingerReversion（~150行）
├── rsi_signal.py        # RSISignal（~150行）
└── zscore_signal.py     # ZScoreReversion（~150行）
```

**影響範囲:** 3ファイル
**工数見積:** 1.5時間

---

## 4. 段階的実行スケジュール

### Phase 1: 高優先度（影響範囲大）
| タスク | 対象 | 工数 | 依存 |
|--------|------|------|------|
| P1-1 | settings.py分割 | 2h | - |
| P1-2 | logger.py分割 | 1.5h | - |
| P1-3 | base.py分割 | 2h | - |

**合計:** 5.5時間
**リスク:** 高（多くのファイルに影響）

### Phase 2: 中優先度（バックテストエンジン）
| タスク | 対象 | 工数 | 依存 |
|--------|------|------|------|
| P2-1 | fast_engine.py分割 | 4h | P1-3 |
| P2-2 | engine.py分割 | 4h | P1-3 |
| P2-3 | vectorbt_engine.py分割 | 2.5h | P1-3 |

**合計:** 10.5時間
**依存:** base.py分割後

### Phase 3: 中優先度（オーケストレータ）
| タスク | 対象 | 工数 | 依存 |
|--------|------|------|------|
| P3-1 | pipeline.py分割 | 3h | P1-1 |
| P3-2 | allocator.py分割 | 1.5h | - |

**合計:** 4.5時間

### Phase 4: 低優先度（シグナル）
| タスク | 対象 | 工数 | 依存 |
|--------|------|------|------|
| P4-1 | trend.py分割 | 2h | - |
| P4-2 | mean_reversion.py分割 | 1.5h | - |

**合計:** 3.5時間

---

## 5. 実行上の注意事項

### インポート互換性
- `__init__.py`でre-exportを維持
- 既存のインポートパスを壊さない

### テスト
- 分割後に既存テストが通ることを確認
- 新規モジュール用のユニットテスト追加

### ドキュメント
- 各新規モジュールにdocstring追加
- CHANGELOG.mdに変更記録

---

## 6. 総工数サマリ

| Phase | 工数 | 累計 |
|-------|------|------|
| Phase 1 | 5.5h | 5.5h |
| Phase 2 | 10.5h | 16h |
| Phase 3 | 4.5h | 20.5h |
| Phase 4 | 3.5h | 24h |

**総見積:** 約24時間（3人日）

---

## 7. 今後の対応

このリファクタリング計画は、優先度に基づいて段階的に実行することを推奨する。
Phase 1から着手し、各フェーズ完了後にテストを実行して品質を確保する。

**推奨実行順序:**
1. settings.py → 最も多くのファイルに影響
2. base.py → エンジン分割の前提条件
3. fast_engine.py → 最大ファイル
4. pipeline.py → オーケストレータの中核
