# テスト改善計画

> **作成日**: 2026-02-01
> **目的**: skipped 8件、failed 13件のテストを分析し改善方針を策定

---

## 概要サマリー

| カテゴリ | 件数 | 対処難易度 | 優先度 |
|---------|------|-----------|--------|
| **SKIPPED** | 8件 | - | - |
| └ 意図的スキップ | 3件 | 維持 | - |
| └ AWS認証依存 | 3件 | 環境設定 | Low |
| └ モジュール未実装 | 2件 | 中 | Medium |
| **FAILED** | 13件 | - | - |
| └ ダッシュボードバグ | 4件 | 簡単 | High |
| └ アーキテクチャ違反 | 5件 | 大規模 | Medium |
| └ ベンチマーク性能 | 4件 | 中 | Low |

---

## SKIPPED テスト（8件）

### 1. 意図的スキップ（3件）- 維持

| テスト | スキップ理由 | 判断 |
|--------|-------------|------|
| `test_static_charts.py:255` | PDF日本語非対応 | 維持（技術的制約） |
| `test_incremental_cache_integration.py:376` | I/O性能テスト | 維持（環境依存） |
| `test_unified_executor_storage.py:168` | StorageConfig検証中 | 維持（設計中） |

**対処**: なし（妥当なスキップ）

---

### 2. AWS認証依存（3件）

| テスト | ファイル |
|--------|---------|
| S3キャッシュテスト | `tests/integration/test_s3_cache.py:374, 396, 415` |

**原因**: `AWS_ACCESS_KEY_ID` 環境変数が未設定

**改善案**:
1. **CI環境**: GitHub Secrets でAWS認証情報を設定
2. **ローカル**: `~/.aws/credentials` または環境変数を設定
3. **代替案**: モックを使用したテストに変更（S3実接続不要）

```bash
# ローカルで実行する場合
export AWS_ACCESS_KEY_ID=xxx
export AWS_SECRET_ACCESS_KEY=xxx
pytest tests/integration/test_s3_cache.py -v
```

---

### 3. シグナルモジュール未実装（2件）

| テスト | スキップ理由 |
|--------|-------------|
| `test_signal_pipeline.py:113` | MomentumSignal, TrendSignal等が存在しない |
| `test_signal_pipeline.py:154` | EnsembleSignal等が存在しない |

**原因**: テストが期待する個別シグナルクラス（`MomentumSignal`, `TrendSignal`, `VolatilitySignal`, `MeanReversionSignal`）が実装されていない

**改善案**:
1. **オプションA**: シグナルモジュールを実装（工数大）
2. **オプションB**: テストを既存シグナル（`LeadLagSignal`, `VIXSignal`等）に変更（推奨）
3. **オプションC**: テストを削除（非推奨）

**推奨**: オプションB - 既存の実装済みシグナルでテストを書き直す

---

## FAILED テスト（13件）

### 1. ダッシュボードバグ（4件）- 優先度: High

| テスト | エラー |
|--------|--------|
| `test_monthly_heatmap` | `ValueError: Length mismatch: Expected 9, got 12` |
| `test_create_app` | 同上 |
| `test_layout_components` | 同上 |
| `test_charts_section` | 同上 |

**根本原因**: `src/analysis/dashboard.py:191` で月名を12列固定で設定しているが、テストデータの期間が1年未満のため9列しかない

```python
# 問題のコード（lines 190-192）
pivot = monthly_df.pivot(index="year", columns="month", values="return")
pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]  # 常に12列を期待
```

**修正案**:
```python
# 修正後
pivot = monthly_df.pivot(index="year", columns="month", values="return")
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# 存在する月のみに対応
pivot.columns = [month_names[m-1] for m in pivot.columns]
```

**追加修正**: `'M'` → `'ME'` のdeprecation警告も対処（line 184）

---

### 2. アーキテクチャ違反（5件）- 優先度: Medium

#### 2.1 重複クラス定義（46クラス）

| 例 | 重複箇所 |
|----|---------|
| `MarketRegime` | 6ファイルで重複定義 |
| `AllocationResult` | 2ファイルで重複定義 |
| `MetricsCalculator` | 3ファイルで重複定義 |

**改善案**:
1. 共通の型定義を `src/types/` または `src/common/` に集約
2. 各モジュールからインポートして使用
3. 段階的に移行（影響範囲が大きいため）

**工数見積**: 大（2-3日）

---

#### 2.2 ファイルサイズ超過（45ファイル）

| ファイル | 行数 | 制限 |
|---------|------|------|
| `signal_precompute.py` | 2,236行 | 500行 |
| `fast_engine.py` | 1,485行 | 500行 |
| `allocator.py` | 1,312行 | 500行 |
| ... | ... | ... |

**改善案**:
1. **オプションA**: 制限を緩和（800-1000行に変更）
2. **オプションB**: ファイル分割リファクタリング（推奨だが工数大）
3. **オプションC**: 一部ファイルを許可リストに追加

**推奨**: 段階的アプローチ
- Phase 1: 制限を800行に緩和
- Phase 2: 1000行超のファイルを優先的に分割

---

#### 2.3 循環依存（4件）

| 依存関係 |
|---------|
| `signals <-> signals` |
| `backtest <-> signals` |
| `backtest <-> backtest` |
| `allocation <-> backtest` |

**改善案**:
1. 依存関係を解析し、共通モジュールを抽出
2. インターフェース/プロトコルを使用して依存を逆転
3. 遅延インポートを活用

---

#### 2.4 print文（168箇所）

**改善案**:
1. `logging` モジュールに置き換え
2. または許可リストに開発用スクリプトを追加

---

#### 2.5 命名規則違反（1件）

| クラス名 | ファイル |
|---------|---------|
| `_WeightsFuncAdapter` | `src/backtest/fast_engine.py:352` |

**原因**: アンダースコアで始まる内部クラス

**改善案**: `WeightsFuncAdapter` にリネーム（内部クラスでも命名規則は適用）

---

### 3. ベンチマーク性能テスト（4件）- 優先度: Low

| テスト | 問題 |
|--------|------|
| `test_benchmark_100_tickers` | 実行時間が長い（数分） |
| `test_benchmark_300_tickers` | 同上 |
| `test_benchmark_500_tickers` | 同上 |
| `test_staged_filter_effect` | Numba依存 |

**原因**:
- ベンチマークテストが通常のpytest実行に含まれている
- Numba未インストールまたはスレッドレイヤー問題

**改善案**:
1. `@pytest.mark.slow` または `@pytest.mark.benchmark` でマーク
2. デフォルトのpytest実行から除外
3. `pytest.ini` に設定追加:

```ini
[pytest]
markers =
    benchmark: marks tests as benchmark (deselect with '-m "not benchmark"')
```

---

## 改善実施の優先順位

### Phase 1: 即時対応（1日）

| タスク | 件数 | 効果 |
|--------|------|------|
| ダッシュボードバグ修正 | 4件 | FAILED → PASSED |
| ベンチマークテストにマーカー追加 | 4件 | FAILED → SKIPPED（意図的） |
| `_WeightsFuncAdapter` リネーム | 1件 | FAILED → PASSED |

**効果**: FAILED 13件 → 4件

---

### Phase 2: 中期対応（1週間）

| タスク | 件数 | 効果 |
|--------|------|------|
| print文をloggingに置換 | 168箇所 | FAILED → PASSED |
| テストのシグナル修正 | 2件 | SKIPPED → PASSED |
| ファイルサイズ制限緩和 | - | FAILED → PASSED（暫定） |

---

### Phase 3: 長期対応（2-4週間）

| タスク | 効果 |
|--------|------|
| 重複クラス定義の解消 | コード品質向上 |
| 循環依存の解消 | 保守性向上 |
| 巨大ファイルの分割 | 可読性向上 |

---

## 具体的な修正コード例

### ダッシュボード修正（dashboard.py）

```python
# Line 184: 'M' を 'ME' に変更
monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)

# Lines 190-192: 動的な月名対応
pivot = monthly_df.pivot(index="year", columns="month", values="return")
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# 存在する月のみマッピング
if len(pivot.columns) < 12:
    pivot.columns = [month_names[m-1] for m in pivot.columns]
else:
    pivot.columns = month_names
```

### ベンチマークテストのマーカー追加

```python
# test_lead_lag_performance.py

@pytest.mark.benchmark
@pytest.mark.slow
def test_benchmark_100_tickers(numba_available: bool = True):
    ...
```

```ini
# pytest.ini または pyproject.toml
[tool.pytest.ini_options]
markers = [
    "benchmark: marks tests as benchmark (deselect with '-m \"not benchmark\"')",
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
```

---

## 結論

| 対応 | FAILED | SKIPPED |
|------|--------|---------|
| 現状 | 13件 | 8件 |
| Phase 1後 | 4件 | 12件（意図的スキップ増） |
| Phase 2後 | 0件 | 5件 |
| Phase 3後 | 0件 | 3件（意図的スキップのみ） |

Phase 1の即時対応で、大部分の問題は解消できる見込み。
