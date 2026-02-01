# CMD016機能統合ガイド

> **Last Updated**: 2026-02-01

---

## 目次

1. [機能概要](#機能概要)
2. [統合順序](#統合順序)
3. [VIXキャッシュ配分](#vixキャッシュ配分)
4. [ペアトレーディング](#ペアトレーディング)
5. [セクターローテーション](#セクターローテーション)
6. [ドローダウンプロテクション](#ドローダウンプロテクション)
7. [シグナルフィルター](#シグナルフィルター)
8. [使用方法](#使用方法)

---

## 機能概要

CMD016Integratorは、パイプラインの各段階で高度な機能を統合するためのモジュールです。

### 統合される機能

| カテゴリ | 機能 | 目的 |
|---------|------|------|
| **リスク管理** | VIXキャッシュ配分 | ボラティリティに応じたキャッシュ比率調整 |
| | 相関ブレイク検出 | 相関構造の急変を検出 |
| | ドローダウンプロテクション | 段階的リスク削減 |
| **戦略** | ペアトレーディング | スプレッド取引 |
| | セクターローテーション | 経済サイクルに応じたセクター調整 |
| | 最低保有期間 | 過剰取引防止 |
| **フィルター** | ヒステリシスフィルター | エントリー/エグジット閾値分離 |
| | シグナル減衰 | 時間経過によるシグナル減衰 |

---

## 統合順序

CMD016の機能は以下の順序で適用されます：

```
┌─────────────────────────────────────────┐
│ 1. VIX取得 → VIXキャッシュ配分計算      │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ 2. レジーム適応シグナルパラメータ設定    │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ 3. シグナル生成                          │
│    - ペアトレーディング                  │
│    - クロスアセットモメンタム            │
│    - デュアルモメンタム                  │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ 4. セクターローテーション調整            │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ 5. ヒステリシスフィルター適用            │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ 6. シグナル減衰適用                      │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ 7. 最低保有期間フィルター                │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ 8. 相関ブレイク検出                      │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ 9. ドローダウンプロテクション            │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ 10. イールドカーブ調整                   │
└─────────────────────────────────────────┘
```

---

## VIXキャッシュ配分

### 概要

VIX水準に応じて、ポートフォリオのキャッシュ比率を動的に調整します。

### 設定

```yaml
# config/default.yaml
cmd_016_features:
  vix_cash_allocation:
    enabled: true
    vix_low: 15       # 低VIX閾値
    vix_high: 25      # 高VIX閾値
    vix_extreme: 35   # 極端VIX閾値
    max_cash_ratio: 0.5  # 最大キャッシュ比率
```

### 動作

| VIX水準 | キャッシュ比率 | 説明 |
|---------|---------------|------|
| < 15 | 0% | 通常モード |
| 15-25 | 0-25% | 段階的にキャッシュ増加 |
| 25-35 | 25-50% | 高警戒モード |
| > 35 | 50% | 最大防御モード |

### 使用例

```python
from src.orchestrator.cmd016_integrator import CMD016Integrator

integrator = CMD016Integrator(settings)

# VIXキャッシュ配分適用
adjusted_weights, vix_info = integrator.apply_vix_cash_allocation(
    vix_value=22.5,
    base_weights=current_weights,
)

print(f"キャッシュ比率: {vix_info['cash_ratio']:.2%}")
print(f"VIXレジーム: {vix_info['vix_regime']}")
```

---

## ペアトレーディング

### 概要

相関の高いアセットペアのスプレッドを利用した取引戦略。

### 設定

```yaml
cmd_016_features:
  pairs_trading:
    enabled: true
    zscore_entry: 2.0    # エントリー閾値
    zscore_exit: 0.5     # エグジット閾値
    lookback: 60         # ルックバック期間
    min_correlation: 0.7 # 最小相関
```

### 利用可能な場合

```python
from src.orchestrator.cmd016_integrator import PAIRS_TRADING_AVAILABLE

if PAIRS_TRADING_AVAILABLE:
    from src.strategy.pairs_trading import PairsTradingStrategy
    # ペアトレーディングが利用可能
```

---

## セクターローテーション

### 概要

経済サイクル（拡大、ピーク、後退、底）に応じてセクター配分を調整。

### 設定

```yaml
cmd_016_features:
  sector_rotation:
    enabled: true
    detection_method: "economic_cycle"  # economic_cycle | momentum
```

### 経済サイクルとセクター

| フェーズ | 有利なセクター | 不利なセクター |
|---------|---------------|---------------|
| 拡大期 | Technology, Consumer Discretionary | Utilities, Consumer Staples |
| ピーク | Energy, Materials | Technology |
| 後退期 | Healthcare, Utilities | Financials, Industrials |
| 底 | Financials, Consumer Discretionary | Energy |

### 使用例

```python
adjusted_weights, sr_info = integrator.apply_sector_rotation(
    base_weights=current_weights,
    macro_indicators={
        "pmi": 55.0,
        "yield_curve_slope": 0.5,
        "unemployment_change": -0.1,
    },
)

print(f"検出フェーズ: {sr_info['detected_phase']}")
```

---

## ドローダウンプロテクション

### 概要

ポートフォリオのドローダウンに応じて、段階的にリスクを削減。

### 設定

```yaml
cmd_016_features:
  drawdown_protection:
    enabled: true
    dd_levels: [0.05, 0.10, 0.15, 0.20]      # ドローダウン閾値
    risk_reductions: [0.9, 0.7, 0.5, 0.3]    # リスク乗数
    recovery_threshold: 0.5                   # 回復閾値
    emergency_dd_level: 0.25                  # 緊急停止レベル
```

### 動作

| ドローダウン | リスク乗数 | 説明 |
|-------------|-----------|------|
| < 5% | 100% | 通常運用 |
| 5-10% | 90% | 軽度リスク削減 |
| 10-15% | 70% | 中度リスク削減 |
| 15-20% | 50% | 高度リスク削減 |
| > 20% | 30% | 最大リスク削減 |
| > 25% | 0% | 緊急停止 |

### 使用例

```python
adjusted_weights, dd_info = integrator.apply_drawdown_protection(
    portfolio_value=950000,  # 現在価値
    base_weights=current_weights,
)

print(f"現在のドローダウン: {dd_info['current_dd']:.2%}")
print(f"リスク乗数: {dd_info['risk_multiplier']:.2f}")
```

---

## シグナルフィルター

### ヒステリシスフィルター

エントリーとエグジットで異なる閾値を使用し、過剰取引を防止。

```yaml
cmd_016_features:
  hysteresis_filter:
    enabled: true
    entry_threshold: 0.3   # エントリー閾値
    exit_threshold: 0.1    # エグジット閾値
```

### シグナル減衰

時間経過に伴いシグナル強度を減衰。

```yaml
cmd_016_features:
  signal_decay:
    enabled: true
    halflife: 5       # 半減期（日数）
    min_signal: 0.01  # 最小シグナル値
```

### 最低保有期間

最低保有期間を設定し、頻繁な売買を防止。

```yaml
cmd_016_features:
  min_holding_period:
    enabled: true
    min_periods: 5                 # 最低保有日数
    force_exit_on_reversal: true   # 反転時強制エグジット
    reversal_threshold: -0.5       # 反転閾値
```

---

## 使用方法

### 基本的な使用

```python
from src.orchestrator.cmd016_integrator import CMD016Integrator, create_integrator

# 設定からIntegratorを作成
integrator = create_integrator(settings)

# 全機能を統合適用
result = integrator.integrate_all(
    base_weights=current_weights,
    signals=signal_values,
    portfolio_value=portfolio_value,
    vix_value=current_vix,
    returns=historical_returns,
    macro_indicators=macro_data,
)

# 結果
print(f"調整後重み: {result.adjusted_weights}")
print(f"キャッシュ比率: {result.cash_ratio:.2%}")
print(f"警告: {result.warnings}")
```

### 個別機能の適用

```python
# VIXキャッシュ配分のみ
weights, info = integrator.apply_vix_cash_allocation(vix, weights)

# シグナルフィルターのみ
filtered, info = integrator.apply_signal_filters(signals, weights)

# ドローダウンプロテクションのみ
weights, info = integrator.apply_drawdown_protection(value, weights)
```

### 機能状態の確認

```python
# 利用可能な機能を確認
status = integrator.get_feature_status()
for feature, enabled in status.items():
    print(f"{feature}: {'有効' if enabled else '無効'}")
```

### 状態リセット

```python
# バックテスト間で状態をリセット
integrator.reset_state()
```

---

## 関連ドキュメント

| ドキュメント | 概要 |
|-------------|------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | システムアーキテクチャ |
| [SIGNAL_PIPELINE.md](SIGNAL_PIPELINE.md) | シグナルパイプライン |
| [ORCHESTRATOR_RESPONSIBILITIES.md](ORCHESTRATOR_RESPONSIBILITIES.md) | オーケストレータ責務 |
| [BACKTEST_STANDARD.md](BACKTEST_STANDARD.md) | バックテスト規格 |
