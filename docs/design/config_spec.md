# 設定ファイル（Config）詳細設計書

> **Version**: 1.0.0
> **作成日**: 2026-01-28
> **タスクID**: task_001_3

## 概要

本設計書は、株式運用システムの設定ファイル（config）の詳細仕様を定義する。
Walk-forward検証、リバランス、コストモデル、ハードゲート、縮退モードの各パラメータを規定する。

---

## 1. リバランス頻度

### 推奨設定: 月次（Monthly）

| 選択肢 | 頻度 | 推奨度 | 理由 |
|--------|------|--------|------|
| 週次 | 52回/年 | △ | 取引コスト増大、短期ノイズに振り回されやすい |
| **月次** | 12回/年 | **◎** | コストと適応性のバランスが良好 |
| 四半期 | 4回/年 | ○ | コスト最小だが市場変化への追従が遅い |

### 推奨理由

1. **取引コストの最適化**: 月次リバランスは、売買回転率を適度に抑えつつ、市場変化に適応可能
2. **税務効率**: 短期売買を避けることで、税務上の不利を軽減
3. **実務的な運用**: 月末に運用レポートと合わせてリバランスを実施可能
4. **ファクターの持続性**: モメンタム・バリュー等のファクターは月次程度で効果を発揮

```yaml
rebalance:
  frequency: monthly      # weekly | monthly | quarterly
  execution_day: last_business_day  # first_business_day | last_business_day | specific_day
  specific_day: null      # execution_day が specific_day の場合のみ使用（1-28）
  min_trade_threshold: 0.02  # ポジション変更が2%未満なら取引しない
```

---

## 2. Walk-Forward検証パラメータ（train/test/step期間）

### 推奨設定

| パラメータ | 推奨値 | 日数 | 理由 |
|------------|--------|------|------|
| **train_period** | 1年 | 252日 | 十分なサンプル数を確保しつつ、古すぎるデータを排除 |
| **test_period** | 3ヶ月 | 63日 | 四半期単位で評価、季節性の影響を捕捉可能 |
| **step_period** | 1ヶ月 | 21日 | リバランス頻度と整合性を持たせる |

### パラメータ設計根拠

```
Timeline:
|<------ Train (252日) ------>|<-- Test (63日) -->|
                              |<- Step (21日) ->|
                                      ↓ スライド
|<------ Train (252日) ------>|<-- Test (63日) -->|
```

1. **train_period = 252日（約1年）**
   - 株式市場の年間営業日数に相当
   - 季節性・四半期決算サイクルを1周期以上含む
   - 統計的に有意なサンプル数を確保

2. **test_period = 63日（約3ヶ月）**
   - 四半期決算サイクルに対応
   - 短すぎると評価が不安定、長すぎると検証回数が減少
   - 21日×3 = 63日で月次リバランスとの整合性を確保

3. **step_period = 21日（約1ヶ月）**
   - リバランス頻度と一致させることで実運用をシミュレート
   - 検証回数を十分に確保（5年データで約60回の検証）

```yaml
walk_forward:
  train_period_days: 252    # トレーニング期間（営業日）
  test_period_days: 63      # テスト期間（営業日）
  step_period_days: 21      # スライド幅（営業日）
  min_train_samples: 200    # 最小トレーニングサンプル数
  purge_gap_days: 5         # train/test間のギャップ（情報リーク防止）
  embargo_days: 5           # テスト後のエンバーゴ期間
```

---

## 3. コストモデル

### 日本株式市場を想定した設定

| コスト項目 | 推奨値 | 説明 |
|------------|--------|------|
| **スプレッド** | 0.10% (10bps) | 大型株0.05%、中小型株0.15%の平均 |
| **手数料** | 0.05% (5bps) | ネット証券の標準的な手数料水準 |
| **スリッページ** | 0.10% (10bps) | 執行時の価格乖離（成行注文想定） |
| **合計片道コスト** | 0.25% (25bps) | 往復で0.50% |

### 設計根拠

1. **スプレッド**
   - 大型株（TOPIX100）: 0.03-0.07%
   - 中型株（TOPIX Mid400）: 0.08-0.15%
   - 小型株: 0.15-0.30%
   - 保守的に0.10%を採用

2. **手数料**
   - 主要ネット証券: 約定代金の0.033-0.099%
   - 0.05%は実務的な平均値

3. **スリッページ**
   - 流動性、発注サイズ、市場環境に依存
   - 中程度の発注サイズで0.10%を想定

```yaml
cost_model:
  spread_bps: 10            # スプレッド（basis points）
  commission_bps: 5         # 取引手数料（basis points）
  slippage_bps: 10          # スリッページ（basis points）

  # 時価総額別の調整係数
  market_cap_adjustment:
    large_cap:              # 時価総額 > 1兆円
      spread_multiplier: 0.5
      slippage_multiplier: 0.5
    mid_cap:                # 1000億円 < 時価総額 <= 1兆円
      spread_multiplier: 1.0
      slippage_multiplier: 1.0
    small_cap:              # 時価総額 <= 1000億円
      spread_multiplier: 1.5
      slippage_multiplier: 2.0

  # 固定コスト
  fixed_costs:
    rebalance_overhead_jpy: 0  # リバランス時の固定費用（円）
```

---

## 4. ハードゲート閾値

### 戦略採用の最低基準

戦略がポートフォリオに採用されるための必須条件を定義する。
**いずれか1つでも満たさない場合、その戦略は不採用となる。**

| 指標 | 閾値 | 理由 |
|------|------|------|
| **Sharpe Ratio** | >= 0.5 | リスク調整後リターンの最低水準 |
| **最大ドローダウン** | <= 25% | 資金管理上の許容限界 |
| **勝率** | >= 45% | 統計的に有意な予測能力 |
| **Profit Factor** | >= 1.2 | 総利益/総損失の最低水準 |
| **取引回数** | >= 30 | 統計的信頼性の確保 |

### 閾値設計根拠

1. **Sharpe Ratio >= 0.5**
   - 0.5未満はノイズと区別困難
   - 年率リターン5%、ボラティリティ10%で達成可能な水準

2. **最大ドローダウン <= 25%**
   - 心理的・資金的に許容可能な範囲
   - 25%のドローダウンから回復には33%のリターンが必要

3. **勝率 >= 45%**
   - 損小利大の戦略でも最低限の勝率は必要
   - 45%未満は連敗リスクが高まる

4. **Profit Factor >= 1.2**
   - 1.0が損益分岐点
   - コスト・スリッページを考慮し20%のマージンを確保

5. **取引回数 >= 30**
   - 中心極限定理により正規分布近似が有効になる最小サンプル

```yaml
hard_gates:
  # 必須条件（全て満たす必要あり）
  min_sharpe_ratio: 0.5
  max_drawdown_pct: 25.0
  min_win_rate_pct: 45.0
  min_profit_factor: 1.2
  min_trades: 30

  # オプション条件
  optional:
    min_calmar_ratio: 0.3       # リターン/最大DD
    max_volatility_pct: 30.0    # 年率ボラティリティ上限
    min_recovery_factor: 1.0    # 累積利益/最大DD

  # 警告条件（満たさなくても採用可能だが警告を出す）
  warnings:
    min_sharpe_ratio: 1.0
    max_drawdown_pct: 15.0
    min_win_rate_pct: 50.0
```

---

## 5. 縮退モード

### 全戦略不採用時の振る舞い

ハードゲートを通過する戦略が存在しない場合、システムは縮退モードに移行する。

### 縮退レベル定義

| レベル | 条件 | アクション |
|--------|------|------------|
| **Level 0** | 通常運用 | ポートフォリオ構築・リバランス実行 |
| **Level 1** | 採用戦略数 < 閾値 | ポジションサイズ縮小 |
| **Level 2** | 採用戦略数 = 0 | 現金退避モード |
| **Level 3** | 異常検知 | 全ポジションクローズ |

### 現金退避条件

以下のいずれかを満たした場合、現金比率を引き上げる：

1. **採用可能な戦略がゼロ**
2. **市場ボラティリティが閾値超過**（VIX相当 > 30）
3. **システム異常検知**（データ欠損、API障害等）

```yaml
degradation_mode:
  # 縮退レベルの定義
  levels:
    level_0:  # 通常運用
      min_adopted_strategies: 3
      cash_ratio: 0.0           # 現金比率0%
    level_1:  # 縮小運用
      min_adopted_strategies: 1
      max_adopted_strategies: 2
      cash_ratio: 0.3           # 現金比率30%
      position_size_multiplier: 0.7
    level_2:  # 現金退避
      adopted_strategies: 0
      cash_ratio: 0.8           # 現金比率80%
      allow_defensive_only: true  # ディフェンシブ銘柄のみ許可
    level_3:  # 緊急停止
      trigger: anomaly_detected
      cash_ratio: 1.0           # 現金比率100%
      close_all_positions: true

  # 現金退避の条件
  cash_evacuation_triggers:
    - condition: no_adopted_strategies
      action: escalate_to_level_2
    - condition: market_volatility_high
      threshold_vix: 30
      action: increase_cash_ratio
      additional_cash_pct: 20
    - condition: system_anomaly
      types:
        - data_gap_detected
        - api_failure
        - price_staleness
      action: escalate_to_level_3

  # 復帰条件
  recovery:
    level_2_to_level_1:
      require_adopted_strategies: 1
      cooldown_days: 5
    level_1_to_level_0:
      require_adopted_strategies: 3
      cooldown_days: 3
    level_3_to_level_2:
      require_manual_approval: true
```

---

## 6. 統合Config例（完全版）

以下は全設定を統合したYAML形式の設定ファイル例である。

```yaml
# =============================================================================
# 株式運用システム設定ファイル
# =============================================================================
# Version: 1.0.0
# Last Updated: 2026-01-28

system:
  name: "multi-strategy-portfolio"
  version: "1.0.0"
  timezone: "Asia/Tokyo"
  base_currency: "JPY"

# -----------------------------------------------------------------------------
# 1. リバランス設定
# -----------------------------------------------------------------------------
rebalance:
  frequency: monthly
  execution_day: last_business_day
  specific_day: null
  min_trade_threshold: 0.02
  max_turnover_pct: 100       # 月次最大回転率

# -----------------------------------------------------------------------------
# 2. Walk-Forward検証パラメータ
# -----------------------------------------------------------------------------
walk_forward:
  train_period_days: 252
  test_period_days: 63
  step_period_days: 21
  min_train_samples: 200
  purge_gap_days: 5
  embargo_days: 5

# -----------------------------------------------------------------------------
# 3. コストモデル
# -----------------------------------------------------------------------------
cost_model:
  spread_bps: 10
  commission_bps: 5
  slippage_bps: 10
  market_cap_adjustment:
    large_cap:
      threshold_jpy: 1_000_000_000_000  # 1兆円
      spread_multiplier: 0.5
      slippage_multiplier: 0.5
    mid_cap:
      threshold_jpy: 100_000_000_000    # 1000億円
      spread_multiplier: 1.0
      slippage_multiplier: 1.0
    small_cap:
      spread_multiplier: 1.5
      slippage_multiplier: 2.0
  fixed_costs:
    rebalance_overhead_jpy: 0

# -----------------------------------------------------------------------------
# 4. ハードゲート閾値
# -----------------------------------------------------------------------------
hard_gates:
  min_sharpe_ratio: 0.5
  max_drawdown_pct: 25.0
  min_win_rate_pct: 45.0
  min_profit_factor: 1.2
  min_trades: 30
  optional:
    min_calmar_ratio: 0.3
    max_volatility_pct: 30.0
    min_recovery_factor: 1.0
  warnings:
    min_sharpe_ratio: 1.0
    max_drawdown_pct: 15.0
    min_win_rate_pct: 50.0

# -----------------------------------------------------------------------------
# 5. 縮退モード
# -----------------------------------------------------------------------------
degradation_mode:
  levels:
    level_0:
      min_adopted_strategies: 3
      cash_ratio: 0.0
    level_1:
      min_adopted_strategies: 1
      max_adopted_strategies: 2
      cash_ratio: 0.3
      position_size_multiplier: 0.7
    level_2:
      adopted_strategies: 0
      cash_ratio: 0.8
      allow_defensive_only: true
    level_3:
      trigger: anomaly_detected
      cash_ratio: 1.0
      close_all_positions: true
  cash_evacuation_triggers:
    - condition: no_adopted_strategies
      action: escalate_to_level_2
    - condition: market_volatility_high
      threshold_vix: 30
      action: increase_cash_ratio
      additional_cash_pct: 20
    - condition: system_anomaly
      types:
        - data_gap_detected
        - api_failure
        - price_staleness
      action: escalate_to_level_3
  recovery:
    level_2_to_level_1:
      require_adopted_strategies: 1
      cooldown_days: 5
    level_1_to_level_0:
      require_adopted_strategies: 3
      cooldown_days: 3
    level_3_to_level_2:
      require_manual_approval: true

# -----------------------------------------------------------------------------
# 6. ログ・通知設定
# -----------------------------------------------------------------------------
logging:
  level: INFO
  output:
    - type: file
      path: "logs/system.log"
    - type: console

notifications:
  enabled: true
  channels:
    - type: slack
      webhook_url: "${SLACK_WEBHOOK_URL}"
      events:
        - rebalance_complete
        - degradation_level_change
        - anomaly_detected
```

---

## 7. 設定値のカスタマイズガイドライン

| 運用スタイル | リバランス | train期間 | Sharpe閾値 | DD許容 |
|--------------|------------|-----------|------------|--------|
| 保守的 | quarterly | 504日（2年） | 0.7 | 15% |
| 標準 | monthly | 252日（1年） | 0.5 | 25% |
| 積極的 | weekly | 126日（6ヶ月） | 0.3 | 35% |

---

## 備考

- 本設計は日本株式市場を前提としている
- 米国株・先物等では、コストモデルの調整が必要
- 本番運用前に、ペーパートレードでの検証を推奨
