# MLコンポーネントガイド

> **Last Updated**: 2026-02-01

---

## 目次

1. [概要と利用可能モジュール](#概要と利用可能モジュール)
2. [リターン予測](#リターン予測)
3. [XGBoostスタッキング](#xgboostスタッキング)
4. [動的アンサンブル重み](#動的アンサンブル重み)
5. [PPO配分](#ppo配分)
6. [有効化・設定方法](#有効化設定方法)

---

## 概要と利用可能モジュール

### モジュール一覧

| モジュール | 機能 | 依存関係 |
|-----------|------|---------|
| `ReturnPredictor` | LightGBM/RandomForestによるリターン予測 | sklearn, lightgbm（オプション） |
| `XGBoostStacker` | XGBoostによる戦略スタッキング | xgboost |
| `LightGBMStacker` | LightGBMによる戦略スタッキング | lightgbm |
| `EnsembleStacker` | 複数モデルのアンサンブル | sklearn |
| `DynamicEnsembleWeightLearner` | 動的アンサンブル重み学習 | sklearn |
| `RegimeAwareEnsemble` | レジーム対応アンサンブル | sklearn |
| `SimplePPOAgent` | PPOによる配分最適化 | numpy |

### インポート

```python
from src.ml import (
    # リターン予測
    ReturnPredictor,
    MultiAssetPredictor,
    PredictorConfig,
    create_return_predictor,

    # スタッキング
    XGBoostStacker,
    LightGBMStacker,
    EnsembleStacker,
    RidgeStacker,
    create_stacker,

    # 動的アンサンブル
    DynamicEnsembleWeightLearner,
    RegimeAwareEnsemble,
    create_dynamic_ensemble_learner,

    # PPO
    SimplePPOAgent,
    TradingEnvironment,
    train_ppo_allocator,
    create_ppo_allocator,
)
```

---

## リターン予測

### 概要

テクニカル特徴量から将来リターンを予測するモジュール。

### 特徴量

| カテゴリ | 特徴量 | 説明 |
|---------|--------|------|
| リターン系 | `ret_{lb}`, `vol_{lb}` | 過去リターン、ボラティリティ |
| テクニカル | `rsi`, `macd`, `bb_position` | RSI、MACD、ボリンジャーバンド位置 |
| 相対強弱 | `rel_str` | ベンチマーク対比の相対強度 |

### 基本使用

```python
from src.ml import ReturnPredictor, PredictorConfig

# 設定
config = PredictorConfig(
    model_type="lightgbm",     # または "random_forest"
    prediction_horizon=20,      # 予測期間（日数）
    lookback_features=(5, 10, 20, 60),
    rsi_period=14,
    benchmark="SPY",
)

# 予測器作成
predictor = ReturnPredictor(config)

# 訓練
predictor.fit(train_prices, train_returns)

# 予測
predictions = predictor.predict(test_prices)
```

### マルチアセット予測

```python
from src.ml import MultiAssetPredictor

# 複数銘柄を一括予測
multi_predictor = MultiAssetPredictor(
    tickers=["SPY", "QQQ", "TLT", "GLD"],
    config=config,
)

# 訓練（各銘柄で個別モデル）
multi_predictor.fit(prices_dict)

# 全銘柄の予測
all_predictions = multi_predictor.predict_all(current_prices)
```

### 特徴量重要度

```python
# 特徴量重要度を取得
importance = predictor.get_feature_importance()
for feature, score in sorted(importance.items(), key=lambda x: -x[1])[:10]:
    print(f"{feature}: {score:.4f}")
```

---

## XGBoostスタッキング

### 概要

複数の戦略シグナルをXGBoost/LightGBMでスタッキングし、最終シグナルを生成。

### 基本使用

```python
from src.ml import XGBoostStacker, create_stacker

# XGBoostスタッカー
stacker = XGBoostStacker(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
)

# または汎用ファクトリ
stacker = create_stacker("xgboost", n_estimators=100)

# 訓練
# X: 各戦略のシグナル (n_samples, n_strategies)
# y: 実際のリターン (n_samples,)
stacker.fit(X_train, y_train)

# 予測
stacked_signal = stacker.predict(X_test)
```

### 利用可能なスタッカー

```python
from src.ml import get_available_stackers

stackers = get_available_stackers()
# ['xgboost', 'lightgbm', 'ridge', 'ensemble']
```

### アンサンブルスタッカー

```python
from src.ml import EnsembleStacker

# 複数モデルのアンサンブル
ensemble = EnsembleStacker(
    models=["xgboost", "lightgbm", "ridge"],
    weights=[0.4, 0.4, 0.2],  # モデル重み
)

ensemble.fit(X_train, y_train)
```

---

## 動的アンサンブル重み

### 概要

戦略パフォーマンスに応じて、アンサンブル重みを動的に調整。

### 基本使用

```python
from src.ml import DynamicEnsembleWeightLearner, create_dynamic_ensemble_learner

# 学習器作成
learner = create_dynamic_ensemble_learner(
    n_strategies=5,
    lookback=60,
    min_weight=0.05,
)

# 過去パフォーマンスから重み学習
# performances: (n_periods, n_strategies)
weights = learner.compute_weights(performances)
```

### レジーム対応アンサンブル

```python
from src.ml import RegimeAwareEnsemble

# レジーム対応アンサンブル
ensemble = RegimeAwareEnsemble(
    n_strategies=5,
    n_regimes=4,  # BULL, BEAR, SIDEWAYS, VOLATILE
)

# レジーム別に重みを学習
ensemble.fit(performances, regimes)

# 現在のレジームで重み取得
weights = ensemble.get_weights(current_regime="BULL")
```

---

## PPO配分

### 概要

強化学習（PPO: Proximal Policy Optimization）によるポートフォリオ配分最適化。

### 環境設定

```python
from src.ml import EnvironmentConfig, TradingEnvironment

# 環境設定
env_config = EnvironmentConfig(
    n_assets=10,
    lookback=20,
    transaction_cost=0.001,
    risk_free_rate=0.02,
)

# 取引環境
env = TradingEnvironment(
    prices=prices_df,
    config=env_config,
)
```

### エージェント訓練

```python
from src.ml import RLConfig, train_ppo_allocator

# RL設定
rl_config = RLConfig(
    learning_rate=3e-4,
    gamma=0.99,
    clip_ratio=0.2,
    n_epochs=10,
    batch_size=64,
)

# 訓練
agent = train_ppo_allocator(
    env=env,
    config=rl_config,
    n_episodes=1000,
)
```

### 配分の取得

```python
from src.ml import create_ppo_allocator

# 訓練済みエージェントで配分
allocator = create_ppo_allocator(agent)

# 現在の状態から配分を決定
state = {
    "prices": current_prices,
    "returns": historical_returns,
    "volatility": current_volatility,
}

allocation = allocator.get_allocation(state)
```

---

## 有効化・設定方法

### 設定ファイル

```yaml
# config/default.yaml

ml:
  return_predictor:
    enabled: true
    model_type: "lightgbm"
    prediction_horizon: 20
    lookback_features: [5, 10, 20, 60]

  stacking:
    enabled: true
    method: "xgboost"
    n_estimators: 100
    max_depth: 3

  dynamic_ensemble:
    enabled: true
    lookback: 60
    min_weight: 0.05

  ppo_allocator:
    enabled: false  # 計算コストが高いためデフォルト無効
    learning_rate: 0.0003
    n_episodes: 1000
```

### 依存関係のインストール

```bash
# 基本ML（sklearn）
pip install scikit-learn

# LightGBM
pip install lightgbm

# XGBoost
pip install xgboost

# 全ML依存関係
pip install -e ".[ml]"
```

### 依存関係の確認

```python
# LightGBMが利用可能か確認
from src.ml.return_predictor import _HAS_LIGHTGBM
print(f"LightGBM: {'利用可能' if _HAS_LIGHTGBM else '利用不可'}")

# 利用可能なスタッカーを確認
from src.ml import get_available_stackers
print(f"利用可能スタッカー: {get_available_stackers()}")
```

### パイプラインでの使用

```python
from src.orchestrator.pipeline import Pipeline

# ML機能を有効化してパイプライン作成
pipeline = Pipeline(
    settings=settings,
    enable_return_predictor=True,
    enable_stacking=True,
)

# 実行
result = pipeline.run(prices_df)
```

---

## ベストプラクティス

### 1. オーバーフィッティング防止

```python
# ウォークフォワード検証を使用
from src.ml import ReturnPredictor

predictor = ReturnPredictor(config)

# 訓練期間とテスト期間を分離
train_end = len(prices) - 252  # 最後の1年をテスト用
predictor.fit(
    prices[:train_end],
    returns[:train_end],
)

# テスト期間で評価
predictions = predictor.predict(prices[train_end:])
```

### 2. 特徴量の正規化

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 3. 計算コストの管理

```python
# PPOは計算コストが高いため、本番環境でのみ使用
if settings.ml.ppo_allocator.enabled and is_production:
    allocator = create_ppo_allocator(agent)
else:
    # 軽量な代替手法を使用
    allocator = create_simple_allocator()
```

---

## 関連ドキュメント

| ドキュメント | 概要 |
|-------------|------|
| [SIGNAL_PIPELINE.md](SIGNAL_PIPELINE.md) | シグナルパイプライン |
| [ARCHITECTURE.md](ARCHITECTURE.md) | システムアーキテクチャ |
| [BACKTEST_STANDARD.md](BACKTEST_STANDARD.md) | バックテスト規格 |
| [backtest_acceleration_options.md](backtest_acceleration_options.md) | 高速化オプション |
