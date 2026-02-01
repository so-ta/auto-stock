# シグナルパイプラインガイド

> **Last Updated**: 2026-02-01

---

## 目次

1. [シグナルカテゴリ概要](#シグナルカテゴリ概要)
2. [シグナル事前計算](#シグナル事前計算)
3. [動的パラメータシステム](#動的パラメータシステム)
4. [レジーム適応パラメータ](#レジーム適応パラメータ)
5. [新規シグナル追加手順](#新規シグナル追加手順)

---

## シグナルカテゴリ概要

### シグナル分類

| カテゴリ | シグナル例 | 特徴 |
|---------|-----------|------|
| **独立シグナル** | momentum_*, rsi_*, volatility_* | 各銘柄の価格データのみから計算 |
| **相対シグナル** | sector_relative_*, cross_asset_*, ranking_* | クロスアセット比較・ランキング |
| **レジームシグナル** | regime_*, market_breadth | 市場全体の状態判定 |
| **テクニカルシグナル** | bollinger_*, stochastic_*, atr_* | テクニカル分析指標 |

### 独立シグナル一覧

```python
INDEPENDENT_SIGNALS = {
    "momentum_*",        # モメンタム
    "rsi_*",             # RSI
    "volatility_*",      # ボラティリティ
    "zscore_*",          # Zスコア
    "sharpe_*",          # ローリングシャープ
    "atr_*",             # ATR
    "bollinger_*",       # ボリンジャーバンド
    "stochastic_*",      # ストキャスティクス
    "breakout_*",        # ブレイクアウト
    "donchian_*",        # ドンチャンチャネル
    "fifty_two_week_high_*",  # 52週高値
}
```

### 相対シグナル一覧

```python
RELATIVE_SIGNALS = {
    "sector_relative_*",   # セクター相対強度
    "cross_asset_*",       # クロスアセットモメンタム
    "momentum_factor",     # モメンタムファクター
    "sector_momentum",     # セクターモメンタム
    "sector_breadth",      # セクターブレッド
    "market_breadth",      # マーケットブレッド
    "ranking_*",           # ランキング
    "lead_lag",            # リードラグ関係
    "short_term_reversal*", # 短期リバーサル
}
```

---

## シグナル事前計算

### SignalPrecomputerの使用

```python
from src.backtest.signal_precompute import SignalPrecomputer

# 初期化
precomputer = SignalPrecomputer(
    cache_dir=".cache/signals",
    storage_backend=backend,  # オプション：S3対応
)

# 全シグナル事前計算
precomputer.precompute_all(prices_df, config)

# 特定シグナルのみ計算
precomputer.precompute_signal(
    signal_name="momentum",
    prices=prices_df,
    params={"lookback": 20},
)
```

### バックテスト中のシグナル取得

```python
# 特定日付のシグナル値取得
signal_value = precomputer.get_signal_at_date(
    signal_name="momentum_20",
    ticker="SPY",
    date=rebalance_date,
)

# 全銘柄のシグナル取得
all_signals = precomputer.get_all_signals_at_date(
    signal_name="momentum_20",
    date=rebalance_date,
)
```

### キャッシュ検証

```python
from src.backtest.signal_precompute import CacheValidationResult

# キャッシュ有効性チェック
result: CacheValidationResult = precomputer.validate_cache_incremental(
    prices_df,
    config,
)

if result.can_use_cache:
    print(f"キャッシュ使用可能: {result.reason}")
    if result.incremental_start:
        print(f"増分更新開始日: {result.incremental_start}")
else:
    print(f"再計算必要: {result.reason}")
    if result.missing_signals:
        print(f"不足シグナル: {result.missing_signals}")
```

---

## 動的パラメータシステム

### パラメータ設定

```yaml
# config/default.yaml
signals:
  momentum:
    lookback: 20
    normalize: true

  rsi:
    period: 14
    overbought: 70
    oversold: 30

  volatility:
    lookback: 20
    annualization_factor: 252
```

### プログラムからのパラメータ取得

```python
from src.config.settings import Settings

settings = Settings()

# シグナルパラメータ取得
momentum_params = settings.signals.get("momentum", {})
lookback = momentum_params.get("lookback", 20)
```

### パラメータのオーバーライド

```python
from src.backtest.fast_engine import FastBacktestConfig

config = FastBacktestConfig(
    signal_params={
        "momentum": {"lookback": 30},  # デフォルトを上書き
        "rsi": {"period": 21},
    },
)
```

---

## レジーム適応パラメータ

### レジーム検出

```python
from src.signals.regime_detector import RegimeDetector

detector = RegimeDetector()
regime = detector.detect(returns_df)

# レジームタイプ
# - BULL: 強気相場
# - BEAR: 弱気相場
# - SIDEWAYS: レンジ相場
# - VOLATILE: 高ボラティリティ
```

### レジーム適応パラメータ

```python
from src.signals.regime_adaptive_params import RegimeAdaptiveParams

# レジームに応じたパラメータ調整
adaptive = RegimeAdaptiveParams(regime)
params = adaptive.get_signal_params("momentum")

# パラメータ例:
# BULL: lookback=15 (短め、トレンド追従)
# BEAR: lookback=30 (長め、慎重)
# VOLATILE: lookback=10 (最短、反応速く)
```

### 設定でのレジーム適応

```yaml
# config/default.yaml
regime_adaptive:
  enabled: true
  params:
    bull:
      momentum_lookback: 15
      rsi_period: 10
    bear:
      momentum_lookback: 30
      rsi_period: 21
    volatile:
      momentum_lookback: 10
      rsi_period: 7
```

---

## 新規シグナル追加手順

### Step 1: シグナルクラスの作成

```python
# src/signals/my_signal.py

from src.signals.base import BaseSignal
import pandas as pd
import numpy as np

class MyCustomSignal(BaseSignal):
    """カスタムシグナルの実装"""

    name = "my_custom_signal"
    category = "technical"  # momentum, mean_reversion, technical, regime

    def __init__(self, lookback: int = 20, threshold: float = 0.5):
        self.lookback = lookback
        self.threshold = threshold

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        シグナル計算

        Args:
            prices: OHLCV DataFrame (columns: Open, High, Low, Close, Volume)

        Returns:
            シグナル値 DataFrame (columns: 銘柄コード, values: シグナル値)
        """
        close = prices["Close"] if "Close" in prices.columns else prices

        # シグナル計算ロジック
        signal = close.pct_change(self.lookback)

        # 正規化 (-1 to 1)
        signal = np.clip(signal / signal.std(), -3, 3) / 3

        return signal

    def get_params_hash(self) -> str:
        """パラメータのハッシュ（キャッシュキー用）"""
        return f"{self.lookback}_{self.threshold}"
```

### Step 2: レジストリへの登録

```python
# src/signals/registry.py

from src.signals.my_signal import MyCustomSignal

# レジストリに追加
SIGNAL_REGISTRY = {
    # 既存シグナル...
    "my_custom_signal": MyCustomSignal,
}

# または動的登録
def register_signal(name: str, signal_class: type):
    SIGNAL_REGISTRY[name] = signal_class

register_signal("my_custom_signal", MyCustomSignal)
```

### Step 3: 設定への追加

```yaml
# config/default.yaml
signals:
  # 既存シグナル...

  my_custom_signal:
    enabled: true
    lookback: 20
    threshold: 0.5
    weight: 0.1  # アンサンブル時の重み
```

### Step 4: 事前計算への追加

```python
# src/backtest/signal_precompute.py

# INDEPENDENT_SIGNALS または RELATIVE_SIGNALS に追加
INDEPENDENT_SIGNALS = {
    # 既存...
    "my_custom_signal_*",  # ワイルドカードパターン
}
```

### Step 5: テストの作成

```python
# tests/signals/test_my_signal.py

import pytest
import pandas as pd
import numpy as np
from src.signals.my_signal import MyCustomSignal

class TestMyCustomSignal:
    def test_compute_basic(self):
        """基本的な計算テスト"""
        signal = MyCustomSignal(lookback=20)

        # テストデータ作成
        dates = pd.date_range("2020-01-01", periods=100)
        prices = pd.DataFrame(
            {"Close": np.random.randn(100).cumsum() + 100},
            index=dates,
        )

        result = signal.compute(prices)

        # 検証
        assert not result.isna().all().all()
        assert result.max().max() <= 1.0
        assert result.min().min() >= -1.0

    def test_params_hash(self):
        """パラメータハッシュのテスト"""
        signal1 = MyCustomSignal(lookback=20)
        signal2 = MyCustomSignal(lookback=30)

        assert signal1.get_params_hash() != signal2.get_params_hash()
```

### Step 6: ドキュメントの更新

1. `docs/SIGNAL_PIPELINE.md` のシグナル一覧に追加
2. `config/README.md` に設定例を追加
3. 必要に応じて `README.md` の機能一覧を更新

---

## シグナル計算のベストプラクティス

### 1. 正規化

```python
# シグナル値は -1 to 1 に正規化
signal = np.clip(raw_signal / raw_signal.std(), -3, 3) / 3
```

### 2. 欠損値処理

```python
# NaN は 0 に置換（中立シグナル）
signal = signal.fillna(0)
```

### 3. ルックフォワードバイアス防止

```python
# shift(1) で1日ずらす（当日のデータを使わない）
signal = prices.pct_change(lookback).shift(1)
```

### 4. メモリ効率

```python
# 大量データ処理時は float32 を使用
signal = signal.astype(np.float32)
```

---

## 関連ドキュメント

| ドキュメント | 概要 |
|-------------|------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | システムアーキテクチャ |
| [CACHE_SYSTEM.md](CACHE_SYSTEM.md) | キャッシュシステム |
| [BACKTEST_STANDARD.md](BACKTEST_STANDARD.md) | バックテスト規格 |
| [CMD016_INTEGRATION.md](CMD016_INTEGRATION.md) | 高度な機能統合 |
