# バックテストエンジン統合ガイド

> **Version**: 1.0.0
> **Last Updated**: 2026-01-29
> **Task Reference**: INT-007

## 概要

本ドキュメントは新規バックテストエンジン追加時の手順を定義する。
全てのバックテストエンジンは `BacktestEngineBase` を継承し、
統一インターフェースに準拠する必要がある。

---

## 新エンジン追加チェックリスト

### 1. 必須要件

- [ ] `BacktestEngineBase` を継承（`src/backtest/base.py`）
- [ ] `ENGINE_NAME` クラス変数を定義（一意の識別子）
- [ ] `run()` メソッドが統一シグネチャに準拠:
  ```python
  def run(
      self,
      universe: List[str],
      prices: Dict[str, pd.DataFrame],
      config: Optional[UnifiedBacktestConfig] = None,
      weights_func: Optional[Callable] = None,
  ) -> UnifiedBacktestResult:
  ```
- [ ] `validate_inputs()` メソッドを実装
- [ ] `UnifiedBacktestResult` を返却（`src/backtest/base.py` で定義）

### 2. テスト要件

- [ ] `tests/integration/test_engine_integration.py` にテスト追加
- [ ] TypeError回帰テストを含める（シグネチャ検証）
- [ ] 結果一致性テスト（Sharpe差異 < 0.01）
- [ ] 基本的なスモークテスト

```python
# テスト例
def test_my_engine_returns_unified_result():
    """新エンジンがUnifiedBacktestResultを返すことを確認"""
    engine = MyNewEngine()
    result = engine.run(universe, prices, config)
    assert isinstance(result, UnifiedBacktestResult)
    assert result.engine_name == "my_engine"
```

### 3. 統合要件

- [ ] `BacktestEngineFactory` に登録（`src/backtest/factory.py`）
- [ ] `src/backtest/__init__.py` にエクスポート追加
- [ ] CI通過確認（`pytest tests/integration/`）

---

## 統一インターフェース詳細

### BacktestEngineBase

```python
from abc import ABC, abstractmethod
from src.backtest.base import BacktestEngineBase, UnifiedBacktestConfig, UnifiedBacktestResult

class MyNewEngine(BacktestEngineBase):
    """新規バックテストエンジン"""

    ENGINE_NAME: str = "my_engine"  # 一意の識別子

    def __init__(self, config: Optional[UnifiedBacktestConfig] = None) -> None:
        super().__init__(config)
        # 追加の初期化

    def run(
        self,
        universe: List[str],
        prices: Dict[str, pd.DataFrame],
        config: Optional[UnifiedBacktestConfig] = None,
        weights_func: Optional[Callable] = None,
    ) -> UnifiedBacktestResult:
        """バックテスト実行"""
        # 設定の解決
        cfg = config or self._config
        if cfg is None:
            raise ValueError("Config must be provided")

        # 入力検証
        self.validate_inputs(universe, prices, cfg)

        # ... バックテストロジック ...

        return UnifiedBacktestResult(
            config=cfg,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            engine_name=self.ENGINE_NAME,
            # ... 結果データ ...
        )

    def validate_inputs(
        self,
        universe: List[str],
        prices: Dict[str, pd.DataFrame],
        config: UnifiedBacktestConfig,
    ) -> bool:
        """入力検証"""
        # 共通検証を利用
        warnings = self._validate_common_inputs(universe, prices, config)
        for w in warnings:
            self._logger.warning(w)
        return True
```

### UnifiedBacktestConfig

```python
@dataclass
class UnifiedBacktestConfig:
    """統一バックテスト設定"""
    start_date: datetime
    end_date: datetime
    rebalance_frequency: str = "monthly"  # daily, weekly, monthly, quarterly
    initial_capital: float = 100000.0
    transaction_cost_bps: float = 10.0
    min_weight: float = 0.0
    max_weight: float = 1.0
    # ... 追加設定 ...
```

### UnifiedBacktestResult

```python
@dataclass
class UnifiedBacktestResult:
    """統一バックテスト結果"""
    config: UnifiedBacktestConfig
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    engine_name: str

    # メトリクス
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0

    # 詳細データ
    daily_returns: Optional[pd.Series] = None
    portfolio_values: Optional[pd.Series] = None
    weights_history: Optional[pd.DataFrame] = None

    # 実行情報
    execution_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
```

---

## 既存エンジン一覧

| エンジン | ファイル | ENGINE_NAME | 特徴 |
|----------|----------|-------------|------|
| BacktestEngine | `engine.py` | `BacktestEngine` | 標準エンジン、Walk-forward対応 |
| FastBacktestEngine | `fast_engine.py` | `FastBacktestEngine` | Numba/Polars最適化、GPU対応 |
| StreamingBacktestEngine | `streaming_engine.py` | `streaming` | ストリーミング処理、メモリ効率 |
| RayBacktestEngine | `ray_engine.py` | `ray` | Ray分散処理、大規模並列 |
| VectorBTStyleEngine | `vectorbt_engine.py` | `vectorbt` | 超高速ベクトル化、Numba JIT |

### エンジン選択ガイド

```
┌─────────────────────────────────────────────────────────────────┐
│                      エンジン選択フローチャート                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  データサイズは?                                                 │
│  ├── 小〜中（〜100銘柄×5年）→ BacktestEngine                   │
│  ├── 中〜大（〜500銘柄×10年）→ FastBacktestEngine              │
│  └── 大（500銘柄×15年超）                                       │
│      ├── メモリ制約あり → StreamingBacktestEngine               │
│      ├── 分散環境あり → RayBacktestEngine                       │
│      └── 単一マシン最速 → VectorBTStyleEngine                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## よくある統合エラーと対策

### 1. TypeError: run() missing required argument

**原因**: シグネチャ不一致

```python
# NG: 古いシグネチャ
def run(self, prices, config):  # universe がない

# OK: 統一シグネチャ
def run(self, universe, prices, config, weights_func=None):
```

### 2. AttributeError: 'dict' object has no attribute 'xxx'

**原因**: 結果型の不一致

```python
# NG: dictを返している
return {"total_return": 0.1, ...}

# OK: UnifiedBacktestResultを返す
return UnifiedBacktestResult(total_return=0.1, ...)
```

### 3. KeyError in factory

**原因**: ENGINE_NAMEが未登録

```python
# src/backtest/factory.py に追加
ENGINES = {
    "BacktestEngine": BacktestEngine,
    "my_engine": MyNewEngine,  # 追加
}
```

---

## PR提出時のチェックリスト

新規エンジン追加PRを提出する前に、以下を確認：

```markdown
## エンジン追加チェックリスト

### 必須
- [ ] `BacktestEngineBase` を継承
- [ ] `ENGINE_NAME` を定義
- [ ] `run()` シグネチャ準拠
- [ ] `validate_inputs()` 実装
- [ ] `UnifiedBacktestResult` 返却

### テスト
- [ ] 統合テスト追加
- [ ] TypeError回帰テスト
- [ ] 結果一致性テスト（Sharpe差異 < 0.01）

### 統合
- [ ] Factory登録
- [ ] __init__.py エクスポート
- [ ] CI通過

### ドキュメント
- [ ] このファイルの「既存エンジン一覧」に追加
- [ ] docstring完備
```

---

## 関連ファイル

- `src/backtest/base.py` - 基底クラス定義
- `src/backtest/factory.py` - エンジンファクトリ
- `src/backtest/__init__.py` - エクスポート
- `tests/integration/test_engine_integration.py` - 統合テスト
- `tests/integration/test_engine_parity.py` - エンジン間一致性テスト

---

## 変更履歴

| 日付 | バージョン | 変更内容 |
|------|-----------|----------|
| 2026-01-29 | 1.0.0 | 初版作成（INT-007） |
