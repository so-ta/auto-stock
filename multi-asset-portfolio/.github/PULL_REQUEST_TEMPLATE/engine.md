## バックテストエンジン追加PR

<!-- エンジン追加時はこのテンプレートを使用してください -->

### 概要

<!-- エンジンの目的と特徴を簡潔に説明 -->

**エンジン名**: `ENGINE_NAME`
**ファイル**: `src/backtest/xxx_engine.py`

### 変更内容

<!-- 主な変更点を箇条書きで -->

- [ ] 新規エンジン実装
- [ ] ファクトリ登録
- [ ] テスト追加

### チェックリスト

<!-- 全項目にチェックを入れてからPRを提出 -->

#### 必須要件
- [ ] `BacktestEngineBase` を継承
- [ ] `ENGINE_NAME` クラス変数を定義
- [ ] `run()` シグネチャが統一インターフェースに準拠
  ```python
  run(universe, prices, config, weights_func) -> UnifiedBacktestResult
  ```
- [ ] `validate_inputs()` メソッドを実装
- [ ] `UnifiedBacktestResult` を返却

#### テスト要件
- [ ] `tests/integration/test_engine_integration.py` にテスト追加
- [ ] TypeError回帰テストを含める
- [ ] 結果一致性テスト（Sharpe差異 < 0.01）
- [ ] 全テストがパス

#### 統合要件
- [ ] `BacktestEngineFactory` に登録
- [ ] `src/backtest/__init__.py` にエクスポート追加
- [ ] `docs/ENGINE_INTEGRATION.md` の「既存エンジン一覧」に追加
- [ ] CI通過確認

### テスト結果

<!-- テスト実行結果を貼り付け -->

```
pytest tests/integration/test_engine_integration.py -v
```

### パフォーマンス比較（任意）

<!-- 既存エンジンとの比較があれば記載 -->

| エンジン | 実行時間 | メモリ使用量 |
|----------|----------|--------------|
| BacktestEngine | - | - |
| 新エンジン | - | - |

### 関連Issue/Task

<!-- 関連するIssueやTaskを記載 -->

- Closes #xxx
- Related to task_xxx

### 参考資料

- [エンジン統合ガイド](../docs/ENGINE_INTEGRATION.md)
