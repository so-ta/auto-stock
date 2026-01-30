"""
Architecture tests configuration.

現状の違反を段階的に解決するための設定。
"""

import pytest


def pytest_configure(config):
    """pytest設定"""
    # アーキテクチャテストマーカーの説明
    config.addinivalue_line(
        "markers",
        "architecture: marks tests as architecture/design rule tests"
    )


# 既知の違反として許容する設定
# これらは今後のリファクタリングで解決予定

# Phase 1: 厳格モード（CI用）- まだ有効化しない
STRICT_MODE = False

# 重複が許容されるクラス（歴史的理由）
LEGACY_DUPLICATES = {
    "OptimizationResult",    # 各オプティマイザで独自定義
    "MarketRegime",          # 複数モジュールで定義
    "RebalanceFrequency",    # エンジン毎に定義
    "SimulationResult",      # バックテストエンジン毎
    "BacktestResult",        # エンジン毎
}

# サイズ超過が許容されるファイル（分割予定）
LEGACY_LARGE_FILES = [
    "orchestrator/pipeline.py",      # 3583行、cmd_027で分割予定
    "backtest/fast_engine.py",       # 統合エンジン
    "backtest/engine.py",            # メインエンジン
    "main.py",                       # エントリポイント
]
