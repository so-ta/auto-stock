"""
Backtest Result Viewer - Web-based viewer for backtest results

独立したWebアプリケーションとしてバックテスト結果を閲覧・比較できる。

Usage:
    python -m scripts.result_viewer.app --port 8080

    または

    cd scripts/result_viewer && uvicorn app:app --port 8080
"""

__version__ = "1.0.0"
