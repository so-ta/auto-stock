#!/usr/bin/env python3
"""
Backtest Result Viewer - FastAPI Web Application

バックテスト結果を閲覧・比較・実行できるWebアプリケーション。

Usage:
    python -m scripts.result_viewer.app --port 8080

    または

    uvicorn scripts.result_viewer.app:app --port 8080
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Try loading from project root first, then from parent directories
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Try parent directory (auto-stock/.env)
        parent_env = PROJECT_ROOT.parent / ".env"
        if parent_env.exists():
            load_dotenv(parent_env)
        else:
            load_dotenv()  # Default behavior
except ImportError:
    pass  # dotenv not installed, rely on system environment variables

try:
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
except ImportError:
    print("FastAPI is required. Install with: pip install fastapi uvicorn")
    sys.exit(1)


# =============================================================================
# Application Setup
# =============================================================================

app = FastAPI(
    title="ポートフォリオ管理",
    description="バックテスト結果の閲覧・比較・実行、ポートフォリオ運用管理ができるWebアプリケーション",
    version="3.0.0",
)

# Templates directory
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Default results directory
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


# =============================================================================
# Router Registration
# =============================================================================

def setup_routers(results_dir: Path = None, project_root: Path = None):
    """Configure and register all routers"""
    from scripts.result_viewer.routers import results as results_router
    from scripts.result_viewer.routers import backtest as backtest_router
    from scripts.result_viewer.routers import config as config_router
    from scripts.result_viewer.routers import portfolio as portfolio_router

    actual_results_dir = results_dir or DEFAULT_RESULTS_DIR
    actual_project_root = project_root or PROJECT_ROOT

    # Configure routers
    results_router.configure(templates, actual_results_dir)
    backtest_router.configure(templates, actual_results_dir, actual_project_root)
    config_router.configure(templates, actual_project_root)
    portfolio_router.configure(templates, actual_project_root)

    # Include routers
    app.include_router(results_router.router)
    app.include_router(backtest_router.router)
    app.include_router(config_router.router)
    app.include_router(portfolio_router.router)


# Initialize with defaults
setup_routers()


# =============================================================================
# Health Check
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "3.0.0",
        "results_dir": str(DEFAULT_RESULTS_DIR),
    }


# =============================================================================
# Scheduler Lifecycle (optional - starts if APScheduler is installed)
# =============================================================================

_scheduler = None


@app.on_event("startup")
async def startup_event():
    """Start scheduler on application startup"""
    global _scheduler
    try:
        from scripts.result_viewer.services.scheduler_service import get_scheduler
        _scheduler = get_scheduler(project_root=PROJECT_ROOT)
        await _scheduler.start()
    except ImportError:
        pass  # APScheduler not installed
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to start scheduler: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop scheduler on application shutdown"""
    global _scheduler
    if _scheduler is not None:
        await _scheduler.stop()


# =============================================================================
# Main entry point
# =============================================================================

def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(
        description="Backtest Result Viewer - Web Application"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="Port to run the server on (default: 8080)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--results-dir", "-d",
        type=str,
        default=None,
        help="Results directory (default: results/)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    # Update configuration if custom paths specified
    global DEFAULT_RESULTS_DIR
    if args.results_dir:
        DEFAULT_RESULTS_DIR = Path(args.results_dir)
        # Re-setup routers with new paths
        setup_routers(results_dir=DEFAULT_RESULTS_DIR)

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required. Install with: pip install uvicorn")
        sys.exit(1)

    print(f"ポートフォリオ管理ビューア v3.0.0 を起動中...")
    print(f"結果ディレクトリ: {DEFAULT_RESULTS_DIR}")
    print(f"サーバー: http://{args.host}:{args.port}")
    print()
    print("ページ一覧:")
    print(f"  - ダッシュボード:      http://{args.host}:{args.port}/")
    print(f"  - ポートフォリオ管理: http://{args.host}:{args.port}/portfolios")
    print(f"  - バックテスト実行:   http://{args.host}:{args.port}/backtest/run")
    print(f"  - バックテスト履歴:   http://{args.host}:{args.port}/history")
    print(f"  - ユニバース管理:     http://{args.host}:{args.port}/config/editor")

    uvicorn.run(
        "scripts.result_viewer.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
