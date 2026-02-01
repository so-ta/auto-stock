"""
Portfolio Router - ポートフォリオ管理APIエンドポイント

ポートフォリオの一覧・詳細・作成・更新・削除・リバランスを提供。

エンドポイント:
- GET  /portfolios              - ポートフォリオ一覧
- GET  /portfolios/{id}         - ポートフォリオ詳細
- POST /portfolios              - ポートフォリオ作成
- PUT  /portfolios/{id}         - ポートフォリオ更新
- DELETE /portfolios/{id}       - ポートフォリオ削除
- GET  /portfolios/{id}/holdings - 保有資産取得
- PUT  /portfolios/{id}/holdings - 保有資産更新
- POST /portfolios/{id}/rebalance - リバランス実行
"""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/portfolios", tags=["portfolios"])

# Configuration - set by main app
_templates: Optional[Jinja2Templates] = None
_project_root: Optional[Path] = None
_config_dir: Optional[Path] = None
_state_dir: Optional[Path] = None


def configure(
    templates: Jinja2Templates,
    project_root: Path,
    config_dir: Path | None = None,
    state_dir: Path | None = None,
) -> None:
    """Configure the router

    Args:
        templates: Jinja2Templates instance
        project_root: Project root directory
        config_dir: Portfolio config directory (default: config/portfolios)
        state_dir: Portfolio state directory (default: data/portfolio_state)
    """
    global _templates, _project_root, _config_dir, _state_dir
    _templates = templates
    _project_root = project_root
    _config_dir = config_dir or (project_root / "config" / "portfolios")
    _state_dir = state_dir or (project_root / "data" / "portfolio_state")


def _get_manager():
    """Get PortfolioManager instance"""
    from src.portfolio.manager import PortfolioManager
    return PortfolioManager(config_dir=_config_dir, state_dir=_state_dir)


def _get_notifier():
    """Get DiscordNotifier instance"""
    from src.notification.discord import DiscordNotifier
    import yaml

    # Load notification config
    notification_config_path = _project_root / "config" / "notification.yaml"
    webhook_url = None

    if notification_config_path.exists():
        try:
            with open(notification_config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
            webhook_url = config.get("discord", {}).get("webhook_url")
        except Exception as e:
            logger.warning(f"Failed to load notification config: {e}")

    return DiscordNotifier(webhook_url=webhook_url)


# =============================================================================
# Request/Response Models
# =============================================================================


class PortfolioCreateRequest(BaseModel):
    """Portfolio creation request"""
    id: str = Field(..., description="Portfolio ID (alphanumeric, underscore)")
    name: str = Field(..., description="Portfolio display name")
    description: str = Field("", description="Portfolio description")
    initial_capital: float = Field(1_000_000, description="Initial capital")
    currency: str = Field("JPY", description="Currency code")

    # Universe
    universe_source: str = Field("asset_master", description="Universe source")
    universe_subset: str | None = Field(None, description="Universe subset name")
    universe_symbols: List[str] | None = Field(None, description="Custom symbols")

    # Lot size
    lot_size_default: int = Field(1, description="Default lot size")
    fractional_allowed: bool = Field(False, description="Allow fractional shares")

    # Schedule
    schedule_enabled: bool = Field(False, description="Enable scheduled execution")
    schedule_hour: int = Field(3, description="Scheduled hour")
    schedule_minute: int = Field(0, description="Scheduled minute")
    schedule_timezone: str = Field("Asia/Tokyo", description="Timezone")

    # Notification
    notification_on_rebalance: bool = Field(True, description="Notify on rebalance")
    notification_on_no_rebalance: bool = Field(False, description="Notify when no rebalance")


class PortfolioUpdateRequest(BaseModel):
    """Portfolio update request"""
    name: str | None = None
    description: str | None = None
    initial_capital: float | None = None
    currency: str | None = None
    universe_source: str | None = None
    universe_subset: str | None = None
    universe_symbols: List[str] | None = None
    lot_size_default: int | None = None
    fractional_allowed: bool | None = None
    schedule_enabled: bool | None = None
    schedule_hour: int | None = None
    schedule_minute: int | None = None
    schedule_timezone: str | None = None
    notification_on_rebalance: bool | None = None
    notification_on_no_rebalance: bool | None = None


class PositionInput(BaseModel):
    """Position input model"""
    symbol: str
    shares: float
    avg_cost: float
    current_price: float | None = None
    lot_size: int = 1


class HoldingsUpdateRequest(BaseModel):
    """Holdings update request"""
    cash: float = Field(0, description="Cash balance")
    positions: List[PositionInput] = Field(default_factory=list, description="Positions")


class RebalanceRequest(BaseModel):
    """Rebalance request"""
    target_weights: Dict[str, float] = Field(..., description="Target weights")
    prices: Dict[str, float] | None = Field(None, description="Current prices")
    threshold: float = Field(0.01, description="Rebalance threshold")
    apply: bool = Field(False, description="Apply rebalance to holdings")
    notify: bool = Field(True, description="Send Discord notification")


class SnapshotPositionInput(BaseModel):
    """Snapshot position input model"""
    symbol: str
    shares: float
    price: float


class SnapshotUpdateRequest(BaseModel):
    """Snapshot update request"""
    cash: float = Field(0, description="Cash balance")
    positions: List[SnapshotPositionInput] = Field(default_factory=list, description="Positions")
    is_rebalance: bool | None = Field(None, description="Is rebalance day (null to keep existing)")
    apply_to_holdings: bool = Field(True, description="Apply to current holdings if this is the latest snapshot")


class SnapshotCreateRequest(BaseModel):
    """Snapshot create request"""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    cash: float = Field(0, description="Cash balance")
    positions: List[SnapshotPositionInput] = Field(default_factory=list, description="Positions")
    is_rebalance: bool = Field(False, description="Is rebalance day")
    apply_to_holdings: bool = Field(False, description="Apply to current holdings if this is the latest snapshot")


# =============================================================================
# HTML Routes
# =============================================================================


@router.get("", response_class=HTMLResponse)
async def portfolios_list_view(request: Request):
    """ポートフォリオ一覧ページ"""
    manager = _get_manager()
    portfolios = manager.list_portfolios()

    # 各ポートフォリオの保有資産サマリを取得
    portfolio_data = []
    for config in portfolios:
        holdings = manager.load_holdings(config.id)
        portfolio_data.append({
            "config": config.to_dict(),
            "holdings": holdings.to_dict() if holdings else None,
        })

    return _templates.TemplateResponse(
        "portfolios.html",
        {
            "request": request,
            "portfolios": portfolio_data,
        },
    )


@router.get("/new", response_class=HTMLResponse)
async def portfolio_create_view(request: Request):
    """ポートフォリオ作成ページ"""
    return _templates.TemplateResponse(
        "portfolio_edit.html",
        {
            "request": request,
            "portfolio": None,
            "is_new": True,
        },
    )


@router.get("/{portfolio_id}", response_class=HTMLResponse)
async def portfolio_detail_view(
    request: Request,
    portfolio_id: str,
    fill_missing: bool = Query(True, description="Fill missing days with dynamic calculation"),
):
    """ポートフォリオ詳細ページ"""
    manager = _get_manager()
    config = manager.get_portfolio_config(portfolio_id)

    if not config:
        raise HTTPException(status_code=404, detail=f"Portfolio not found: {portfolio_id}")

    holdings = manager.load_holdings(portfolio_id)
    history = manager.get_history(portfolio_id)

    # fill_missingがTrueの場合は動的計算で補完
    if fill_missing:
        snapshots = history.get_filled_history(fill_missing=True)
    else:
        snapshots = history.get_history()

    metrics = history.get_performance_metrics()

    # 時系列データをJSON化（日次リターンも追加）
    timeseries_data = {
        "dates": [s.date.isoformat() for s in snapshots],
        "total_value": [s.total_value for s in snapshots],
        "cumulative_return": [s.cumulative_return * 100 for s in snapshots],
        "daily_return": [(s.daily_return or 0) * 100 for s in snapshots],
        "is_rebalance": [s.is_rebalance_day for s in snapshots],
    }

    # 直近30日を日付でソートして取得（追加順ではなく日付順）
    sorted_snapshots = sorted(snapshots, key=lambda s: s.date)
    recent_snapshots = sorted_snapshots[-30:]

    return _templates.TemplateResponse(
        "portfolio_detail.html",
        {
            "request": request,
            "portfolio": config.to_dict(),
            "holdings": holdings.to_dict() if holdings else None,
            "metrics": metrics.to_dict(),
            "timeseries": json.dumps(timeseries_data),
            "snapshots": [s.to_dict() for s in recent_snapshots],
            "snapshot_count": len(snapshots),
        },
    )


@router.get("/{portfolio_id}/edit", response_class=HTMLResponse)
async def portfolio_edit_view(request: Request, portfolio_id: str):
    """ポートフォリオ編集ページ"""
    manager = _get_manager()
    config = manager.get_portfolio_config(portfolio_id)

    if not config:
        raise HTTPException(status_code=404, detail=f"Portfolio not found: {portfolio_id}")

    return _templates.TemplateResponse(
        "portfolio_edit.html",
        {
            "request": request,
            "portfolio": config.to_dict(),
            "is_new": False,
        },
    )


@router.get("/{portfolio_id}/holdings", response_class=HTMLResponse)
async def holdings_editor_view(request: Request, portfolio_id: str):
    """保有資産編集ページ"""
    manager = _get_manager()
    config = manager.get_portfolio_config(portfolio_id)

    if not config:
        raise HTTPException(status_code=404, detail=f"Portfolio not found: {portfolio_id}")

    holdings = manager.get_or_create_holdings(portfolio_id)

    return _templates.TemplateResponse(
        "holdings_editor.html",
        {
            "request": request,
            "portfolio": config.to_dict(),
            "holdings": holdings.to_dict(),
        },
    )


@router.get("/{portfolio_id}/performance", response_class=HTMLResponse)
async def portfolio_performance_view(request: Request, portfolio_id: str):
    """パフォーマンス詳細ページ（統合のためリダイレクト）"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"/portfolios/{portfolio_id}?tab=performance", status_code=302)


# =============================================================================
# API Routes
# =============================================================================


@router.get("/api/list")
async def api_list_portfolios() -> List[Dict[str, Any]]:
    """Get all portfolios with summary"""
    manager = _get_manager()
    portfolios = manager.list_portfolios()

    result = []
    for config in portfolios:
        holdings = manager.load_holdings(config.id)
        result.append({
            "id": config.id,
            "name": config.name,
            "description": config.description,
            "currency": config.currency,
            "initial_capital": config.initial_capital,
            "schedule_enabled": config.schedule.enabled,
            "total_value": holdings.total_value if holdings else config.initial_capital,
            "position_count": holdings.position_count if holdings else 0,
            "last_updated": holdings.updated_at.isoformat() if holdings else None,
        })

    return result


@router.post("/api/create")
async def api_create_portfolio(request: PortfolioCreateRequest) -> Dict[str, Any]:
    """Create a new portfolio"""
    from src.portfolio.config import (
        PortfolioConfig,
        ScheduleConfig,
        NotificationConfig,
        UniverseConfig,
        LotSizeConfig,
    )

    manager = _get_manager()

    # Check if already exists
    if manager.get_portfolio_config(request.id):
        raise HTTPException(status_code=400, detail=f"Portfolio already exists: {request.id}")

    # Create config
    config = PortfolioConfig(
        id=request.id,
        name=request.name,
        description=request.description,
        initial_capital=request.initial_capital,
        currency=request.currency,
        universe=UniverseConfig(
            source=request.universe_source,
            subset=request.universe_subset,
            symbols=request.universe_symbols,
        ),
        lot_size=LotSizeConfig(
            default=request.lot_size_default,
            fractional_allowed=request.fractional_allowed,
        ),
        schedule=ScheduleConfig(
            enabled=request.schedule_enabled,
            hour=request.schedule_hour,
            minute=request.schedule_minute,
            timezone=request.schedule_timezone,
        ),
        notification=NotificationConfig(
            on_rebalance=request.notification_on_rebalance,
            on_no_rebalance=request.notification_on_no_rebalance,
        ),
    )

    if not manager.save_portfolio_config(config):
        raise HTTPException(status_code=500, detail="Failed to save portfolio config")

    # Create empty holdings
    holdings = manager.get_or_create_holdings(request.id)
    manager.save_holdings(holdings)

    return {
        "status": "created",
        "portfolio_id": request.id,
        "config": config.to_dict(),
    }


@router.put("/api/{portfolio_id}")
async def api_update_portfolio(
    portfolio_id: str,
    request: PortfolioUpdateRequest,
) -> Dict[str, Any]:
    """Update a portfolio"""
    from src.portfolio.config import (
        ScheduleConfig,
        NotificationConfig,
        UniverseConfig,
        LotSizeConfig,
    )

    manager = _get_manager()
    config = manager.get_portfolio_config(portfolio_id)

    if not config:
        raise HTTPException(status_code=404, detail=f"Portfolio not found: {portfolio_id}")

    # Update fields
    if request.name is not None:
        config.name = request.name
    if request.description is not None:
        config.description = request.description
    if request.initial_capital is not None:
        config.initial_capital = request.initial_capital
    if request.currency is not None:
        config.currency = request.currency

    # Universe
    if request.universe_source is not None:
        config.universe.source = request.universe_source
    if request.universe_subset is not None:
        config.universe.subset = request.universe_subset
    if request.universe_symbols is not None:
        config.universe.symbols = request.universe_symbols

    # Lot size
    if request.lot_size_default is not None:
        config.lot_size.default = request.lot_size_default
    if request.fractional_allowed is not None:
        config.lot_size.fractional_allowed = request.fractional_allowed

    # Schedule
    if request.schedule_enabled is not None:
        config.schedule.enabled = request.schedule_enabled
    if request.schedule_hour is not None:
        config.schedule.hour = request.schedule_hour
    if request.schedule_minute is not None:
        config.schedule.minute = request.schedule_minute
    if request.schedule_timezone is not None:
        config.schedule.timezone = request.schedule_timezone

    # Notification
    if request.notification_on_rebalance is not None:
        config.notification.on_rebalance = request.notification_on_rebalance
    if request.notification_on_no_rebalance is not None:
        config.notification.on_no_rebalance = request.notification_on_no_rebalance

    if not manager.save_portfolio_config(config):
        raise HTTPException(status_code=500, detail="Failed to save portfolio config")

    return {
        "status": "updated",
        "portfolio_id": portfolio_id,
        "config": config.to_dict(),
    }


@router.delete("/api/{portfolio_id}")
async def api_delete_portfolio(portfolio_id: str) -> Dict[str, str]:
    """Delete a portfolio"""
    manager = _get_manager()

    if not manager.get_portfolio_config(portfolio_id):
        raise HTTPException(status_code=404, detail=f"Portfolio not found: {portfolio_id}")

    if not manager.delete_portfolio(portfolio_id):
        raise HTTPException(status_code=500, detail="Failed to delete portfolio")

    return {
        "status": "deleted",
        "portfolio_id": portfolio_id,
    }


@router.get("/api/{portfolio_id}/holdings")
async def api_get_holdings(portfolio_id: str) -> Dict[str, Any]:
    """Get portfolio holdings"""
    manager = _get_manager()
    config = manager.get_portfolio_config(portfolio_id)

    if not config:
        raise HTTPException(status_code=404, detail=f"Portfolio not found: {portfolio_id}")

    holdings = manager.get_or_create_holdings(portfolio_id)
    return holdings.to_dict()


@router.put("/api/{portfolio_id}/holdings")
async def api_update_holdings(
    portfolio_id: str,
    request: HoldingsUpdateRequest,
) -> Dict[str, Any]:
    """Update portfolio holdings"""
    from src.portfolio.holdings import Holdings

    manager = _get_manager()
    config = manager.get_portfolio_config(portfolio_id)

    if not config:
        raise HTTPException(status_code=404, detail=f"Portfolio not found: {portfolio_id}")

    # Create new holdings
    holdings = Holdings.create_empty(
        portfolio_id=portfolio_id,
        initial_capital=request.cash,
        currency=config.currency,
    )

    # Add positions
    for pos in request.positions:
        holdings.add_position(
            symbol=pos.symbol,
            shares=pos.shares,
            avg_cost=pos.avg_cost,
            current_price=pos.current_price or pos.avg_cost,
            lot_size=pos.lot_size,
        )

    holdings.cash = request.cash

    if not manager.save_holdings(holdings):
        raise HTTPException(status_code=500, detail="Failed to save holdings")

    # Record snapshot
    manager.record_snapshot(portfolio_id, is_rebalance=False)

    return holdings.to_dict()


@router.post("/api/{portfolio_id}/rebalance")
async def api_rebalance(
    portfolio_id: str,
    request: RebalanceRequest,
) -> Dict[str, Any]:
    """Calculate and optionally apply rebalance"""
    manager = _get_manager()
    config = manager.get_portfolio_config(portfolio_id)

    if not config:
        raise HTTPException(status_code=404, detail=f"Portfolio not found: {portfolio_id}")

    holdings = manager.load_holdings(portfolio_id)
    if not holdings:
        raise HTTPException(status_code=400, detail="No holdings found")

    # Calculate rebalance
    result = manager.calculate_rebalance(
        portfolio_id=portfolio_id,
        target_weights=request.target_weights,
        prices=request.prices,
        threshold=request.threshold,
    )

    response = {
        "portfolio_id": portfolio_id,
        "needs_rebalance": result.needs_rebalance,
        "message": result.message,
        "target_weights": result.target_weights,
        "current_weights": result.current_weights,
    }

    if result.orders:
        response["orders"] = result.orders.to_dict()
    if result.adjustment:
        response["adjustment"] = result.adjustment.to_dict()

    # Apply rebalance if requested
    if request.apply and result.needs_rebalance:
        new_holdings = manager.apply_rebalance(portfolio_id, result)
        if new_holdings:
            response["applied"] = True
            response["new_holdings"] = new_holdings.to_dict()

            # Record snapshot
            manager.record_snapshot(portfolio_id, is_rebalance=True)
        else:
            response["applied"] = False
            response["apply_error"] = "Failed to apply rebalance"

    # Send notification if requested
    if request.notify and result.needs_rebalance and result.orders:
        notifier = _get_notifier()
        try:
            notifier.send_portfolio_rebalance(
                portfolio_name=config.name,
                holdings=holdings,
                orders=result.orders,
                adjustment=result.adjustment,
            )
            response["notification_sent"] = True
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            response["notification_sent"] = False
            response["notification_error"] = str(e)

    return response


@router.post("/api/{portfolio_id}/snapshot")
async def api_record_snapshot(
    portfolio_id: str,
    is_rebalance: bool = Query(False, description="Is rebalance day"),
) -> Dict[str, Any]:
    """Manually record a snapshot"""
    manager = _get_manager()

    if not manager.get_portfolio_config(portfolio_id):
        raise HTTPException(status_code=404, detail=f"Portfolio not found: {portfolio_id}")

    holdings = manager.load_holdings(portfolio_id)
    if not holdings:
        raise HTTPException(status_code=400, detail="No holdings found")

    manager.record_snapshot(portfolio_id, is_rebalance=is_rebalance)

    return {
        "status": "recorded",
        "portfolio_id": portfolio_id,
        "is_rebalance": is_rebalance,
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/api/{portfolio_id}/history")
async def api_get_history(
    portfolio_id: str,
    start_date: str | None = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: str | None = Query(None, description="End date (YYYY-MM-DD)"),
    fill_missing: bool = Query(False, description="Fill missing days with dynamic calculation"),
) -> Dict[str, Any]:
    """Get portfolio history

    Args:
        portfolio_id: Portfolio ID
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        fill_missing: If True, fill missing days with dynamically calculated snapshots
    """
    from datetime import date as date_type

    manager = _get_manager()

    if not manager.get_portfolio_config(portfolio_id):
        raise HTTPException(status_code=404, detail=f"Portfolio not found: {portfolio_id}")

    history = manager.get_history(portfolio_id)

    start = date_type.fromisoformat(start_date) if start_date else None
    end = date_type.fromisoformat(end_date) if end_date else None

    if fill_missing:
        snapshots = history.get_filled_history(start, end, fill_missing=True)
    else:
        snapshots = history.get_history(start, end)

    metrics = history.get_performance_metrics(start, end)

    return {
        "portfolio_id": portfolio_id,
        "metrics": metrics.to_dict(),
        "snapshots": [s.to_dict() for s in snapshots],
        "count": len(snapshots),
    }


@router.get("/api/{portfolio_id}/export")
async def api_export_history(portfolio_id: str) -> Dict[str, Any]:
    """Export history to CSV"""
    manager = _get_manager()

    if not manager.get_portfolio_config(portfolio_id):
        raise HTTPException(status_code=404, detail=f"Portfolio not found: {portfolio_id}")

    history = manager.get_history(portfolio_id)
    output_path = _state_dir / portfolio_id / f"history_export_{datetime.now().strftime('%Y%m%d')}.csv"

    if not history.export_to_csv(output_path):
        raise HTTPException(status_code=500, detail="Failed to export history")

    return {
        "status": "exported",
        "portfolio_id": portfolio_id,
        "file_path": str(output_path),
    }


@router.post("/api/{portfolio_id}/test-notification")
async def api_test_notification(portfolio_id: str) -> Dict[str, Any]:
    """Send a test notification"""
    manager = _get_manager()
    config = manager.get_portfolio_config(portfolio_id)

    if not config:
        raise HTTPException(status_code=404, detail=f"Portfolio not found: {portfolio_id}")

    notifier = _get_notifier()

    if not notifier.webhook_url:
        raise HTTPException(status_code=400, detail="Discord webhook URL not configured")

    success = notifier._send_message(
        f"🧪 **テスト通知**\n\nポートフォリオ: {config.name}\nID: {config.id}\n\nこれはテスト通知です。"
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to send test notification")

    return {
        "status": "sent",
        "portfolio_id": portfolio_id,
    }


@router.post("/api/{portfolio_id}/run-now")
async def api_run_rebalance_now(portfolio_id: str) -> Dict[str, Any]:
    """Execute rebalance check immediately

    Calls the scheduler's run_now() method which:
    1. Fetches latest prices from yfinance
    2. Updates holdings prices
    3. Records daily snapshot
    4. Runs rebalance calculation
    5. Sends Discord notification (if enabled)

    Returns:
        {
            "portfolio_id": str,
            "timestamp": str,
            "status": "rebalance_needed" | "no_rebalance" | "error",
            "needs_rebalance": bool,
            "message": str,
            "error": str (only on error)
        }
    """
    from scripts.result_viewer.services.scheduler_service import get_scheduler

    manager = _get_manager()
    config = manager.get_portfolio_config(portfolio_id)

    if not config:
        raise HTTPException(status_code=404, detail=f"Portfolio not found: {portfolio_id}")

    try:
        scheduler = get_scheduler(
            project_root=_project_root,
            config_dir=_config_dir,
            state_dir=_state_dir,
        )
        result = await scheduler.run_now(portfolio_id)
        return result
    except Exception as e:
        logger.error(f"Failed to run rebalance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Symbol/Price/CSV API Routes
# =============================================================================


def _load_asset_master() -> Dict[str, Any]:
    """Load asset master configuration"""
    asset_master_path = _project_root / "config" / "asset_master.yaml"
    if not asset_master_path.exists():
        return {"symbols": [], "taxonomy": {}}

    try:
        with open(asset_master_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {"symbols": [], "taxonomy": {}}
    except Exception as e:
        logger.warning(f"Failed to load asset master: {e}")
        return {"symbols": [], "taxonomy": {}}


@router.get("/api/symbols")
async def api_get_symbols(
    market: str | None = Query(None, description="Filter by market (us, japan, etc.)"),
    asset_class: str | None = Query(None, description="Filter by asset class (equity, etf, etc.)"),
    search: str | None = Query(None, description="Search query for ticker or name"),
    limit: int = Query(100, le=500, description="Max results"),
) -> List[Dict[str, Any]]:
    """Get symbols from asset master"""
    master = _load_asset_master()
    symbols = master.get("symbols", [])

    results = []
    for sym in symbols:
        ticker = sym.get("ticker", "")
        taxonomy = sym.get("taxonomy", {})

        # Filter by market
        if market and taxonomy.get("market") != market:
            continue

        # Filter by asset class
        if asset_class and taxonomy.get("asset_class") != asset_class:
            continue

        # Search filter
        if search:
            search_lower = search.lower()
            name = sym.get("name", "")
            if search_lower not in ticker.lower() and search_lower not in name.lower():
                continue

        results.append({
            "ticker": ticker,
            "name": sym.get("name", ticker),
            "market": taxonomy.get("market"),
            "asset_class": taxonomy.get("asset_class"),
            "sector": taxonomy.get("sector"),
            "lot_size": sym.get("lot_size", 1),
        })

        if len(results) >= limit:
            break

    return results


@router.get("/api/prices")
async def api_get_prices(
    symbols: str = Query(..., description="Comma-separated symbols"),
) -> Dict[str, float | None]:
    """Get latest prices for symbols via yfinance"""
    from src.data.batch_fetcher import BatchDataFetcher

    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        return {}

    try:
        fetcher = BatchDataFetcher(
            max_concurrent=5,
            rate_limit_per_sec=2.0,
            cache_max_age_days=0,
        )

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        batch_result = await fetcher.fetch_all(symbol_list, start_date, end_date)

        prices: Dict[str, float | None] = {}
        for symbol, result in batch_result.results.items():
            if result.data is not None and len(result.data) > 0:
                df = result.data
                close_col = None
                for col in df.columns:
                    if 'close' in str(col).lower():
                        close_col = col
                        break
                if close_col:
                    prices[symbol] = float(df[close_col].to_list()[-1])
                else:
                    prices[symbol] = None
            else:
                prices[symbol] = None

        return prices

    except Exception as e:
        logger.error(f"Failed to fetch prices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/lot-sizes")
async def api_get_lot_sizes(
    symbols: str = Query(..., description="Comma-separated symbols"),
) -> Dict[str, int]:
    """Get lot sizes for symbols from asset master"""
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        return {}

    master = _load_asset_master()
    symbol_map = {s.get("ticker"): s for s in master.get("symbols", [])}

    result = {}
    for symbol in symbol_list:
        if symbol in symbol_map:
            result[symbol] = symbol_map[symbol].get("lot_size", 1)
        else:
            # Default lot size based on market
            if symbol.endswith(".T"):
                result[symbol] = 100  # Japanese stocks
            else:
                result[symbol] = 1

    return result


@router.get("/api/subsets")
async def api_get_subsets() -> List[Dict[str, str]]:
    """Get available universe subsets from asset master"""
    master = _load_asset_master()
    taxonomy = master.get("taxonomy", {})

    subsets = []

    # Add markets as subsets
    for key, name in taxonomy.get("market", {}).items():
        subsets.append({"id": f"market:{key}", "name": f"Market: {name}"})

    # Add asset classes as subsets
    for key, name in taxonomy.get("asset_class", {}).items():
        subsets.append({"id": f"asset_class:{key}", "name": f"Asset Class: {name}"})

    return subsets


@router.get("/api/csv-template")
async def api_get_csv_template() -> StreamingResponse:
    """Download CSV template for holdings import"""
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(["symbol", "shares", "avg_cost"])

    # Sample data
    writer.writerow(["7203.T", "100", "2500"])
    writer.writerow(["9984.T", "200", "7800"])
    writer.writerow(["6758.T", "50", "12000"])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=holdings_template.csv"},
    )


@router.get("/api/{portfolio_id}/holdings/download")
async def api_download_holdings(portfolio_id: str) -> StreamingResponse:
    """Download current holdings as CSV"""
    manager = _get_manager()
    config = manager.get_portfolio_config(portfolio_id)

    if not config:
        raise HTTPException(status_code=404, detail=f"Portfolio not found: {portfolio_id}")

    holdings = manager.load_holdings(portfolio_id)
    if not holdings:
        raise HTTPException(status_code=400, detail="No holdings found")

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(["symbol", "shares", "avg_cost", "current_price", "market_value", "weight"])

    # Data
    for symbol, pos in holdings.positions.items():
        writer.writerow([
            symbol,
            pos.shares,
            pos.avg_cost,
            pos.current_price,
            pos.market_value,
            f"{pos.weight:.4f}",
        ])

    output.seek(0)
    filename = f"{portfolio_id}_holdings_{datetime.now().strftime('%Y%m%d')}.csv"
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.post("/api/{portfolio_id}/holdings/import")
async def api_import_holdings(
    portfolio_id: str,
    request: Request,
) -> Dict[str, Any]:
    """Import holdings from CSV data"""
    from src.portfolio.holdings import Holdings

    manager = _get_manager()
    config = manager.get_portfolio_config(portfolio_id)

    if not config:
        raise HTTPException(status_code=404, detail=f"Portfolio not found: {portfolio_id}")

    # Parse request body
    body = await request.json()
    csv_data = body.get("csv_data", "")
    cash = body.get("cash", 0.0)

    if not csv_data:
        raise HTTPException(status_code=400, detail="No CSV data provided")

    # Parse CSV
    reader = csv.DictReader(io.StringIO(csv_data))
    positions = []

    for row in reader:
        try:
            symbol = row.get("symbol", "").strip()
            shares = float(row.get("shares", 0))
            avg_cost = float(row.get("avg_cost", 0))
            current_price = row.get("current_price")

            if symbol and shares > 0:
                positions.append({
                    "symbol": symbol,
                    "shares": shares,
                    "avg_cost": avg_cost,
                    "current_price": float(current_price) if current_price else None,
                })
        except (ValueError, KeyError) as e:
            logger.warning(f"Skipping invalid row: {row}, error: {e}")
            continue

    # Create holdings
    holdings = Holdings.create_empty(
        portfolio_id=portfolio_id,
        initial_capital=cash,
        currency=config.currency,
    )

    for pos in positions:
        holdings.add_position(
            symbol=pos["symbol"],
            shares=pos["shares"],
            avg_cost=pos["avg_cost"],
            current_price=pos["current_price"] or pos["avg_cost"],
            lot_size=1,
        )

    holdings.cash = cash

    if not manager.save_holdings(holdings):
        raise HTTPException(status_code=500, detail="Failed to save holdings")

    return {
        "status": "imported",
        "portfolio_id": portfolio_id,
        "position_count": len(positions),
        "holdings": holdings.to_dict(),
    }


@router.post("/api/{portfolio_id}/update-prices")
async def api_update_prices(portfolio_id: str) -> Dict[str, Any]:
    """Fetch and update latest prices for portfolio holdings"""
    from src.data.batch_fetcher import BatchDataFetcher

    manager = _get_manager()
    config = manager.get_portfolio_config(portfolio_id)

    if not config:
        raise HTTPException(status_code=404, detail=f"Portfolio not found: {portfolio_id}")

    holdings = manager.load_holdings(portfolio_id)
    if not holdings or not holdings.positions:
        raise HTTPException(status_code=400, detail="No holdings to update")

    symbols = list(holdings.positions.keys())

    try:
        fetcher = BatchDataFetcher(
            max_concurrent=5,
            rate_limit_per_sec=2.0,
            cache_max_age_days=0,
        )

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        batch_result = await fetcher.fetch_all(symbols, start_date, end_date)

        prices = {}
        for symbol, result in batch_result.results.items():
            if result.data is not None and len(result.data) > 0:
                df = result.data
                close_col = None
                for col in df.columns:
                    if 'close' in str(col).lower():
                        close_col = col
                        break
                if close_col:
                    prices[symbol] = float(df[close_col].to_list()[-1])

        # Update holdings
        if prices:
            manager.update_prices(portfolio_id, prices)
            holdings = manager.load_holdings(portfolio_id)

        return {
            "status": "updated",
            "portfolio_id": portfolio_id,
            "updated_count": len(prices),
            "total_symbols": len(symbols),
            "prices": prices,
            "holdings": holdings.to_dict() if holdings else None,
        }

    except Exception as e:
        logger.error(f"Failed to update prices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Snapshot Edit/Delete API Routes
# =============================================================================


@router.post("/api/{portfolio_id}/snapshot")
async def api_create_snapshot(
    portfolio_id: str,
    request: SnapshotCreateRequest,
) -> Dict[str, Any]:
    """Create a new snapshot for a specific date

    Args:
        portfolio_id: Portfolio ID
        request: Snapshot creation data including date
    """
    from datetime import date as date_type
    from src.portfolio.holdings import Holdings

    manager = _get_manager()
    config = manager.get_portfolio_config(portfolio_id)

    if not config:
        raise HTTPException(status_code=404, detail=f"Portfolio not found: {portfolio_id}")

    try:
        target_date = date_type.fromisoformat(request.date)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {request.date}")

    history = manager.get_history(portfolio_id)

    # Check if snapshot already exists
    existing = history.get_snapshot_by_date(target_date)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Snapshot already exists for {request.date}. Use PUT to update."
        )

    # Convert positions to dict format
    positions_dict = {
        pos.symbol: {"shares": pos.shares, "price": pos.price}
        for pos in request.positions
    }

    # Create the snapshot using update_snapshot (which handles new creation too)
    created_snapshot = history.update_snapshot(
        target_date=target_date,
        cash=request.cash,
        positions=positions_dict,
        is_rebalance=request.is_rebalance,
    )

    if not created_snapshot:
        raise HTTPException(status_code=500, detail="Failed to create snapshot")

    result = {
        "status": "created",
        "portfolio_id": portfolio_id,
        "date": request.date,
        "snapshot": created_snapshot.to_dict(),
        "applied_to_holdings": False,
    }

    # Check if this is now the latest snapshot and apply_to_holdings is True
    latest = history.get_latest_snapshot()
    if request.apply_to_holdings and latest and latest.date == target_date:
        # Create holdings from snapshot
        holdings = Holdings.create_empty(
            portfolio_id=portfolio_id,
            initial_capital=request.cash,
            currency=config.currency,
        )

        for pos in request.positions:
            holdings.add_position(
                symbol=pos.symbol,
                shares=pos.shares,
                avg_cost=pos.price,
                current_price=pos.price,
                lot_size=1,
            )

        holdings.cash = request.cash

        if manager.save_holdings(holdings):
            result["applied_to_holdings"] = True
            result["holdings"] = holdings.to_dict()

    return result


@router.get("/api/{portfolio_id}/snapshot/{snapshot_date}")
async def api_get_snapshot(
    portfolio_id: str,
    snapshot_date: str,
) -> Dict[str, Any]:
    """Get a specific snapshot by date

    Args:
        portfolio_id: Portfolio ID
        snapshot_date: Date in YYYY-MM-DD format
    """
    from datetime import date as date_type

    manager = _get_manager()

    if not manager.get_portfolio_config(portfolio_id):
        raise HTTPException(status_code=404, detail=f"Portfolio not found: {portfolio_id}")

    try:
        target_date = date_type.fromisoformat(snapshot_date)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {snapshot_date}")

    history = manager.get_history(portfolio_id)
    snapshot = history.get_snapshot_by_date(target_date)

    if not snapshot:
        raise HTTPException(status_code=404, detail=f"Snapshot not found for {snapshot_date}")

    return snapshot.to_dict()


@router.put("/api/{portfolio_id}/snapshot/{snapshot_date}")
async def api_update_snapshot(
    portfolio_id: str,
    snapshot_date: str,
    request: SnapshotUpdateRequest,
) -> Dict[str, Any]:
    """Update a snapshot

    If the updated snapshot is the latest one and apply_to_holdings is True,
    the current holdings will also be updated.

    Args:
        portfolio_id: Portfolio ID
        snapshot_date: Date in YYYY-MM-DD format
        request: Snapshot update data
    """
    from datetime import date as date_type
    from src.portfolio.holdings import Holdings

    manager = _get_manager()
    config = manager.get_portfolio_config(portfolio_id)

    if not config:
        raise HTTPException(status_code=404, detail=f"Portfolio not found: {portfolio_id}")

    try:
        target_date = date_type.fromisoformat(snapshot_date)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {snapshot_date}")

    history = manager.get_history(portfolio_id)

    # Convert positions to dict format
    positions_dict = {
        pos.symbol: {"shares": pos.shares, "price": pos.price}
        for pos in request.positions
    }

    # Update the snapshot
    updated_snapshot = history.update_snapshot(
        target_date=target_date,
        cash=request.cash,
        positions=positions_dict,
        is_rebalance=request.is_rebalance,
    )

    if not updated_snapshot:
        raise HTTPException(status_code=500, detail="Failed to update snapshot")

    result = {
        "status": "updated",
        "portfolio_id": portfolio_id,
        "date": snapshot_date,
        "snapshot": updated_snapshot.to_dict(),
        "applied_to_holdings": False,
    }

    # Check if this is the latest snapshot and apply_to_holdings is True
    latest = history.get_latest_snapshot()
    if request.apply_to_holdings and latest and latest.date == target_date:
        # Create holdings from snapshot
        holdings = Holdings.create_empty(
            portfolio_id=portfolio_id,
            initial_capital=request.cash,
            currency=config.currency,
        )

        for pos in request.positions:
            holdings.add_position(
                symbol=pos.symbol,
                shares=pos.shares,
                avg_cost=pos.price,  # Use current price as avg_cost
                current_price=pos.price,
                lot_size=1,
            )

        holdings.cash = request.cash

        if manager.save_holdings(holdings):
            result["applied_to_holdings"] = True
            result["holdings"] = holdings.to_dict()

    return result


@router.delete("/api/{portfolio_id}/snapshot/{snapshot_date}")
async def api_delete_snapshot(
    portfolio_id: str,
    snapshot_date: str,
) -> Dict[str, str]:
    """Delete a snapshot

    Args:
        portfolio_id: Portfolio ID
        snapshot_date: Date in YYYY-MM-DD format
    """
    from datetime import date as date_type

    manager = _get_manager()

    if not manager.get_portfolio_config(portfolio_id):
        raise HTTPException(status_code=404, detail=f"Portfolio not found: {portfolio_id}")

    try:
        target_date = date_type.fromisoformat(snapshot_date)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {snapshot_date}")

    history = manager.get_history(portfolio_id)

    if not history.delete_snapshot(target_date):
        raise HTTPException(status_code=404, detail=f"Snapshot not found or failed to delete: {snapshot_date}")

    return {
        "status": "deleted",
        "portfolio_id": portfolio_id,
        "date": snapshot_date,
    }
