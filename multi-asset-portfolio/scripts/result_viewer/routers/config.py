"""
Config Router - Configuration management API endpoints

ユニバース、コスト設定、システム設定の管理用APIエンドポイント。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

router = APIRouter(prefix="/config", tags=["config"])

# These will be set by the main app
_templates: Optional[Jinja2Templates] = None
_project_root: Optional[Path] = None


def configure(templates: Jinja2Templates, project_root: Path) -> None:
    """Configure the router"""
    global _templates, _project_root
    _templates = templates
    _project_root = project_root


def _get_config_service():
    """Get ConfigService instance"""
    from scripts.result_viewer.services.config_service import get_config_service
    return get_config_service(project_root=_project_root)


# =============================================================================
# Request/Response Models
# =============================================================================


class UniverseUpdateRequest(BaseModel):
    """Universe update request"""
    name: str = Field(..., description="Universe name")
    symbols: List[str] = Field(..., description="List of symbols")
    description: Optional[str] = Field(None, description="Description")


class CostOverrideRequest(BaseModel):
    """Custom cost settings request"""
    spread_bps: float = Field(10.0, ge=0, le=100, description="Spread in bps")
    commission_bps: float = Field(5.0, ge=0, le=100, description="Commission in bps")
    slippage_bps: float = Field(10.0, ge=0, le=100, description="Slippage in bps")


# =============================================================================
# HTML Routes
# =============================================================================


@router.get("/settings", response_class=HTMLResponse)
async def settings_view(request: Request):
    """Settings view page"""
    config_service = _get_config_service()

    defaults = config_service.get_default_settings()
    backtest_defaults = config_service.get_backtest_defaults()
    cost_profiles = config_service.list_cost_profiles()

    return _templates.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "defaults": defaults,
            "backtest_defaults": backtest_defaults,
            "cost_profiles": [c.to_dict() for c in cost_profiles],
        },
    )


# =============================================================================
# Universe API Routes
# =============================================================================


@router.get("/api/universes")
async def api_list_universes() -> List[Dict[str, Any]]:
    """List all available universes"""
    config_service = _get_config_service()
    universes = config_service.list_universes()
    return [u.to_dict() for u in universes]


@router.get("/api/universes/{name}")
async def api_get_universe(name: str) -> Dict[str, Any]:
    """Get universe details"""
    config_service = _get_config_service()
    universe = config_service.get_universe(name)

    if not universe:
        raise HTTPException(status_code=404, detail=f"Universe not found: {name}")

    # Include symbols
    symbols = config_service.get_universe_symbols(name)

    return {
        **universe.to_dict(),
        "symbols": symbols,
    }


@router.get("/api/universes/{name}/symbols")
async def api_get_universe_symbols(name: str) -> List[str]:
    """Get symbols in a universe"""
    config_service = _get_config_service()
    symbols = config_service.get_universe_symbols(name)

    if not symbols:
        raise HTTPException(status_code=404, detail=f"Universe not found or empty: {name}")

    return symbols


@router.get("/api/universes/{name}/validate")
async def api_validate_universe(name: str) -> Dict[str, Any]:
    """Validate a universe file"""
    config_service = _get_config_service()
    return config_service.validate_universe_file(name)


# =============================================================================
# Cost Profile API Routes
# =============================================================================


@router.get("/api/cost-profiles")
async def api_list_cost_profiles() -> List[Dict[str, Any]]:
    """List all cost profiles"""
    config_service = _get_config_service()
    profiles = config_service.list_cost_profiles()
    return [p.to_dict() for p in profiles]


@router.get("/api/cost-profiles/{name}")
async def api_get_cost_profile(name: str) -> Dict[str, Any]:
    """Get a specific cost profile"""
    config_service = _get_config_service()
    profile = config_service.get_cost_profile(name)

    if not profile:
        raise HTTPException(status_code=404, detail=f"Cost profile not found: {name}")

    return profile.to_dict()


# =============================================================================
# Preset API Routes
# =============================================================================


@router.get("/api/frequency-presets")
async def api_list_frequency_presets() -> List[Dict[str, Any]]:
    """List rebalance frequency presets"""
    config_service = _get_config_service()
    presets = config_service.list_frequency_presets()
    return [p.to_dict() for p in presets]


@router.get("/api/period-presets")
async def api_list_period_presets() -> List[Dict[str, Any]]:
    """List period presets"""
    config_service = _get_config_service()
    presets = config_service.list_period_presets()
    return [p.to_dict() for p in presets]


# =============================================================================
# Default Settings API Routes
# =============================================================================


@router.get("/api/defaults")
async def api_get_defaults() -> Dict[str, Any]:
    """Get all default settings"""
    config_service = _get_config_service()
    return config_service.get_default_settings()


@router.get("/api/defaults/backtest")
async def api_get_backtest_defaults() -> Dict[str, Any]:
    """Get backtest-specific defaults"""
    config_service = _get_config_service()
    return config_service.get_backtest_defaults()


# =============================================================================
# Estimation API Routes
# =============================================================================


@router.get("/api/estimate")
async def api_estimate_backtest(
    universe: str = Query(..., description="Universe name"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    frequency: str = Query("monthly", description="Rebalance frequency"),
) -> Dict[str, Any]:
    """Estimate backtest execution time"""
    config_service = _get_config_service()

    # Get universe size
    universe_info = config_service.get_universe(universe)
    if not universe_info:
        raise HTTPException(status_code=404, detail=f"Universe not found: {universe}")

    return config_service.estimate_backtest_duration(
        universe_size=universe_info.symbol_count,
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
    )


# =============================================================================
# Subset API Routes (CRUD)
# =============================================================================


class SubsetDefinition(BaseModel):
    """Subset definition for create/update"""
    name: str = Field(..., description="Display name")
    description: str = Field("", description="Description")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Filter criteria")
    manual_include: List[str] = Field(default_factory=list, description="Manual inclusions")
    manual_exclude: List[str] = Field(default_factory=list, description="Manual exclusions")


class FilterPreviewRequest(BaseModel):
    """Request for filter preview"""
    tags: List[str] = Field(default_factory=list)
    exclude_tags: List[str] = Field(default_factory=list)
    taxonomy: Dict[str, List[str]] = Field(default_factory=dict)
    manual_include: List[str] = Field(default_factory=list)
    manual_exclude: List[str] = Field(default_factory=list)


@router.get("/api/subsets")
async def api_list_subsets() -> Dict[str, Any]:
    """List all subsets"""
    config_service = _get_config_service()
    data = config_service._load_master()
    return data.get("subsets", {})


@router.get("/api/subsets/{name}")
async def api_get_subset(name: str) -> Dict[str, Any]:
    """Get a specific subset"""
    config_service = _get_config_service()
    subset = config_service.get_subset(name)
    if not subset:
        raise HTTPException(status_code=404, detail=f"Subset not found: {name}")
    return subset


@router.post("/api/subsets")
async def api_create_subset(
    name: str = Query(..., description="Subset identifier"),
    body: SubsetDefinition = ...,
) -> Dict[str, Any]:
    """Create a new subset"""
    config_service = _get_config_service()
    try:
        definition = {
            "name": body.name,
            "description": body.description,
            "filters": body.filters,
        }
        if body.manual_include:
            definition["manual_include"] = body.manual_include
        if body.manual_exclude:
            definition["manual_exclude"] = body.manual_exclude

        config_service.create_subset(name, definition)
        return {"success": True, "name": name}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/api/subsets/{name}")
async def api_update_subset(name: str, body: SubsetDefinition) -> Dict[str, Any]:
    """Update an existing subset"""
    config_service = _get_config_service()
    try:
        definition = {
            "name": body.name,
            "description": body.description,
            "filters": body.filters,
        }
        if body.manual_include:
            definition["manual_include"] = body.manual_include
        if body.manual_exclude:
            definition["manual_exclude"] = body.manual_exclude

        config_service.update_subset(name, definition)
        return {"success": True, "name": name}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/api/subsets/{name}")
async def api_delete_subset(name: str) -> Dict[str, Any]:
    """Delete a subset"""
    config_service = _get_config_service()
    try:
        config_service.delete_subset(name)
        return {"success": True, "name": name}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/subsets/preview")
async def api_preview_filter(body: FilterPreviewRequest) -> Dict[str, Any]:
    """Preview symbols matching filter criteria"""
    config_service = _get_config_service()

    filters = {
        "tags": body.tags,
        "exclude_tags": body.exclude_tags,
        "taxonomy": body.taxonomy,
        "manual_include": body.manual_include,
        "manual_exclude": body.manual_exclude,
    }
    return config_service.preview_filter(filters)


# =============================================================================
# Symbol API Routes (CRUD)
# =============================================================================


class SymbolCreateRequest(BaseModel):
    """Request to create a new symbol"""
    ticker: str = Field(..., description="Ticker symbol")
    taxonomy: Dict[str, str] = Field(..., description="Taxonomy (market, asset_class, etc.)")
    tags: List[str] = Field(default_factory=list, description="Tags")
    name: Optional[str] = Field(None, description="Display name")


class SymbolUpdateRequest(BaseModel):
    """Request to update a symbol"""
    taxonomy: Optional[Dict[str, str]] = Field(None, description="New taxonomy")
    tags: Optional[List[str]] = Field(None, description="New tags")
    name: Optional[str] = Field(None, description="Display name")


class BulkTagRequest(BaseModel):
    """Request for bulk tag operations"""
    tickers: List[str] = Field(..., description="List of tickers")
    add_tags: List[str] = Field(default_factory=list, description="Tags to add")
    remove_tags: List[str] = Field(default_factory=list, description="Tags to remove")


@router.get("/api/symbols")
async def api_list_symbols(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(50, ge=1, le=500, description="Page size"),
    search: Optional[str] = Query(None, description="Search ticker"),
    market: Optional[str] = Query(None, description="Filter by market"),
    asset_class: Optional[str] = Query(None, description="Filter by asset class"),
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    cache_filter: Optional[str] = Query(None, description="cached, not_cached, or all (price cache)"),
    signal_filter: Optional[str] = Query(None, description="has_signals, no_signals, or all"),
) -> Dict[str, Any]:
    """List symbols with pagination, filters, and cache status"""
    config_service = _get_config_service()

    tag_list = tags.split(",") if tags else None

    return config_service.list_symbols_with_cache(
        page=page,
        size=size,
        search=search,
        market=market,
        asset_class=asset_class,
        tags=tag_list,
        cache_filter=cache_filter,
        signal_filter=signal_filter,
    )


@router.get("/api/symbols/{ticker}")
async def api_get_symbol(ticker: str) -> Dict[str, Any]:
    """Get a specific symbol's details"""
    config_service = _get_config_service()
    symbol = config_service.get_symbol(ticker)
    if not symbol:
        raise HTTPException(status_code=404, detail=f"Symbol not found: {ticker}")
    return symbol


@router.post("/api/symbols")
async def api_add_symbol(body: SymbolCreateRequest) -> Dict[str, Any]:
    """Add a new symbol"""
    config_service = _get_config_service()
    try:
        config_service.add_symbol(
            ticker=body.ticker,
            taxonomy=body.taxonomy,
            tags=body.tags,
            name=body.name,
        )
        return {"success": True, "ticker": body.ticker}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/api/symbols/{ticker}")
async def api_update_symbol(ticker: str, body: SymbolUpdateRequest) -> Dict[str, Any]:
    """Update a symbol's taxonomy, tags, or name"""
    config_service = _get_config_service()
    try:
        updates = {}
        if body.taxonomy is not None:
            updates["taxonomy"] = body.taxonomy
        if body.tags is not None:
            updates["tags"] = body.tags
        if body.name is not None:
            updates["name"] = body.name

        config_service.update_symbol(ticker, updates)
        return {"success": True, "ticker": ticker}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/api/symbols/{ticker}")
async def api_delete_symbol(ticker: str) -> Dict[str, Any]:
    """Delete a symbol"""
    config_service = _get_config_service()
    try:
        config_service.delete_symbol(ticker)
        return {"success": True, "ticker": ticker}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/symbols/bulk-tag")
async def api_bulk_tag(body: BulkTagRequest) -> Dict[str, Any]:
    """Bulk add/remove tags from symbols"""
    config_service = _get_config_service()
    updated = config_service.bulk_update_tags(
        tickers=body.tickers,
        add_tags=body.add_tags,
        remove_tags=body.remove_tags,
    )
    return {"success": True, "updated": updated}


# =============================================================================
# Taxonomy API Routes (CRUD)
# =============================================================================


class TaxonomyKeyRequest(BaseModel):
    """Request to create/update taxonomy key"""
    values: Dict[str, str] = Field(..., description="Code -> Label mappings")


class TaxonomyValueRequest(BaseModel):
    """Request to add taxonomy value"""
    code: str = Field(..., description="Value code")
    label: str = Field(..., description="Value label")


@router.get("/api/taxonomy")
async def api_get_taxonomy() -> Dict[str, Dict[str, str]]:
    """Get all taxonomy definitions"""
    config_service = _get_config_service()
    return config_service.get_taxonomy()


@router.get("/api/taxonomy/{key}")
async def api_get_taxonomy_key(key: str) -> Dict[str, str]:
    """Get a specific taxonomy key"""
    config_service = _get_config_service()
    values = config_service.get_taxonomy_key(key)
    if values is None:
        raise HTTPException(status_code=404, detail=f"Taxonomy key not found: {key}")
    return values


@router.post("/api/taxonomy/{key}")
async def api_add_taxonomy_key(key: str, body: TaxonomyKeyRequest) -> Dict[str, Any]:
    """Add a new taxonomy key"""
    config_service = _get_config_service()
    try:
        config_service.add_taxonomy_key(key, body.values)
        return {"success": True, "key": key}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/api/taxonomy/{key}")
async def api_update_taxonomy_key(key: str, body: TaxonomyKeyRequest) -> Dict[str, Any]:
    """Update a taxonomy key's values"""
    config_service = _get_config_service()
    try:
        config_service.update_taxonomy_key(key, body.values)
        return {"success": True, "key": key}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/api/taxonomy/{key}")
async def api_delete_taxonomy_key(key: str) -> Dict[str, Any]:
    """Delete a taxonomy key"""
    config_service = _get_config_service()
    try:
        config_service.delete_taxonomy_key(key)
        return {"success": True, "key": key}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/taxonomy/{key}/values")
async def api_add_taxonomy_value(key: str, body: TaxonomyValueRequest) -> Dict[str, Any]:
    """Add a value to a taxonomy key"""
    config_service = _get_config_service()
    try:
        config_service.add_taxonomy_value(key, body.code, body.label)
        return {"success": True, "key": key, "code": body.code}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
# Global Save/Reload API Routes
# =============================================================================


@router.post("/api/save")
async def api_save() -> Dict[str, Any]:
    """Save all changes to asset_master.yaml"""
    config_service = _get_config_service()
    try:
        config_service.save_master()
        return {"success": True, "message": "Changes saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/reload")
async def api_reload() -> Dict[str, Any]:
    """Reload asset_master.yaml (discard unsaved changes)"""
    config_service = _get_config_service()
    config_service.reload_master()
    return {"success": True, "message": "Data reloaded"}


@router.get("/api/tags")
async def api_get_all_tags() -> List[str]:
    """Get all unique tags used across symbols"""
    config_service = _get_config_service()
    return config_service.get_all_tags()


@router.get("/api/status")
async def api_get_status() -> Dict[str, Any]:
    """Get current status including unsaved changes"""
    config_service = _get_config_service()
    data = config_service._load_master()

    return {
        "has_unsaved_changes": config_service.has_unsaved_changes(),
        "symbol_count": len(data.get("symbols", [])),
        "subset_count": len(data.get("subsets", {})),
        "taxonomy_keys": list(data.get("taxonomy", {}).keys()),
    }


@router.get("/api/cache-summary")
async def api_get_cache_summary() -> Dict[str, Any]:
    """Get cache summary statistics"""
    config_service = _get_config_service()
    return config_service.get_cache_summary()


# =============================================================================
# Universe Editor HTML Route
# =============================================================================


@router.get("/editor", response_class=HTMLResponse)
async def editor_view(request: Request):
    """Universe Editor page"""
    config_service = _get_config_service()

    data = config_service._load_master()
    subsets = data.get("subsets", {})
    taxonomy = data.get("taxonomy", {})
    all_tags = config_service.get_all_tags()

    # Get universes for counts
    universes = config_service.list_universes()
    universe_counts = {u.name: u.symbol_count for u in universes}

    return _templates.TemplateResponse(
        "universe_editor.html",
        {
            "request": request,
            "subsets": subsets,
            "taxonomy": taxonomy,
            "all_tags": all_tags,
            "universe_counts": universe_counts,
        },
    )
