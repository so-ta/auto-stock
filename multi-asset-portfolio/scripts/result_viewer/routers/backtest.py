"""
Backtest Router - Backtest execution API endpoints

バックテストの実行・管理用APIエンドポイント。
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

router = APIRouter(prefix="/backtest", tags=["backtest"])

# These will be set by the main app
_templates: Optional[Jinja2Templates] = None
_results_dir: Optional[Path] = None
_project_root: Optional[Path] = None


def configure(
    templates: Jinja2Templates,
    results_dir: Path,
    project_root: Path,
) -> None:
    """Configure the router"""
    global _templates, _results_dir, _project_root
    _templates = templates
    _results_dir = results_dir
    _project_root = project_root


def _get_job_manager():
    """Get JobManager instance"""
    from scripts.result_viewer.services.job_manager import get_job_manager
    return get_job_manager(
        project_root=_project_root,
        results_dir=_results_dir,
    )


def _get_config_service():
    """Get ConfigService instance"""
    from scripts.result_viewer.services.config_service import get_config_service
    return get_config_service(project_root=_project_root)


def _get_view_service():
    """Get BacktestViewService instance"""
    from scripts.result_viewer.services.backtest_view_service import get_backtest_view_service
    return get_backtest_view_service(
        results_dir=_results_dir,
        progress_dir=_results_dir / ".progress" if _results_dir else None,
    )


# =============================================================================
# Request/Response Models
# =============================================================================


class BacktestRequest(BaseModel):
    """Backtest execution request"""
    universe_file: str = Field(..., description="Universe file name (e.g., universe_sbi.yaml)")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    frequency: str = Field("monthly", description="Rebalance frequency")
    config_overrides: Optional[Dict[str, Any]] = Field(None, description="Config overrides")


class BacktestResponse(BaseModel):
    """Backtest job response"""
    job_id: str
    run_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    """Job status response"""
    job_id: str
    run_id: str
    status: str
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    archive_id: Optional[str]
    error_message: Optional[str]
    progress: Optional[Dict[str, Any]]


# =============================================================================
# HTML Routes
# =============================================================================


@router.get("/run", response_class=HTMLResponse)
async def run_view(request: Request):
    """バックテスト実行ページ（1画面完結）"""
    config_service = _get_config_service()

    # Get available options
    universes = config_service.list_universes()
    frequency_presets = config_service.list_frequency_presets()
    period_presets = config_service.list_period_presets()
    defaults = config_service.get_backtest_defaults()

    return _templates.TemplateResponse(
        "run.html",
        {
            "request": request,
            "universes": [u.to_dict() for u in universes],
            "frequency_presets": [f.to_dict() for f in frequency_presets],
            "period_presets": [p.to_dict() for p in period_presets],
            "defaults": defaults,
        },
    )


@router.get("/launcher", response_class=HTMLResponse)
async def launcher_view(request: Request):
    """Backtest wizard launcher page (redirect to /run)"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/backtest/run", status_code=302)


@router.get("/monitor", response_class=HTMLResponse)
async def monitor_view(request: Request):
    """Live backtest monitoring page (redirect to dashboard)"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/", status_code=302)


@router.get("/progress/{run_id}", response_class=HTMLResponse)
async def progress_view(request: Request, run_id: str):
    """実行中のバックテスト詳細ページ"""
    from src.utils.progress_tracker import ProgressTracker
    from fastapi.responses import RedirectResponse

    progress_dir = _results_dir / ".progress"
    progress_file = progress_dir / f"{run_id}.json"

    progress_data = ProgressTracker.load_progress(progress_file)

    if progress_data is None:
        # Progress not found - redirect to dashboard
        return RedirectResponse(url="/", status_code=302)

    # If completed or failed, redirect to dashboard
    if progress_data.status in ("completed", "failed"):
        return RedirectResponse(url="/", status_code=302)

    return _templates.TemplateResponse(
        "progress.html",
        {
            "request": request,
            "progress": progress_data.to_dict(),
        },
    )


@router.get("/view/{run_id}", response_class=HTMLResponse)
async def legacy_view(request: Request, run_id: str):
    """レガシービュー - 新しい統一ビューにリダイレクト"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"/backtest/{run_id}", status_code=302)


@router.get("/{id}", response_class=HTMLResponse)
async def unified_backtest_view(request: Request, id: str):
    """
    統一バックテストビュー

    /backtest/{id} で実行中・完了済み・アーカイブ済みすべてに対応。
    URLを変えずにリアルタイム表示から結果表示に自動切り替え。

    ID判定フロー:
    1. .progress/{id}.json が存在 → 進捗データを読み込み
       - status = running/initializing → 実行中モード
       - status = completed/failed → 完了モード（archive検索）
    2. results/{id}/metadata.json が存在 → アーカイブモード
    3. どちらも無し → 404
    """
    from scripts.result_viewer.services.backtest_view_service import BacktestState

    view_service = _get_view_service()
    view_data = view_service.get_view_data(id)

    if view_data is None or view_data.state == BacktestState.NOT_FOUND:
        raise HTTPException(status_code=404, detail=f"Backtest not found: {id}")

    # 状態判定
    state = view_data.state
    is_running = state in (BacktestState.INITIALIZING, BacktestState.RUNNING)
    is_completed = state == BacktestState.COMPLETED
    is_failed = state == BacktestState.FAILED
    is_archived = state == BacktestState.ARCHIVED

    # アーカイブデータを読み込み（完了/アーカイブの場合）
    archive = None
    timeseries_json = "null"
    rebalances = None
    reliability = None

    archive_id = view_data.archive_id
    if is_archived:
        archive_id = id

    if archive_id:
        try:
            from src.analysis.result_store import BacktestResultStore
            store = BacktestResultStore(str(_results_dir))
            archive = store.load(archive_id)

            # 信頼性情報を取得
            if archive and hasattr(archive, "reliability") and archive.reliability:
                reliability = archive.reliability

            # 時系列データ
            try:
                ts_df = store.load_timeseries(archive_id)
                timeseries_data = {
                    "dates": ts_df.index.strftime("%Y-%m-%d").tolist(),
                    "portfolio_value": ts_df["portfolio_value"].tolist(),
                    "cumulative_return": (ts_df["cumulative_return"] * 100).tolist(),
                    "drawdown": (ts_df["drawdown"] * 100).tolist(),
                }
                timeseries_json = json.dumps(timeseries_data)
            except (FileNotFoundError, KeyError):
                pass

            # リバランスデータ
            try:
                reb_df = store.load_rebalances(archive_id)
                rebalances = reb_df.to_dict(orient="records")
                for r in rebalances:
                    if "timestamp" in r:
                        r["date"] = str(r["timestamp"])[:10]
            except (FileNotFoundError, KeyError):
                pass

        except FileNotFoundError:
            pass

    # 初期資本（デフォルト値）
    initial_capital = 1_000_000
    if view_data.trading_stats:
        initial_capital = view_data.trading_stats.get("initial_value", 1_000_000)

    return _templates.TemplateResponse(
        "unified_view.html",
        {
            "request": request,
            "id": id,
            "run_id": id,  # 後方互換性
            "state": state.value,
            "status": state.value,
            "is_running": is_running,
            "is_completed": is_completed,
            "is_failed": is_failed,
            "is_archived": is_archived,
            "progress": view_data.progress,
            "archive": archive,
            "archive_id": archive_id,
            "timeseries": timeseries_json,
            "rebalances": rebalances,
            "initial_capital": initial_capital,
            "execution_info": view_data.execution_info,
            "reliability": reliability,
        },
    )


# =============================================================================
# API Routes
# =============================================================================


@router.post("/api/start", response_model=BacktestResponse)
async def api_start_backtest(request: BacktestRequest) -> BacktestResponse:
    """Start a new backtest"""
    job_manager = _get_job_manager()
    config_service = _get_config_service()

    # Validate universe file
    validation = config_service.validate_universe_file(request.universe_file)
    if not validation["valid"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid universe file: {', '.join(validation['errors'])}"
        )

    config_overrides = request.config_overrides or {}

    try:
        # Create and start job
        job = job_manager.create_job(
            universe_file=request.universe_file,
            start_date=request.start_date,
            end_date=request.end_date,
            frequency=request.frequency,
            config_overrides=config_overrides,
        )

        job = job_manager.start_job(job.job_id)

        return BacktestResponse(
            job_id=job.job_id,
            run_id=job.run_id,
            status=job.status.value,
            message="Backtest started successfully",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start backtest: {e}")


@router.get("/api/jobs", response_model=List[Dict[str, Any]])
async def api_list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
) -> List[Dict[str, Any]]:
    """List all backtest jobs"""
    from scripts.result_viewer.services.job_manager import JobStatus

    job_manager = _get_job_manager()

    job_status = None
    if status:
        try:
            job_status = JobStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    jobs = job_manager.list_jobs(status=job_status, limit=limit)
    return [j.to_dict() for j in jobs]


@router.get("/api/job/{job_id}", response_model=JobStatusResponse)
async def api_get_job_status(job_id: str) -> JobStatusResponse:
    """Get job status with progress info"""
    from src.utils.progress_tracker import ProgressTracker

    job_manager = _get_job_manager()
    job = job_manager.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    # Get progress info
    progress = None
    if job.status.value == "running":
        progress_dir = _results_dir / ".progress"
        progress_file = progress_dir / f"{job.run_id}.json"
        progress_data = ProgressTracker.load_progress(progress_file)
        if progress_data:
            progress = progress_data.to_dict()

    return JobStatusResponse(
        job_id=job.job_id,
        run_id=job.run_id,
        status=job.status.value,
        created_at=job.created_at.isoformat(),
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        archive_id=job.archive_id,
        error_message=job.error_message,
        progress=progress,
    )


@router.post("/api/job/{job_id}/stop")
async def api_stop_job(job_id: str) -> Dict[str, Any]:
    """Stop a running backtest"""
    job_manager = _get_job_manager()

    try:
        job = job_manager.stop_job(job_id)
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "message": "Job stopped successfully",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/estimate")
async def api_estimate_duration(
    universe: str = Query(..., description="Universe file name"),
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


@router.get("/api/cache-coverage")
async def api_cache_coverage(
    universe: str = Query(..., description="Universe name"),
) -> Dict[str, Any]:
    """
    ユニバースのキャッシュカバレッジを取得

    Returns:
        total_symbols: ユニバース内の銘柄数
        price_cached: 価格キャッシュ済み銘柄数
        price_coverage: 価格キャッシュカバレッジ (%)
        signals_cached: シグナルキャッシュ済み銘柄数
        signal_coverage: シグナルキャッシュカバレッジ (%)
    """
    config_service = _get_config_service()

    # Get universe symbols
    symbols = config_service.get_universe_symbols(universe)
    if not symbols:
        raise HTTPException(status_code=404, detail=f"Universe not found: {universe}")

    total_symbols = len(symbols)

    # Get cache status for symbols
    cache_status = config_service.get_cache_status(symbols)

    # Count cached symbols
    price_cached = sum(1 for s in cache_status.values() if s.get("has_price"))
    signals_cached = sum(1 for s in cache_status.values() if s.get("signal_count", 0) > 0)

    # Get total signal types for coverage calculation
    cache_summary = config_service.get_cache_summary()
    total_signal_types = cache_summary.get("total_signal_types", 1)

    # Calculate signal coverage (average across all symbols)
    total_signal_coverage = 0
    for status in cache_status.values():
        if status.get("total_signals", 0) > 0:
            total_signal_coverage += status.get("signal_count", 0) / status["total_signals"]
    signal_coverage = (total_signal_coverage / total_symbols * 100) if total_symbols > 0 else 0

    return {
        "total_symbols": total_symbols,
        "price_cached": price_cached,
        "price_coverage": round(price_cached / total_symbols * 100, 1) if total_symbols > 0 else 0,
        "signals_cached": signals_cached,
        "signal_coverage": round(signal_coverage, 1),
        "total_signal_types": total_signal_types,
    }


# =============================================================================
# Log Retrieval API Routes
# =============================================================================


@router.get("/api/logs/{archive_id}")
async def api_get_logs(
    archive_id: str,
    level: Optional[str] = Query(None, description="Filter by log level (ERROR, WARNING, INFO, DEBUG)"),
    limit: int = Query(500, ge=1, le=5000, description="Maximum number of log entries to return"),
) -> Dict[str, Any]:
    """
    アーカイブのログを取得

    Args:
        archive_id: アーカイブID
        level: ログレベルでフィルタ（ERROR, WARNING, INFO, DEBUG）
        limit: 取得するログエントリの最大数

    Returns:
        logs: ログエントリのリスト
        total: 総ログ数（フィルタ前）
        filtered: フィルタ後のログ数
    """
    logs_path = _results_dir / archive_id / "logs.jsonl"

    if not logs_path.exists():
        return {"logs": [], "total": 0, "filtered": 0}

    all_logs = []
    filtered_logs = []

    try:
        with open(logs_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    all_logs.append(entry)

                    # Apply level filter
                    if level is None or entry.get("level") == level.upper():
                        filtered_logs.append(entry)
                except json.JSONDecodeError:
                    continue
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {e}")

    # Apply limit (return last N entries)
    result_logs = filtered_logs[-limit:] if len(filtered_logs) > limit else filtered_logs

    return {
        "logs": result_logs,
        "total": len(all_logs),
        "filtered": len(filtered_logs),
    }


# =============================================================================
# Progress Monitoring API Routes
# =============================================================================


@router.get("/api/progress")
async def api_list_progress(
    status: Optional[str] = Query(None, description="Filter by status"),
) -> List[Dict[str, Any]]:
    """Get all progress entries"""
    from src.utils.progress_tracker import ProgressTracker

    progress_dir = _results_dir / ".progress"
    all_progress = ProgressTracker.list_active_progress(progress_dir)

    if status:
        all_progress = [p for p in all_progress if p.status == status]

    return [p.to_dict() for p in all_progress]


@router.get("/api/progress/{run_id}")
async def api_get_progress(run_id: str) -> Dict[str, Any]:
    """Get specific progress entry"""
    from src.utils.progress_tracker import ProgressTracker

    progress_dir = _results_dir / ".progress"
    progress_file = progress_dir / f"{run_id}.json"

    progress = ProgressTracker.load_progress(progress_file)
    if progress is None:
        raise HTTPException(status_code=404, detail=f"Progress not found: {run_id}")

    return progress.to_dict()


@router.get("/api/{id}/state")
async def api_get_state(id: str) -> Dict[str, Any]:
    """
    Get current state of a backtest.

    Returns state info including archive_id if completed.
    """
    from scripts.result_viewer.services.backtest_view_service import BacktestState

    view_service = _get_view_service()
    view_data = view_service.get_view_data(id)

    if view_data is None:
        raise HTTPException(status_code=404, detail=f"Backtest not found: {id}")

    return {
        "id": id,
        "state": view_data.state.value,
        "archive_id": view_data.archive_id,
        "has_timeseries": view_data.has_timeseries,
    }


@router.delete("/api/progress/{run_id}")
async def api_delete_progress(run_id: str) -> Dict[str, str]:
    """Delete a progress file"""
    progress_dir = _results_dir / ".progress"
    progress_file = progress_dir / f"{run_id}.json"

    if not progress_file.exists():
        raise HTTPException(status_code=404, detail=f"Progress not found: {run_id}")

    progress_file.unlink()
    return {"status": "deleted", "run_id": run_id}


@router.post("/api/progress/cleanup")
async def api_cleanup_progress(
    max_age_hours: int = Query(24, ge=1, le=168, description="Max age in hours"),
) -> Dict[str, Any]:
    """Cleanup old progress files"""
    from src.utils.progress_tracker import ProgressTracker

    progress_dir = _results_dir / ".progress"
    deleted = ProgressTracker.cleanup_old_progress(progress_dir, max_age_hours)
    return {"deleted_count": deleted}


# =============================================================================
# SSE (Server-Sent Events) for Real-time Progress
# =============================================================================


@router.get("/api/progress/{run_id}/stream")
async def api_progress_stream(run_id: str):
    """
    SSE endpoint for real-time progress updates.

    Usage (JavaScript):
        const eventSource = new EventSource('/backtest/api/progress/<run_id>/stream');
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log(data);
        };
    """
    from src.utils.progress_tracker import ProgressTracker

    async def event_generator():
        progress_dir = _results_dir / ".progress"
        progress_file = progress_dir / f"{run_id}.json"

        last_update = None
        init_wait_count = 0
        max_init_wait = 30  # 最大30秒待機

        while True:
            progress = ProgressTracker.load_progress(progress_file)

            if progress is None:
                # 初期化中: ファイルがまだ存在しない場合は待機して再試行
                init_wait_count += 1
                if init_wait_count >= max_init_wait:
                    yield f"data: {json.dumps({'error': 'Progress not found after timeout'})}\n\n"
                    break
                # 初期化中であることをクライアントに通知
                yield f"data: {json.dumps({'status': 'initializing', 'wait_count': init_wait_count})}\n\n"
                await asyncio.sleep(1)
                continue

            # Only send if updated
            if progress.last_update != last_update:
                last_update = progress.last_update
                yield f"data: {json.dumps(progress.to_dict())}\n\n"

            # Stop streaming if completed or failed
            if progress.status in ("completed", "failed"):
                yield f"data: {json.dumps({'status': 'done', 'final': progress.to_dict()})}\n\n"
                break

            await asyncio.sleep(1)  # Poll every second

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# Timeseries API for Real-time Equity Curve
# =============================================================================


@router.get("/api/progress/{run_id}/timeseries")
async def api_get_timeseries(run_id: str) -> Dict[str, Any]:
    """
    実行中バックテストの時系列データを取得

    Returns:
        timeseries: エクイティカーブスナップショットのリスト
        count: データポイント数
    """
    from src.utils.progress_tracker import ProgressTracker

    progress_dir = _results_dir / ".progress"
    timeseries = ProgressTracker.load_timeseries(progress_dir, run_id)

    return {
        "run_id": run_id,
        "timeseries": timeseries,
        "count": len(timeseries),
    }


@router.get("/api/progress/{run_id}/timeseries/stream")
async def api_timeseries_stream(run_id: str):
    """
    時系列データのSSEストリーミング

    新規データポイントをリアルタイム配信。
    フロントエンドのリアルタイムエクイティカーブ更新に使用。

    Usage (JavaScript):
        const eventSource = new EventSource('/backtest/api/progress/<run_id>/timeseries/stream');
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            // data.timeseries に新しいデータポイント配列
            updateChart(data.timeseries);
        };
    """
    from src.utils.progress_tracker import ProgressTracker

    async def event_generator():
        progress_dir = _results_dir / ".progress"
        progress_file = progress_dir / f"{run_id}.json"
        timeseries_file = progress_dir / f"{run_id}_timeseries.jsonl"

        last_count = 0
        init_wait_count = 0
        max_init_wait = 30  # 最大30秒待機

        while True:
            # 進捗ファイルを確認
            progress = ProgressTracker.load_progress(progress_file)

            if progress is None:
                # 初期化中: ファイルがまだ存在しない場合は待機して再試行
                init_wait_count += 1
                if init_wait_count >= max_init_wait:
                    yield f"data: {json.dumps({'error': 'Progress not found after timeout'})}\n\n"
                    break
                await asyncio.sleep(1)
                continue

            # 時系列データを取得
            timeseries = ProgressTracker.load_timeseries(progress_dir, run_id)
            current_count = len(timeseries)

            # 新しいデータがあれば送信
            if current_count > last_count:
                new_points = timeseries[last_count:]
                yield f"data: {json.dumps({'timeseries': new_points, 'total_count': current_count})}\n\n"
                last_count = current_count

            # 完了または失敗時は最終データを送信して終了
            if progress.status in ("completed", "failed"):
                yield f"data: {json.dumps({'status': 'done', 'total_count': current_count, 'final_status': progress.status})}\n\n"
                break

            await asyncio.sleep(2)  # 2秒間隔でポーリング

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
