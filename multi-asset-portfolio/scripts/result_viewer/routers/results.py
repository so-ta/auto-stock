"""
Results Router - Archive management API endpoints

バックテスト結果のアーカイブ管理用APIエンドポイント。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

router = APIRouter(tags=["results"])


def _get_attr(obj, key: str, default=None):
    """Get attribute from dict or object safely."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_metric(metrics, key: str, default: float = 0.0) -> float:
    """Get metric value from dict or object safely."""
    value = _get_attr(metrics, key, default)
    return value if value is not None else default

# These will be set by the main app
_templates: Optional[Jinja2Templates] = None
_results_dir: Optional[Path] = None


def configure(templates: Jinja2Templates, results_dir: Path) -> None:
    """Configure the router with templates and results directory"""
    global _templates, _results_dir
    _templates = templates
    _results_dir = results_dir


def _get_store():
    """Get BacktestResultStore instance"""
    from src.analysis.result_store import BacktestResultStore
    return BacktestResultStore(str(_results_dir or Path("results")))


# =============================================================================
# HTML Routes
# =============================================================================


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Dashboard - ダッシュボード"""
    from src.utils.progress_tracker import ProgressTracker

    store = _get_store()
    archives = store.list_archives(limit=10)  # 最近の10件
    stats = store.get_stats()

    # 実行中のパイプラインを取得
    progress_dir = _results_dir / ".progress"
    all_progress = ProgressTracker.list_active_progress(progress_dir)
    active_pipelines = [p for p in all_progress if p.status == "running"]
    recent_completed = [p for p in all_progress if p.status in ("completed", "failed")][:5]

    # 最新の保有銘柄を取得
    latest_holdings = []
    latest_archive_id = None
    if archives:
        try:
            latest_archive = store.load(archives[0]["archive_id"])
            latest_archive_id = archives[0]["archive_id"]
            final_weights = _get_attr(latest_archive, "final_weights")
            if final_weights:
                latest_holdings = [
                    {"ticker": k, "weight": v}
                    for k, v in sorted(
                        final_weights.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:10]
                ]
        except Exception:
            pass

    return _templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "archives": archives,
            "stats": stats,
            "active_pipelines": [p.to_dict() for p in active_pipelines],
            "recent_completed": [p.to_dict() for p in recent_completed],
            "latest_holdings": latest_holdings,
            "latest_archive_id": latest_archive_id,
        },
    )


@router.get("/archive/{archive_id}", response_class=HTMLResponse)
async def archive_detail(request: Request, archive_id: str):
    """Archive detail view - redirects to unified backtest view"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"/backtest/{archive_id}", status_code=302)


@router.get("/history", response_class=HTMLResponse)
async def history_view(request: Request):
    """バックテスト履歴ページ"""
    store = _get_store()
    archives = store.list_archives(limit=100)
    stats = store.get_stats()

    # 全タグを収集
    all_tags = set()
    for archive in archives:
        if archive.get("tags"):
            all_tags.update(archive["tags"])

    return _templates.TemplateResponse(
        "history.html",
        {
            "request": request,
            "archives": archives,
            "all_tags": sorted(all_tags),
        },
    )


@router.get("/compare", response_class=HTMLResponse)
async def compare_view(
    request: Request,
    ids: Optional[str] = Query(None, description="Comma-separated archive IDs"),
):
    """Comparison view (redirect to history)"""
    # 旧/compareルートは/historyにリダイレクト
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/history", status_code=302)


# =============================================================================
# API Routes
# =============================================================================


@router.get("/api/archives")
async def api_list_archives(
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    name: Optional[str] = Query(None, description="Name contains"),
    limit: int = Query(100, ge=1, le=1000),
) -> List[Dict[str, Any]]:
    """Get archive list"""
    store = _get_store()

    tag_list = None
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    return store.list_archives(tags=tag_list, name_contains=name, limit=limit)


@router.get("/api/archive/{archive_id}")
async def api_get_archive(archive_id: str) -> Dict[str, Any]:
    """Get archive details"""
    store = _get_store()

    try:
        archive = store.load(archive_id)
        return archive.to_metadata_dict()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Archive not found: {archive_id}")


@router.get("/api/archive/{archive_id}/timeseries")
async def api_get_timeseries(archive_id: str) -> Dict[str, Any]:
    """Get timeseries data"""
    store = _get_store()

    try:
        ts_df = store.load_timeseries(archive_id)
        return {
            "dates": ts_df.index.strftime("%Y-%m-%d").tolist(),
            "portfolio_value": ts_df["portfolio_value"].tolist(),
            "daily_return": ts_df["daily_return"].tolist(),
            "cumulative_return": ts_df["cumulative_return"].tolist(),
            "drawdown": ts_df["drawdown"].tolist(),
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Archive not found: {archive_id}")


@router.get("/api/archive/{archive_id}/rebalances")
async def api_get_rebalances(archive_id: str) -> List[Dict[str, Any]]:
    """Get rebalance data"""
    store = _get_store()

    try:
        reb_df = store.load_rebalances(archive_id)
        records = reb_df.to_dict(orient="records")
        # Convert datetime to string
        for r in records:
            if "timestamp" in r:
                r["timestamp"] = str(r["timestamp"])
        return records
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Archive not found: {archive_id}")


@router.get("/api/stats")
async def api_get_stats() -> Dict[str, Any]:
    """Get storage statistics"""
    store = _get_store()
    return store.get_stats()


@router.get("/api/latest-holdings")
async def api_latest_holdings() -> Dict[str, Any]:
    """最新バックテストの保有銘柄を取得"""
    store = _get_store()
    archives = store.list_archives(limit=1)

    if not archives:
        return {
            "holdings": [],
            "archive_id": None,
            "as_of": None,
        }

    try:
        archive = store.load(archives[0]["archive_id"])
        holdings = []

        final_weights = _get_attr(archive, "final_weights")
        if final_weights:
            holdings = [
                {"ticker": k, "weight": round(v, 4)}
                for k, v in sorted(
                    final_weights.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            ]

        return {
            "holdings": holdings,
            "archive_id": archives[0]["archive_id"],
            "as_of": _get_attr(archive, "end_date"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load holdings: {e}")


@router.get("/api/compare")
async def api_compare(
    ids: str = Query(..., description="Comma-separated archive IDs"),
) -> Dict[str, Any]:
    """Compare multiple archives"""
    from src.analysis.backtest_comparator import BacktestComparator

    store = _get_store()
    comparator = BacktestComparator(store)

    archive_ids = [id.strip() for id in ids.split(",") if id.strip()]

    if len(archive_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 archive IDs required")

    try:
        result = comparator.compare(archive_ids)
        return {
            "archive_ids": result.archive_ids,
            "metric_comparison": result.metric_comparison.to_dict(),
            "config_diffs": result.config_diffs,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/api/archive/{archive_id}")
async def api_delete_archive(archive_id: str) -> Dict[str, str]:
    """Delete an archive"""
    store = _get_store()

    success = store.delete(archive_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Archive not found: {archive_id}")

    return {"status": "deleted", "archive_id": archive_id}


# =============================================================================
# Benchmark Comparison APIs
# =============================================================================


@router.get("/api/benchmarks")
async def api_list_benchmarks() -> Dict[str, Any]:
    """利用可能なベンチマーク一覧を取得"""
    from src.analysis.benchmark_fetcher import BenchmarkFetcher

    return {
        "available": BenchmarkFetcher.get_available_benchmarks(),
        "default": ["SPY", "QQQ"],
    }


@router.get("/api/archive/{archive_id}/benchmark-comparison")
async def api_benchmark_comparison(
    archive_id: str,
    benchmarks: str = Query("SPY,QQQ", description="Comma-separated benchmark tickers"),
    custom_tickers: Optional[str] = Query(None, description="Comma-separated custom tickers"),
) -> Dict[str, Any]:
    """
    ポートフォリオとベンチマークを比較

    Returns:
        portfolio: ポートフォリオの累積リターン時系列
        benchmarks: 各ベンチマークの累積リターン時系列
        stats: 比較統計情報
    """
    from src.analysis.benchmark_fetcher import BenchmarkFetcher, BenchmarkFetcherError
    from src.utils.storage_backend import get_storage_backend, StorageConfig
    from src.config.settings import Settings

    store = _get_store()

    # アーカイブ読み込み
    try:
        archive = store.load(archive_id)
        ts_df = store.load_timeseries(archive_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Archive not found: {archive_id}")

    # ベンチマークリスト作成
    benchmark_list = [b.strip() for b in benchmarks.split(",") if b.strip()]
    if custom_tickers:
        custom_list = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
        benchmark_list.extend(custom_list)

    if not benchmark_list:
        benchmark_list = ["SPY", "QQQ"]

    # 日付範囲取得
    archive_start = _get_attr(archive, "start_date")
    archive_end = _get_attr(archive, "end_date")
    start_date = archive_start.strftime("%Y-%m-%d") if archive_start else None
    end_date = archive_end.strftime("%Y-%m-%d") if archive_end else None

    if not start_date or not end_date:
        raise HTTPException(status_code=400, detail="Archive missing date range")

    # ベンチマークデータ取得
    try:
        settings = Settings()

        # S3バケットを環境変数または設定から取得
        s3_bucket = os.environ.get("S3_BUCKET") or settings.storage.s3_bucket

        # S3が設定されていない場合はエラーメッセージを返す
        if not s3_bucket:
            logger.warning("S3 bucket not configured, benchmark comparison unavailable")
            # ポートフォリオ統計のみを返す
            m = archive.metrics
            portfolio_stats = {
                "name": "Portfolio",
                "annual_return": round(_get_metric(m, "annual_return") * 100, 2),
                "annual_volatility": round(_get_metric(m, "volatility") * 100, 2),
                "sharpe_ratio": round(_get_metric(m, "sharpe_ratio"), 3),
                "max_drawdown": round(_get_metric(m, "max_drawdown") * 100, 2),
                "total_return": round(_get_metric(m, "total_return") * 100, 2),
            }
            return {
                "portfolio": {
                    "dates": ts_df.index.strftime("%Y-%m-%d").tolist(),
                    "cumulative_return": (ts_df["cumulative_return"] * 100).tolist(),
                },
                "benchmarks": {},
                "stats": {"Portfolio": portfolio_stats},
                "error": "S3が設定されていないため、ベンチマーク比較は利用できません。.envファイルにS3_BUCKETを設定してください。",
            }

        # StorageConfigを作成（環境変数のs3_bucketを優先）
        storage_config = StorageConfig(
            s3_bucket=s3_bucket,
            s3_prefix=os.environ.get("S3_PREFIX", settings.storage.s3_prefix),
            s3_region=os.environ.get("S3_REGION", settings.storage.s3_region),
            base_path=settings.storage.base_path,
            local_cache_ttl_hours=settings.storage.local_cache_ttl_hours,
        )
        backend = get_storage_backend(storage_config)
        fetcher = BenchmarkFetcher(storage_backend=backend)

        prices = fetcher.fetch_benchmarks(start_date, end_date, benchmark_list)
        returns = fetcher.calculate_returns(prices, frequency="daily")
        cumulative = fetcher.get_cumulative_returns(returns)
        stats = fetcher.get_benchmark_stats(returns)
    except BenchmarkFetcherError as e:
        logger.warning(f"Failed to fetch benchmark data: {e}")
        return {
            "portfolio": {
                "dates": ts_df.index.strftime("%Y-%m-%d").tolist(),
                "cumulative_return": (ts_df["cumulative_return"] * 100).tolist(),
            },
            "benchmarks": {},
            "stats": {},
            "error": str(e),
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Unexpected error fetching benchmarks: {e}\n{tb}")
        return {
            "portfolio": {
                "dates": ts_df.index.strftime("%Y-%m-%d").tolist(),
                "cumulative_return": (ts_df["cumulative_return"] * 100).tolist(),
            },
            "benchmarks": {},
            "stats": {},
            "error": str(e),
        }

    # ポートフォリオデータ準備
    portfolio_data = {
        "dates": ts_df.index.strftime("%Y-%m-%d").tolist(),
        "cumulative_return": (ts_df["cumulative_return"] * 100).tolist(),
    }

    # ベンチマークデータ準備
    benchmark_data = {}
    for ticker in cumulative.columns:
        # ポートフォリオの日付に合わせてリサンプリング
        aligned = cumulative[ticker].reindex(ts_df.index, method="ffill")
        benchmark_data[ticker] = {
            "cumulative_return": ((aligned - 1) * 100).fillna(0).tolist(),
            "name": BenchmarkFetcher.BENCHMARKS.get(ticker, ticker),
        }

    # 統計情報準備（ポートフォリオ統計も追加）
    m = archive.metrics
    portfolio_stats = {
        "name": "Portfolio",
        "annual_return": round(_get_metric(m, "annual_return") * 100, 2),
        "annual_volatility": round(_get_metric(m, "volatility") * 100, 2),
        "sharpe_ratio": round(_get_metric(m, "sharpe_ratio"), 3),
        "max_drawdown": round(_get_metric(m, "max_drawdown") * 100, 2),
        "total_return": round(_get_metric(m, "total_return") * 100, 2),
    }

    stats_dict = {"Portfolio": portfolio_stats}
    # statsがDataFrameの場合とdictの場合に対応
    if stats is not None and len(stats) > 0:
        if hasattr(stats, "index"):
            # DataFrame
            for ticker in stats.index:
                stats_dict[ticker] = {
                    "name": stats.loc[ticker, "name"],
                    "annual_return": round(stats.loc[ticker, "annual_return"] * 100, 2),
                    "annual_volatility": round(stats.loc[ticker, "annual_volatility"] * 100, 2),
                    "sharpe_ratio": round(stats.loc[ticker, "sharpe_ratio"], 3),
                    "max_drawdown": round(stats.loc[ticker, "max_drawdown"] * 100, 2),
                    "total_return": round(stats.loc[ticker, "total_return"] * 100, 2),
                }
        elif isinstance(stats, dict):
            # dict
            for ticker, s in stats.items():
                stats_dict[ticker] = {
                    "name": s.get("name", ticker),
                    "annual_return": round(s.get("annual_return", 0) * 100, 2),
                    "annual_volatility": round(s.get("annual_volatility", 0) * 100, 2),
                    "sharpe_ratio": round(s.get("sharpe_ratio", 0), 3),
                    "max_drawdown": round(s.get("max_drawdown", 0) * 100, 2),
                    "total_return": round(s.get("total_return", 0) * 100, 2),
                }

    return {
        "portfolio": portfolio_data,
        "benchmarks": benchmark_data,
        "stats": stats_dict,
    }


# =============================================================================
# Portfolio Composition APIs
# =============================================================================


@router.get("/api/archive/{archive_id}/portfolio-composition")
async def api_portfolio_composition(
    archive_id: str,
    date: Optional[str] = Query(None, description="Date for composition (YYYY-MM-DD), defaults to latest"),
    top_n: int = Query(10, ge=1, le=50, description="Number of top holdings to return"),
) -> Dict[str, Any]:
    """
    ポートフォリオ構成を取得

    Returns:
        date: 対象日付
        weights: 全銘柄のウェイト
        top_holdings: 上位N銘柄
        total_positions: ポジション数
    """
    store = _get_store()

    try:
        archive = store.load(archive_id)
        reb_df = store.load_rebalances(archive_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Archive not found: {archive_id}")

    if reb_df.empty:
        return {
            "date": None,
            "weights": {},
            "top_holdings": [],
            "total_positions": 0,
        }

    # 対象日付のウェイトを取得
    if date:
        target_date = pd.to_datetime(date)
        # 指定日以前の最新リバランスを取得
        reb_df["timestamp"] = pd.to_datetime(reb_df["timestamp"])
        valid_rows = reb_df[reb_df["timestamp"] <= target_date]
        if valid_rows.empty:
            # 指定日以前のデータがない場合は最初のリバランスを使用
            row = reb_df.iloc[0]
        else:
            row = valid_rows.iloc[-1]
    else:
        # 最新のリバランス
        row = reb_df.iloc[-1]

    weights = row["weights_after"]
    selected_date = str(row["timestamp"])[:10] if "timestamp" in row else None

    # ゼロウェイトを除外してソート
    non_zero_weights = {k: v for k, v in weights.items() if v > 0.0001}
    sorted_weights = sorted(non_zero_weights.items(), key=lambda x: x[1], reverse=True)

    # 上位N銘柄
    top_holdings = [
        {"ticker": ticker, "weight": round(weight * 100, 2)}
        for ticker, weight in sorted_weights[:top_n]
    ]

    # その他の合計
    if len(sorted_weights) > top_n:
        others_weight = sum(w for _, w in sorted_weights[top_n:])
        top_holdings.append({"ticker": "Others", "weight": round(others_weight * 100, 2)})

    return {
        "date": selected_date,
        "weights": {k: round(v * 100, 4) for k, v in sorted_weights},
        "top_holdings": top_holdings,
        "total_positions": len(non_zero_weights),
    }


@router.get("/api/archive/{archive_id}/weight-history")
async def api_weight_history(
    archive_id: str,
    top_n: int = Query(10, ge=1, le=20, description="Number of top holdings to track"),
) -> Dict[str, Any]:
    """
    上位銘柄のウェイト推移を取得

    Returns:
        dates: 日付リスト
        tickers: トラッキング銘柄リスト
        weights: {ticker: [weight1, weight2, ...]} の形式
    """
    store = _get_store()

    try:
        reb_df = store.load_rebalances(archive_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Archive not found: {archive_id}")

    if reb_df.empty:
        return {
            "dates": [],
            "tickers": [],
            "weights": {},
        }

    # 全期間で最も頻繁に上位に現れる銘柄を特定
    ticker_importance = {}
    for _, row in reb_df.iterrows():
        weights = row["weights_after"]
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        for rank, (ticker, weight) in enumerate(sorted_weights[:top_n * 2]):
            if ticker not in ticker_importance:
                ticker_importance[ticker] = 0
            ticker_importance[ticker] += weight * (top_n * 2 - rank)

    # 上位N銘柄を選択
    top_tickers = sorted(
        ticker_importance.keys(),
        key=lambda t: ticker_importance[t],
        reverse=True
    )[:top_n]

    # 日付とウェイトを収集
    dates = []
    weights_by_ticker = {t: [] for t in top_tickers}

    for _, row in reb_df.iterrows():
        date_str = str(row["timestamp"])[:10] if "timestamp" in row else ""
        dates.append(date_str)

        current_weights = row["weights_after"]
        for ticker in top_tickers:
            weights_by_ticker[ticker].append(
                round(current_weights.get(ticker, 0) * 100, 2)
            )

    return {
        "dates": dates,
        "tickers": top_tickers,
        "weights": weights_by_ticker,
    }


@router.get("/api/archive/{archive_id}/rebalance-detail/{index}")
async def api_rebalance_detail(
    archive_id: str,
    index: int,
) -> Dict[str, Any]:
    """
    特定リバランスの詳細を取得

    Returns:
        date: リバランス日
        weights_before: リバランス前ウェイト
        weights_after: リバランス後ウェイト
        changes: 変更詳細（追加/削除/増減）
        portfolio_value: ポートフォリオ価値
        turnover: 回転率
        transaction_cost: 取引コスト
    """
    store = _get_store()

    try:
        reb_df = store.load_rebalances(archive_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Archive not found: {archive_id}")

    if index < 0 or index >= len(reb_df):
        raise HTTPException(status_code=404, detail=f"Rebalance index out of range: {index}")

    row = reb_df.iloc[index]
    weights_before = row.get("weights_before", {})
    weights_after = row.get("weights_after", {})

    # 変更を計算
    all_tickers = set(weights_before.keys()) | set(weights_after.keys())
    changes = {
        "added": [],    # 新規追加（before=0, after>0）
        "removed": [],  # 削除（before>0, after=0）
        "increased": [],  # 増加
        "decreased": [],  # 減少
    }

    for ticker in all_tickers:
        before = weights_before.get(ticker, 0)
        after = weights_after.get(ticker, 0)
        diff = after - before

        # 微小な変更は無視
        if abs(diff) < 0.0001:
            continue

        change_item = {
            "ticker": ticker,
            "before": round(before * 100, 2),
            "after": round(after * 100, 2),
            "diff": round(diff * 100, 2),
        }

        if before < 0.0001 and after > 0.0001:
            changes["added"].append(change_item)
        elif before > 0.0001 and after < 0.0001:
            changes["removed"].append(change_item)
        elif diff > 0:
            changes["increased"].append(change_item)
        else:
            changes["decreased"].append(change_item)

    # 各カテゴリを変化量でソート
    for key in changes:
        changes[key].sort(key=lambda x: abs(x["diff"]), reverse=True)

    return {
        "index": index,
        "date": str(row.get("timestamp", ""))[:10],
        "weights_before": {k: round(v * 100, 2) for k, v in weights_before.items() if v > 0.0001},
        "weights_after": {k: round(v * 100, 2) for k, v in weights_after.items() if v > 0.0001},
        "changes": changes,
        "portfolio_value": row.get("portfolio_value", 0),
        "turnover": round(row.get("turnover", 0) * 100, 2),
        "transaction_cost": row.get("transaction_cost", 0),
    }
