"""
Backtest Initializer - CLI/Viewer共通の初期化処理

バックテスト実行前の初期化処理を統一し、
CLI/Viewerどちらから実行しても同じ設定が適用されるようにする。

統一される初期化項目:
1. プロジェクトルートの設定 (sys.path)
2. .env ファイルの読み込み (AWS認証情報等)
3. Settings の読み込み (config/default.yaml)
4. ロギングの初期化 (structlog + PipelineLogCollector連携)
5. ProgressTracker の初期化 (進捗追跡)

Usage:
    from src.orchestrator.backtest_initializer import initialize_backtest_process

    init_result = initialize_backtest_process(
        project_root=Path("/path/to/project"),
        run_id="bt_20260201_143022",
        results_dir=Path("/path/to/results"),
    )

    settings = init_result["settings"]
    progress_tracker = init_result["progress_tracker"]
    log_collector = init_result["log_collector"]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from src.config.settings import Settings
    from src.utils.pipeline_log_collector import PipelineLogCollector
    from src.utils.progress_tracker import ProgressTracker


def initialize_backtest_process(
    project_root: Optional[Path] = None,
    run_id: Optional[str] = None,
    results_dir: Optional[Path] = None,
    universe_size: int = 0,
    frequency: str = "monthly",
    enable_progress_tracking: bool = True,
    enable_log_collector: bool = True,
) -> Dict[str, Any]:
    """
    バックテスト実行プロセスの統一初期化

    CLI/Viewer両方から呼び出され、同一の設定を適用する。
    これにより、どちらから実行しても同じログ出力・キャッシュ動作が保証される。

    Args:
        project_root: プロジェクトルートパス（省略時は自動検出）
        run_id: 実行ID（ProgressTracker用、省略時は自動生成）
        results_dir: 結果保存ディレクトリ（省略時は project_root/results）
        universe_size: ユニバースサイズ（進捗表示用）
        frequency: リバランス頻度
        enable_progress_tracking: ProgressTrackerを有効化
        enable_log_collector: PipelineLogCollectorを有効化

    Returns:
        {
            "settings": Settings,
            "progress_tracker": ProgressTracker or None,
            "log_collector": PipelineLogCollector or None,
        }
    """
    # 1. プロジェクトルートの設定
    if project_root is None:
        # このファイルは src/orchestrator/ にあるので、2階層上がプロジェクトルート
        project_root = Path(__file__).parent.parent.parent

    # sys.path に追加（モジュール解決用）
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    # カレントディレクトリを設定（相対パス解決用）
    os.chdir(project_root)

    # 2. .env ファイルの読み込み（AWS認証情報等）
    from dotenv import load_dotenv
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # 3. Settings の読み込み（config/default.yaml）
    from src.config.settings import load_settings_from_yaml
    settings = load_settings_from_yaml()

    # 4. PipelineLogCollector の初期化（ロギング設定前に行う）
    log_collector: Optional["PipelineLogCollector"] = None
    if enable_log_collector:
        from src.utils.pipeline_log_collector import PipelineLogCollector
        from src.utils.logger import set_log_collector

        # run_id が未設定の場合は仮のIDを使用
        collector_run_id = run_id or f"bt_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_collector = PipelineLogCollector(run_id=collector_run_id)
        set_log_collector(log_collector)

    # 5. ロギングの初期化（structlog + PipelineLogCollector連携）
    from src.utils.logger import setup_logging
    setup_logging(settings=settings, enable_log_collector=enable_log_collector)

    # 6. ProgressTracker の初期化
    progress_tracker: Optional["ProgressTracker"] = None
    if enable_progress_tracking:
        from src.utils.progress_tracker import ProgressTracker

        if results_dir is None:
            results_dir = project_root / "results"

        progress_dir = Path(results_dir) / ".progress"
        progress_tracker = ProgressTracker(
            run_id=run_id,
            progress_dir=progress_dir,
            universe_size=universe_size,
            frequency=frequency,
        )

        # ProgressTracker と LogCollector を連携
        if log_collector is not None:
            log_collector.attach_progress_tracker(progress_tracker)

    return {
        "settings": settings,
        "progress_tracker": progress_tracker,
        "log_collector": log_collector,
    }
