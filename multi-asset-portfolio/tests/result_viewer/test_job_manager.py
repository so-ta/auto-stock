"""
Tests for JobManager - Background job management for backtest execution.

Tests cover:
1. OHLCVData attribute access (data, not df)
2. Job creation and lifecycle
3. Job status management
"""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
import polars as pl

from scripts.result_viewer.services.job_manager import (
    JobManager,
    BacktestJob,
    JobStatus,
    get_job_manager,
    _run_backtest_process,
)


class TestOHLCVDataAccess:
    """Test OHLCVData attribute access (data, not df)."""

    def test_ohlcv_data_attribute_exists(self):
        """OHLCVData should have 'data' attribute, not 'df'."""
        from src.data.adapters import OHLCVData, AssetType, DataFrequency

        # Create sample data
        df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1)],
            "open": [100.0],
            "high": [105.0],
            "low": [95.0],
            "close": [102.0],
            "volume": [1000.0],
        })

        ohlcv = OHLCVData(
            symbol="TEST",
            asset_type=AssetType.STOCK,
            frequency=DataFrequency.DAILY,
            data=df,
            source="test",
            fetched_at=datetime.now(),
        )

        # Should have 'data' attribute
        assert hasattr(ohlcv, "data")
        assert ohlcv.data is not None
        assert not ohlcv.data.is_empty()

        # Should NOT have 'df' attribute
        assert not hasattr(ohlcv, "df")

    def test_ohlcv_data_to_pandas_conversion(self):
        """OHLCVData.data should convert to pandas correctly."""
        from src.data.adapters import OHLCVData, AssetType, DataFrequency

        df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "open": [100.0, 101.0],
            "high": [105.0, 106.0],
            "low": [95.0, 96.0],
            "close": [102.0, 103.0],
            "volume": [1000.0, 1100.0],
        })

        ohlcv = OHLCVData(
            symbol="TEST",
            asset_type=AssetType.STOCK,
            frequency=DataFrequency.DAILY,
            data=df,
            source="test",
            fetched_at=datetime.now(),
        )

        # Convert to pandas
        pdf = ohlcv.data.to_pandas()

        assert len(pdf) == 2
        assert "timestamp" in pdf.columns
        assert "close" in pdf.columns
        assert pdf["close"].iloc[0] == 102.0


class TestJobCreation:
    """Test job creation functionality."""

    def test_create_job(self, tmp_path):
        """Test creating a new backtest job."""
        manager = JobManager(
            project_root=tmp_path,
            results_dir=tmp_path / "results",
        )

        job = manager.create_job(
            universe_file="test_universe",
            start_date="2024-01-01",
            end_date="2024-12-31",
            frequency="monthly",
        )

        assert job.job_id is not None
        assert job.run_id.startswith("bt_")
        assert job.status == JobStatus.PENDING
        assert job.universe_file == "test_universe"
        assert job.start_date == "2024-01-01"
        assert job.end_date == "2024-12-31"
        assert job.frequency == "monthly"

    def test_create_job_with_overrides(self, tmp_path):
        """Test creating a job with config overrides."""
        manager = JobManager(
            project_root=tmp_path,
            results_dir=tmp_path / "results",
        )

        overrides = {"backtest": {"initial_capital": 500000}}
        job = manager.create_job(
            universe_file="test_universe",
            start_date="2024-01-01",
            end_date="2024-12-31",
            config_overrides=overrides,
        )

        assert job.config_overrides == overrides

    def test_job_to_dict(self, tmp_path):
        """Test job serialization to dict."""
        manager = JobManager(
            project_root=tmp_path,
            results_dir=tmp_path / "results",
        )

        job = manager.create_job(
            universe_file="test_universe",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        job_dict = job.to_dict()

        assert job_dict["job_id"] == job.job_id
        assert job_dict["run_id"] == job.run_id
        assert job_dict["status"] == "pending"
        assert "created_at" in job_dict


class TestJobLifecycle:
    """Test job lifecycle management."""

    def test_job_listing(self, tmp_path):
        """Test listing jobs."""
        manager = JobManager(
            project_root=tmp_path,
            results_dir=tmp_path / "results",
        )

        # Create multiple jobs
        job1 = manager.create_job(
            universe_file="test1",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
        job2 = manager.create_job(
            universe_file="test2",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        jobs = manager.list_jobs()
        assert len(jobs) == 2

        # Filter by status
        pending_jobs = manager.list_jobs(status=JobStatus.PENDING)
        assert len(pending_jobs) == 2

    def test_get_job(self, tmp_path):
        """Test getting a specific job."""
        manager = JobManager(
            project_root=tmp_path,
            results_dir=tmp_path / "results",
        )

        job = manager.create_job(
            universe_file="test_universe",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        retrieved = manager.get_job(job.job_id)
        assert retrieved is not None
        assert retrieved.job_id == job.job_id

        # Non-existent job
        non_existent = manager.get_job("non_existent_id")
        assert non_existent is None

    def test_start_job_updates_status(self, tmp_path):
        """Test that starting a job updates its status."""
        manager = JobManager(
            project_root=tmp_path,
            results_dir=tmp_path / "results",
            max_concurrent_jobs=1,
        )

        job = manager.create_job(
            universe_file="test_universe",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        assert job.status == JobStatus.PENDING

        # Mock the multiprocessing to avoid actual execution
        with patch("multiprocessing.Process") as mock_process:
            mock_instance = MagicMock()
            mock_instance.pid = 12345
            mock_process.return_value = mock_instance

            started_job = manager.start_job(job.job_id)

            assert started_job.status == JobStatus.RUNNING
            assert started_job.started_at is not None

    def test_start_non_existent_job_raises_error(self, tmp_path):
        """Test starting a non-existent job raises ValueError."""
        manager = JobManager(
            project_root=tmp_path,
            results_dir=tmp_path / "results",
        )

        with pytest.raises(ValueError, match="Job not found"):
            manager.start_job("non_existent_id")

    def test_start_already_running_job_raises_error(self, tmp_path):
        """Test starting an already running job raises ValueError."""
        manager = JobManager(
            project_root=tmp_path,
            results_dir=tmp_path / "results",
            max_concurrent_jobs=2,
        )

        job = manager.create_job(
            universe_file="test_universe",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        with patch("multiprocessing.Process") as mock_process:
            mock_instance = MagicMock()
            mock_instance.pid = 12345
            mock_process.return_value = mock_instance

            manager.start_job(job.job_id)

            with pytest.raises(ValueError, match="not pending"):
                manager.start_job(job.job_id)

    def test_max_concurrent_jobs(self, tmp_path):
        """Test max concurrent jobs limit."""
        manager = JobManager(
            project_root=tmp_path,
            results_dir=tmp_path / "results",
            max_concurrent_jobs=1,
        )

        job1 = manager.create_job(
            universe_file="test1",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
        job2 = manager.create_job(
            universe_file="test2",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        with patch("multiprocessing.Process") as mock_process:
            mock_instance = MagicMock()
            mock_instance.pid = 12345
            mock_instance.is_alive.return_value = True
            mock_process.return_value = mock_instance

            manager.start_job(job1.job_id)

            with pytest.raises(ValueError, match="Max concurrent jobs"):
                manager.start_job(job2.job_id)


class TestJobManagerSingleton:
    """Test JobManager singleton behavior."""

    def test_get_job_manager_returns_same_instance(self, tmp_path):
        """Test that get_job_manager returns singleton."""
        # Reset singleton
        import scripts.result_viewer.services.job_manager as jm
        jm._job_manager = None

        manager1 = get_job_manager(
            project_root=tmp_path,
            results_dir=tmp_path / "results",
        )
        manager2 = get_job_manager()

        assert manager1 is manager2

        # Clean up
        jm._job_manager = None


class TestJobStatusUpdate:
    """Test job status update from process state."""

    def test_completed_job_detection(self, tmp_path):
        """Test detecting completed job from result file."""
        results_dir = tmp_path / "results"
        progress_dir = results_dir / ".progress"
        progress_dir.mkdir(parents=True)

        manager = JobManager(
            project_root=tmp_path,
            results_dir=results_dir,
        )

        job = manager.create_job(
            universe_file="test_universe",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        with patch("multiprocessing.Process") as mock_process:
            mock_instance = MagicMock()
            mock_instance.pid = 12345
            mock_instance.is_alive.return_value = False
            mock_process.return_value = mock_instance

            manager.start_job(job.job_id)

            # Create result file
            result_file = progress_dir / f"{job.run_id}.result"
            result_file.write_text("archive_123")

            # Get job to trigger status update
            updated_job = manager.get_job(job.job_id)

            assert updated_job.status == JobStatus.COMPLETED
            assert updated_job.archive_id == "archive_123"

    def test_failed_job_detection(self, tmp_path):
        """Test detecting failed job from error file."""
        results_dir = tmp_path / "results"
        progress_dir = results_dir / ".progress"
        progress_dir.mkdir(parents=True)

        manager = JobManager(
            project_root=tmp_path,
            results_dir=results_dir,
        )

        job = manager.create_job(
            universe_file="test_universe",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        with patch("multiprocessing.Process") as mock_process:
            mock_instance = MagicMock()
            mock_instance.pid = 12345
            mock_instance.is_alive.return_value = False
            mock_process.return_value = mock_instance

            manager.start_job(job.job_id)

            # Create error file
            error_file = progress_dir / f"{job.run_id}.error"
            error_file.write_text("ValueError: Something went wrong")

            # Get job to trigger status update
            updated_job = manager.get_job(job.job_id)

            assert updated_job.status == JobStatus.FAILED
            assert "Something went wrong" in updated_job.error_message


class TestJobCleanup:
    """Test job cleanup functionality."""

    def test_cleanup_old_jobs(self, tmp_path):
        """Test cleaning up old completed jobs."""
        from datetime import timedelta

        manager = JobManager(
            project_root=tmp_path,
            results_dir=tmp_path / "results",
        )

        job = manager.create_job(
            universe_file="test_universe",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        # Manually set job to completed with old timestamp
        job.status = JobStatus.COMPLETED
        job.created_at = datetime.now() - timedelta(hours=48)

        deleted_count = manager.cleanup_old_jobs(max_age_hours=24)

        assert deleted_count == 1
        assert manager.get_job(job.job_id) is None
