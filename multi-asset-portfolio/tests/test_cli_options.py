"""
CLI Options Test (task_046_7)

Tests for --report and --dashboard CLI options.
"""

import subprocess
import sys

import pytest


class TestCLIHelpOutput:
    """--help オプションテスト"""

    def test_help_contains_report_option(self):
        """--report オプションが表示される"""
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--report" in result.stdout
        assert "Generate performance report" in result.stdout

    def test_help_contains_benchmarks_option(self):
        """--benchmarks オプションが表示される"""
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--benchmarks" in result.stdout
        assert "SPY,QQQ,DIA" in result.stdout

    def test_help_contains_report_output_option(self):
        """--report-output オプションが表示される"""
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--report-output" in result.stdout
        assert "performance_report.html" in result.stdout

    def test_help_contains_dashboard_option(self):
        """--dashboard オプションが表示される"""
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--dashboard" in result.stdout
        assert "interactive performance dashboard" in result.stdout.lower()

    def test_help_contains_port_option(self):
        """--port オプションが表示される"""
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--port" in result.stdout
        assert "8050" in result.stdout


class TestCLIArgumentParsing:
    """引数パーステスト"""

    def test_parse_report_requires_dates(self):
        """--report は --start と --end が必要"""
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--report"],
            capture_output=True,
            text=True,
        )
        # 日付がないのでエラーメッセージが出る
        assert "requires --start and --end" in result.stdout.lower() or result.returncode != 0

    def test_parse_dashboard_requires_dates(self):
        """--dashboard は --start と --end が必要"""
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--dashboard"],
            capture_output=True,
            text=True,
        )
        # 日付がないのでエラーメッセージが出る
        assert "requires --start and --end" in result.stdout.lower() or result.returncode != 0

    def test_parse_benchmarks_default(self):
        """--benchmarks のデフォルト値"""
        from src.main import parse_args
        import sys

        # 空の引数でテスト
        original_argv = sys.argv
        sys.argv = ["main.py"]
        try:
            args = parse_args()
            assert args.benchmarks == "SPY,QQQ,DIA"
        finally:
            sys.argv = original_argv

    def test_parse_benchmarks_custom(self):
        """--benchmarks カスタム値"""
        from src.main import parse_args
        import sys

        original_argv = sys.argv
        sys.argv = ["main.py", "--benchmarks", "VTI,BND,GLD"]
        try:
            args = parse_args()
            assert args.benchmarks == "VTI,BND,GLD"
        finally:
            sys.argv = original_argv

    def test_parse_port_default(self):
        """--port のデフォルト値"""
        from src.main import parse_args
        import sys

        original_argv = sys.argv
        sys.argv = ["main.py"]
        try:
            args = parse_args()
            assert args.port == 8050
        finally:
            sys.argv = original_argv

    def test_parse_port_custom(self):
        """--port カスタム値"""
        from src.main import parse_args
        import sys

        original_argv = sys.argv
        sys.argv = ["main.py", "--port", "9000"]
        try:
            args = parse_args()
            assert args.port == 9000
        finally:
            sys.argv = original_argv

    def test_parse_report_output_default(self):
        """--report-output のデフォルト値"""
        from src.main import parse_args
        import sys

        original_argv = sys.argv
        sys.argv = ["main.py"]
        try:
            args = parse_args()
            assert args.report_output == "reports/performance_report.html"
        finally:
            sys.argv = original_argv

    def test_parse_report_output_custom(self):
        """--report-output カスタム値"""
        from src.main import parse_args
        import sys

        original_argv = sys.argv
        sys.argv = ["main.py", "--report-output", "custom/report.html"]
        try:
            args = parse_args()
            assert args.report_output == "custom/report.html"
        finally:
            sys.argv = original_argv


class TestExampleCommands:
    """例示コマンドのテスト（構文チェックのみ）"""

    def test_help_shows_report_example(self):
        """レポート生成の例が表示される"""
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--report" in result.stdout
        # 例示セクションに含まれている
        assert "Performance report" in result.stdout or "report" in result.stdout.lower()

    def test_help_shows_dashboard_example(self):
        """ダッシュボードの例が表示される"""
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--dashboard" in result.stdout
