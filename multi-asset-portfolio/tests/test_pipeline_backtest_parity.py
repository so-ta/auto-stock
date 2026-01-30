"""
Pipeline/Backtest同期監査テスト - SYNC-007

pipeline.pyとfast_engine.pyの機能乖離を検出するテスト。
CI/CDで自動実行することで、機能乖離の早期発見を実現。

Usage:
    pytest tests/test_pipeline_backtest_parity.py -v
    pytest tests/test_pipeline_backtest_parity.py -v --strict-markers
"""

import sys
from pathlib import Path

import pytest

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from scripts.audit_pipeline_backtest_parity import (
    PipelineBacktestAuditor,
    AuditResult,
    FeatureInfo,
)


class TestPipelineBacktestParity:
    """パイプライン/バックテスト同期テスト"""

    @pytest.fixture
    def auditor(self):
        """監査インスタンスを作成"""
        return PipelineBacktestAuditor()

    def test_files_exist(self, auditor):
        """対象ファイルが存在することを確認"""
        assert auditor.pipeline_path.exists(), (
            f"Pipeline file not found: {auditor.pipeline_path}"
        )
        assert auditor.backtest_path.exists(), (
            f"Backtest file not found: {auditor.backtest_path}"
        )

    def test_required_features_in_backtest(self, auditor):
        """必須機能がバックテストに実装されていることを確認"""
        result = auditor.audit()

        # 必須機能をフィルタ
        required_features = [
            f.name for f in auditor.REQUIRED_FEATURES if f.required
        ]
        missing_required = [
            f for f in result.missing_in_backtest if f in required_features
        ]

        assert len(missing_required) == 0, (
            f"必須機能がバックテストで未実装: {missing_required}\n"
            f"詳細: {result.warnings}"
        )

    def test_required_features_in_pipeline(self, auditor):
        """必須機能がパイプラインに実装されていることを確認"""
        result = auditor.audit()

        # 必須機能をフィルタ
        required_features = [
            f.name for f in auditor.REQUIRED_FEATURES if f.required
        ]
        missing_required = [
            f for f in result.missing_in_pipeline if f in required_features
        ]

        assert len(missing_required) == 0, (
            f"必須機能がパイプラインで未実装: {missing_required}\n"
            f"詳細: {result.warnings}"
        )

    def test_audit_pass_or_warn(self, auditor):
        """監査結果がPASSまたはWARNであることを確認"""
        result = auditor.audit()

        assert result.status in ["PASS", "WARN"], (
            f"監査失敗: {result.status}\n"
            f"メッセージ: {result.message}\n"
            f"バックテスト未実装: {result.missing_in_backtest}\n"
            f"パイプライン未実装: {result.missing_in_pipeline}\n"
            f"警告: {result.warnings}"
        )

    def test_transaction_cost_optimizer_parity(self, auditor):
        """TransactionCostOptimizerの同期を確認"""
        pipeline_content = auditor.pipeline_path.read_text()
        backtest_content = auditor.backtest_path.read_text()

        # フラグの存在確認
        assert "TRANSACTION_COST_OPTIMIZER_AVAILABLE" in backtest_content, (
            "TransactionCostOptimizer flag missing in backtest"
        )

        # 設定フラグの確認
        assert "use_cost_optimizer" in backtest_content, (
            "use_cost_optimizer config missing in backtest"
        )

    def test_drawdown_protection_parity(self, auditor):
        """DrawdownProtectionの同期を確認"""
        backtest_content = auditor.backtest_path.read_text()

        assert "DRAWDOWN_PROTECTION_AVAILABLE" in backtest_content, (
            "DrawdownProtection flag missing in backtest"
        )
        assert "use_drawdown_protection" in backtest_content, (
            "use_drawdown_protection config missing in backtest"
        )

    def test_regime_detector_parity(self, auditor):
        """RegimeDetectorの同期を確認"""
        backtest_content = auditor.backtest_path.read_text()

        assert "REGIME_DETECTOR_AVAILABLE" in backtest_content, (
            "RegimeDetector flag missing in backtest"
        )
        assert "use_regime_detection" in backtest_content, (
            "use_regime_detection config missing in backtest"
        )

    def test_vix_signal_parity(self, auditor):
        """VIXSignalの同期を確認"""
        backtest_content = auditor.backtest_path.read_text()

        assert "VIX_SIGNAL_AVAILABLE" in backtest_content, (
            "VIXSignal flag missing in backtest"
        )
        assert "vix_cash_enabled" in backtest_content, (
            "vix_cash_enabled config missing in backtest"
        )

    def test_no_critical_divergence(self, auditor):
        """致命的な乖離がないことを確認"""
        result = auditor.audit()

        # SYNC-001 ~ SYNC-006 の機能が揃っていることを確認
        sync_features = [
            "TransactionCostOptimizer",  # SYNC-001
            "DrawdownProtection",        # SYNC-002
            "RegimeDetector",            # SYNC-003
            "VIXSignal",                 # SYNC-004
        ]

        for feature in sync_features:
            assert feature not in result.missing_in_backtest, (
                f"{feature} (SYNC機能) がバックテストで未実装"
            )

    def test_audit_result_structure(self, auditor):
        """監査結果の構造が正しいことを確認"""
        result = auditor.audit()

        assert hasattr(result, "status")
        assert hasattr(result, "pipeline_features")
        assert hasattr(result, "backtest_features")
        assert hasattr(result, "missing_in_backtest")
        assert hasattr(result, "missing_in_pipeline")
        assert hasattr(result, "warnings")
        assert hasattr(result, "message")

        # ステータスは有効な値
        assert result.status in ["PASS", "WARN", "FAIL"]

        # リストは正しい型
        assert isinstance(result.pipeline_features, list)
        assert isinstance(result.backtest_features, list)
        assert isinstance(result.missing_in_backtest, list)
        assert isinstance(result.missing_in_pipeline, list)
        assert isinstance(result.warnings, list)


class TestAuditResultSerialization:
    """監査結果のシリアライズテスト"""

    def test_to_dict(self):
        """to_dictメソッドのテスト"""
        result = AuditResult(
            status="PASS",
            pipeline_features=["A", "B"],
            backtest_features=["A", "B"],
            missing_in_backtest=[],
            missing_in_pipeline=[],
            warnings=[],
            message="All good",
        )

        d = result.to_dict()

        assert d["status"] == "PASS"
        assert d["pipeline_features"] == ["A", "B"]
        assert d["message"] == "All good"


class TestStrictMode:
    """厳格モードのテスト"""

    def test_strict_mode_fails_on_optional(self):
        """厳格モードではオプション機能の乖離もFAIL"""
        auditor = PipelineBacktestAuditor(strict=True)
        result = auditor.audit()

        # オプション機能に乖離があればFAIL
        if result.missing_in_backtest or result.missing_in_pipeline:
            assert result.status == "FAIL"

    def test_normal_mode_warns_on_optional(self):
        """通常モードではオプション機能の乖離はWARN"""
        auditor = PipelineBacktestAuditor(strict=False)
        result = auditor.audit()

        # 必須機能に乖離がなければPASSまたはWARN
        if result.status != "FAIL":
            assert result.status in ["PASS", "WARN"]


# CI/CD用のマーカー
pytestmark = pytest.mark.parity


def test_quick_parity_check():
    """
    クイック同期チェック - CI/CDで高速実行

    最低限の同期チェックを行い、致命的な乖離を検出。
    """
    auditor = PipelineBacktestAuditor()
    result = auditor.audit()

    # 必須機能のみチェック（厳格モードではない）
    required_features = [f.name for f in auditor.REQUIRED_FEATURES if f.required]

    critical_missing = [
        f for f in result.missing_in_backtest if f in required_features
    ]

    assert len(critical_missing) == 0, (
        f"致命的な機能乖離を検出: {critical_missing}"
    )
