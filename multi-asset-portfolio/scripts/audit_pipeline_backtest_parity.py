#!/usr/bin/env python3
"""
Pipeline/Backtest同期監査システム - SYNC-007

pipeline.pyとfast_engine.pyの機能乖離を自動検出する監査ツール。
CI/CD統合やpre-commitフックでの自動実行を想定。

Usage:
    python scripts/audit_pipeline_backtest_parity.py
    python scripts/audit_pipeline_backtest_parity.py --strict
    python scripts/audit_pipeline_backtest_parity.py --json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class FeatureInfo:
    """機能情報"""
    name: str
    available_flag: str
    config_flag: str
    description: str = ""
    required: bool = True


@dataclass
class AuditResult:
    """監査結果"""
    status: str  # 'PASS', 'WARN', 'FAIL'
    pipeline_features: List[str] = field(default_factory=list)
    backtest_features: List[str] = field(default_factory=list)
    missing_in_backtest: List[str] = field(default_factory=list)
    missing_in_pipeline: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "status": self.status,
            "pipeline_features": self.pipeline_features,
            "backtest_features": self.backtest_features,
            "missing_in_backtest": self.missing_in_backtest,
            "missing_in_pipeline": self.missing_in_pipeline,
            "warnings": self.warnings,
            "message": self.message,
        }


class PipelineBacktestAuditor:
    """パイプラインとバックテストの機能同期を監査"""

    # 同期が必要な機能リスト
    REQUIRED_FEATURES = [
        FeatureInfo(
            name="TransactionCostOptimizer",
            available_flag="TRANSACTION_COST_OPTIMIZER_AVAILABLE",
            config_flag="use_cost_optimizer",
            description="取引コスト最適化",
            required=True,
        ),
        FeatureInfo(
            name="DrawdownProtection",
            available_flag="DRAWDOWN_PROTECTION_AVAILABLE",
            config_flag="use_drawdown_protection",
            description="ドローダウン保護",
            required=True,
        ),
        FeatureInfo(
            name="RegimeDetector",
            available_flag="REGIME_DETECTOR_AVAILABLE",
            config_flag="use_regime_detection",
            description="レジーム検出",
            required=True,
        ),
        FeatureInfo(
            name="VIXSignal",
            available_flag="VIX_SIGNAL_AVAILABLE",
            config_flag="vix_cash_enabled",
            description="VIX動的キャッシュ配分",
            required=True,
        ),
        FeatureInfo(
            name="DynamicWeighter",
            available_flag="DYNAMIC_WEIGHTER_AVAILABLE",
            config_flag="use_dynamic_weighting",
            description="動的ウェイト調整",
            required=False,  # オプション機能
        ),
        FeatureInfo(
            name="DividendHandler",
            available_flag="DIVIDEND_HANDLER_AVAILABLE",
            config_flag="use_dividends",
            description="配当処理",
            required=False,  # オプション機能
        ),
        FeatureInfo(
            name="CorrelationBreakDetector",
            available_flag="CORRELATION_BREAK_AVAILABLE",
            config_flag="use_correlation_break",
            description="相関ブレイク検出",
            required=False,
        ),
        FeatureInfo(
            name="KellyAllocator",
            available_flag="KELLY_ALLOCATOR_AVAILABLE",
            config_flag="use_kelly",
            description="ケリー基準配分",
            required=False,
        ),
    ]

    # パイプライン専用機能（バックテストに統合不要）
    PIPELINE_ONLY_FEATURES = [
        "UNIVERSE_EXPANSION_AVAILABLE",
        "RETURN_MAXIMIZATION_AVAILABLE",
        "CMD016_AVAILABLE",
        "CMD017_AVAILABLE",
    ]

    # バックテスト専用機能（パイプラインに統合不要）
    BACKTEST_ONLY_FEATURES = [
        "GPU_AVAILABLE",
        "NUMBA_AVAILABLE",
    ]

    def __init__(
        self,
        pipeline_path: Optional[Path] = None,
        backtest_path: Optional[Path] = None,
        strict: bool = False,
    ):
        """
        初期化

        Args:
            pipeline_path: pipeline.pyのパス
            backtest_path: fast_engine.pyのパス
            strict: 厳格モード（オプション機能もエラー扱い）
        """
        self.project_root = Path(__file__).parent.parent
        self.pipeline_path = pipeline_path or (
            self.project_root / "src" / "orchestrator" / "pipeline.py"
        )
        self.backtest_path = backtest_path or (
            self.project_root / "src" / "backtest" / "fast_engine.py"
        )
        self.strict = strict

    def _scan_file_for_features(self, file_path: Path) -> Set[str]:
        """
        ファイルから AVAILABLE フラグをスキャン

        Args:
            file_path: スキャン対象ファイル

        Returns:
            検出された機能フラグのセット
        """
        if not file_path.exists():
            return set()

        content = file_path.read_text(encoding="utf-8")
        features = set()

        # パターン: XXX_AVAILABLE = True/False
        pattern = r"([A-Z_]+_AVAILABLE)\s*=\s*(True|False)"
        matches = re.findall(pattern, content)

        for flag, _ in matches:
            features.add(flag)

        return features

    def _check_feature_usage(
        self,
        file_path: Path,
        feature: FeatureInfo,
    ) -> bool:
        """
        機能が使用されているか確認（インポートや使用パターン）

        Args:
            file_path: チェック対象ファイル
            feature: 機能情報

        Returns:
            機能が使用されているか
        """
        if not file_path.exists():
            return False

        content = file_path.read_text(encoding="utf-8")

        # 機能名のパターンマッチング（クラス名やメソッド名）
        feature_patterns = {
            "TransactionCostOptimizer": [
                r"TransactionCostOptimizer",
                r"transaction_cost",
                r"tc_optimizer",
            ],
            "DrawdownProtection": [
                r"DrawdownProtect",
                r"dd_protect",
                r"drawdown_protection",
            ],
            "RegimeDetector": [
                r"RegimeDetector",
                r"regime_detect",
                r"detect.*regime",
            ],
            "VIXSignal": [
                r"VIXSignal",
                r"VIX.*cash",
                r"vix_value",
                r"_vix",
            ],
            "DynamicWeighter": [
                r"DynamicWeight",
                r"dynamic_weight",
            ],
            "DividendHandler": [
                r"DividendHandler",
                r"dividend",
            ],
            "CorrelationBreakDetector": [
                r"CorrelationBreak",
                r"correlation_break",
            ],
            "KellyAllocator": [
                r"Kelly",
                r"kelly",
            ],
        }

        patterns = feature_patterns.get(feature.name, [feature.name])

        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True

        return False

    def _scan_file_for_config_flags(self, file_path: Path) -> Set[str]:
        """
        ファイルから設定フラグ（config.use_xxx）をスキャン

        Args:
            file_path: スキャン対象ファイル

        Returns:
            検出された設定フラグのセット
        """
        if not file_path.exists():
            return set()

        content = file_path.read_text(encoding="utf-8")
        flags = set()

        # パターン: config.use_xxx or self.config.use_xxx
        pattern = r"(?:self\.)?config\.([a-z_]+)"
        matches = re.findall(pattern, content)

        for flag in matches:
            if flag.startswith("use_") or flag.endswith("_enabled"):
                flags.add(flag)

        return flags

    def _check_feature_implementation(
        self,
        file_path: Path,
        feature: FeatureInfo,
    ) -> Tuple[bool, str]:
        """
        特定の機能が実装されているか確認

        Args:
            file_path: チェック対象ファイル
            feature: 機能情報

        Returns:
            (実装済みか, 詳細メッセージ)
        """
        if not file_path.exists():
            return False, f"File not found: {file_path}"

        content = file_path.read_text(encoding="utf-8")

        # AVAILABLE フラグをチェック
        has_flag = feature.available_flag in content

        # config フラグをチェック
        has_config = feature.config_flag in content

        # 使用パターンをチェック
        has_usage = self._check_feature_usage(file_path, feature)

        if has_flag and has_config:
            return True, f"{feature.name}: OK (flag+config)"
        elif has_flag:
            return True, f"{feature.name}: OK (flag)"
        elif has_usage:
            return True, f"{feature.name}: OK (usage pattern)"
        elif has_config:
            return False, f"{feature.name}: config only (no implementation)"
        else:
            return False, f"{feature.name}: not found"

    def audit(self) -> AuditResult:
        """
        監査を実行

        Returns:
            AuditResult: 監査結果
        """
        result = AuditResult(status="PASS")

        # ファイル存在チェック
        if not self.pipeline_path.exists():
            result.status = "FAIL"
            result.message = f"Pipeline file not found: {self.pipeline_path}"
            return result

        if not self.backtest_path.exists():
            result.status = "FAIL"
            result.message = f"Backtest file not found: {self.backtest_path}"
            return result

        # 機能をスキャン
        pipeline_features = self._scan_file_for_features(self.pipeline_path)
        backtest_features = self._scan_file_for_features(self.backtest_path)

        # パイプライン専用・バックテスト専用を除外
        pipeline_features -= set(self.PIPELINE_ONLY_FEATURES)
        backtest_features -= set(self.BACKTEST_ONLY_FEATURES)

        result.pipeline_features = sorted(pipeline_features)
        result.backtest_features = sorted(backtest_features)

        # 各必須機能をチェック
        missing_in_backtest = []
        missing_in_pipeline = []

        for feature in self.REQUIRED_FEATURES:
            # パイプラインをチェック
            pipeline_ok, pipeline_msg = self._check_feature_implementation(
                self.pipeline_path, feature
            )
            # バックテストをチェック
            backtest_ok, backtest_msg = self._check_feature_implementation(
                self.backtest_path, feature
            )

            if pipeline_ok and not backtest_ok:
                missing_in_backtest.append(feature.name)
                if feature.required or self.strict:
                    result.warnings.append(
                        f"{feature.name} ({feature.description}): "
                        f"pipeline=OK, backtest=MISSING"
                    )

            if backtest_ok and not pipeline_ok:
                missing_in_pipeline.append(feature.name)
                if feature.required or self.strict:
                    result.warnings.append(
                        f"{feature.name} ({feature.description}): "
                        f"pipeline=MISSING, backtest=OK"
                    )

        result.missing_in_backtest = missing_in_backtest
        result.missing_in_pipeline = missing_in_pipeline

        # ステータス判定
        required_missing_backtest = [
            f for f in missing_in_backtest
            if any(rf.name == f and rf.required for rf in self.REQUIRED_FEATURES)
        ]
        required_missing_pipeline = [
            f for f in missing_in_pipeline
            if any(rf.name == f and rf.required for rf in self.REQUIRED_FEATURES)
        ]

        if required_missing_backtest or required_missing_pipeline:
            result.status = "FAIL"
            result.message = (
                f"機能乖離検出: バックテスト未実装={len(required_missing_backtest)}, "
                f"パイプライン未実装={len(required_missing_pipeline)}"
            )
        elif missing_in_backtest or missing_in_pipeline:
            result.status = "WARN" if not self.strict else "FAIL"
            result.message = (
                f"オプション機能の乖離: バックテスト={len(missing_in_backtest)}, "
                f"パイプライン={len(missing_in_pipeline)}"
            )
        else:
            result.status = "PASS"
            result.message = "全機能が同期済み"

        return result

    def print_report(self, result: AuditResult) -> None:
        """監査レポートを出力"""
        print("=" * 60)
        print("Pipeline/Backtest 同期監査レポート")
        print("=" * 60)
        print()

        # ステータス
        status_color = {
            "PASS": "\033[92m",  # Green
            "WARN": "\033[93m",  # Yellow
            "FAIL": "\033[91m",  # Red
        }
        reset = "\033[0m"
        color = status_color.get(result.status, "")

        print(f"Status: {color}{result.status}{reset}")
        print(f"Message: {result.message}")
        print()

        # 検出機能
        print("Pipeline Features:")
        for f in result.pipeline_features:
            print(f"  - {f}")
        print()

        print("Backtest Features:")
        for f in result.backtest_features:
            print(f"  - {f}")
        print()

        # 乖離
        if result.missing_in_backtest:
            print("Missing in Backtest (要実装):")
            for f in result.missing_in_backtest:
                print(f"  - {f}")
            print()

        if result.missing_in_pipeline:
            print("Missing in Pipeline:")
            for f in result.missing_in_pipeline:
                print(f"  - {f}")
            print()

        # 警告
        if result.warnings:
            print("Warnings:")
            for w in result.warnings:
                print(f"  - {w}")
            print()

        print("=" * 60)


def main():
    """メインエントリポイント"""
    parser = argparse.ArgumentParser(
        description="Pipeline/Backtest同期監査システム"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="厳格モード（オプション機能もエラー扱い）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="JSON形式で出力",
    )
    parser.add_argument(
        "--pipeline",
        type=Path,
        help="pipeline.pyのパス",
    )
    parser.add_argument(
        "--backtest",
        type=Path,
        help="fast_engine.pyのパス",
    )

    args = parser.parse_args()

    auditor = PipelineBacktestAuditor(
        pipeline_path=args.pipeline,
        backtest_path=args.backtest,
        strict=args.strict,
    )

    result = auditor.audit()

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        auditor.print_report(result)

    # 終了コード
    if result.status == "FAIL":
        sys.exit(1)
    elif result.status == "WARN":
        sys.exit(0)  # 警告は成功扱い
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
