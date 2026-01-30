#!/usr/bin/env python3
"""
Backtest Validation Script

バックテスト結果を検証し、品質基準を満たしているか確認する。
CI/CDパイプラインでの使用を想定。

Usage:
    python scripts/validate_backtest.py results/backtest_daily_standard.json
    python scripts/validate_backtest.py results/backtest_*.json --compare-baseline results/baseline.json

Exit codes:
    0: All validations passed
    1: Validation errors found
    2: Warnings only (configurable)
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ValidationConfig:
    """検証設定"""

    # n_rebalances の許容誤差（±%）
    rebalance_tolerance: float = 0.05

    # Sharpe Ratio の最小値
    min_sharpe: float = -1.0

    # 前回比較時のSharpe低下許容（%）
    sharpe_decline_tolerance: float = 0.10

    # Max Drawdown の警告閾値
    mdd_warning_threshold: float = -0.40

    # Max Drawdown のエラー閾値
    mdd_error_threshold: float = -0.60


@dataclass
class ValidationResult:
    """検証結果"""

    passed: bool
    errors: list[str]
    warnings: list[str]
    metrics: dict


def get_expected_rebalances(frequency: str, n_days: int) -> int:
    """期待されるリバランス回数を計算"""
    years = n_days / 252
    if frequency == "daily":
        return n_days
    elif frequency == "weekly":
        return int(years * 52)
    elif frequency == "monthly":
        return int(years * 12)
    else:
        return n_days


def validate_backtest(
    result_path: str,
    config: ValidationConfig,
    baseline_path: Optional[str] = None,
) -> ValidationResult:
    """
    バックテスト結果を検証

    Args:
        result_path: 結果JSONファイルのパス
        config: 検証設定
        baseline_path: ベースライン比較用ファイル（オプション）

    Returns:
        ValidationResult
    """
    errors = []
    warnings = []

    # 結果読み込み
    try:
        with open(result_path, "r") as f:
            result = json.load(f)
    except Exception as e:
        return ValidationResult(
            passed=False,
            errors=[f"Failed to load result file: {e}"],
            warnings=[],
            metrics={},
        )

    metrics = result.get("metrics", {})
    trading_stats = result.get("trading_stats", {})
    frequency = result.get("frequency", "unknown")

    # 1. 基本メトリクス存在チェック
    required_metrics = ["sharpe_ratio", "annual_return", "max_drawdown"]
    for metric in required_metrics:
        if metric not in metrics:
            errors.append(f"Required metric missing: {metric}")

    if errors:
        return ValidationResult(
            passed=False, errors=errors, warnings=warnings, metrics=metrics
        )

    # 2. n_rebalances チェック
    n_rebalances = trading_stats.get("n_rebalances", 0)
    n_days = trading_stats.get("n_days", 0)

    if n_rebalances == 0:
        errors.append("n_rebalances is 0")
    elif n_days > 0:
        expected = get_expected_rebalances(frequency, n_days)
        tolerance = expected * config.rebalance_tolerance
        if abs(n_rebalances - expected) > tolerance:
            warnings.append(
                f"n_rebalances ({n_rebalances}) differs from expected ({expected}) "
                f"by more than {config.rebalance_tolerance*100:.0f}%"
            )

    # 3. Sharpe Ratio チェック
    sharpe = metrics.get("sharpe_ratio", 0)
    if sharpe < config.min_sharpe:
        errors.append(f"Sharpe Ratio ({sharpe:.3f}) below minimum ({config.min_sharpe})")
    elif sharpe < 0:
        warnings.append(f"Negative Sharpe Ratio: {sharpe:.3f}")

    # 4. Max Drawdown チェック
    mdd = metrics.get("max_drawdown", 0)
    if mdd < config.mdd_error_threshold:
        errors.append(f"Max Drawdown ({mdd*100:.1f}%) exceeds error threshold")
    elif mdd < config.mdd_warning_threshold:
        warnings.append(f"Max Drawdown ({mdd*100:.1f}%) exceeds warning threshold")

    # 5. ベースライン比較（オプション）
    if baseline_path:
        try:
            with open(baseline_path, "r") as f:
                baseline = json.load(f)
            baseline_metrics = baseline.get("metrics", {})
            baseline_sharpe = baseline_metrics.get("sharpe_ratio", 0)

            if baseline_sharpe > 0:
                decline = (baseline_sharpe - sharpe) / baseline_sharpe
                if decline > config.sharpe_decline_tolerance:
                    errors.append(
                        f"Sharpe declined by {decline*100:.1f}% from baseline "
                        f"(threshold: {config.sharpe_decline_tolerance*100:.0f}%)"
                    )
        except Exception as e:
            warnings.append(f"Baseline comparison failed: {e}")

    passed = len(errors) == 0
    return ValidationResult(
        passed=passed, errors=errors, warnings=warnings, metrics=metrics
    )


def main():
    parser = argparse.ArgumentParser(description="Validate backtest results")
    parser.add_argument("result_file", help="Path to backtest result JSON")
    parser.add_argument(
        "--baseline", "-b", help="Baseline result file for comparison"
    )
    parser.add_argument(
        "--fail-on-warning", action="store_true", help="Treat warnings as errors"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )

    args = parser.parse_args()

    config = ValidationConfig()
    result = validate_backtest(args.result_file, config, args.baseline)

    if args.json:
        output = {
            "passed": result.passed,
            "errors": result.errors,
            "warnings": result.warnings,
            "metrics": result.metrics,
        }
        print(json.dumps(output, indent=2))
    else:
        print("=" * 60)
        print("BACKTEST VALIDATION RESULTS")
        print("=" * 60)
        print(f"File: {args.result_file}")
        print(f"Status: {'PASSED' if result.passed else 'FAILED'}")
        print()

        if result.metrics:
            print("Metrics:")
            print(f"  Sharpe Ratio:    {result.metrics.get('sharpe_ratio', 0):.3f}")
            print(f"  Annual Return:   {result.metrics.get('annual_return', 0)*100:.2f}%")
            print(f"  Max Drawdown:    {result.metrics.get('max_drawdown', 0)*100:.2f}%")
            print()

        if result.errors:
            print("ERRORS:")
            for e in result.errors:
                print(f"  - {e}")
            print()

        if result.warnings:
            print("WARNINGS:")
            for w in result.warnings:
                print(f"  - {w}")
            print()

        print("=" * 60)

    # Exit code
    if not result.passed:
        sys.exit(1)
    elif result.warnings and args.fail_on_warning:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
