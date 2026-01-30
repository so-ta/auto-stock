#!/usr/bin/env python3
"""
Static Analysis Runner - Run mypy, ruff, and vulture for code quality.

This script runs all static analysis tools and generates a summary report.
All errors do not need to be fixed immediately; the goal is to establish
a baseline and track improvements over time.

Usage:
    python scripts/run_static_analysis.py [--fix] [--report]

Options:
    --fix       Run ruff with auto-fix enabled
    --report    Generate detailed report in results/
    --mypy      Run only mypy
    --ruff      Run only ruff
    --vulture   Run only vulture
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(cmd: list[str], capture: bool = True) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"


def get_python_cmd() -> str:
    """Get the appropriate python command for the system."""
    import shutil
    for cmd in ["python3", "python"]:
        if shutil.which(cmd):
            return cmd
    return "python3"


def run_mypy(verbose: bool = True) -> dict:
    """Run mypy type checker."""
    print("\n" + "=" * 60)
    print("Running mypy (Type Checker)")
    print("=" * 60)

    python_cmd = get_python_cmd()
    cmd = [python_cmd, "-m", "mypy", "src/", "--show-error-codes", "--no-error-summary"]
    exit_code, stdout, stderr = run_command(cmd)

    # Count errors
    error_count = stdout.count(": error:")
    warning_count = stdout.count(": warning:")
    note_count = stdout.count(": note:")

    if verbose:
        if stdout.strip():
            print(stdout[:5000])  # Limit output
            if len(stdout) > 5000:
                print(f"... ({len(stdout)} chars total, truncated)")
        if stderr.strip():
            print(f"stderr: {stderr[:1000]}")

    result = {
        "tool": "mypy",
        "exit_code": exit_code,
        "errors": error_count,
        "warnings": warning_count,
        "notes": note_count,
        "status": "pass" if exit_code == 0 else "fail",
    }

    print(f"\nmypy: {error_count} errors, {warning_count} warnings")
    return result


def run_ruff(fix: bool = False, verbose: bool = True) -> dict:
    """Run ruff linter."""
    print("\n" + "=" * 60)
    print(f"Running ruff (Linter) {'with --fix' if fix else ''}")
    print("=" * 60)

    python_cmd = get_python_cmd()
    cmd = [python_cmd, "-m", "ruff", "check", "src/"]
    if fix:
        cmd.append("--fix")

    exit_code, stdout, stderr = run_command(cmd)

    # Count issues
    lines = stdout.strip().split("\n") if stdout.strip() else []
    issue_count = len([l for l in lines if l.strip() and not l.startswith("Found")])

    # Parse "Found X errors" line
    found_line = [l for l in lines if l.startswith("Found")]
    if found_line:
        try:
            issue_count = int(found_line[0].split()[1])
        except (IndexError, ValueError):
            pass

    if verbose:
        if stdout.strip():
            print(stdout[:5000])
            if len(stdout) > 5000:
                print(f"... ({len(stdout)} chars total, truncated)")
        if stderr.strip():
            print(f"stderr: {stderr[:1000]}")

    result = {
        "tool": "ruff",
        "exit_code": exit_code,
        "issues": issue_count,
        "fixed": fix,
        "status": "pass" if exit_code == 0 else "fail",
    }

    print(f"\nruff: {issue_count} issues")
    return result


def run_vulture(verbose: bool = True) -> dict:
    """Run vulture dead code detector."""
    print("\n" + "=" * 60)
    print("Running vulture (Dead Code Detector)")
    print("=" * 60)

    python_cmd = get_python_cmd()
    cmd = [python_cmd, "-m", "vulture", "src/", "--min-confidence", "80"]
    exit_code, stdout, stderr = run_command(cmd)

    # Count unused code
    lines = stdout.strip().split("\n") if stdout.strip() else []
    unused_count = len([l for l in lines if l.strip()])

    if verbose:
        if stdout.strip():
            print(stdout[:5000])
            if len(stdout) > 5000:
                print(f"... ({len(stdout)} chars total, truncated)")
        if stderr.strip():
            print(f"stderr: {stderr[:1000]}")

    result = {
        "tool": "vulture",
        "exit_code": exit_code,
        "unused_code": unused_count,
        "status": "pass" if exit_code == 0 else "fail",
    }

    print(f"\nvulture: {unused_count} potentially unused items")
    return result


def generate_report(results: list[dict], output_dir: Path) -> None:
    """Generate detailed report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at": datetime.now().isoformat(),
        "tools": results,
        "summary": {
            "total_tools": len(results),
            "passed": sum(1 for r in results if r["status"] == "pass"),
            "failed": sum(1 for r in results if r["status"] == "fail"),
        },
    }

    report_path = output_dir / "static_analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_path}")


def print_summary(results: list[dict]) -> int:
    """Print summary and return exit code."""
    print("\n" + "=" * 60)
    print("STATIC ANALYSIS SUMMARY")
    print("=" * 60)

    total_issues = 0

    for result in results:
        tool = result["tool"]
        status = result["status"]
        status_icon = "✓" if status == "pass" else "✗"

        if tool == "mypy":
            issues = result.get("errors", 0)
            total_issues += issues
            print(f"  {status_icon} mypy: {issues} errors, {result.get('warnings', 0)} warnings")

        elif tool == "ruff":
            issues = result.get("issues", 0)
            total_issues += issues
            print(f"  {status_icon} ruff: {issues} issues")

        elif tool == "vulture":
            issues = result.get("unused_code", 0)
            total_issues += issues
            print(f"  {status_icon} vulture: {issues} unused items")

    print(f"\nTotal issues: {total_issues}")
    print("=" * 60)

    # Return non-zero if any tool failed
    return 1 if any(r["status"] == "fail" for r in results) else 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run static analysis tools")
    parser.add_argument("--fix", action="store_true", help="Run ruff with auto-fix")
    parser.add_argument("--report", action="store_true", help="Generate detailed report")
    parser.add_argument("--mypy", action="store_true", help="Run only mypy")
    parser.add_argument("--ruff", action="store_true", help="Run only ruff")
    parser.add_argument("--vulture", action="store_true", help="Run only vulture")

    args = parser.parse_args()

    # If no specific tool is selected, run all
    run_all = not (args.mypy or args.ruff or args.vulture)

    results = []

    print("=" * 60)
    print("STATIC ANALYSIS RUNNER")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if run_all or args.mypy:
        results.append(run_mypy())

    if run_all or args.ruff:
        results.append(run_ruff(fix=args.fix))

    if run_all or args.vulture:
        results.append(run_vulture())

    if args.report:
        output_dir = Path(__file__).parent.parent / "results"
        generate_report(results, output_dir)

    return print_summary(results)


if __name__ == "__main__":
    sys.exit(main())
