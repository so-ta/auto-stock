#!/usr/bin/env python3
"""
アーキテクチャ違反レポート生成スクリプト

使用方法:
    python tests/architecture/report_violations.py
"""

import ast
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# 設定
MAX_FILE_LINES = 500
SKIP_SIZE_CHECK_FILES = [
    "orchestrator/pipeline.py",
    "backtest/fast_engine.py",
]


def get_python_files(directory: Path) -> List[Path]:
    """指定ディレクトリ以下の.pyファイルを取得"""
    if not directory.exists():
        return []
    return [f for f in directory.rglob("*.py") if "__pycache__" not in str(f)]


def analyze_duplicates() -> Dict[str, List[Path]]:
    """重複クラス定義を分析"""
    class_locations: Dict[str, List[Path]] = defaultdict(list)
    py_files = get_python_files(SRC_DIR)

    for py_file in py_files:
        try:
            content = py_file.read_text(encoding="utf-8")
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_locations[node.name].append(py_file)
        except (SyntaxError, UnicodeDecodeError):
            continue

    return {k: v for k, v in class_locations.items() if len(v) > 1}


def analyze_file_sizes() -> List[Tuple[Path, int]]:
    """ファイルサイズ違反を分析"""
    py_files = get_python_files(SRC_DIR)
    violations = []

    for py_file in py_files:
        try:
            content = py_file.read_text(encoding="utf-8")
            lines = content.count("\n") + 1
            if lines > MAX_FILE_LINES:
                violations.append((py_file, lines))
        except UnicodeDecodeError:
            continue

    return sorted(violations, key=lambda x: -x[1])


def main():
    """メイン処理"""
    print("=" * 70)
    print("Architecture Violation Report")
    print("=" * 70)

    # 重複クラス定義
    print("\n## Duplicate Class Definitions")
    print("-" * 40)
    duplicates = analyze_duplicates()
    if duplicates:
        # カテゴリ分け
        critical = []  # 同一機能の重複
        acceptable = []  # 許容可能な重複

        acceptable_patterns = ["Test", "Mock", "Error", "Exception", "Base"]

        for name, paths in sorted(duplicates.items()):
            is_acceptable = any(p in name for p in acceptable_patterns)
            if is_acceptable:
                acceptable.append((name, paths))
            else:
                critical.append((name, paths))

        print(f"\n### Critical Duplicates ({len(critical)} classes)")
        for name, paths in critical[:20]:
            relative_paths = [str(p.relative_to(PROJECT_ROOT)) for p in paths]
            print(f"  {name}:")
            for rp in relative_paths:
                print(f"    - {rp}")

        if len(critical) > 20:
            print(f"  ... and {len(critical) - 20} more")

        print(f"\n### Acceptable Duplicates ({len(acceptable)} classes)")
        print("  (Test*, Mock*, *Error, *Exception, Base*)")
    else:
        print("  No duplicate classes found.")

    # ファイルサイズ違反
    print("\n## File Size Violations (> 500 lines)")
    print("-" * 40)
    size_violations = analyze_file_sizes()
    if size_violations:
        # 重要度でグループ化
        critical_files = []  # 1000行超
        warning_files = []   # 500-1000行

        for path, lines in size_violations:
            relative = path.relative_to(PROJECT_ROOT)
            skip_note = ""
            for skip in SKIP_SIZE_CHECK_FILES:
                if str(relative).endswith(skip):
                    skip_note = " [SKIP]"
                    break

            if lines > 1000:
                critical_files.append((relative, lines, skip_note))
            else:
                warning_files.append((relative, lines, skip_note))

        print(f"\n### Critical (> 1000 lines): {len(critical_files)} files")
        for path, lines, note in critical_files:
            print(f"  {path}: {lines} lines{note}")

        print(f"\n### Warning (500-1000 lines): {len(warning_files)} files")
        for path, lines, note in warning_files[:30]:
            print(f"  {path}: {lines} lines{note}")
        if len(warning_files) > 30:
            print(f"  ... and {len(warning_files) - 30} more")
    else:
        print("  No size violations found.")

    # サマリー
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Duplicate classes: {len(duplicates)}")
    print(f"  Files > 500 lines: {len(size_violations)}")
    print(f"  Files > 1000 lines: {len([v for v in size_violations if v[1] > 1000])}")

    # 推奨アクション
    print("\n## Recommended Actions")
    print("-" * 40)
    print("  1. 重複クラスの統一（config/types.py への集約検討）")
    print("  2. 大規模ファイルの分割（cmd_027 タスク）")
    print("  3. 段階的なCIへの組み込み")

    return 0


if __name__ == "__main__":
    sys.exit(main())
