"""
アーキテクチャテスト - 設計ルール違反検出 【QA-006】

設計ルール違反を自動検出する仕組み:
- 型定義重複の検出
- 循環依存の検出
- ファイル肥大化の検出
- モジュール依存関係ルールの検証

使用方法:
    pytest tests/architecture/ -m architecture -v
"""

import ast
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pytest

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"


# =============================================================================
# 設定
# =============================================================================

# ファイルサイズ上限（行数）
MAX_FILE_LINES = 500

# 行サイズ上限を超えてもスキップするファイル（分割予定）
SKIP_SIZE_CHECK_FILES = [
    "orchestrator/pipeline.py",  # task_027_3/7で分割予定
    "backtest/fast_engine.py",   # 統合モジュール
]

# 重複が許可されるクラス名（Test*, Mock*, ベース例外など）
ALLOWED_DUPLICATE_PATTERNS = [
    r"^Test.*",        # テストクラス
    r"^Mock.*",        # モッククラス
    r".*Error$",       # 例外クラス（各モジュールで定義可能）
    r".*Exception$",   # 例外クラス
    r"^Base.*",        # ベースクラス（抽象）
]

# 依存禁止ルール: {from_module: [to_modules]}
# from_module は to_modules に依存してはならない
FORBIDDEN_DEPENDENCIES = {
    "backtest": ["orchestrator"],    # バックテストはオーケストレータに依存禁止
    "signals": ["orchestrator"],     # シグナルはオーケストレータに依存禁止
    "allocation": ["orchestrator"],  # アロケーションはオーケストレータに依存禁止
    "risk": ["orchestrator"],        # リスクはオーケストレータに依存禁止
    "data": ["orchestrator", "backtest", "signals"],  # データは上位層に依存禁止
}


# =============================================================================
# ヘルパー関数
# =============================================================================

def get_python_files(directory: Path) -> List[Path]:
    """指定ディレクトリ以下の.pyファイルを取得"""
    if not directory.exists():
        return []
    return list(directory.rglob("*.py"))


def is_skip_file(file_path: Path, skip_list: List[str]) -> bool:
    """スキップ対象ファイルかチェック"""
    relative = file_path.relative_to(SRC_DIR)
    return any(str(relative) == skip or str(relative).endswith(skip) for skip in skip_list)


def is_allowed_duplicate(class_name: str) -> bool:
    """重複が許可されるクラスかチェック"""
    for pattern in ALLOWED_DUPLICATE_PATTERNS:
        if re.match(pattern, class_name):
            return True
    return False


def extract_classes(file_path: Path) -> List[Tuple[str, int]]:
    """ファイルからクラス定義を抽出（クラス名, 行番号）"""
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content)
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append((node.name, node.lineno))
        return classes
    except (SyntaxError, UnicodeDecodeError) as e:
        pytest.skip(f"Could not parse {file_path}: {e}")
        return []


def extract_imports(file_path: Path) -> List[Tuple[str, int]]:
    """ファイルからインポートを抽出（モジュール名, 行番号）"""
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content)
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append((alias.name, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append((node.module, node.lineno))

        return imports
    except (SyntaxError, UnicodeDecodeError):
        return []


def get_module_name(file_path: Path) -> Optional[str]:
    """ファイルパスからモジュール名（最上位ディレクトリ）を取得"""
    try:
        relative = file_path.relative_to(SRC_DIR)
        parts = relative.parts
        if len(parts) > 0 and parts[0] != "__pycache__":
            return parts[0]
    except ValueError:
        pass
    return None


# =============================================================================
# テスト
# =============================================================================

@pytest.mark.architecture
class TestDesignRules:
    """設計ルールテスト"""

    def test_no_duplicate_class_definitions(self):
        """同一クラスが複数ファイルで定義されていないことを確認"""
        class_locations: Dict[str, List[Path]] = defaultdict(list)
        py_files = get_python_files(SRC_DIR)

        for py_file in py_files:
            # __pycache__ をスキップ
            if "__pycache__" in str(py_file):
                continue

            classes = extract_classes(py_file)
            for class_name, lineno in classes:
                # 許可されるパターンはスキップ
                if is_allowed_duplicate(class_name):
                    continue
                class_locations[class_name].append(py_file)

        # 重複を検出
        duplicates = {
            name: paths
            for name, paths in class_locations.items()
            if len(paths) > 1
        }

        if duplicates:
            messages = []
            for name, paths in duplicates.items():
                path_str = ", ".join(str(p.relative_to(PROJECT_ROOT)) for p in paths)
                messages.append(f"  - {name}: {path_str}")

            pytest.fail(
                f"Found {len(duplicates)} duplicate class definitions:\n"
                + "\n".join(messages)
            )

    def test_file_size_limits(self):
        """ファイルサイズ上限（{MAX_FILE_LINES}行）を確認"""
        py_files = get_python_files(SRC_DIR)
        violations = []

        for py_file in py_files:
            # __pycache__ をスキップ
            if "__pycache__" in str(py_file):
                continue

            # スキップ対象ファイルをチェック
            if is_skip_file(py_file, SKIP_SIZE_CHECK_FILES):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                lines = content.count("\n") + 1

                if lines > MAX_FILE_LINES:
                    relative = py_file.relative_to(PROJECT_ROOT)
                    violations.append((relative, lines))
            except UnicodeDecodeError:
                continue

        if violations:
            messages = [
                f"  - {path}: {lines} lines (limit: {MAX_FILE_LINES})"
                for path, lines in sorted(violations, key=lambda x: -x[1])
            ]
            pytest.fail(
                f"Found {len(violations)} files exceeding {MAX_FILE_LINES} lines:\n"
                + "\n".join(messages)
            )

    def test_module_dependencies(self):
        """モジュール依存関係ルールを確認"""
        py_files = get_python_files(SRC_DIR)
        violations = []

        for py_file in py_files:
            # __pycache__ をスキップ
            if "__pycache__" in str(py_file):
                continue

            from_module = get_module_name(py_file)
            if not from_module or from_module not in FORBIDDEN_DEPENDENCIES:
                continue

            forbidden = FORBIDDEN_DEPENDENCIES[from_module]
            imports = extract_imports(py_file)

            for import_name, lineno in imports:
                # src. プレフィックスを処理
                if import_name.startswith("src."):
                    import_name = import_name[4:]

                # 相対インポートを処理
                if import_name.startswith("."):
                    continue

                # 依存先モジュールをチェック
                for forbidden_module in forbidden:
                    if import_name.startswith(forbidden_module + ".") or import_name == forbidden_module:
                        relative = py_file.relative_to(PROJECT_ROOT)
                        violations.append((
                            from_module,
                            forbidden_module,
                            str(relative),
                            lineno,
                            import_name,
                        ))

        if violations:
            messages = []
            for from_mod, to_mod, file_path, lineno, import_name in violations:
                messages.append(
                    f"  - {file_path}:{lineno}: {from_mod} -> {to_mod} "
                    f"(import {import_name})"
                )
            pytest.fail(
                f"Found {len(violations)} forbidden dependencies:\n"
                + "\n".join(messages)
            )

    def test_no_circular_imports_basic(self):
        """基本的な循環インポートチェック（軽量版）"""
        # import-linter の代わりに簡易チェック
        # 各モジュールのインポートグラフを構築

        import_graph: Dict[str, Set[str]] = defaultdict(set)
        py_files = get_python_files(SRC_DIR)

        for py_file in py_files:
            if "__pycache__" in str(py_file):
                continue

            from_module = get_module_name(py_file)
            if not from_module:
                continue

            imports = extract_imports(py_file)
            for import_name, _ in imports:
                # src. プレフィックスを処理
                if import_name.startswith("src."):
                    import_name = import_name[4:]

                # モジュール名を抽出
                parts = import_name.split(".")
                if len(parts) > 0 and parts[0] in FORBIDDEN_DEPENDENCIES:
                    import_graph[from_module].add(parts[0])

        # 単純な循環検出（深さ2まで）
        cycles = []
        for mod_a, deps_a in import_graph.items():
            for mod_b in deps_a:
                if mod_b in import_graph and mod_a in import_graph[mod_b]:
                    cycle = tuple(sorted([mod_a, mod_b]))
                    if cycle not in cycles:
                        cycles.append(cycle)

        if cycles:
            messages = [f"  - {a} <-> {b}" for a, b in cycles]
            pytest.fail(
                f"Found {len(cycles)} potential circular dependencies:\n"
                + "\n".join(messages)
            )


@pytest.mark.architecture
class TestCodeQuality:
    """コード品質テスト"""

    def test_no_star_imports(self):
        """ワイルドカードインポート（from x import *）がないことを確認"""
        py_files = get_python_files(SRC_DIR)
        violations = []

        for py_file in py_files:
            if "__pycache__" in str(py_file):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        for alias in node.names:
                            if alias.name == "*":
                                relative = py_file.relative_to(PROJECT_ROOT)
                                violations.append((relative, node.lineno, node.module))

            except (SyntaxError, UnicodeDecodeError):
                continue

        if violations:
            messages = [
                f"  - {path}:{lineno}: from {module} import *"
                for path, lineno, module in violations
            ]
            pytest.fail(
                f"Found {len(violations)} star imports (use explicit imports):\n"
                + "\n".join(messages)
            )

    def test_no_print_statements(self):
        """本番コードにprint文がないことを確認（デバッグ用を除く）"""
        py_files = get_python_files(SRC_DIR)
        violations = []

        # スキップパターン
        skip_patterns = [
            "__pycache__",
            "test_",
            "_test.py",
        ]

        for py_file in py_files:
            if any(pattern in str(py_file) for pattern in skip_patterns):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name) and node.func.id == "print":
                            relative = py_file.relative_to(PROJECT_ROOT)
                            violations.append((relative, node.lineno))

            except (SyntaxError, UnicodeDecodeError):
                continue

        # 5個以上のprint文がある場合のみ警告
        if len(violations) > 5:
            messages = [f"  - {path}:{lineno}" for path, lineno in violations[:10]]
            if len(violations) > 10:
                messages.append(f"  ... and {len(violations) - 10} more")
            pytest.fail(
                f"Found {len(violations)} print statements in production code:\n"
                + "\n".join(messages)
            )

    def test_functions_have_docstrings(self):
        """主要な関数にdocstringがあることを確認"""
        py_files = get_python_files(SRC_DIR)
        violations = []
        checked = 0

        # 公開関数（_で始まらない）のみチェック
        for py_file in py_files:
            if "__pycache__" in str(py_file):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        # プライベート関数をスキップ
                        if node.name.startswith("_"):
                            continue
                        # テスト関数をスキップ
                        if node.name.startswith("test"):
                            continue

                        checked += 1

                        # docstringをチェック
                        docstring = ast.get_docstring(node)
                        if not docstring:
                            relative = py_file.relative_to(PROJECT_ROOT)
                            violations.append((relative, node.lineno, node.name))

            except (SyntaxError, UnicodeDecodeError):
                continue

        # 20%以上の関数にdocstringがない場合のみ失敗
        if checked > 0 and len(violations) / checked > 0.2:
            messages = [
                f"  - {path}:{lineno}: {name}()"
                for path, lineno, name in violations[:10]
            ]
            if len(violations) > 10:
                messages.append(f"  ... and {len(violations) - 10} more")
            pytest.fail(
                f"Found {len(violations)}/{checked} public functions without docstrings:\n"
                + "\n".join(messages)
            )


@pytest.mark.architecture
class TestNamingConventions:
    """命名規則テスト"""

    def test_class_names_are_pascal_case(self):
        """クラス名がPascalCaseであることを確認"""
        py_files = get_python_files(SRC_DIR)
        violations = []

        for py_file in py_files:
            if "__pycache__" in str(py_file):
                continue

            classes = extract_classes(py_file)
            for class_name, lineno in classes:
                # UPPER_CASE (定数クラス) や snake_case は違反
                if "_" in class_name and not class_name.isupper():
                    relative = py_file.relative_to(PROJECT_ROOT)
                    violations.append((relative, lineno, class_name))

        if violations:
            messages = [
                f"  - {path}:{lineno}: {name} (should be PascalCase)"
                for path, lineno, name in violations
            ]
            pytest.fail(
                f"Found {len(violations)} classes not following PascalCase:\n"
                + "\n".join(messages)
            )


# =============================================================================
# 違反レポート生成
# =============================================================================

def generate_violation_report() -> str:
    """現状の違反レポートを生成"""
    report = []
    report.append("=" * 60)
    report.append("Architecture Violation Report")
    report.append("=" * 60)

    # ファイルサイズ違反
    report.append("\n## File Size Violations (> 500 lines)")
    py_files = get_python_files(SRC_DIR)
    size_violations = []

    for py_file in py_files:
        if "__pycache__" in str(py_file):
            continue
        try:
            content = py_file.read_text(encoding="utf-8")
            lines = content.count("\n") + 1
            if lines > MAX_FILE_LINES:
                relative = py_file.relative_to(PROJECT_ROOT)
                size_violations.append((relative, lines))
        except UnicodeDecodeError:
            continue

    for path, lines in sorted(size_violations, key=lambda x: -x[1]):
        skip_note = " (SKIP)" if is_skip_file(SRC_DIR / path, SKIP_SIZE_CHECK_FILES) else ""
        report.append(f"  {path}: {lines} lines{skip_note}")

    if not size_violations:
        report.append("  No violations found.")

    return "\n".join(report)


if __name__ == "__main__":
    # 直接実行時はレポート生成
    print(generate_violation_report())
