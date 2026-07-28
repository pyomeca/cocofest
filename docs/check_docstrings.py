"""Fail if any cocofest module, class or public function/method has no description."""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "cocofest"
HEADERS = {"Parameters", "Returns", "-------", "----------"}


def is_empty(doc):
    if doc is None:
        return True
    lines = [line.strip() for line in doc.strip().splitlines() if line.strip()]
    return not any(line not in HEADERS for line in lines)


def module_issues(path, tree):
    if is_empty(ast.get_docstring(tree)):
        yield path, 1, "module"


def function_issues(path, node, qualname):
    if node.name.startswith("__"):
        return
    if is_empty(ast.get_docstring(node)):
        yield path, node.lineno, qualname


def class_issues(path, node):
    if is_empty(ast.get_docstring(node)):
        yield path, node.lineno, f"class {node.name}"
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield from function_issues(path, item, f"{node.name}.{item.name}")


def top_level_issues(path, tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            yield from class_issues(path, node)
        elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
            yield from function_issues(path, node, node.name)


def file_issues(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    yield from module_issues(path, tree)
    yield from top_level_issues(path, tree)


def package_files():
    for path in sorted(ROOT.rglob("*.py")):
        if "__pycache__" not in path.parts:
            yield path


def collect_issues():
    for path in package_files():
        yield from file_issues(path)


def report(path, lineno, name):
    rel = path.relative_to(ROOT.parent)
    print(f"::error file={rel},line={lineno}::Missing description for {name}")


def main():
    issues = list(collect_issues())
    for path, lineno, name in issues:
        report(path, lineno, name)

    if issues:
        print(f"{len(issues)} missing description(s), see annotations above.", file=sys.stderr)
        sys.exit(1)

    print("All modules, classes and functions have a description.")


if __name__ == "__main__":
    main()
