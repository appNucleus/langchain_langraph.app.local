from __future__ import annotations

import ast
import re
from pathlib import Path

_NUMBERED_DELIVERY = re.compile(r"(?:phase|stage)[_-]?\d+", re.IGNORECASE)
_SCANNED_ROOTS = (Path("app"), Path("tests"), Path("scripts"))

# These are the bounded compatibility identifiers retained while existing
# deployments migrate from the former environment/configuration names. No new
# entry may be added without an explicit deprecation and removal plan.
_ALLOWED_COMPATIBILITY_IDENTIFIERS = {
    ("app/settings.py", "phase2_max_iterations"),
    ("app/settings.py", "phase2_max_research_rounds"),
    ("app/settings.py", "phase2_max_replans"),
    ("app/settings.py", "phase2_max_context_chars"),
    ("app/graph.py", "phase2_max_context_chars"),
    ("app/graph.py", "phase2_max_research_rounds"),
    ("app/graph.py", "phase2_max_replans"),
}


def _python_tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _python_identifiers(path: Path) -> set[str]:
    tree = _python_tree(path)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def test_source_and_test_filenames_use_domain_names() -> None:
    violations: list[str] = []
    for root in _SCANNED_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if _NUMBERED_DELIVERY.search(path.name):
                violations.append(path.as_posix())

    assert not violations, (
        "Numbered delivery terminology is prohibited in maintained filenames:\n"
        + "\n".join(sorted(violations))
    )


def test_python_identifiers_use_domain_names_except_bounded_aliases() -> None:
    violations: list[str] = []
    for root in (Path("app"), Path("tests")):
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            relative = path.as_posix()
            for name in _python_identifiers(path):
                if not _NUMBERED_DELIVERY.search(name):
                    continue
                if (relative, name) in _ALLOWED_COMPATIBILITY_IDENTIFIERS:
                    continue
                violations.append(f"{relative}:{name}")

    assert not violations, (
        "Numbered delivery terminology is prohibited in maintained Python identifiers:\n"
        + "\n".join(sorted(violations))
    )


def test_tests_do_not_reference_json_fixture_files() -> None:
    """Keep pytest data code-only; runtime request examples are not test fixtures."""

    suffix = "." + "json"
    violations: list[str] = []
    for path in Path("tests").rglob("*.py"):
        for node in ast.walk(_python_tree(path)):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if node.value.strip().casefold().endswith(suffix):
                violations.append(f"{path.as_posix()}:{node.lineno}:{node.value}")

    assert not violations, (
        "Tests must use code-defined or injected data instead of JSON fixture files:\n"
        + "\n".join(sorted(violations))
    )
