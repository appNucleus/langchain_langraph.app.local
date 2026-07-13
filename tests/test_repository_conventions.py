from __future__ import annotations

import ast
from pathlib import Path


_FORBIDDEN_MARKER = "phase"


def test_test_and_script_filenames_use_domain_language() -> None:
    candidates = [
        *Path("tests").glob("test_*.py"),
        *Path("scripts").glob("*.sh"),
    ]
    offenders = sorted(
        str(path)
        for path in candidates
        if _FORBIDDEN_MARKER in path.name.casefold()
    )

    assert not offenders, (
        "Milestone-oriented filenames are not allowed; use domain-oriented names:\n"
        + "\n".join(offenders)
    )


def test_test_functions_use_domain_language() -> None:
    offenders: list[str] = []

    for path in Path("tests").glob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if _FORBIDDEN_MARKER in node.name.casefold():
                    offenders.append(f"{path}:{node.lineno}:{node.name}")

    assert not offenders, (
        "Milestone-oriented test function names are not allowed:\n"
        + "\n".join(sorted(offenders))
    )
