from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

_REMOVED_WORD = "ph" + "ase"
_NUMBERED_DELIVERY_WORD = "st" + "age"
_NUMBERED_DELIVERY_TERM = re.compile(
    r"\b" + re.escape(_NUMBERED_DELIVERY_WORD) + r"[ _-]?\d+\b",
    re.IGNORECASE,
)


def _tracked_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [Path(raw.decode("utf-8")) for raw in result.stdout.split(b"\0") if raw]


def _text(path: Path) -> str | None:
    data = path.read_bytes()
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _contains_removed_terminology(value: str) -> bool:
    normalized = value.casefold()
    return (
        _REMOVED_WORD in normalized
        or _NUMBERED_DELIVERY_TERM.search(value) is not None
    )


def test_tracked_source_uses_domain_terminology() -> None:
    violations: list[str] = []
    for path in _tracked_paths():
        path_text = path.as_posix()
        if _contains_removed_terminology(path_text):
            violations.append(f"filename:{path_text}")
            continue

        content = _text(path)
        if content is None:
            continue
        for line_number, line in enumerate(content.splitlines(), start=1):
            if _contains_removed_terminology(line):
                violations.append(f"content:{path_text}:{line_number}:{line.strip()}")

    assert not violations, (
        "Removed delivery terminology remains in tracked source files:\n"
        + "\n".join(violations)
    )


def _python_tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


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
