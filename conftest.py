"""Repository-wide pytest hooks used only by the test runner."""

from __future__ import annotations

import os
from typing import Any

_PYTEST_CONFIG: Any | None = None


def pytest_configure(config: Any) -> None:
    global _PYTEST_CONFIG
    _PYTEST_CONFIG = config


def pytest_unconfigure(config: Any) -> None:
    global _PYTEST_CONFIG
    if _PYTEST_CONFIG is config:
        _PYTEST_CONFIG = None


def _escape_workflow_command(value: object, *, property_value: bool = False) -> str:
    text = str(value).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    if property_value:
        text = text.replace(":", "%3A").replace(",", "%2C")
    return text


def pytest_warning_recorded(
    warning_message: Any,
    when: str,
    nodeid: str,
    location: tuple[str, int, str] | None,
) -> None:
    """Mirror pytest warnings as GitHub Actions annotations.

    Pytest still prints its normal warnings summary. This hook only adds the
    yellow, clickable Actions annotation so warnings are not hidden behind a
    failed test collection or a JUnit-only view.
    """

    if os.getenv("GITHUB_ACTIONS", "").lower() != "true":
        return

    filename = getattr(warning_message, "filename", "") or ""
    lineno = int(getattr(warning_message, "lineno", 0) or 0)
    if location:
        filename = location[0] or filename
        lineno = int(location[1] or lineno)

    category = getattr(getattr(warning_message, "category", None), "__name__", "Warning")
    message = getattr(warning_message, "message", warning_message)
    title = f"pytest {category} ({when})"
    properties = [f"title={_escape_workflow_command(title, property_value=True)}"]
    if filename:
        properties.append(
            f"file={_escape_workflow_command(filename, property_value=True)}"
        )
    if lineno > 0:
        properties.append(f"line={lineno}")
    annotation = (
        f"::warning {','.join(properties)}::"
        f"{_escape_workflow_command(message)}"
    )

    terminal = (
        _PYTEST_CONFIG.pluginmanager.get_plugin("terminalreporter")
        if _PYTEST_CONFIG is not None
        else None
    )
    if terminal is not None:
        terminal.write_line(annotation)
    else:
        print(annotation, flush=True)
