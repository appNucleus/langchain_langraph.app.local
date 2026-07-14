"""Repository-wide pytest diagnostics and GitHub Actions annotations."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

_PYTEST_CONFIG: Any | None = None
_FAILURE_RECORDS: list[dict[str, Any]] = []
_RESULTS_DIR = Path("test-results")
_FAILURE_LOG = _RESULTS_DIR / "failure-details.txt"
_FAILURE_JSON = _RESULTS_DIR / "failure-details.json"
_CRITICAL_PATHS = (
    "app/factory.py",
    "app/graph.py",
    "app/graphs/routes.py",
    "app/graphs/state.py",
    "app/llm/ollama.py",
    "app/mcp/client.py",
    "app/orchestration/execution_meter.py",
    "app/orchestration/run_context.py",
    "app/schemas/chat.py",
    "app/schemas/execution.py",
    "app/schemas/verification.py",
    "app/schemas/worker.py",
)


def _run_git(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    output = (completed.stdout or completed.stderr).strip()
    return output or "clean"


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "missing"


def _repository_snapshot() -> dict[str, Any]:
    files = {
        name: {"exists": Path(name).is_file(), "sha256": _file_sha256(Path(name))}
        for name in _CRITICAL_PATHS
    }
    return {
        "head": _run_git("rev-parse", "HEAD"),
        "branch": _run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "last_commit": _run_git("log", "-1", "--format=%H %cI %s"),
        "status": _run_git("status", "--short"),
        "changed_files": _run_git("diff", "--name-status", "HEAD"),
        "critical_files": files,
    }


def pytest_configure(config: Any) -> None:
    global _PYTEST_CONFIG
    _PYTEST_CONFIG = config
    _FAILURE_RECORDS.clear()
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (_RESULTS_DIR / "repository-snapshot.json").write_text(
        json.dumps(_repository_snapshot(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def pytest_unconfigure(config: Any) -> None:
    global _PYTEST_CONFIG
    if _PYTEST_CONFIG is config:
        _PYTEST_CONFIG = None


def pytest_report_header(config: Any) -> list[str]:
    del config
    snapshot = _repository_snapshot()
    return [
        f"git head: {snapshot['head']}",
        f"git branch: {snapshot['branch']}",
        "failure diagnostics: test-results/failure-details.txt",
        "repository snapshot: test-results/repository-snapshot.json",
    ]


def _terminal() -> Any | None:
    if _PYTEST_CONFIG is None:
        return None
    return _PYTEST_CONFIG.pluginmanager.get_plugin("terminalreporter")


def _captured_sections(report: Any) -> str:
    sections: list[str] = []
    for title, content in getattr(report, "sections", ()):
        if content:
            sections.append(f"\n--- {title} ---\n{content.rstrip()}\n")
    return "".join(sections)


def _escape_workflow_command(value: object, *, property_value: bool = False) -> str:
    text = str(value).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    if property_value:
        text = text.replace(":", "%3A").replace(",", "%2C")
    return text


def pytest_runtest_logreport(report: Any) -> None:
    """Persist every failed setup/call/teardown report with captured output."""

    if not getattr(report, "failed", False):
        return
    longrepr = getattr(report, "longreprtext", None) or str(report.longrepr)
    details = (
        f"NODEID: {report.nodeid}\n"
        f"WHEN: {report.when}\n"
        f"DURATION_SECONDS: {getattr(report, 'duration', 0.0):.6f}\n\n"
        f"{longrepr.rstrip()}\n"
        f"{_captured_sections(report)}"
    )
    record = {
        "nodeid": report.nodeid,
        "when": report.when,
        "duration_seconds": float(getattr(report, "duration", 0.0)),
        "details": details,
    }
    _FAILURE_RECORDS.append(record)

    terminal = _terminal()
    if terminal is not None:
        terminal.write_line(f"::group::pytest failure: {report.nodeid} [{report.when}]")
        for line in details.splitlines():
            terminal.write_line(line)
        terminal.write_line("::endgroup::")

    if os.getenv("GITHUB_ACTIONS", "").lower() == "true":
        first_line = (
            longrepr.strip().splitlines()[-1] if longrepr.strip() else "test failed"
        )
        annotation = (
            "::error "
            f"title={_escape_workflow_command('pytest failure', property_value=True)}::"
            f"{_escape_workflow_command(f'{report.nodeid} [{report.when}] - {first_line}')}"
        )
        if terminal is not None:
            terminal.write_line(annotation)
        else:
            print(annotation, flush=True)


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    del session
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = _repository_snapshot()
    text_parts = [
        "PYTEST FAILURE DIAGNOSTICS",
        f"exit_status: {exitstatus}",
        f"failure_reports: {len(_FAILURE_RECORDS)}",
        f"git_head: {snapshot['head']}",
        f"git_branch: {snapshot['branch']}",
        f"git_status:\n{snapshot['status']}",
        "",
    ]
    for index, record in enumerate(_FAILURE_RECORDS, start=1):
        text_parts.extend(
            [
                "=" * 100,
                f"FAILURE {index}/{len(_FAILURE_RECORDS)}",
                "=" * 100,
                record["details"],
                "",
            ]
        )
    _FAILURE_LOG.write_text("\n".join(text_parts), encoding="utf-8")
    _FAILURE_JSON.write_text(
        json.dumps(
            {
                "exit_status": exitstatus,
                "repository": snapshot,
                "failures": _FAILURE_RECORDS,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    terminal = _terminal()
    if terminal is not None:
        terminal.write_sep("=", "failure diagnostic artifacts")
        terminal.write_line(str(_FAILURE_LOG))
        terminal.write_line(str(_FAILURE_JSON))
        terminal.write_line(str(_RESULTS_DIR / "repository-snapshot.json"))


def pytest_warning_recorded(
    warning_message: Any,
    when: str,
    nodeid: str,
    location: tuple[str, int, str] | None,
) -> None:
    """Mirror warnings as GitHub Actions annotations."""

    del nodeid
    if os.getenv("GITHUB_ACTIONS", "").lower() != "true":
        return
    filename = getattr(warning_message, "filename", "") or ""
    lineno = int(getattr(warning_message, "lineno", 0) or 0)
    if location:
        filename = location[0] or filename
        lineno = int(location[1] or lineno)
    category = getattr(
        getattr(warning_message, "category", None), "__name__", "Warning"
    )
    message = getattr(warning_message, "message", warning_message)
    properties = [
        f"title={_escape_workflow_command(f'pytest {category} ({when})', property_value=True)}"
    ]
    if filename:
        properties.append(
            f"file={_escape_workflow_command(filename, property_value=True)}"
        )
    if lineno > 0:
        properties.append(f"line={lineno}")
    annotation = (
        f"::warning {','.join(properties)}::{_escape_workflow_command(message)}"
    )

    terminal = _terminal()
    if terminal is not None:
        terminal.write_line(annotation)
    else:
        print(annotation, flush=True)
