from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

_EPSILON = 1e-9


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enforce total and per-file branch-coverage thresholds."
    )
    parser.add_argument("coverage_json", type=Path)
    parser.add_argument("--config", type=Path, default=Path("pyproject.toml"))
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to read coverage JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Coverage JSON {path} must contain an object.")
    return value


def _load_policy(path: Path) -> tuple[float, set[str]]:
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(f"Unable to read coverage policy {path}: {exc}") from exc

    policy = (payload.get("tool") or {}).get("project_coverage") or {}
    minimum = float(policy.get("minimum", 70.0))
    raw_waivers = policy.get("per_file_waivers", [])
    if not isinstance(raw_waivers, list) or not all(
        isinstance(item, str) and item.strip() for item in raw_waivers
    ):
        raise RuntimeError("tool.project_coverage.per_file_waivers must be a string list.")

    normalized = [item.replace("\\", "/").lstrip("./") for item in raw_waivers]
    if len(normalized) != len(set(normalized)):
        raise RuntimeError("Duplicate paths exist in the per-file coverage waiver list.")
    return minimum, set(normalized)


def _percent(summary: dict[str, Any], *, path: str) -> float:
    value = summary.get("percent_covered")
    if not isinstance(value, (int, float)):
        raise RuntimeError(f"Coverage summary for {path} has no numeric percent_covered.")
    return float(value)


def main() -> int:
    args = _parse_args()
    try:
        coverage = _load_json(args.coverage_json)
        minimum, waivers = _load_policy(args.config)

        meta = coverage.get("meta") or {}
        if meta.get("branch_coverage") is not True:
            raise RuntimeError("Coverage data must be collected with branch coverage enabled.")

        totals = coverage.get("totals")
        files = coverage.get("files")
        if not isinstance(totals, dict) or not isinstance(files, dict):
            raise RuntimeError("Coverage JSON is missing totals or files.")

        total_percent = _percent(totals, path="TOTAL")
        failures: list[tuple[str, float]] = []
        reported_paths: set[str] = set()
        evaluated = 0

        for raw_path, file_data in sorted(files.items()):
            path = str(raw_path).replace("\\", "/").lstrip("./")
            reported_paths.add(path)
            if path in waivers:
                continue
            if not isinstance(file_data, dict) or not isinstance(file_data.get("summary"), dict):
                raise RuntimeError(f"Coverage entry for {path} is malformed.")
            summary = file_data["summary"]
            opportunities = int(summary.get("num_statements", 0)) + int(
                summary.get("num_branches", 0)
            )
            if opportunities == 0:
                continue
            evaluated += 1
            percent = _percent(summary, path=path)
            if percent + _EPSILON < minimum:
                failures.append((path, percent))

        print(f"Coverage threshold: {minimum:.2f}%")
        print(f"Total measured coverage: {total_percent:.2f}%")
        print(f"Files evaluated by the per-file gate: {evaluated}")
        print(f"Temporary per-file waivers: {len(waivers & reported_paths)}")

        stale_waivers = sorted(waivers - reported_paths)
        for path in stale_waivers:
            print(f"::warning file={path}::Coverage waiver is not present in the report.")

        if total_percent + _EPSILON < minimum:
            print(
                f"::error::Total coverage {total_percent:.2f}% is below "
                f"the required {minimum:.2f}% threshold."
            )

        for path, percent in failures:
            print(
                f"::error file={path}::Coverage {percent:.2f}% is below "
                f"the required {minimum:.2f}% threshold."
            )

        if total_percent + _EPSILON < minimum or failures:
            return 1

        print("Total and per-file coverage thresholds passed.")
        return 0
    except RuntimeError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
