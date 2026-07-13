"""Regression tests for application package metadata and startup imports."""

from __future__ import annotations

import re

from app import __version__
from app.settings import Settings


_SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")


def test_package_exports_valid_version() -> None:
    """The package version is authoritative and follows semantic-version syntax."""

    assert _SEMVER.fullmatch(__version__)


def test_version_uses_package_as_single_runtime_source() -> None:
    assert "app_version" not in Settings.model_fields
    assert "mcp_client_version" not in Settings.model_fields


def test_factory_imports_with_package_version() -> None:
    """Prevent deployment failures caused by a missing package version."""

    from app.factory import create_app

    assert callable(create_app)
