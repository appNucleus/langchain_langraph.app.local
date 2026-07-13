"""Regression tests for application package metadata and startup imports."""

from __future__ import annotations
from app.factory import create_app
import re
from app import __version__

_SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")


def test_package_exports_valid_version() -> None:
    """The package version is authoritative and follows semantic-version syntax."""
    assert _SEMVER.fullmatch(__version__)


def test_factory_imports_with_package_version() -> None:
    """Prevent deployment failures caused by a missing package version."""
    assert callable(create_app)
