"""Regression tests for application package metadata and startup imports."""

from app import __version__


def test_package_exports_version() -> None:
    """The FastAPI factory must always be able to import the app version."""
    assert __version__ == "0.4.0"


def test_factory_imports_with_package_version() -> None:
    """Prevent deployment failures caused by a missing package version."""
    from app.factory import create_app

    assert callable(create_app)
