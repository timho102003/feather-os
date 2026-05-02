"""Feather package."""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("feather-os")
except PackageNotFoundError:
    try:
        from feather._version import __version__  # type: ignore[no-redef]
    except ImportError:
        __version__ = "0.0.0+unknown"
