"""Human-readable logging setup for Feather."""

from __future__ import annotations

import logging
from pathlib import Path

from feather.observability.context import build_context_filter
from feather.models import LoggingConfig


def configure_logging(root: Path, config: LoggingConfig) -> logging.Logger:
    """Configure the root logger for the application.

    The format string embeds ``agent_ctx`` and ``session_ctx`` between
    the level and the module name; they are populated by the context
    filter installed once on the root logger (see
    :mod:`feather.observability.context`). Feather code sets the contextvars
    inside :meth:`BaseAgent.run_loop`; third-party libraries that
    don't set them render as ``-`` so the column layout stays stable.

    Args:
        root: Repository root.
        config: Logging configuration.

    Returns:
        Configured root logger.
    """

    log_path = (root / config.path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)s | %(agent_ctx)s | %(session_ctx)s | "
            "%(name)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(formatter)
    handler.addFilter(build_context_filter())
    logger.addHandler(handler)
    return logger
