"""Structured logging configuration for llm-gateway."""

from __future__ import annotations

import logging
from typing import Any

_CONFIGURED = False

# ── Optional structlog import ───────────────────────────────────
try:
    import structlog
    from structlog.contextvars import merge_contextvars

    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False


def configure_logging(
    level: str = "INFO",
    fmt: str = "json",
) -> None:
    """Configure logging for llm-gateway.

    If structlog is installed, sets up structured logging with JSON or
    console rendering. Otherwise falls back to stdlib logging.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        fmt: Output format — "json" or "console".
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    if HAS_STRUCTLOG:
        _configure_structlog(numeric_level, fmt)
    else:
        logging.basicConfig(level=numeric_level, format="%(levelname)s %(name)s %(message)s")


def _configure_structlog(level: int, fmt: str) -> None:
    """Set up structlog processors and rendering."""
    shared_processors: list[Any] = [
        merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if fmt == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler()
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=renderer,
            foreign_pre_chain=shared_processors,
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> Any:
    """Return a logger instance.

    Returns a structlog BoundLogger if structlog is installed,
    otherwise a stdlib logger.
    """
    if HAS_STRUCTLOG:
        return structlog.get_logger(name)
    return logging.getLogger(name)
