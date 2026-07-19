"""Loguru-based structured logging with rotation and dual output."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_INITIALIZED = False


def setup_logging(
    log_level: str = "INFO",
    log_dir: str | Path = "logs",
    rotation: str = "10 MB",
    retention: str = "30 days",
    compression: str = "gz",
) -> None:
    """Configure loguru for file + console output with rotation.

    Parameters
    ----------
    log_level:
        Minimum severity level to emit.
    log_dir:
        Directory for log files.
    rotation:
        When to rotate the log file (loguru expression).
    retention:
        How long to keep old log files.
    compression:
        Compression format for rotated files.
    """
    global _INITIALIZED  # noqa: PLW0603
    if _INITIALIZED:
        return

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger.remove()

    log_fmt = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )

    logger.add(
        sys.stderr,
        level=log_level,
        format=log_fmt,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    logger.add(
        str(log_path / "trademind_{time:YYYY-MM-DD}.log"),
        level=log_level,
        format=log_fmt,
        rotation=rotation,
        retention=retention,
        compression=compression,
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    logger.add(
        str(log_path / "errors_{time:YYYY-MM-DD}.log"),
        level="ERROR",
        format=log_fmt,
        rotation="5 MB",
        retention="90 days",
        compression=compression,
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    logger.info("Logging initialised — level={} dir={}", log_level, log_path)
    _INITIALIZED = True


def get_logger(name: str | None = None):
    """Return a logger instance optionally bound to a context name."""
    if name:
        return logger.bind(component=name)
    return logger
