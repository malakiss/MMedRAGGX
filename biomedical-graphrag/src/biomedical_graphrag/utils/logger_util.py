from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger


def setup_logging(log_level: str | None = None) -> Logger:
    """Set up logging configuration.
    Args:
        log_level: Logging level (e.g., "DEBUG", "INFO").
        If None, defaults to environment variable LOG_LEVEL or "DEBUG".
    Returns:
        Configured logger instance.
    """
    log_level = log_level or os.getenv("LOG_LEVEL", "DEBUG").upper()

    # Outside Prefect → Loguru
    logger.remove()
    logger.add(
        sys.stdout,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan> - "
        "<level>{message}</level>",
    )
    logger.debug(f"Logging initialized at {log_level} level (Loguru).")
    return logger
