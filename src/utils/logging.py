from __future__ import annotations

import logging
import os
from typing import Optional


def configure_logging(level: Optional[str] = None) -> None:
    """Configure root logging for the project.

    - `level` can be a logging level string (e.g. 'INFO'/'DEBUG'). If omitted
      the function reads the `LOG_LEVEL` environment variable and falls back to
      'INFO'.
    """
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    fmt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    logging.basicConfig(level=numeric_level, format=fmt)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name)
