from __future__ import annotations

import logging
import os
from typing import Optional


def configure_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
    force: bool = False,
) -> None:
    """Configure root logging for the project.

    Parameters
    - level: logging level string (e.g. 'INFO'/'DEBUG'). Falls back to `LOG_LEVEL`
      env var and then to 'INFO'.
    - log_file: optional file path to write logs. Falls back to `LOG_FILE` env var.
    - max_bytes, backup_count: parameters for rotating file handler.
    - force: if True, reconfigure even if logging was previously configured.
    """
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()

    # Avoid adding duplicate handlers on repeated calls unless forced.
    if getattr(root, "_configured", False) and not force:
        root.setLevel(numeric_level)
        return

    root.setLevel(numeric_level)

    fmt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    formatter = logging.Formatter(fmt)

    if force:
        for h in list(root.handlers):
            root.removeHandler(h)

    # Add a stream handler if none exists
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(numeric_level)
        sh.setFormatter(formatter)
        root.addHandler(sh)

    # File handler (rotating) when requested
    log_file = log_file or os.environ.get("LOG_FILE")
    if log_file:
        try:
            from logging.handlers import RotatingFileHandler

            fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
            fh.setLevel(numeric_level)
            fh.setFormatter(formatter)
            root.addHandler(fh)
        except Exception as exc:  # pragma: no cover - best-effort logging setup
            # If file handler fails, ensure we still have stream handler
            root.warning("Failed to create log file handler %s: %s", log_file, exc)

    root._configured = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name)
