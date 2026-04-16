"""
Structured logging configuration.
Enterprise-grade logging like Salesforce/HubSpot observability systems.
Logs are structured JSON for easy parsing by log aggregators (Splunk, DataDog etc).
"""

import logging
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class StructuredJsonFormatter(logging.Formatter):
    """
    Formats log records as structured JSON.
    Every log line is a parseable JSON object — industry standard
    for production systems at Microsoft, Salesforce, HubSpot.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Attach correlation_id if present (for request tracing)
        if hasattr(record, "correlation_id"):
            log_entry["correlation_id"] = record.correlation_id

        # Attach extra context fields if present
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        # Attach exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class ContextLogger(logging.LoggerAdapter):
    """
    Logger adapter that automatically injects correlation_id
    into every log line — like HubSpot's request tracing system.
    Allows end-to-end tracing of a single email generation request.
    """

    def __init__(self, logger: logging.Logger, correlation_id: Optional[str] = None):
        self.correlation_id = correlation_id or str(uuid.uuid4())[:8]
        super().__init__(logger, extra={"correlation_id": self.correlation_id})

    def process(self, msg: str, kwargs: Dict) -> tuple:
        kwargs.setdefault("extra", {})
        kwargs["extra"]["correlation_id"] = self.correlation_id
        return msg, kwargs

    def with_fields(self, **fields) -> "ContextLogger":
        """Attach additional context fields to subsequent log lines."""
        new_logger = ContextLogger(self.logger, self.correlation_id)
        new_logger.extra.update(fields)
        return new_logger


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Initialize the global logging configuration.
    Call once at application startup (in app.py).

    Args:
        log_level: Logging level string e.g. "INFO", "DEBUG", "WARNING"
        log_file: Optional file path to also write logs to disk
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicate logs
    root_logger.handlers.clear()

    # Console handler — structured JSON to stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(StructuredJsonFormatter())
    root_logger.addHandler(console_handler)

    # File handler — optional persistent log file
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(StructuredJsonFormatter())
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger. Use module __name__ as convention.

    Usage:
        logger = get_logger(__name__)
        logger.info("Something happened")
    """
    return logging.getLogger(name)