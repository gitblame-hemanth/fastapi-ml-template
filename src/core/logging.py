"""Structured JSON logging configuration."""

import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

import structlog

_request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)


def get_request_id() -> str | None:
    """Get the current request ID from context."""
    return _request_id_ctx.get()


def set_request_id(request_id: str | None) -> None:
    """Set the current request ID in context."""
    _request_id_ctx.set(request_id)


def _add_request_id(logger: logging.Logger, method_name: str, event_dict: dict) -> dict:
    """Inject request_id from context into every log entry."""
    request_id = get_request_id()
    if request_id is not None:
        event_dict["request_id"] = request_id
    return event_dict


def _add_timestamp(logger: logging.Logger, method_name: str, event_dict: dict) -> dict:
    """Add ISO 8601 UTC timestamp."""
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured JSON logging with structlog.

    Args:
        log_level: Python log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure standard library logging as structlog's output sink
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
        force=True,
    )

    shared_processors: list[structlog.types.Processor] = [
        _add_timestamp,
        _add_request_id,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.contextvars.merge_contextvars,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Apply structlog formatting to all stdlib handlers
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a named structlog logger.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
    """
    return structlog.get_logger(name)
