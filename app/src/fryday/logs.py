"""Structured logging with structlog.

Pretty, coloured lines when stderr is a terminal (development); one JSON object
per line otherwise, or when FRYDAY_LOG_JSON=true (files, log tooling).
Library logs (httpx, openai) go through the same pipeline, so every line has
the same shape.

Usage:
    log = structlog.stdlib.get_logger(__name__)
    bind_contextvars(turn_id="ab12")  # attached to every log line until cleared
    log.info("turn", ttft_s=0.05)
"""

import logging
import sys

import structlog
from structlog.typing import EventDict, Processor, WrappedLogger

from fryday.tracing import current_trace_id


def add_trace_id(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """Stamp the current trace's id, so a log line links to its trace in Langfuse."""
    if trace_id := current_trace_id():
        event_dict["trace_id"] = trace_id
    return event_dict


def setup_logging(as_json: bool | None = None, level: int = logging.INFO) -> None:
    if as_json is None:
        as_json = not sys.stderr.isatty()

    # Steps every log line passes through, ours and the libraries'.
    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,  # adds bound fields like turn_id
        add_trace_id,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    renderer: list[Processor] = (
        [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ]
        if as_json
        else [structlog.dev.ConsoleRenderer()]  # pretty-prints tracebacks itself
    )

    # Our loggers: run the shared steps, then hand the event to the stdlib handler below.
    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    # One handler renders everything; foreign_pre_chain gives library records the same fields.
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, *renderer],
    )
    # stderr keeps logs out of the chat on stdout: `fryday-chat 2>turns.log`.
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logging.basicConfig(level=level, handlers=[handler], force=True)

    # The HTTP client logs every request at INFO; our own turn log already covers it.
    for noisy in ("httpx", "httpx2", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
