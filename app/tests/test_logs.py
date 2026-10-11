import json
import logging
from collections.abc import Callable

import pytest
import structlog
from fryday.logs import setup_logging
from fryday.tracing import current_trace_id, trace_root
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from structlog.contextvars import bind_contextvars, clear_contextvars


def test_json_mode_has_bound_context_on_our_and_library_lines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup_logging(as_json=True)
    bind_contextvars(turn_id="t1")
    try:
        structlog.stdlib.get_logger("fryday.test").info("turn", ttft_s=0.1)
        logging.getLogger("some.library").warning("slow response")
    finally:
        clear_contextvars()
        structlog.reset_defaults()

    ours, library = (json.loads(line) for line in capsys.readouterr().err.splitlines())
    assert ours["event"] == "turn"
    assert ours["ttft_s"] == 0.1
    assert ours["turn_id"] == "t1"
    assert ours["level"] == "info"
    assert "timestamp" in ours
    # A plain stdlib logger (like httpx) gets the same shape and the same turn_id.
    assert library["event"] == "slow response"
    assert library["turn_id"] == "t1"
    assert library["logger"] == "some.library"


def test_log_lines_inside_a_trace_carry_its_trace_id(
    capsys: pytest.CaptureFixture[str], traced: Callable[..., InMemorySpanExporter]
) -> None:
    traced()
    setup_logging(as_json=True)
    log = structlog.stdlib.get_logger("fryday.test")
    try:
        log.info("before")
        with trace_root("turn", session_id="s1"):
            trace_id = current_trace_id()
            log.info("inside")
    finally:
        structlog.reset_defaults()

    before, inside = (json.loads(line) for line in capsys.readouterr().err.splitlines())
    assert "trace_id" not in before
    assert trace_id is not None
    assert inside["trace_id"] == trace_id  # the id you search for in Langfuse
