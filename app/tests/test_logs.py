import json
import logging

import pytest
import structlog
from fryday.logs import setup_logging
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
