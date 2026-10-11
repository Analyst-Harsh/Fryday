import logging
import os
from collections.abc import Callable, Iterator

import pytest
from fryday.config import Settings
from fryday.tracing import init_tracing
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr

# One global tracer provider for the whole test run (OTel allows setting it once).
# Simple = export synchronously, so a span is visible the moment it ends.
_SPANS = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_SPANS))
trace.set_tracer_provider(_provider)


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: os.PathLike[str]) -> None:
    """Every test sees Settings' code defaults: no developer .env, no FRYDAY_*/LANGFUSE_* vars."""
    monkeypatch.chdir(tmp_path)  # .env is looked up relative to the working directory
    for name in os.environ:
        if name.startswith(("FRYDAY_", "LANGFUSE_")):
            monkeypatch.delenv(name)


@pytest.fixture(autouse=True)
def restore_root_logger() -> Iterator[None]:
    """setup_logging() reconfigures the root logger; undo it so tests can't leak."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:], root.level = handlers, level


@pytest.fixture
def spans() -> InMemorySpanExporter:
    """Finished spans from this test only."""
    _SPANS.clear()
    return _SPANS


@pytest.fixture
def traced() -> Iterator[Callable[..., InMemorySpanExporter]]:
    """Turn real Langfuse tracing on, captured in memory with no network.

    Usage: spans = traced(trace_content=True)
    """

    def enable(*, trace_content: bool = False) -> InMemorySpanExporter:
        settings = Settings(
            langfuse_tracing_enabled=True,
            langfuse_public_key="pk-lf-test",
            langfuse_secret_key=SecretStr("sk-lf-test"),
            langfuse_base_url="http://127.0.0.1:9",  # closed port, never contacted
            trace_content=trace_content,
        )
        # should_export_span=False: Langfuse's own exporter sends nothing; the
        # in-memory processor on the same provider still sees every span.
        init_tracing(settings, tracer_provider=_provider, should_export_span=lambda _s: False)
        _SPANS.clear()
        return _SPANS

    yield enable
    init_tracing(Settings())  # back to disabled for the next test
