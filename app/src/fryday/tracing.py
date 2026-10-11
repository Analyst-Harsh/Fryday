"""Tracing helpers (Langfuse, built on OpenTelemetry).

The ONLY module that imports langfuse: every other file traces through these
helpers, so switching tools later is a one-file change.

    init_tracing(settings)                      # once, at startup
    with trace_root("turn", session_id=sid, input=text) as turn:   # one trace per request
        with span("memory.search", kind="retriever", input=q):     # a step inside it
            ...
        with generation("llm.stream", model=m, params=p, input=msgs) as gen:
            gen.chunk(piece)         # per streamed piece; first one records TTFT
            gen.set_usage(38, 12)    # token counts
        record(turn, output=reply, metadata={"ttft_s": 0.12})
    shutdown_tracing()                          # flush on exit

Before init_tracing(), or with tracing disabled, all of this is a cheap no-op.
Text passed as input/output is dropped unless FRYDAY_TRACE_CONTENT=true.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from langfuse import (
    Langfuse,
    LangfuseAgent,
    LangfuseEmbedding,
    LangfuseGeneration,
    LangfuseRetriever,
    LangfuseSpan,
    LangfuseTool,
    propagate_attributes,
)
from opentelemetry import trace as otel_trace

from fryday.config import Settings

if TYPE_CHECKING:  # type hints only; the OTel SDK is installed via langfuse
    from opentelemetry.sdk.trace import ReadableSpan, TracerProvider

SpanKind = Literal["span", "tool", "retriever", "embedding", "agent"]
Observation = LangfuseSpan | LangfuseTool | LangfuseRetriever | LangfuseEmbedding | LangfuseAgent

# Module state, set by init_tracing(). Langfuse also keeps one client per public
# key internally, so a second Langfuse(...) with the same key reuses the first.
_client: Langfuse | None = None
_capture_content = False


def init_tracing(
    settings: Settings,
    *,
    tracer_provider: TracerProvider | None = None,  # tests: an in-memory provider
    should_export_span: Callable[[ReadableSpan], bool] | None = None,  # tests: export nothing
) -> None:
    global _client, _capture_content
    _capture_content = settings.trace_content
    if not settings.langfuse_tracing_enabled:
        _client = _disabled_client()
        return
    _client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url,
        environment=settings.langfuse_environment,
        tracing_enabled=True,
        tracer_provider=tracer_provider,
        should_export_span=should_export_span,
    )


def current_trace_id() -> str | None:
    """The active trace's id (32 hex chars, as shown in Langfuse), or None.

    Read straight from OpenTelemetry's current span: a pure context lookup.
    Not langfuse.get_current_trace_id(), which logs; the log pipeline calls this,
    so a logging lookup would recurse forever.
    """
    ctx = otel_trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None


def shutdown_tracing() -> None:
    """Flush buffered traces. Also runs at interpreter exit; explicit is safer."""
    if _client is not None:
        _client.shutdown()


def content[T](value: T) -> T | None:
    """Privacy gate: user text only leaves the machine when trace_content is on."""
    return value if _capture_content else None


def _disabled_client() -> Langfuse:
    # Langfuse logs an "Authentication error" whenever keys are absent, even with
    # tracing deliberately off. Placeholder keys keep it quiet; nothing is ever sent.
    return Langfuse(
        public_key="pk-lf-disabled",
        secret_key="sk-lf-disabled",  # noqa: S106 -- a placeholder, not a credential
        tracing_enabled=False,
    )


def _lf() -> Langfuse:
    global _client
    if _client is None:  # not initialised: a disabled client, so helpers are no-ops
        _client = _disabled_client()
    return _client


@contextmanager
def trace_root(name: str, *, session_id: str, input: object = None) -> Generator[LangfuseSpan]:
    """Start a new trace (one per turn/request); every child gets session_id."""
    with (
        _lf().start_as_current_observation(as_type="span", name=name, input=content(input)) as root,
        propagate_attributes(session_id=session_id),
    ):
        yield root


@contextmanager
def span(name: str, *, kind: SpanKind = "span", input: object = None) -> Generator[Observation]:
    """A step inside the current trace. kind sets how Langfuse displays it."""
    with _lf().start_as_current_observation(as_type=kind, name=name, input=content(input)) as obs:
        yield obs


def record(
    obs: Observation,
    *,
    output: object = None,
    metadata: Mapping[str, object] | None = None,
    error: str | None = None,
) -> None:
    """Attach results to a span/trace_root; output goes through the privacy gate."""
    obs.update(output=content(output), metadata=dict(metadata) if metadata else None)
    if error is not None:
        obs.update(level="ERROR", status_message=error)


class Generation:
    """Handle for one LLM call; see generation()."""

    def __init__(self, obs: LangfuseGeneration) -> None:
        self._obs = obs
        self._pieces: list[str] = []

    def chunk(self, text: str) -> None:
        """Call for every streamed piece; the first one marks time-to-first-token."""
        if not self._pieces:
            self._obs.update(completion_start_time=datetime.now(UTC))
        self._pieces.append(text)

    def set_usage(self, input_tokens: int, output_tokens: int) -> None:
        self._obs.update(usage_details={"input": input_tokens, "output": output_tokens})

    def end(self, **extra: object) -> None:
        """Called by generation(); you never need to call it yourself."""
        self._obs.update(output=content("".join(self._pieces)), metadata=extra or None)
        self._obs.end()


@contextmanager
def generation(
    name: str, *, model: str, params: Mapping[str, str | int | float | bool], input: object = None
) -> Generator[Generation]:
    """Trace one LLM call. Always ends it: normally, on error (level ERROR, re-raised)
    or on early stop (generator closed, e.g. barge-in: stopped_early, not an error)."""
    obs = _lf().start_observation(
        as_type="generation",
        name=name,
        model=model,
        model_parameters=dict(params),
        input=content(input),
    )
    gen = Generation(obs)
    try:
        yield gen
    except GeneratorExit:
        gen.end(stopped_early=True)
        raise
    except Exception as exc:
        obs.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
        gen.end()
        raise
    else:
        gen.end()
