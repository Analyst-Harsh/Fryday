"""The tracing helpers, against real Langfuse spans captured in memory."""

from collections.abc import Callable

import pytest
from fryday.tracing import generation, record, span, trace_root
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

Traced = Callable[..., InMemorySpanExporter]
OBS = "langfuse.observation."


def by_name(spans: InMemorySpanExporter) -> dict[str, dict[str, object]]:
    finished: tuple[ReadableSpan, ...] = spans.get_finished_spans()
    return {s.name: dict(s.attributes or {}) for s in finished}


def test_helpers_are_noops_when_tracing_is_disabled(spans: InMemorySpanExporter) -> None:
    with trace_root("turn", session_id="s1", input="hi") as turn:
        record(turn, output="hello")
    assert spans.get_finished_spans() == ()


def test_trace_root_tags_children_with_session_and_kind(traced: Traced) -> None:
    spans = traced()
    with trace_root("turn", session_id="s1"), span("tool.create_reminder", kind="tool"):
        pass

    got = by_name(spans)
    assert got["turn"]["session.id"] == "s1"
    assert got["tool.create_reminder"]["session.id"] == "s1"
    assert got["tool.create_reminder"][OBS + "type"] == "tool"


@pytest.mark.parametrize("trace_content", [False, True])
def test_content_gate_controls_input_and_output(traced: Traced, trace_content: bool) -> None:
    spans = traced(trace_content=trace_content)
    with trace_root("turn", session_id="s1", input="secret question") as turn:
        record(turn, output="secret answer", metadata={"ttft_s": 0.1})

    attrs = by_name(spans)["turn"]
    assert (OBS + "input" in attrs) is trace_content
    assert (OBS + "output" in attrs) is trace_content
    assert attrs[OBS + "metadata.ttft_s"] is not None  # metadata is never gated


def test_generation_records_ttft_usage_and_joined_output(traced: Traced) -> None:
    spans = traced(trace_content=True)
    with generation("llm.stream", model="qwen", params={"top_k": 20}, input="hi") as gen:
        gen.chunk("Na")
        gen.chunk("maste")
        gen.set_usage(input_tokens=38, output_tokens=2)

    attrs = by_name(spans)["llm.stream"]
    assert attrs[OBS + "type"] == "generation"
    assert attrs[OBS + "model.name"] == "qwen"
    assert OBS + "completion_start_time" in attrs
    assert attrs[OBS + "output"] == "Namaste"
    assert attrs[OBS + "usage_details"] == '{"input": 38, "output": 2}'


def test_generation_marks_errors_and_reraises(traced: Traced) -> None:
    spans = traced()
    with (
        pytest.raises(TimeoutError),
        generation("llm.stream", model="qwen", params={}),
    ):
        raise TimeoutError("model stalled")

    attrs = by_name(spans)["llm.stream"]
    assert attrs[OBS + "level"] == "ERROR"
    assert "model stalled" in str(attrs[OBS + "status_message"])


def test_generation_closed_early_is_stopped_not_failed(traced: Traced) -> None:
    spans = traced()

    def stream():  # like LLMClient.stream: a generator with a generation inside
        with generation("llm.stream", model="qwen", params={}) as gen:
            for piece in ("Na", "mas", "te"):
                gen.chunk(piece)
                yield piece

    s = stream()
    next(s)
    s.close()  # barge-in

    attrs = by_name(spans)["llm.stream"]
    assert attrs.get(OBS + "level") != "ERROR"
    assert attrs[OBS + "metadata.stopped_early"] is not None
