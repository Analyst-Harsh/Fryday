"""Contract tests for the LLM seam.

The real LLMClient, openai SDK and httpx2 run unchanged; only the network at the
bottom is replaced by a handler function that plays the model server.
"""

import json
from collections.abc import Callable, Iterator

import httpx2
import openai
import pytest
from fryday.config import Settings
from fryday.llm import LLMClient, Message
from fryday.tracing import trace_root
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

Handler = Callable[[httpx2.Request], httpx2.Response]
SSE_HEADERS = {"content-type": "text/event-stream"}
HI: list[Message] = [{"role": "user", "content": "hi"}]

# What an OpenAI-compatible server streams back (Server-Sent Events).
SSE_OK = (
    'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'  # role-only chunk
    'data: {"choices":[{"index":0,"delta":{"content":"Na"}}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{"content":"mas"}}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{"content":"te"}}]}\n\n'
    # Final usage chunk (stream_options.include_usage): no choices, exact token counts.
    'data: {"choices":[],"usage":{"prompt_tokens":38,"completion_tokens":3,"total_tokens":41}}\n\n'
    "data: [DONE]\n\n"
)


def fake_llm(handler: Handler) -> LLMClient:
    """A real LLMClient whose network is replaced by `handler`."""
    transport = httpx2.MockTransport(handler)
    return LLMClient(Settings(), http_client=httpx2.Client(transport=transport))


def test_stream_joins_deltas_and_skips_empty_chunks() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, text=SSE_OK, headers=SSE_HEADERS)

    assert "".join(fake_llm(handler).stream(HI)) == "Namaste"


def test_request_carries_sampling_settings_including_top_k() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)  # keep it so the test can look inside
        return httpx2.Response(200, text=SSE_OK, headers=SSE_HEADERS)

    list(fake_llm(handler).stream(HI))

    s = Settings()  # conftest.py isolates it from .env and FRYDAY_* vars
    request = requests[0]
    body = json.loads(request.content)
    assert request.url.path == "/v1/chat/completions"
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["top_k"] == s.llm_top_k  # extra_body really reached the wire
    assert body["top_p"] == s.llm_top_p
    assert body["temperature"] == s.llm_temperature
    assert body["max_tokens"] == s.llm_max_tokens


@pytest.mark.parametrize(
    "mid_stream_error",
    [
        httpx2.RemoteProtocolError("peer closed connection"),  # server died mid-reply
        httpx2.ReadTimeout("no chunk for too long"),  # server stalled mid-reply
    ],
)
def test_mid_stream_failure_raises_an_error_cli_can_catch(
    mid_stream_error: httpx2.TransportError,
) -> None:
    def body() -> Iterator[bytes]:
        yield b'data: {"choices":[{"index":0,"delta":{"content":"Na"}}]}\n\n'
        yield b'data: {"choices":[{"index":0,"delta":{"content":"mas"}}]}\n\n'
        raise mid_stream_error  # the connection breaks here

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=body(), headers=SSE_HEADERS)

    stream = fake_llm(handler).stream(HI)
    assert [next(stream), next(stream)] == ["Na", "mas"]  # the first two pieces did arrive
    with pytest.raises(openai.APIError):  # what cli.py catches
        next(stream)


def test_timeout_before_reply_raises_api_timeout_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("server never answered", request=request)

    with pytest.raises(openai.APITimeoutError):  # a subclass of APIError
        list(fake_llm(handler).stream(HI))


# --- tracing: each stream() call is a Langfuse generation ------------------------

Traced = Callable[..., InMemorySpanExporter]
OBS = "langfuse.observation."


def ok_handler(_request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(200, text=SSE_OK, headers=SSE_HEADERS)


def attrs_of(spans: InMemorySpanExporter, name: str) -> dict[str, object]:
    (span,) = [s for s in spans.get_finished_spans() if s.name == name]
    return dict(span.attributes or {})


def test_generation_records_model_params_usage_and_content(traced: Traced) -> None:
    spans = traced(trace_content=True)
    list(fake_llm(ok_handler).stream(HI))

    s, attrs = Settings(), attrs_of(spans, "llm.stream")
    assert attrs[OBS + "type"] == "generation"
    assert attrs[OBS + "model.name"] == s.llm_model
    assert '"top_k": 20' in str(attrs[OBS + "model.parameters"])
    assert OBS + "completion_start_time" in attrs  # TTFT
    assert attrs[OBS + "usage_details"] == '{"input": 38, "output": 3}'  # from the usage chunk
    assert '"content": "hi"' in str(attrs[OBS + "input"])
    assert attrs[OBS + "output"] == "Namaste"


def test_generation_is_a_child_of_the_turn_and_inherits_its_session(traced: Traced) -> None:
    spans = traced()
    with trace_root("turn", session_id="sess-1"):
        list(fake_llm(ok_handler).stream(HI))

    finished = {s.name: s for s in spans.get_finished_spans()}
    llm_span, turn_span = finished["llm.stream"], finished["turn"]
    assert llm_span.parent is not None
    assert turn_span.context is not None
    assert llm_span.parent.span_id == turn_span.context.span_id
    assert dict(llm_span.attributes or {})["session.id"] == "sess-1"


def test_generation_is_error_when_the_stream_fails(traced: Traced) -> None:
    spans = traced()

    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("server never answered", request=request)

    with pytest.raises(openai.APITimeoutError):  # still re-raised for cli.py
        list(fake_llm(handler).stream(HI))

    assert attrs_of(spans, "llm.stream")[OBS + "level"] == "ERROR"


def test_stopping_early_is_recorded_but_not_an_error(traced: Traced) -> None:
    spans = traced(trace_content=True)
    stream = fake_llm(ok_handler).stream(HI)
    assert next(stream) == "Na"
    stream.close()  # barge-in (v5)

    attrs = attrs_of(spans, "llm.stream")
    assert attrs.get(OBS + "level") != "ERROR"
    assert attrs[OBS + "metadata.stopped_early"] is not None
    assert attrs[OBS + "output"] == "Na"  # the partial reply that was actually spoken
