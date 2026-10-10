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

Handler = Callable[[httpx2.Request], httpx2.Response]
SSE_HEADERS = {"content-type": "text/event-stream"}
HI: list[Message] = [{"role": "user", "content": "hi"}]

# What an OpenAI-compatible server streams back (Server-Sent Events).
SSE_OK = (
    'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'  # role-only chunk
    'data: {"choices":[{"index":0,"delta":{"content":"Na"}}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{"content":"mas"}}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{"content":"te"}}]}\n\n'
    'data: {"choices":[]}\n\n'  # usage-style chunk with no choices
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
