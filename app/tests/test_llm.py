"""Contract tests for the LLM seam.

The real LLMClient, openai SDK and httpx2 run unchanged; only the network at the
bottom is replaced by a handler function that plays the model server.
"""

from collections.abc import Callable

import httpx2
from fryday.config import Settings
from fryday.llm import LLMClient

Handler = Callable[[httpx2.Request], httpx2.Response]

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
        return httpx2.Response(200, text=SSE_OK, headers={"content-type": "text/event-stream"})

    reply = "".join(fake_llm(handler).stream([{"role": "user", "content": "hi"}]))
    assert reply == "Namaste"
