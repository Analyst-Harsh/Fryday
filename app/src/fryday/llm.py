"""The LLM seam: the only module that talks to the model server.

Anything speaking the OpenAI chat-completions contract works behind it:
mlx_lm.server on the Mac today, vLLM on the GPU box later (v13).
"""

from collections.abc import Generator

import httpx2
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from fryday.config import Settings
from fryday.tracing import generation

Message = ChatCompletionMessageParam


class LLMClient:
    def __init__(self, settings: Settings, http_client: httpx2.Client | None = None) -> None:
        # http_client: tests inject a fake network here; None = the SDK's normal client.
        self._settings = settings
        self._client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value(),
            timeout=settings.llm_timeout_s,
            # No hidden retries (the SDK default is 2, with backoff). A voice turn
            # can't wait out several timeouts, the local server won't recover in
            # milliseconds, and the caller (turn manager, v5) decides what to do.
            max_retries=0,
            http_client=http_client,
        )

    def stream(self, messages: list[Message]) -> Generator[str, None, None]:
        """Yield pieces of the reply as the server generates them (SSE under the hood).

        A generator, so callers can .close() it to stop a reply early (barge-in, v5).
        Each call is traced as a Langfuse generation (see tracing.generation).
        """
        s = self._settings
        params = {
            "temperature": s.llm_temperature,
            "top_p": s.llm_top_p,
            "top_k": s.llm_top_k,
            "max_tokens": s.llm_max_tokens,
        }
        # Inside the generator, so it starts on the first next() and ends when the
        # stream finishes, fails (ERROR, re-raised) or is closed (stopped_early).
        with generation("llm.stream", model=s.llm_model, params=params, input=messages) as gen:
            chunks = self._client.chat.completions.create(
                model=s.llm_model,
                messages=messages,
                stream=True,
                temperature=s.llm_temperature,
                top_p=s.llm_top_p,
                max_tokens=s.llm_max_tokens,
                # Ask for exact token counts in a final chunk (mlx_lm.server supports it).
                stream_options={"include_usage": True},
                # top_k isn't in the OpenAI spec; mlx_lm.server and vLLM read it from the body.
                extra_body={"top_k": s.llm_top_k},
            )
            for chunk in chunks:
                if chunk.usage:  # the final chunk: no choices, just token counts
                    gen.set_usage(chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
                # The first chunk carries only the role; later ones carry text deltas.
                if chunk.choices and (text := chunk.choices[0].delta.content):
                    gen.chunk(text)  # first one marks time-to-first-token
                    yield text
