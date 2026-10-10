"""The LLM seam: the only module that talks to the model server.

Anything speaking the OpenAI chat-completions contract works behind it:
mlx_lm.server on the Mac today, vLLM on the GPU box later (v13).
"""

from collections.abc import Iterator

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from fryday.config import Settings

Message = ChatCompletionMessageParam


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value(),
            timeout=settings.llm_timeout_s,
            # No automatic retries: re-sending after half a reply has streamed
            # would duplicate text. The caller decides what to do on failure.
            max_retries=0,
        )

    def stream(self, messages: list[Message]) -> Iterator[str]:
        """Yield pieces of the reply as the server generates them (SSE under the hood)."""
        s = self._settings
        chunks = self._client.chat.completions.create(
            model=s.llm_model,
            messages=messages,
            stream=True,
            temperature=s.llm_temperature,
            top_p=s.llm_top_p,
            max_tokens=s.llm_max_tokens,
            # top_k isn't in the OpenAI spec; mlx_lm.server and vLLM both read it from the body.
            extra_body={"top_k": s.llm_top_k},
        )
        for chunk in chunks:
            # The first chunk carries only the role; later ones carry text deltas.
            if chunk.choices and (text := chunk.choices[0].delta.content):
                yield text
