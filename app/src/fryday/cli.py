"""v0: a terminal chat with Fryday. Streams replies and keeps conversation history.

Run the model server first (`make llm`), then `make chat`.
"""

import time
import uuid

import openai
import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars

from fryday.config import Settings
from fryday.llm import LLMClient, Message
from fryday.logs import setup_logging

log = structlog.stdlib.get_logger(__name__)


def trim_history(history: list[Message], max_messages: int) -> list[Message]:
    """Keep the system prompt plus the newest `max_messages`, starting at a user turn.

    The system prompt stays first and unchanged, so the server's prefix cache keeps hitting.
    """
    system, rest = history[:1], history[1:]
    tail = rest[-max_messages:] if max_messages > 0 else []
    while tail and tail[0]["role"] != "user":  # never open on an orphaned assistant reply
        tail = tail[1:]
    return system + tail


def main() -> None:
    settings = Settings()
    setup_logging(settings.log_json)
    llm = LLMClient(settings)
    history: list[Message] = [{"role": "system", "content": settings.system_prompt}]
    print("Fryday v0 — type a message. /reset clears history, Ctrl-D quits.")

    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user:
            continue
        if user == "/reset":
            history = history[:1]
            print("(history cleared)")
            continue

        history.append({"role": "user", "content": user})
        history = trim_history(history, settings.max_history_messages)
        clear_contextvars()
        bind_contextvars(turn_id=uuid.uuid4().hex[:8])  # on every log line of this turn
        t0 = time.perf_counter()
        ttft: float | None = None
        pieces: list[str] = []

        print("fryday> ", end="", flush=True)
        try:
            for piece in llm.stream(history):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                pieces.append(piece)
                print(piece, end="", flush=True)
        except openai.APIError:
            # User gets a short message; the log gets the details.
            log.exception("llm_failed")
            print("\n(Fryday couldn't reach the model. Is `make llm` running?)")
            history.pop()  # drop the unanswered user turn
            continue
        print()

        history.append({"role": "assistant", "content": "".join(pieces)})
        total = time.perf_counter() - t0
        log.info(
            "turn",
            ttft_s=round(ttft, 3) if ttft is not None else None,
            total_s=round(total, 3),
            chunks=len(pieces),
            history_messages=len(history),
        )


if __name__ == "__main__":
    main()
