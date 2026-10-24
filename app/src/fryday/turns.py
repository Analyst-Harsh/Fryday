"""Turn-taking: at most one turn runs, and a new one (or barge-in) cancels it.

Pure asyncio with no audio or models, so it's tested with fakes.
"""

import asyncio
from collections.abc import Callable, Coroutine

import structlog

log = structlog.stdlib.get_logger(__name__)
RunTurn = Callable[[bytes, int], Coroutine[object, object, None]]


class TurnManager:
    """One turn at a time. A new turn or a barge-in cancels the current one and silences
    its audio: the task stops at its next await, and the speaker drops the old turn's
    chunks, including any an orphaned TTS thread still finishes rendering."""

    def __init__(self, run_turn: RunTurn, allow_audio: Callable[[int], None]) -> None:
        self._run_turn = run_turn
        self._allow_audio = allow_audio
        self.turn_id = 0
        self._task: asyncio.Task[None] | None = None

    def interrupt(self) -> asyncio.Task[None] | None:
        """Silence and cancel the current turn; returns its task if one was still running,
        for callers that must wait until it has stopped (gather(..., return_exceptions))."""
        self.turn_id += 1
        self._allow_audio(self.turn_id)  # before cancel: silence is the urgent part
        if self._task is not None and not self._task.done():
            self._task.cancel()
            return self._task
        return None

    def start(self, pcm: bytes) -> asyncio.Task[None]:
        self.interrupt()
        self._task = asyncio.create_task(self._run_turn(pcm, self.turn_id))
        self._task.add_done_callback(_report_failure)
        return self._task


def _report_failure(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and (error := task.exception()) is not None:
        log.error("turn_failed", exc_info=error)
