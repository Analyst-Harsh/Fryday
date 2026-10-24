"""Talk to Fryday: voice chat in the terminal (mic → ASR → LLM → TTS → speaker).

Run the model server first (`make llm`), then `make talk`, or `FRYDAY_HANDS_FREE=1 make talk`
to end each turn on silence (Silero VAD) instead of Enter. Speaking (or pressing Enter)
while Fryday talks interrupts it. Use headphones: with no echo cancellation, Fryday's
own voice would interrupt it hands-free. For the browser, see gateway.py (`make serve`).

This module wires the parts together: mic.py (listeners), turns.py (turn-taking),
voice.py (reply → speech), speaker.py (playback). VoiceSession only talks to a
voice.Channel, so the browser gateway reuses it unchanged.
"""

import asyncio
import threading
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from functools import partial
from typing import cast

import openai
import structlog
from langfuse import LangfuseSpan
from structlog.contextvars import bind_contextvars, clear_contextvars

from fryday import mic
from fryday.asr import SAMPLE_RATE, ASRClient
from fryday.cli import trim_history
from fryday.config import Settings
from fryday.llm import LLMClient, Message
from fryday.logs import setup_logging
from fryday.normalise import normalise
from fryday.speaker import PLAY_PIECE_S, Speaker
from fryday.tracing import content, init_tracing, record, shutdown_tracing, span, trace_root
from fryday.tts import TTSClient
from fryday.turns import TurnManager
from fryday.voice import NOTICES, Channel, SpokenReply, heard_text, speak_reply

log = structlog.get_logger(__name__)


@dataclass
class _LastReply:
    """The previous reply's audio, so a barge-in can trim history to what was heard."""

    turn: int
    message: Message  # the history entry, edited in place
    chunks: list[tuple[str, int]]  # (text, PCM bytes) in play order
    cut: bool  # cancelled mid-generation: text ran past the audio


class VoiceSession:
    """One conversation: its history, and how a single turn runs (ASR → LLM → TTS)."""

    def __init__(
        self,
        settings: Settings,
        asr: ASRClient,
        llm: LLMClient,
        tts: TTSClient,
        out: Channel,
        vad_wait_s: float = 0.0,
    ) -> None:
        self._settings = settings
        self._asr, self._llm, self._tts = asr, llm, tts
        self.out = out  # re-pointed when a browser reconnects; history stays
        self._history: list[Message] = [{"role": "system", "content": settings.voice_system_prompt}]
        self._session_id = uuid.uuid4().hex
        # Hands-free, the turn only ends after this much silence: part of the user's wait.
        self._vad_wait_s = vad_wait_s
        self._last: _LastReply | None = None

    async def run_turn(self, pcm: bytes, turn_id: int) -> None:
        clear_contextvars()  # the task's own copy: other turns' log context is untouched
        bind_contextvars(turn_id=turn_id, session_id=self._session_id)
        # A new turn always follows interrupt(), so the last reply's audio has stopped.
        self.settle()
        self.out.event("thinking", turn=turn_id)
        # One trace per turn: asr, llm.stream and each tts sentence are its children.
        with trace_root("voice_turn", session_id=self._session_id) as turn:
            user, asr_s = await self._transcribe(pcm)
            if not user:
                self.out.event("notice", turn=turn_id, code="heard_nothing")
                record(turn, error="empty_transcript")
                return
            turn.update(input=content(user))  # through the privacy gate, like trace_root
            self.out.event("transcript", turn=turn_id, text=user)
            self._remember({"role": "user", "content": user})

            said: list[str] = []  # filled as deltas arrive, so a cancelled turn keeps them
            chunks: list[tuple[str, int]] = []  # filled as chunks are rendered
            try:
                reply = await speak_reply(
                    self._deltas(said),
                    partial(self._speak, chunks, turn_id),
                    partial(self.out.play, turn=turn_id),
                )
            except openai.APIError:
                log.exception("llm_failed")
                self.out.event("notice", turn=turn_id, code="llm_down")
                self._history.pop()  # drop the unanswered user turn
                record(turn, error="llm_failed")
                return
            except asyncio.CancelledError:
                self._interrupted(turn, turn_id, "".join(said), chunks)
                raise
            message: Message = {"role": "assistant", "content": reply.text}
            self._remember(message)
            self._last = _LastReply(turn_id, message, chunks, cut=False)
            self._report(turn, turn_id, reply, asr_s, pcm)

    async def _transcribe(self, pcm: bytes) -> tuple[str, float]:
        start = time.perf_counter()
        with span("asr") as asr_span:
            user = await asyncio.to_thread(self._asr.transcribe, pcm, SAMPLE_RATE)
            record(asr_span, output=user)
        return user, time.perf_counter() - start

    async def _deltas(self, said: list[str]) -> AsyncIterator[str]:
        async for event in self._llm.reply(list(self._history)):
            said.append(event.text)
            yield event.text

    async def _speak(
        self, chunks: list[tuple[str, int]], turn_id: int, sentence: str
    ) -> list[bytes]:
        self.out.event("reply", turn=turn_id, text=sentence)  # here, on the loop: not in _render
        pcm = await asyncio.to_thread(self._render, sentence)  # Kokoro blocks
        chunks.append((sentence, sum(map(len, pcm))))
        return pcm

    def _render(self, sentence: str) -> list[bytes]:
        spoken = normalise(sentence)  # layer 2 safety net: ₹, times, decimals
        with span("tts", input=spoken):
            # ponytail: renders the whole chunk before playing it; stream Kokoro's
            # segments through a queue if long chunks delay first audio.
            return list(self._tts.speak(spoken))

    def _remember(self, message: Message) -> None:
        self._history.append(message)
        self._history[:] = trim_history(self._history, self._settings.max_history_messages)

    def settle(self) -> None:
        """If the last reply was cut short, keep only what was heard of it in history.

        Usually the turn had already finished (TTS renders ~20x faster than playback),
        so barge-in found only queued audio to drop, not a task to cancel.
        """
        last, self._last = self._last, None
        if last is None:
            return
        played = self.out.played(last.turn)
        if last.cut or played < sum(size for _, size in last.chunks):
            last.message["content"] = heard_text(last.chunks, played) + "…"
            log.info("reply_trimmed", heard_bytes=played)

    def _interrupted(
        self, turn: LangfuseSpan, turn_id: int, said: str, chunks: list[tuple[str, int]]
    ) -> None:
        if said:  # trimmed to what was heard when the next turn starts
            message: Message = {"role": "assistant", "content": said}
            self._remember(message)
            self._last = _LastReply(turn_id, message, chunks, cut=True)
        self.out.event("interrupted", turn=turn_id)
        log.info("barge_in", said_chars=len(said))
        record(turn, output=said, error="barge_in")

    def _report(
        self, turn: LangfuseSpan, turn_id: int, reply: SpokenReply, asr_s: float, pcm: bytes
    ) -> None:
        stats = {
            "asr_s": round(asr_s, 3),
            "llm_first_token_s": _round(reply.first_token_s),
            "first_audio_s": _round(reply.first_audio_s),
            "speech_in_s": round(len(pcm) / 2 / SAMPLE_RATE, 2),
            "vad_wait_s": self._vad_wait_s,
            # You stop talking → Fryday speaks, as measured on the server.
            "heard_s": round(self._vad_wait_s + asr_s + (reply.first_audio_s or 0), 3),
        }
        record(turn, output=reply.text, metadata=stats)
        log.info("voice_turn", **stats)
        self.out.event("done", turn=turn_id, stats=stats)


class Terminal(Speaker):
    """The terminal as a voice.Channel: Speaker plays the audio, print() shows the rest."""

    def event(self, kind: str, **fields: object) -> None:
        match kind:
            case "transcript":
                print(f"you>    {fields['text']}")
            case "reply":
                print(f"fryday> {fields['text']}")
            case "interrupted":
                print("        (interrupted)")
            case "notice":
                print(NOTICES[str(fields["code"])])
            case "done":
                print(_stats_line(cast(Mapping[str, float | None], fields["stats"])))
            case _:  # thinking: the terminal shows nothing
                pass


def _stats_line(stats: Mapping[str, float | None]) -> str:
    return (
        f"        ⏱  silence {_ms(stats['vad_wait_s'])} → ASR {_ms(stats['asr_s'])}"
        f" → first token +{_ms(stats['llm_first_token_s'])}"
        f" → first audio +{_ms(stats['first_audio_s'])}"
        f"  (you stop talking → Fryday speaks: {_ms(stats['heard_s'])})"
    )


async def converse(settings: Settings, session: VoiceSession, speaker: Terminal) -> None:
    """The turn loop: a mic thread feeds utterances in, one turn runs at a time."""
    loop = asyncio.get_running_loop()
    utterances: asyncio.Queue[bytes | None] = asyncio.Queue()
    turns = TurnManager(session.run_turn, speaker.allow)

    # The mic thread must never touch asyncio objects directly: hand calls to the loop.
    def on_speech() -> None:
        loop.call_soon_threadsafe(turns.interrupt)

    def on_utterance(pcm: bytes | None) -> None:
        loop.call_soon_threadsafe(utterances.put_nowait, pcm)

    listen = mic.make_listener(settings, on_speech, on_utterance)
    print(f"(barge-in: audio stops within {_ms(PLAY_PIECE_S + speaker.latency_s)} of the cut)")
    threading.Thread(target=listen, daemon=True).start()

    while (pcm := await utterances.get()) is not None:  # None: the user quit
        if problem := mic.problem_with(pcm):
            print(NOTICES[problem])
            continue
        turns.start(pcm)
    turns.interrupt()


def main() -> None:
    settings = Settings()
    setup_logging(settings.log_json)
    init_tracing(settings)
    llm = LLMClient(settings)
    print("Loading Whisper and Kokoro…")
    asr, tts = ASRClient(settings), TTSClient(settings)
    if not _warm_up(settings, asr, llm, tts):
        return
    speaker = Terminal(tts.sample_rate)
    vad_wait_s = settings.vad_min_silence_ms / 1000 if settings.hands_free else 0.0
    session = VoiceSession(settings, asr, llm, tts, speaker, vad_wait_s)
    try:
        asyncio.run(converse(settings, session, speaker))
    except KeyboardInterrupt:
        print()
    finally:
        speaker.close()
        shutdown_tracing()


def _warm_up(settings: Settings, asr: ASRClient, llm: LLMClient, tts: TTSClient) -> bool:
    """Pay first-call costs now, not on the first turn. False if the LLM is unreachable."""
    warm_speech(asr, tts)
    try:
        warm_llm(settings, llm)
    except openai.APIError:
        # Also raised on timeout (e.g. the server is still downloading or loading a model).
        print("Model server unreachable or too slow. Run `make llm` and wait until it's loaded.")
        return False
    return True


def warm_speech(asr: ASRClient, tts: TTSClient) -> None:
    asr.transcribe(bytes(SAMPLE_RATE * 2), SAMPLE_RATE)  # 1 s of silence: compiles kernels
    list(tts.speak("नमस्ते।"))


def warm_llm(settings: Settings, llm: LLMClient) -> None:
    """Warm the server's prefix cache for the voice system prompt (smoke run: first
    token 8.5 s cold vs 0.1 s warm). Raises openai.APIError if the server is down."""
    warm: list[Message] = [
        {"role": "system", "content": settings.voice_system_prompt},
        {"role": "user", "content": "नमस्ते"},
    ]
    for _ in llm.stream(warm):
        pass


def _round(seconds: float | None) -> float | None:
    return round(seconds, 3) if seconds is not None else None


def _ms(seconds: float | None) -> str:
    return f"{seconds * 1000:.0f} ms" if seconds is not None else "—"


if __name__ == "__main__":
    main()
