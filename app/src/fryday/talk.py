"""Talk to Fryday: push-to-talk voice chat in the terminal (mic → ASR → LLM → TTS → speaker).

Run the model server first (`make llm`), then `make talk`. Use headphones: there is
no echo cancellation. VAD endpointing, barge-in and the browser come in v5.
"""

import queue
import threading
import time
import uuid
from collections.abc import Iterator

import numpy as np
import openai
import sounddevice as sd
import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars

from fryday.asr import SAMPLE_RATE, ASRClient
from fryday.cli import trim_history
from fryday.config import Settings
from fryday.llm import LLMClient, Message
from fryday.logs import setup_logging
from fryday.tracing import (
    content,
    init_tracing,
    record,
    shutdown_tracing,
    span,
    trace_root,
)
from fryday.tts import TTSClient
from fryday.voice import speak_reply

log = structlog.stdlib.get_logger(__name__)
MIN_SPEECH_S = 0.3  # shorter than this is a stray key press, not speech
SILENT_PEAK = 100  # of 32767 (about -50 dBFS): macOS records pure zeros without mic permission


def record_until_enter() -> bytes:
    """Record mono PCM16 at 16 kHz from the default mic until Enter is pressed."""
    frames: list[bytes] = []

    def on_audio(data: memoryview, _frames: int, _time: object, _status: object) -> None:
        frames.append(bytes(data))  # PortAudio's thread; list.append is thread-safe

    # macOS resamples the mic's native 48 kHz to the 16 kHz we ask for.
    with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=on_audio):
        input("🎙  listening… press Enter to stop ")
    return b"".join(frames)


class Speaker:
    """Plays PCM16 chunks in order on a background thread, so play() returns at once
    and the next sentence can render while this one is heard."""

    def __init__(self, sample_rate: int) -> None:
        self._chunks: queue.Queue[bytes | None] = queue.Queue()
        self._stream = sd.RawOutputStream(samplerate=sample_rate, channels=1, dtype="int16")
        self._stream.start()
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while (chunk := self._chunks.get()) is not None:
            try:
                if self._error is None:  # after a failure, just drain so wait() returns
                    # Blocks until the device has room, which paces playback.
                    self._stream.write(chunk)  # pyright: ignore[reportUnknownMemberType]
            except Exception as e:  # handed to the main thread in wait()
                self._error = e
            finally:
                self._chunks.task_done()
        self._chunks.task_done()

    def play(self, pcm: bytes) -> None:
        self._chunks.put(pcm)

    def wait(self) -> None:
        """Block until every queued chunk has been handed to the device; re-raise its error."""
        self._chunks.join()
        if (error := self._error) is not None:
            self._error = None
            raise error

    def close(self) -> None:
        self._chunks.put(None)
        self._thread.join()
        self._stream.stop()
        self._stream.close()


def main() -> None:
    settings = Settings()
    setup_logging(settings.log_json)
    init_tracing(settings)
    llm = LLMClient(settings)
    print("Loading Whisper and Kokoro…")
    asr, tts = ASRClient(settings), TTSClient(settings)
    # First calls compile GPU kernels; pay that now, not on the first turn.
    asr.transcribe(bytes(SAMPLE_RATE * 2), SAMPLE_RATE)  # 1 s of silence
    list(tts.speak("नमस्ते।"))
    # Also warms the server's prefix cache for the voice system prompt (smoke run: first
    # token 8.5 s cold vs 0.1 s warm). Fails fast here if `make llm` isn't running.
    warm: list[Message] = [
        {"role": "system", "content": settings.voice_system_prompt},
        {"role": "user", "content": "नमस्ते"},
    ]
    try:
        for _ in llm.stream(warm):
            pass
    except openai.APIError:
        print("Can't reach the model server. Run `make llm` in another terminal first.")
        return
    speaker = Speaker(tts.sample_rate)

    history: list[Message] = [{"role": "system", "content": settings.voice_system_prompt}]
    session_id = uuid.uuid4().hex
    print("Fryday voice — use headphones. Enter to talk, Ctrl-D quits.")

    try:
        while True:
            try:
                input("\n⏎  press Enter and speak ")
                pcm = record_until_enter()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if len(pcm) < MIN_SPEECH_S * SAMPLE_RATE * 2:  # 2 bytes per sample
                print("(too short, try again)")
                continue

            if np.abs(np.frombuffer(pcm, dtype="<i2")).max() < SILENT_PEAK:
                print("(silence: check System Settings > Privacy > Microphone for your terminal)")
                continue

            clear_contextvars()
            bind_contextvars(turn_id=uuid.uuid4().hex[:8], session_id=session_id)

            def say(sentence: str) -> Iterator[bytes]:
                print(f"fryday> {sentence}")
                with span("tts", input=sentence):
                    yield from tts.speak(sentence)

            # One trace per turn: asr, llm.stream and each tts sentence are its children.
            with trace_root("voice_turn", session_id=session_id) as turn:
                t0 = time.perf_counter()
                with span("asr") as asr_span:
                    user = asr.transcribe(pcm, SAMPLE_RATE)
                    record(asr_span, output=user)
                asr_s = time.perf_counter() - t0
                if not user:
                    print("(heard nothing, try again)")
                    record(turn, error="empty_transcript")
                    continue
                turn.update(input=content(user))  # through the privacy gate, like trace_root
                print(f"you>    {user}")
                history.append({"role": "user", "content": user})
                history = trim_history(history, settings.max_history_messages)
                try:
                    reply = speak_reply(llm.stream(history), say, speaker.play)
                except openai.APIError:
                    log.exception("llm_failed")
                    print("(Fryday couldn't reach the model. Is `make llm` running?)")
                    history.pop()  # drop the unanswered user turn
                    record(turn, error="llm_failed")
                    continue
                speaker.wait()
                history.append({"role": "assistant", "content": reply.text})
                stats = {
                    "asr_s": round(asr_s, 3),
                    "llm_first_token_s": _round(reply.first_token_s),
                    "first_audio_s": _round(reply.first_audio_s),
                    "speech_in_s": round(len(pcm) / 2 / SAMPLE_RATE, 2),
                }
                record(turn, output=reply.text, metadata=stats)
                log.info("voice_turn", **stats)
                heard_s = asr_s + (reply.first_audio_s or 0)
                print(
                    f"        ⏱  ASR {asr_s:.2f} s → first token +{_ms(reply.first_token_s)}"
                    f" → first audio +{_ms(reply.first_audio_s)}"
                    f"  (you stop talking → Fryday speaks: {heard_s:.2f} s)"
                )
    finally:
        speaker.close()
        shutdown_tracing()


def _round(seconds: float | None) -> float | None:
    return round(seconds, 3) if seconds is not None else None


def _ms(seconds: float | None) -> str:
    return f"{seconds * 1000:.0f} ms" if seconds is not None else "—"


if __name__ == "__main__":
    main()
