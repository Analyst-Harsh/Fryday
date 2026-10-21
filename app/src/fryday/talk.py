"""Talk to Fryday: push-to-talk voice chat in the terminal (mic → ASR → LLM → TTS → speaker).

Run the model server first (`make llm`), then `make talk`, or `FRYDAY_HANDS_FREE=1 make talk`
to end each turn on silence (Silero VAD) instead of Enter. Use headphones: there is
no echo cancellation. Barge-in and the browser come later in v5.
"""

import queue
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from typing import cast

import numpy as np
import openai
import sounddevice as sd
import structlog
from silero_vad import VADIterator, load_silero_vad
from structlog.contextvars import bind_contextvars, clear_contextvars

from fryday.asr import SAMPLE_RATE, ASRClient
from fryday.cli import trim_history
from fryday.config import Settings
from fryday.llm import LLMClient, Message
from fryday.logs import setup_logging
from fryday.normalise import normalise
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
VAD_FRAME = 512  # samples (32 ms at 16 kHz): the only frame size Silero accepts at 16 kHz
VAD_PAD_MS = 100  # kept before and after speech so first and last syllables aren't clipped
MAX_UTTERANCE_S = 30  # hands-free stops here even if the VAD never hears silence
# Frames kept while waiting for speech: Silero's start reaches back the pad plus one frame.
PRE_SPEECH_FRAMES = VAD_PAD_MS * SAMPLE_RATE // 1000 // VAD_FRAME + 2


def record_until_enter() -> bytes:
    """Record mono PCM16 at 16 kHz from the default mic until Enter is pressed."""
    frames: list[bytes] = []

    def on_audio(data: memoryview, _frames: int, _time: object, _status: object) -> None:
        frames.append(bytes(data))  # PortAudio's thread; list.append is thread-safe

    # macOS resamples the mic's native 48 kHz to the 16 kHz we ask for.
    with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=on_audio):
        input("🎙  listening… press Enter to stop ")
    return b"".join(frames)


def endpoint(
    frames: Iterable[np.ndarray], vad: Callable[[np.ndarray], dict[str, int | float] | None]
) -> bytes:
    """Cut one utterance out of a stream of VAD_FRAME-sample int16 frames, from the
    VAD's start event to its end event (sample indices, padding included)."""
    audio: deque[np.ndarray] = deque()
    dropped = 0  # samples trimmed off the front while idle; VAD indices count them
    start: int | None = None  # index into `audio`, not the VAD's turn-wide count
    for frame in frames:
        audio.append(frame)
        event = vad(frame) or {}
        if "start" in event:
            start = max(0, int(event["start"]) - dropped)
        if start is None:
            if len(audio) > PRE_SPEECH_FRAMES:  # idle: memory stays flat however long we wait
                audio.popleft()
                dropped += VAD_FRAME
            continue
        end = int(event["end"]) - dropped if "end" in event else None
        if end is None and len(audio) * VAD_FRAME - start >= MAX_UTTERANCE_S * SAMPLE_RATE:
            end = len(audio) * VAD_FRAME  # never heard an end (TV, fan): send what we have
        if end is not None:
            return np.concatenate(audio)[start:end].tobytes()
    return b""


def record_until_silence(model: object, min_silence_ms: int) -> bytes:
    """Record from the mic until Silero VAD hears speech and then min_silence_ms of silence."""
    chunks: queue.Queue[bytes] = queue.Queue()

    def on_audio(data: memoryview, _frames: int, _time: object, _status: object) -> None:
        chunks.put(bytes(data))  # PortAudio's thread; Queue is thread-safe

    def frames() -> Iterator[np.ndarray]:
        while True:
            yield np.frombuffer(chunks.get(), dtype="<i2")

    # A fresh iterator per turn resets the model's state and the sample counter.
    vad = VADIterator(
        model,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=VAD_PAD_MS,
    )
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=VAD_FRAME, callback=on_audio
    ):
        print("🎙  listening… just speak, and pause to send ")
        # VADIterator turns the float32 array into a tensor itself.
        return endpoint(frames(), lambda f: vad(f.astype(np.float32) / 32768))


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
        # Also raised on timeout (e.g. the server is still downloading or loading a model).
        print("Model server unreachable or too slow. Run `make llm` and wait until it's loaded.")
        return
    speaker = Speaker(tts.sample_rate)
    # Silero ships untyped; the model is only handed back to its own VADIterator.
    vad_model = cast(object, load_silero_vad()) if settings.hands_free else None
    # Hands-free, the turn only ends after this much silence: part of the user's wait.
    vad_wait_s = settings.vad_min_silence_ms / 1000 if vad_model else 0.0

    history: list[Message] = [{"role": "system", "content": settings.voice_system_prompt}]
    session_id = uuid.uuid4().hex
    if vad_model:
        print(f"Fryday voice, hands-free ({settings.vad_min_silence_ms} ms). Ctrl-C quits.")
    else:
        print("Fryday voice — use headphones. Enter to talk, Ctrl-D quits.")

    try:
        while True:
            try:
                if vad_model:
                    print()
                    pcm = record_until_silence(vad_model, settings.vad_min_silence_ms)
                else:
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
                spoken = normalise(sentence)  # layer 2 safety net: ₹, times, decimals
                with span("tts", input=spoken):
                    yield from tts.speak(spoken)

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
                    "vad_wait_s": vad_wait_s,
                }
                record(turn, output=reply.text, metadata=stats)
                log.info("voice_turn", **stats)
                heard_s = vad_wait_s + asr_s + (reply.first_audio_s or 0)
                print(
                    f"        ⏱  silence {_ms(vad_wait_s)} → ASR {asr_s:.2f} s"
                    f" → first token +{_ms(reply.first_token_s)}"
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
