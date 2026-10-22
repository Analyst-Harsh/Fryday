"""The microphone side of a voice chat: listeners that turn mic audio into utterances.

A listener runs forever on its own thread and reports through two callbacks:
on_speech when the user starts talking (barge-in) and on_utterance with each finished
recording, or None when the user quits.
"""

import queue
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from functools import partial
from typing import cast

import numpy as np
from silero_vad import VADIterator, load_silero_vad

from fryday.asr import SAMPLE_RATE
from fryday.config import Settings

OnSpeech = Callable[[], None]
OnUtterance = Callable[[bytes | None], None]

MIN_SPEECH_S = 0.3  # shorter than this is a stray key press, not speech
SILENT_PEAK = 100  # of 32767 (about -50 dBFS): macOS records pure zeros without mic permission
VAD_FRAME = 512  # samples (32 ms at 16 kHz): the only frame size Silero accepts at 16 kHz
VAD_PAD_MS = 100  # kept before and after speech so first and last syllables aren't clipped
MAX_UTTERANCE_S = 30  # hands-free stops here even if the VAD never hears silence
# Frames kept while waiting for speech: Silero's start reaches back the pad plus one frame.
PRE_SPEECH_FRAMES = VAD_PAD_MS * SAMPLE_RATE // 1000 // VAD_FRAME + 2


def make_listener(
    settings: Settings, on_speech: OnSpeech, on_utterance: OnUtterance
) -> Callable[[], None]:
    """The listener settings ask for, ready to run on a thread."""
    if settings.hands_free:
        print(f"Fryday voice, hands-free ({settings.vad_min_silence_ms} ms). Ctrl-C quits.")
        # Silero ships untyped; the model is only handed back to its own VADIterator.
        model = cast(object, load_silero_vad())
        return partial(
            listen_hands_free, model, settings.vad_min_silence_ms, on_speech, on_utterance
        )
    print("Fryday voice — use headphones. Enter to talk, Ctrl-D quits.")
    return partial(listen_push_to_talk, on_speech, on_utterance)


def problem_with(pcm: bytes) -> str | None:
    """Why this recording isn't worth transcribing, or None if it is."""
    if len(pcm) < MIN_SPEECH_S * SAMPLE_RATE * 2:  # 2 bytes per sample
        return "(too short, try again)"
    if np.abs(np.frombuffer(pcm, dtype="<i2")).max() < SILENT_PEAK:
        return "(silence: check System Settings > Privacy > Microphone for your terminal)"
    return None


def listen_push_to_talk(on_speech: OnSpeech, on_utterance: OnUtterance) -> None:
    """Forever: Enter starts and stops each recording; Ctrl-D quits.

    Pressing Enter while Fryday is talking interrupts it, like speaking does hands-free.
    """
    while True:
        try:
            input("\n⏎  press Enter and speak ")
            on_speech()
            pcm = record_until_enter()
        except EOFError:
            on_utterance(None)
            return
        on_utterance(pcm)


def record_until_enter() -> bytes:
    """Record mono PCM16 at 16 kHz from the default mic until Enter is pressed."""
    import sounddevice as sd  # here, not at the top: it loads PortAudio, which CI lacks

    frames: list[bytes] = []

    def on_audio(data: memoryview, _frames: int, _time: object, _status: object) -> None:
        frames.append(bytes(data))  # PortAudio's thread; list.append is thread-safe

    # macOS resamples the mic's native 48 kHz to the 16 kHz we ask for.
    with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=on_audio):
        input("🎙  listening… press Enter to stop ")
    return b"".join(frames)


def listen_hands_free(
    model: object, min_silence_ms: int, on_speech: OnSpeech, on_utterance: OnUtterance
) -> None:
    """Forever: cut utterances out of the open mic with Silero VAD.

    on_speech fires at every VAD start, also while Fryday is talking: that's barge-in.
    """
    import sounddevice as sd  # see record_until_enter

    chunks: queue.Queue[bytes] = queue.Queue()

    def on_audio(data: memoryview, _frames: int, _time: object, _status: object) -> None:
        chunks.put(bytes(data))  # PortAudio's thread; Queue is thread-safe

    def frames() -> Iterator[np.ndarray]:
        while True:
            yield np.frombuffer(chunks.get(), dtype="<i2")

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=VAD_FRAME, callback=on_audio
    ):
        print("🎙  listening… just speak, and pause to send ")
        while True:
            # A fresh iterator per utterance resets the model's state and sample counter.
            vad = VADIterator(
                model,
                sampling_rate=SAMPLE_RATE,
                min_silence_duration_ms=min_silence_ms,
                speech_pad_ms=VAD_PAD_MS,
            )

            def detect(frame: np.ndarray, vad: VADIterator = vad) -> dict[str, int | float] | None:
                # VADIterator turns the float32 array into a tensor itself.
                event = vad(frame.astype(np.float32) / 32768)
                if event and "start" in event:
                    # ponytail: any speech interrupts, even a cough; add a min-speech
                    # or semantic check (production gap #32) if that annoys.
                    on_speech()
                return event

            on_utterance(endpoint(frames(), detect))


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
