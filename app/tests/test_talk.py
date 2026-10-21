import tracemalloc
from collections.abc import Callable, Iterator

import numpy as np
from fryday.asr import SAMPLE_RATE
from fryday.talk import MAX_UTTERANCE_S, VAD_FRAME, endpoint


def frames(n: int) -> Iterator[np.ndarray]:
    """n frames whose samples are their own index, so a cut shows exactly where it fell."""
    for i in range(n):
        yield np.arange(i * VAD_FRAME, (i + 1) * VAD_FRAME, dtype=np.int32).astype("<i2")


def fake_vad(
    events: dict[int, dict[str, int | float]],
) -> Callable[[np.ndarray], dict[str, int | float] | None]:
    calls = iter(range(10**9))
    return lambda _frame: events.get(next(calls))


def test_endpoint_cuts_from_start_to_end_event():
    pcm = endpoint(frames(20), fake_vad({3: {"start": 1000}, 10: {"end": 5000}}))
    samples = np.frombuffer(pcm, dtype="<i2")
    assert samples[0] == 1000
    assert samples[-1] == 4999


def test_endpoint_stops_after_end_without_reading_more_frames():
    consumed: list[int] = []

    def counting() -> Iterator[np.ndarray]:
        for i, f in enumerate(frames(1000)):
            consumed.append(i)
            yield f

    endpoint(counting(), fake_vad({2: {"start": 0}, 5: {"end": 3000}}))
    assert len(consumed) == 6  # the turn ends on the end event, not when the mic does


def test_endpoint_ignores_silence_with_no_speech():
    assert endpoint(frames(50), fake_vad({})) == b""


def test_endpoint_caps_an_utterance_that_never_ends():
    cap_frames = MAX_UTTERANCE_S * SAMPLE_RATE // VAD_FRAME + 2
    pcm = endpoint(frames(cap_frames + 100), fake_vad({0: {"start": 0}}))
    assert len(pcm) // 2 <= MAX_UTTERANCE_S * SAMPLE_RATE + VAD_FRAME


def test_endpoint_cuts_correctly_after_a_long_idle_wait():
    # Silero-shaped indices after 10,000 frames (5+ min) of silence: start reaches back
    # the 100 ms pad plus one frame, which must still be in the trimmed buffer.
    speech_frame = 10_000
    start = (speech_frame + 1) * VAD_FRAME - 1600 - VAD_FRAME
    end = start + 8000
    events: dict[int, dict[str, int | float]] = {speech_frame: {"start": start}}
    events[speech_frame + 20] = {"end": end}
    samples = np.frombuffer(endpoint(frames(speech_frame + 100), fake_vad(events)), "<i2")
    assert np.array_equal(samples, np.arange(start, end, dtype=np.int32).astype("<i2"))


def test_endpoint_memory_stays_flat_while_idle():
    tracemalloc.start()
    endpoint(frames(10_000), fake_vad({}))  # 10 MB of audio if every frame were kept
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 100_000
