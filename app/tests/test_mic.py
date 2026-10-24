import tracemalloc
from collections.abc import Callable, Iterator

import numpy as np
from fryday.asr import SAMPLE_RATE
from fryday.mic import MAX_UTTERANCE_S, VAD_FRAME, endpoint, problem_with, vad_loop


def frames(n: int) -> Iterator[np.ndarray]:
    """n frames whose samples are their own index, so a cut shows exactly where it fell."""
    for i in range(n):
        yield np.arange(i * VAD_FRAME, (i + 1) * VAD_FRAME, dtype=np.int32).astype("<i2")


def fake_vad(
    events: dict[int, dict[str, int | float]],
) -> Callable[[np.ndarray], dict[str, int | float] | None]:
    calls = iter(range(10**9))
    return lambda _frame: events.get(next(calls))


def cut(
    frames: Iterator[np.ndarray], vad: Callable[[np.ndarray], dict[str, int | float] | None]
) -> bytes:
    pcm = endpoint(frames, vad)
    assert pcm is not None, "frames ran out before the utterance ended"
    return pcm


def test_endpoint_cuts_from_start_to_end_event():
    pcm = cut(frames(20), fake_vad({3: {"start": 1000}, 10: {"end": 5000}}))
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
    assert endpoint(frames(50), fake_vad({})) is None


def test_endpoint_caps_an_utterance_that_never_ends():
    cap_frames = MAX_UTTERANCE_S * SAMPLE_RATE // VAD_FRAME + 2
    pcm = cut(frames(cap_frames + 100), fake_vad({0: {"start": 0}}))
    assert len(pcm) // 2 <= MAX_UTTERANCE_S * SAMPLE_RATE + VAD_FRAME


def test_endpoint_cuts_correctly_after_a_long_idle_wait():
    # Silero-shaped indices after 10,000 frames (5+ min) of silence: start reaches back
    # the 100 ms pad plus one frame, which must still be in the trimmed buffer.
    speech_frame = 10_000
    start = (speech_frame + 1) * VAD_FRAME - 1600 - VAD_FRAME
    end = start + 8000
    events: dict[int, dict[str, int | float]] = {speech_frame: {"start": start}}
    events[speech_frame + 20] = {"end": end}
    samples = np.frombuffer(cut(frames(speech_frame + 100), fake_vad(events)), "<i2")
    assert np.array_equal(samples, np.arange(start, end, dtype=np.int32).astype("<i2"))


def test_endpoint_memory_stays_flat_while_idle():
    tracemalloc.start()
    endpoint(frames(10_000), fake_vad({}))  # 10 MB of audio if every frame were kept
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 100_000


def test_problem_with_rejects_a_stray_key_press_and_a_silent_mic() -> None:
    loud = np.full(SAMPLE_RATE, 3000, dtype="<i2")
    assert problem_with(loud.tobytes()) is None
    assert problem_with(loud[: SAMPLE_RATE // 10].tobytes()) == "too_short"
    assert problem_with(bytes(SAMPLE_RATE * 2)) == "silent"  # all zeros


def test_vad_loop_reports_each_utterance_and_returns_when_frames_run_out():
    # Every VAD sees a start on its 2nd frame and an end on its 5th: utterances of 4 frames.
    speeches: list[int] = []
    utterances: list[bytes | None] = []

    def new_vad() -> Callable[[np.ndarray], dict[str, int | float] | None]:
        calls = iter(range(10**9))
        events: dict[int, dict[str, int | float]] = {
            1: {"start": VAD_FRAME},
            4: {"end": 5 * VAD_FRAME},
        }
        return lambda _frame: events.get(next(calls))

    vad_loop(frames(12), new_vad, lambda: speeches.append(1), utterances.append)

    # 12 frames: two whole utterances, then 2 frames with a start but no end before the
    # frames ran out (the socket closed): a third start, but no partial utterance.
    assert len(speeches) == 3
    assert len(utterances) == 2
    assert None not in utterances
