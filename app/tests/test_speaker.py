import threading
from collections.abc import Callable

import pytest
from fryday import speaker as speaker_module
from fryday.speaker import Speaker


class FakeOutput:
    """Stands in for sd.RawOutputStream: records each write instead of playing it."""

    latency = 0.02

    def __init__(self, **_kwargs: object) -> None:
        self.writes: list[bytes] = []
        self.on_write: Callable[[int], None] = lambda _n: None

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...

    def write(self, data: bytes) -> None:
        self.writes.append(data)
        self.on_write(len(self.writes))


def test_speaker_drops_a_cancelled_turn_mid_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    out = FakeOutput()

    def output(**_kwargs: object) -> FakeOutput:
        return out

    monkeypatch.setattr(speaker_module.sd, "RawOutputStream", output)
    speaker = Speaker(sample_rate=1000)  # 50 ms pieces = 100 bytes
    speaker.allow(1)
    done = threading.Event()

    def on_write(n: int) -> None:
        if n == 3:
            speaker.allow(2)  # barge-in at piece 3 of turn 1
        if n == 5:
            done.set()

    out.on_write = on_write
    speaker.play(b"\x01" * 1000, turn=1)  # 10 pieces
    speaker.play(b"\x02" * 200, turn=2)
    assert done.wait(timeout=5)
    speaker.close()

    assert [w[0] for w in out.writes] == [1, 1, 1, 2, 2]
    assert speaker.played(2) == 200
