"""The speaker side of a voice chat: ordered playback that barge-in can cut short."""

import queue
import threading

import sounddevice as sd
import structlog

log = structlog.stdlib.get_logger(__name__)
PLAY_PIECE_S = 0.05  # written in pieces this long, so barge-in can stop mid-sentence


class Speaker:
    """Plays PCM16 chunks in order on a background thread, so play() returns at once
    and the next sentence can render while this one is heard.

    Each chunk carries its turn's id. allow(n) makes the thread drop every chunk of an
    older turn, mid-chunk too, so after barge-in the audio stops within one piece
    (PLAY_PIECE_S) plus the device's own buffer (latency_s).
    """

    def __init__(self, sample_rate: int) -> None:
        self._chunks: queue.Queue[tuple[int, bytes] | None] = queue.Queue()
        self._stream = sd.RawOutputStream(samplerate=sample_rate, channels=1, dtype="int16")
        self._stream.start()
        self.latency_s = float(self._stream.latency)  # pyright: ignore[reportArgumentType]
        self._piece = int(sample_rate * PLAY_PIECE_S) * 2  # bytes: 2 per sample
        self._turn = 0  # one int, written by the event loop and read here: atomic in CPython
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def play(self, pcm: bytes, turn: int) -> None:
        self._chunks.put((turn, pcm))

    def allow(self, turn: int) -> None:
        """Play only chunks of this turn from now on."""
        self._turn = turn

    def close(self) -> None:
        self.allow(-1)  # no turn has this id: drop whatever is still queued
        self._chunks.put(None)
        self._thread.join()
        self._stream.stop()
        self._stream.close()

    def _run(self) -> None:
        while (item := self._chunks.get()) is not None:
            turn, chunk = item
            try:
                for i in range(0, len(chunk), self._piece):
                    if turn != self._turn:
                        break  # barge-in: this turn was cancelled
                    # Blocks until the device has room, which paces playback.
                    self._stream.write(chunk[i : i + self._piece])  # pyright: ignore[reportUnknownMemberType]
            except Exception:  # keep playing later turns; a dead device shows up in the log
                log.exception("speaker_failed")
