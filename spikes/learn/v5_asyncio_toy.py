"""v5 learn 1 toy: an async voice pipeline with fakes. Backpressure, barge-in, and the blocking trap.

fake LLM (a word every 30 ms) → chunker → [sentence queue] → fake TTS (200 ms of blocking
"maths") → [audio queue, maxsize=1] → fake speaker (1 s per chunk). A fake mic ticks every 50 ms
and reports when it was starved. At 1.4 s the "user speaks", and the reply is cancelled.

Run twice: TTS via asyncio.to_thread (right), then TTS called directly (the trap).
Run: uv run python spikes/learn/v5_asyncio_toy.py
"""

import asyncio
import time

REPLY = "पहला वाक्य यहाँ है। दूसरा वाक्य यहाँ है। तीसरा वाक्य यहाँ है। चौथा वाक्य यहाँ है। पाँचवाँ वाक्य यहाँ है।"
BARGE_IN_S = 1.4
t0 = 0.0


def log(who: str, msg: str) -> None:
    print(f"  {time.perf_counter() - t0:5.2f}s  {who:<8} {msg}")


def fake_synth(sentence: str) -> str:
    time.sleep(0.2)  # stands in for Kokoro: CPU/GPU work with no `await` inside
    log("tts", f"done   {sentence.split()[0]}")
    return sentence


async def run(blocking: bool) -> None:
    global t0
    t0 = time.perf_counter()
    sentence_q: asyncio.Queue[str | None] = asyncio.Queue()
    audio_q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=1)  # bounded: backpressure

    async def llm_and_chunker() -> None:
        buf = []
        for word in REPLY.split():
            await asyncio.sleep(0.03)  # waiting for the next token: other tasks run
            buf.append(word)
            if word.endswith("।"):
                await sentence_q.put(" ".join(buf))
                buf = []
        await sentence_q.put(None)

    async def tts() -> None:
        while (s := await sentence_q.get()) is not None:
            log("tts", f"start  {s.split()[0]}")
            audio = fake_synth(s) if blocking else await asyncio.to_thread(fake_synth, s)
            if audio_q.full():
                log("tts", "audio queue full → waiting (backpressure)")
            await audio_q.put(audio)
        await audio_q.put(None)

    async def speaker() -> None:
        while (a := await audio_q.get()) is not None:
            log("speaker", f"play   {a.split()[0]}")
            await asyncio.sleep(1.0)  # playing 1 s of audio

    async def reply() -> None:
        async with asyncio.TaskGroup() as tg:
            for job in (llm_and_chunker(), tts(), speaker()):
                tg.create_task(job)

    async def mic() -> None:
        last = time.perf_counter()
        while True:
            await asyncio.sleep(0.05)
            now = time.perf_counter()
            if now - last > 0.1:
                log("mic", f"⚠ STALLED {(now - last) * 1000:.0f} ms (no audio read, VAD blind)")
            last = now

    mic_task = asyncio.create_task(mic())
    reply_task = asyncio.create_task(reply())
    await asyncio.sleep(BARGE_IN_S)
    late = time.perf_counter() - t0 - BARGE_IN_S
    log("user", f"starts speaking → cancel reply (timer fired {late * 1000:.0f} ms late)")
    t_cancel = time.perf_counter()
    reply_task.cancel()
    try:
        await reply_task
    except asyncio.CancelledError:
        pass
    log("reply", f"cancelled: audio stopped {(time.perf_counter() - t_cancel) * 1000:.1f} ms "
                 f"after cancel(); {audio_q.qsize()} queued chunk(s) left to drop")
    await asyncio.sleep(0.4)  # let any orphaned TTS thread finish and report
    mic_task.cancel()


async def main() -> None:
    for blocking in (False, True):
        print(f"\n=== TTS {'called directly (blocking the loop)' if blocking else 'via to_thread'} ===")
        await run(blocking)


if __name__ == "__main__":
    asyncio.run(main())
