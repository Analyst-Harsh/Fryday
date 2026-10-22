import asyncio

from fryday.turns import TurnManager


def test_turn_manager_new_turn_cancels_the_old_one_and_silences_it() -> None:
    allowed: list[int] = []
    started: list[int] = []

    async def run_turn(_pcm: bytes, turn_id: int) -> None:
        started.append(turn_id)
        await asyncio.sleep(10)  # still speaking

    async def scenario() -> tuple[asyncio.Task[None], asyncio.Task[None]]:
        manager = TurnManager(run_turn, allowed.append)
        first = manager.start(b"one")
        await asyncio.sleep(0)
        second = manager.start(b"two")
        await asyncio.sleep(0)
        manager.interrupt()  # barge-in during the second turn
        await asyncio.gather(first, second, return_exceptions=True)
        return first, second

    first, second = asyncio.run(scenario())
    assert first.cancelled()
    assert second.cancelled()
    assert started == [1, 2]
    assert allowed == [1, 2, 3]  # each id silences every older turn's audio
