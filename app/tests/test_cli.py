from fryday.cli import trim_history
from fryday.llm import Message

SYS: Message = {"role": "system", "content": "s"}


def u(n: int) -> Message:
    return {"role": "user", "content": f"u{n}"}


def a(n: int) -> Message:
    return {"role": "assistant", "content": f"a{n}"}


def test_short_history_is_untouched() -> None:
    history = [SYS, u(1), a(1), u(2)]
    assert trim_history(history, 10) == history


def test_keeps_system_and_newest_messages() -> None:
    history = [SYS, u(1), a(1), u(2), a(2), u(3)]
    assert trim_history(history, 3) == [SYS, u(2), a(2), u(3)]


def test_never_starts_on_assistant_reply() -> None:
    # The newest 2 would be [a(2), u(3)]: the orphaned reply is dropped.
    history = [SYS, u(1), a(1), u(2), a(2), u(3)]
    assert trim_history(history, 2) == [SYS, u(3)]


def test_zero_keeps_only_system() -> None:
    assert trim_history([SYS, u(1)], 0) == [SYS]
