"""Tests for the reply → speech plumbing: sentence chunking and speak_reply, with fakes."""

import asyncio
from collections.abc import AsyncIterator, Iterator

from fryday.voice import heard_text, sentences, speak_reply


def test_splits_on_hindi_and_latin_sentence_ends() -> None:
    deltas = ["नमस्ते। कैसे ", "हो? ठीक हूँ! Bye. "]
    assert list(sentences(deltas)) == ["नमस्ते।", "कैसे हो?", "ठीक हूँ!", "Bye."]


def test_sentence_split_across_deltas_is_yielded_once_complete() -> None:
    assert list(sentences(["कल सु", "बह जि", "म है।", " ठीक"])) == ["कल सुबह जिम है।", "ठीक"]


def test_decimal_point_does_not_end_a_sentence() -> None:
    assert list(sentences(["Meeting 7.", "30 pe hai. OK"])) == ["Meeting 7.30 pe hai.", "OK"]


def test_line_break_ends_a_sentence_and_blanks_are_skipped() -> None:
    assert list(sentences(["पहली लाइन\n\n", "दूसरी लाइन", "  "])) == ["पहली लाइन", "दूसरी लाइन"]


def test_each_sentence_is_yielded_before_the_rest_of_the_reply_arrives() -> None:
    # The point of chunking: TTS can start on sentence 1 while the LLM is still writing.
    pulled: list[str] = []

    def deltas() -> Iterator[str]:
        for d in ["एक। ", "दो। ", "तीन।"]:
            pulled.append(d)
            yield d

    chunks = sentences(deltas())
    assert next(chunks) == "एक।"
    assert pulled == ["एक। "]  # the space after "।" arrived with it: nothing more was read


def words(text: str) -> list[str]:
    """Stream text the way the LLM does: one word (plus its space) per delta."""
    return [w + " " for w in text.split()]


def test_first_chunk_is_cut_after_six_words_when_there_is_no_break() -> None:
    # The first audio waits only for 6 words, not the whole sentence (EXPERIMENTS 2026-10-21).
    reply = "मैं आपको कल सुबह सात बजे याद दिला दूँगा। ठीक है?"
    assert list(sentences(words(reply))) == [
        "मैं आपको कल सुबह सात बजे",
        "याद दिला दूँगा।",
        "ठीक है?",
    ]


def test_first_chunk_is_cut_at_a_comma_after_three_words() -> None:
    reply = "आज मौसम बादलों में था, ठंढा लग रहा है। कल धूप होगी।"
    assert list(sentences(words(reply))) == [
        "आज मौसम बादलों में था,",
        "ठंढा लग रहा है।",
        "कल धूप होगी।",
    ]


def test_short_first_sentence_is_not_cut() -> None:
    assert list(sentences(words("हाँ, मैं यह कर सकता हूँ। बताओ।"))) == [
        "हाँ, मैं यह कर सकता हूँ।",
        "बताओ।",
    ]


def test_first_chunk_never_splits_an_amount_from_its_unit() -> None:
    # "₹2.5 | लाख" would make the normaliser say "ढाई रुपये… लाख": a wrong amount.
    reply = "इस फ़ोन की कीमत ₹2.5 लाख रुपये है, जो काफ़ी ज़्यादा है।"
    assert next(sentences(words(reply))) == "इस फ़ोन की कीमत ₹2.5 लाख रुपये"


async def stream(deltas: list[str]) -> AsyncIterator[str]:
    for delta in deltas:
        yield delta


def test_speak_reply_speaks_each_sentence_and_plays_in_order() -> None:
    spoken: list[str] = []
    played: list[bytes] = []

    async def speak(sentence: str) -> list[bytes]:
        spoken.append(sentence)
        return [sentence.encode()]

    reply = asyncio.run(speak_reply(stream(["नमस्ते। ", "कैसे हो?"]), speak, played.append))

    assert spoken == ["नमस्ते।", "कैसे हो?"]
    assert played == ["नमस्ते।".encode(), "कैसे हो?".encode()]
    assert reply.text == "नमस्ते। कैसे हो?"
    assert reply.first_token_s is not None
    assert reply.first_audio_s is not None
    assert reply.first_token_s <= reply.first_audio_s


def test_speak_reply_with_empty_reply_plays_nothing() -> None:
    played: list[bytes] = []

    async def speak(_sentence: str) -> list[bytes]:
        return [b"x"]

    reply = asyncio.run(speak_reply(stream([]), speak, played.append))

    assert played == []
    assert reply.text == ""
    assert reply.first_token_s is None
    assert reply.first_audio_s is None


def test_heard_text_keeps_whole_chunks_then_a_share_of_the_cut_one() -> None:
    chunks = [("Kal dhoop rahegi.", 100), ("Shaam ko halki baarish hogi.", 200)]
    assert heard_text(chunks, 300) == "Kal dhoop rahegi. Shaam ko halki baarish hogi."
    assert heard_text(chunks, 180) == "Kal dhoop rahegi. Shaam ko"  # 80/200 of 5 words
    assert heard_text(chunks, 50) == "Kal"  # 50/100 of 3 words, rounded down
    assert heard_text(chunks, 0) == ""
