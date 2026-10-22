"""F9 toy: split Fryday's ~1.3 s "first token → first audio" into its two halves.

A. LLM: first token → first complete sentence (sentence aggregation).
B. Kokoro: that sentence → its audio (TTS time to first audio; Kokoro returns a whole
   sentence at once, so this is the full synthesis time).

Needs `make llm` running. Run: uv run python spikes/learn/f9_budget.py
"""

import statistics
import time

from fryday.config import Settings
from fryday.llm import LLMClient, Message
from fryday.normalise import normalise
from fryday.tts import TTSClient
from fryday.voice import sentences

QUESTIONS = [
    "आज मौसम कैसा है?",
    "मुझे कल सुबह सात बजे याद दिलाना।",
    "एक अच्छी किताब बताओ।",
    "चाय बनाने का आसान तरीका बताओ।",
    "तुम कौन हो?",
]


def main() -> None:
    settings = Settings()
    llm, tts = LLMClient(settings), TTSClient(settings)
    list(tts.speak("नमस्ते।"))  # warm-up

    rows = []
    for q in [QUESTIONS[0], *QUESTIONS]:  # first run is a warm-up, dropped below
        history: list[Message] = [
            {"role": "system", "content": settings.voice_system_prompt},
            {"role": "user", "content": q},
        ]
        t0 = time.perf_counter()
        first_token = None

        def deltas(history: list[Message] = history):
            nonlocal first_token
            for d in llm.stream(history):
                if first_token is None:
                    first_token = time.perf_counter()
                yield d

        sentence = next(sentences(deltas()))
        t_sentence = time.perf_counter()
        audio = b"".join(tts.speak(normalise(sentence)))
        t_audio = time.perf_counter()
        assert first_token is not None
        rows.append((first_token - t0, t_sentence - first_token, t_audio - t_sentence,
                     len(audio) / 2 / tts.sample_rate, len(sentence.split()), sentence))

    rows = rows[1:]
    for ttft, agg, synth, dur, words, s in rows:
        print(f"ttft {ttft*1000:4.0f}  aggregate {agg*1000:5.0f}  kokoro {synth*1000:5.0f} ms"
              f"  ({words} words, {dur:.1f} s audio)  {s}")
    med = lambda i: statistics.median(r[i] for r in rows) * 1000  # noqa: E731
    print(f"\nmedian: ttft {med(0):.0f} ms | A. aggregation {med(1):.0f} ms | "
          f"B. kokoro {med(2):.0f} ms | first token → first audio {med(1) + med(2):.0f} ms")


if __name__ == "__main__":
    main()
