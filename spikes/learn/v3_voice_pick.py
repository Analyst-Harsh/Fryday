"""v3 build 3: pick Kokoro's Hindi voice by ear, blind (half of S0-5).

Writes the same 3 sentences (through normalise(), as talk.py does) in each Hindi voice
to data/audio/v3_voice_{A..D}.wav under shuffled letters. The key goes to
data/audio/v3_voice_key.txt: score the files 1-5 before opening it.

Run: uv run python spikes/learn/v3_voice_pick.py
"""

import random
import wave
from pathlib import Path

from fryday.config import Settings
from fryday.normalise import normalise
from fryday.tts import TTSClient

VOICES = ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"]
SENTENCES = [
    "नमस्ते! मैं फ्राइडे हूँ, आपकी मदद के लिए हमेशा तैयार।",
    "कल सुबह 7:30 बजे आपकी मीटिंग है, टाइम पर पहुँच जाना।",
    "ये फ़ोन ₹2.5 लाख का नहीं, सिर्फ़ ₹15,999 का है।",
]
OUT = Path("data/audio")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    letters = list("ABCD")
    random.shuffle(letters)  # unseeded on purpose: you shouldn't know the mapping
    key = []
    for voice, letter in zip(VOICES, letters, strict=True):
        tts = TTSClient(Settings(tts_voice=voice))
        gap = b"\x00\x00" * (tts.sample_rate // 2)  # 0.5 s of silence between sentences
        pcm = gap.join(b"".join(tts.speak(normalise(s))) for s in SENTENCES)
        with wave.open(str(OUT / f"v3_voice_{letter}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(tts.sample_rate)
            w.writeframes(pcm)
        key.append(f"{letter} = {voice}")
        print(f"wrote v3_voice_{letter}.wav")
    (OUT / "v3_voice_key.txt").write_text("\n".join(sorted(key)) + "\n")


if __name__ == "__main__":
    main()
