"""Layer 2 of the output normaliser (HLD §3.1): rewrite the forms espeak-ng reads wrong.

Kokoro's Hindi G2P is espeak-ng (through misaki), which already reads plain numbers well
(99, 150000 → एक लाख पचास हज़ार, 50%). So we only fix what it breaks (EXPERIMENTS
2026-10-20) and leave the remaining digits to espeak.
"""

import re

_INDIAN_COMMA = re.compile(r"(?<=\d),(?=\d)")
_MONEY = re.compile(
    r"(?:₹|\bRs\.?|\bINR)\s?(\d+)(?:\.(\d+))?"
    r"(\s?(?:लाख|करोड़|हज़ार|हजार))?(?:\s?(?:रुपये|रुपए|rupees?))?",
    re.IGNORECASE,
)
_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b(?:\s?(?:बजे|baje))?")
_DECIMAL = re.compile(r"\b(\d+)\.(\d+)\b")


def _half(n: int) -> str:
    """n and a half: 1.5 and 2.5 have their own words, the rest are साढ़े n."""
    return {0: "आधा", 1: "डेढ़", 2: "ढाई"}.get(n, f"साढ़े {n}")


def _number(whole: str, frac: str | None) -> str:
    if not frac:
        return whole
    if frac.rstrip("0") == "5":
        return _half(int(whole))
    return f"{whole} दशमलव {' '.join(frac)}"  # digits after the point are read one by one


def _money(m: re.Match[str]) -> str:
    unit = f" {m[3].strip()}" if m[3] else ""
    return f"{_number(m[1], m[2])}{unit} रुपये"


def _time(m: re.Match[str]) -> str:
    h, mins = int(m[1]), int(m[2])
    if h > 23 or mins > 59:
        return m[0]
    h = h % 12 or 12  # ponytail: no सुबह/शाम for 24h times; add when tools produce them (v1)
    if mins == 0:
        return f"{h} बजे"
    if mins == 15:
        return f"सवा {h} बजे"
    if mins == 30:
        return f"{_half(h)} बजे"
    if mins == 45:
        return f"पौने {h % 12 + 1} बजे"
    return f"{h} बजकर {mins} मिनट"


def normalise(text: str) -> str:
    text = _INDIAN_COMMA.sub("", text)
    text = _MONEY.sub(_money, text)
    text = _TIME.sub(_time, text)
    return _DECIMAL.sub(lambda m: _number(m[1], m[2]), text)
