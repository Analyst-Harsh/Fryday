"""Golden tests for the output normaliser (S0-9): what the LLM writes → what TTS gets."""

import pytest
from fryday.normalise import normalise

GOLDEN = [
    # Money: espeak drops ₹, so it becomes रुपये after the amount.
    ("₹250 दो।", "250 रुपये दो।"),
    ("₹250 रुपये", "250 रुपये"),  # the LLM's own रुपये isn't doubled
    ("Rs. 99 का है", "99 रुपये का है"),
    ("₹1,50,000", "150000 रुपये"),  # Indian commas break espeak
    ("₹2.5 लाख", "ढाई लाख रुपये"),
    ("₹99.50", "साढ़े 99 रुपये"),
    # Time: espeak reads 7:30 as "सात तीस".
    ("मीटिंग 7:30 बजे है।", "मीटिंग साढ़े 7 बजे है।"),
    ("7:00", "7 बजे"),
    ("1:30", "डेढ़ बजे"),
    ("2:30", "ढाई बजे"),
    ("9:15", "सवा 9 बजे"),
    ("12:45", "पौने 1 बजे"),
    ("18:00", "6 बजे"),
    ("7:05", "7 बजकर 5 मिनट"),
    # Decimals: espeak reads 2.5 as "दो . पाँच".
    ("0.5 किलो", "आधा किलो"),
    ("3.14", "3 दशमलव 1 4"),
    # Left for espeak, which reads these well.
    ("50% छूट", "50% छूट"),
    ("150000", "150000"),
    ("iPhone 7", "iPhone 7"),
    ("v3.5", "v3.5"),
    ("25:99", "25:99"),  # not a time
]


@pytest.mark.parametrize(("text", "spoken"), GOLDEN)
def test_golden(text: str, spoken: str) -> None:
    assert normalise(text) == spoken
