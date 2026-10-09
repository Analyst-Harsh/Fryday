# /// script
# requires-python = ">=3.12"
# dependencies = ["tokenizers", "huggingface_hub"]
# ///
"""F session 3 toy: how many tokens does the same sentence cost in each script?

Uses Fryday's real LLM tokenizer (Qwen3-4B-Instruct-2507). Downloads only
tokenizer.json (a few MB), no weights. Throwaway learning code.

Run: uv run spikes/learn/f3_tokens.py
"""

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

tok = Tokenizer.from_file(hf_hub_download("Qwen/Qwen3-4B-Instruct-2507", "tokenizer.json"))

SENTENCES = {
    "English": "Set a reminder for 7 tomorrow morning",
    "Roman Hinglish": "Kal subah 7 baje ka reminder laga do",
    "Devanagari": "कल सुबह 7 बजे का रिमाइंडर लगा दो",
}

for name, text in SENTENCES.items():
    enc = tok.encode(text)
    words = len(text.split())
    print(f"\n{name}: {text}")
    print(
        f"  chars={len(text)}  utf8_bytes={len(text.encode())}  words={words}  "
        f"tokens={len(enc.ids)}  tokens/word={len(enc.ids) / words:.2f}"
    )
    # Decode each token on its own to see where BPE split the text.
    print("  pieces:", " | ".join(tok.decode([i]) for i in enc.ids))
