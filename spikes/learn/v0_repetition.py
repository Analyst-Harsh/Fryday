# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""v0 learn toy: how often does the reply fall into a repetition loop, per sampling config?

Needs `make llm` running. Run: uv run spikes/learn/v0_repetition.py   (throwaway learning code)
"""

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="x", timeout=60)
MESSAGES = [
    {"role": "system", "content": "You are Fryday, a helpful assistant. Reply in short, friendly Hinglish."},
    {"role": "user", "content": "Namaste! Mera naam Harshit hai."},
    {"role": "assistant", "content": "Namaste Harshit! Main Fryday hoon. Kya karna hai?"},
    {"role": "user", "content": "Mujhe chai pasand hai, coffee nahi."},
]
CONFIGS = {
    "ours: top_p 0.9, no top_k": {"top_p": 0.9},
    "Qwen recommended: top_p 0.8, top_k 20": {"top_p": 0.8, "extra_body": {"top_k": 20}},
    "recommended + presence_penalty 1.0": {
        "top_p": 0.8, "presence_penalty": 1.0, "extra_body": {"top_k": 20},
    },
}
RUNS, MAX_TOKENS = 6, 300

for name, kwargs in CONFIGS.items():
    loops, lengths = 0, []
    for _ in range(RUNS):
        r = client.chat.completions.create(
            model="mlx-community/Qwen3-4B-Instruct-2507-4bit", messages=MESSAGES,
            temperature=0.7, max_tokens=MAX_TOKENS, **kwargs,
        )
        n = r.usage.completion_tokens if r.usage else 0
        lengths.append(n)
        loops += n >= MAX_TOKENS  # hit the cap = never stopped on its own
    print(f"{name:42s} loops {loops}/{RUNS}   reply tokens {lengths}")
