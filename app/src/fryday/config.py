"""Runtime configuration. Every field can be overridden with a FRYDAY_<NAME> env var or .env."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FRYDAY_", env_file=".env", extra="ignore")

    # LLM server: anything that speaks the OpenAI chat-completions contract.
    llm_base_url: str = "http://127.0.0.1:8080/v1"
    llm_model: str = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    # mlx_lm.server ignores the key; vLLM on the GPU box requires one (v13).
    llm_api_key: SecretStr = SecretStr("not-needed")
    # Applies to connecting and to the gap between streamed chunks, so a stalled
    # server fails fast instead of hanging the turn.
    llm_timeout_s: float = 30.0
    # Qwen3-4B-Instruct-2507's published defaults (its generation_config.json).
    # top_k cuts the long tail of unlikely tokens; see EXPERIMENTS 2026-10-10.
    llm_temperature: float = 0.7
    llm_top_p: float = 0.8
    llm_top_k: int = 20
    llm_max_tokens: int = 512

    # None = auto: pretty logs in a terminal, JSON otherwise. true forces JSON.
    log_json: bool | None = None

    # Messages kept after the system prompt; older ones are dropped so the
    # prompt never outgrows the context window.
    max_history_messages: int = 20
    system_prompt: str = "You are Fryday, a helpful assistant. Reply in short, friendly Hinglish."
