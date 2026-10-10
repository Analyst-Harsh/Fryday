"""Runtime configuration. Every field can be overridden with a FRYDAY_<NAME> env var or .env."""

from typing import Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # populate_by_name: code/tests can use field names even where a field reads an alias.
    model_config = SettingsConfigDict(
        env_prefix="FRYDAY_", env_file=".env", extra="ignore", populate_by_name=True
    )

    # LLM server: anything that speaks the OpenAI chat-completions contract.
    llm_base_url: str = "http://127.0.0.1:8080/v1"
    llm_model: str = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    # mlx_lm.server ignores the key; vLLM on the GPU box requires one (v13).
    llm_api_key: SecretStr = SecretStr("not-needed")
    # Applies to connecting and to the gap between streamed chunks, so a stalled
    # server fails fast instead of hanging the turn.
    llm_timeout_s: float = Field(default=30.0, gt=0)
    # Qwen3-4B-Instruct-2507's published defaults (its generation_config.json).
    # top_k cuts the long tail of unlikely tokens; see EXPERIMENTS 2026-10-10.
    # Bounds reject bad values at startup instead of mid-conversation.
    llm_temperature: float = Field(default=0.7, ge=0, le=2)
    llm_top_p: float = Field(default=0.8, gt=0, le=1)
    llm_top_k: int = Field(default=20, ge=0)  # 0 = no top-k filter
    llm_max_tokens: int = Field(default=512, ge=1)

    # None = auto: pretty logs in a terminal, JSON otherwise. true forces JSON.
    log_json: bool | None = None

    # Tracing to Langfuse Cloud (EU). These read Langfuse's standard env names
    # (no FRYDAY_ prefix) so the same .env works with Langfuse's own tooling.
    langfuse_tracing_enabled: bool = Field(
        default=False, validation_alias="LANGFUSE_TRACING_ENABLED"
    )
    langfuse_public_key: str = Field(default="", validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: SecretStr = Field(
        default=SecretStr(""), validation_alias="LANGFUSE_SECRET_KEY"
    )
    langfuse_base_url: str = Field(
        default="https://cloud.langfuse.com", validation_alias="LANGFUSE_BASE_URL"
    )
    langfuse_environment: str = Field(
        default="dev", validation_alias="LANGFUSE_TRACING_ENVIRONMENT"
    )
    # Send prompt/reply text with traces. Dev only: it is user data (HLD §7);
    # masking is required before any non-dev use (v7b).
    trace_content: bool = False

    # Messages kept after the system prompt; older ones are dropped so the
    # prompt never outgrows the context window.
    max_history_messages: int = Field(default=20, ge=0)
    system_prompt: str = "You are Fryday, a helpful assistant. Reply in short, friendly Hinglish."

    @model_validator(mode="after")
    def _langfuse_keys_present_when_enabled(self) -> Self:
        if self.langfuse_tracing_enabled and not (
            self.langfuse_public_key and self.langfuse_secret_key.get_secret_value()
        ):
            raise ValueError(
                "LANGFUSE_TRACING_ENABLED=true needs LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY"
            )
        return self
