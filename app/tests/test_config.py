import pytest
from fryday.config import Settings
from pydantic import ValidationError


@pytest.mark.parametrize(
    ("env_var", "bad_value"),
    [
        ("FRYDAY_LLM_TEMPERATURE", "-1"),
        ("FRYDAY_LLM_TOP_P", "0"),
        ("FRYDAY_LLM_MAX_TOKENS", "0"),
        ("FRYDAY_LLM_TIMEOUT_S", "abc"),
    ],
)
def test_bad_config_fails_at_startup(
    monkeypatch: pytest.MonkeyPatch, env_var: str, bad_value: str
) -> None:
    monkeypatch.setenv(env_var, bad_value)
    with pytest.raises(ValidationError):
        Settings()


def test_langfuse_enabled_without_keys_fails_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    with pytest.raises(ValidationError, match="LANGFUSE_PUBLIC_KEY"):
        Settings()


def test_langfuse_fields_read_standard_env_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-x")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    s = Settings()
    assert s.langfuse_public_key == "pk-lf-x"
    assert s.langfuse_tracing_enabled is True
