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
