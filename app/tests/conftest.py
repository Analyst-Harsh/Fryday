import os

import pytest


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: os.PathLike[str]) -> None:
    """Every test sees Settings' code defaults: no developer .env, no FRYDAY_* env vars."""
    monkeypatch.chdir(tmp_path)  # .env is looked up relative to the working directory
    for name in os.environ:
        if name.startswith("FRYDAY_"):
            monkeypatch.delenv(name)
