"""Test-wide defaults.

Tests must be hermetic: nothing here may depend on a developer `.env`, a shell variable
or a live API key. `load_settings()` reads the environment on every call (it does not
cache), and `load_dotenv()` never overrides an existing variable, so setting the limiter
off here is sufficient. Limiter tests that need it on set `RATE_LIMIT_ENABLED=1` in
their own bodies, which runs after this fixture.
"""

import pytest


@pytest.fixture(autouse=True)
def _rate_limit_off(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "0")
