"""Test-wide defaults.

`test_api.py::test_chat_returns_answer_and_citations` exercises the happy path, which the
route only reaches when a key is configured. No test may call the model, and every chat
test replaces `api._answer` with a fake, so a dummy value here is safe and never sent
anywhere. The 503 test patches `api._settings` to an empty key explicitly.
"""

import pytest


@pytest.fixture(autouse=True)
def _dummy_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-dummy-key-not-used")
