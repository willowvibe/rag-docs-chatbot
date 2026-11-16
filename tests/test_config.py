import pytest

from ragchat.config import load_settings


def test_defaults(monkeypatch):
    for key in (
        "DATABASE_URL",
        "ANTHROPIC_API_KEY",
        "RATE_LIMIT_ENABLED",
        "RAG_TOP_K",
        "RAG_MODEL",
        "RAG_CHEAP_MODEL",
        "PER_IP_PER_HOUR",
        "DAILY_CAP",
        "MAX_QUESTION_CHARS",
        "UPSTASH_REDIS_REST_URL",
        "UPSTASH_REDIS_REST_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)
    s = load_settings()
    assert s.top_k == 5
    assert s.rate_limit_enabled is False
    assert s.model == "claude-sonnet-5-5"
    assert s.cheap_model == "claude-haiku-5-5"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x/y")
    monkeypatch.setenv("RAG_TOP_K", "9")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "1")
    monkeypatch.setenv("PER_IP_PER_HOUR", "3")
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "https://example.invalid")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "t")
    s = load_settings()
    assert s.database_url == "postgresql://x/y"
    assert s.top_k == 9
    assert s.rate_limit_enabled is True
    assert s.per_ip_per_hour == 3
    assert s.upstash_redis_rest_url == "https://example.invalid"
    assert s.upstash_redis_rest_token == "t"


def test_demo_lane_requires_redis(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "1")
    monkeypatch.delenv("UPSTASH_REDIS_REST_URL", raising=False)
    monkeypatch.delenv("UPSTASH_REDIS_REST_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="UPSTASH_REDIS_REST_URL"):
        load_settings()


def test_missing_key_is_not_fatal(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = load_settings()
    assert s.anthropic_api_key == ""
