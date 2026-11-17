import pytest

from ragchat.config import load_settings


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    """Exercise the environment only: a developer's .env must not repopulate it."""
    monkeypatch.setattr("ragchat.config.load_dotenv", lambda *args, **kwargs: None)


def test_defaults(monkeypatch):
    for key in (
        "DATABASE_URL",
        "LLM_PROVIDER",
        "OLLAMA_API_KEY",
        "OLLAMA_HOST",
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
    assert s.llm_provider == "ollama"
    assert s.ollama_api_key == ""
    assert s.ollama_host == "https://ollama.com"
    assert s.model == "gpt-oss:120b"
    assert s.cheap_model == "gpt-oss:20b"
    assert s.llm_api_key == ""
    assert s.llm_key_env == "OLLAMA_API_KEY"


def test_anthropic_provider_uses_claude_defaults(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("RAG_MODEL", raising=False)
    monkeypatch.delenv("RAG_CHEAP_MODEL", raising=False)
    s = load_settings()
    assert s.llm_provider == "anthropic"
    assert s.model == "claude-sonnet-5-5"
    assert s.cheap_model == "claude-haiku-5-5"
    assert s.llm_key_env == "ANTHROPIC_API_KEY"


def test_llm_key_follows_the_active_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    s = load_settings()
    assert s.llm_api_key == "anthropic-key"
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    s = load_settings()
    assert s.llm_api_key == "ollama-key"


def test_invalid_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(RuntimeError, match="LLM_PROVIDER"):
        load_settings()


def test_ollama_host_strips_a_trailing_slash(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "https://example.invalid/")
    assert load_settings().ollama_host == "https://example.invalid"


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
    with pytest.raises(RuntimeError, match="UPSTASH_REDIS_REST_URL"):
        load_settings()


def test_non_integer_env_value_raises(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "abc")
    with pytest.raises(RuntimeError, match="RAG_TOP_K"):
        load_settings()


def test_missing_key_is_not_fatal(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    s = load_settings()
    assert s.ollama_api_key == ""
    assert s.llm_api_key == ""
