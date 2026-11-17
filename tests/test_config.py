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
        "RAG_FALLBACK_MODEL",
        "RAG_RERANK",
        "RAG_RERANK_MODEL",
        "RAG_RERANK_CANDIDATES",
        "RAG_HYBRID",
        "PER_IP_PER_HOUR",
        "DAILY_CAP",
        "MAX_QUESTION_CHARS",
        "UPSTASH_REDIS_REST_URL",
        "UPSTASH_REDIS_REST_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)
    s = load_settings()
    assert s.top_k == 5
    assert s.rerank_enabled is False
    assert s.rerank_model == "Xenova/ms-marco-MiniLM-L-12-v2"
    assert s.rerank_candidates == 15
    assert s.hybrid_enabled is False
    assert s.rate_limit_enabled is False
    assert s.llm_provider == "ollama"
    assert s.ollama_api_key == ""
    assert s.ollama_host == "https://ollama.com"
    assert s.model == "glm-5.3-flash"
    assert s.cheap_model == "glm-5.3-flash"
    assert s.fallback_model == "deepseek-v4.1-flash:1017"
    assert s.llm_api_key == ""
    assert s.llm_key_env == "OLLAMA_API_KEY"


def test_anthropic_provider_uses_claude_defaults(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("RAG_MODEL", raising=False)
    monkeypatch.delenv("RAG_CHEAP_MODEL", raising=False)
    monkeypatch.delenv("RAG_FALLBACK_MODEL", raising=False)
    s = load_settings()
    assert s.llm_provider == "anthropic"
    assert s.model == "claude-sonnet-5-5"
    assert s.cheap_model == "claude-haiku-5-5"
    assert s.fallback_model == ""
    assert s.llm_key_env == "ANTHROPIC_API_KEY"


def test_fallback_model_env_override(monkeypatch):
    monkeypatch.setenv("RAG_FALLBACK_MODEL", "backup-model:1")
    assert load_settings().fallback_model == "backup-model:1"


@pytest.mark.parametrize("value", ["none", "NONE", "off", "OFF"])
def test_fallback_model_can_be_disabled(monkeypatch, value):
    monkeypatch.setenv("RAG_FALLBACK_MODEL", value)
    assert load_settings().fallback_model == ""


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
    monkeypatch.setenv("RAG_RERANK", "yes")
    monkeypatch.setenv("RAG_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
    monkeypatch.setenv("RAG_RERANK_CANDIDATES", "30")
    monkeypatch.setenv("RAG_HYBRID", "yes")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "1")
    monkeypatch.setenv("PER_IP_PER_HOUR", "3")
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "https://example.invalid")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "t")
    s = load_settings()
    assert s.database_url == "postgresql://x/y"
    assert s.top_k == 9
    assert s.rerank_enabled is True
    assert s.rerank_model == "Xenova/ms-marco-MiniLM-L-6-v2"
    assert s.rerank_candidates == 30
    assert s.hybrid_enabled is True
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


def test_rerank_candidates_must_cover_top_k(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "10")
    monkeypatch.setenv("RAG_RERANK_CANDIDATES", "5")
    with pytest.raises(RuntimeError, match="RAG_RERANK_CANDIDATES"):
        load_settings()


def test_non_integer_rerank_candidates_raises(monkeypatch):
    monkeypatch.setenv("RAG_RERANK_CANDIDATES", "abc")
    with pytest.raises(RuntimeError, match="RAG_RERANK_CANDIDATES"):
        load_settings()


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_rerank_flag_accepts_truthy_values(monkeypatch, value):
    monkeypatch.setenv("RAG_RERANK", value)
    assert load_settings().rerank_enabled is True


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_hybrid_flag_accepts_truthy_values(monkeypatch, value):
    monkeypatch.setenv("RAG_HYBRID", value)
    assert load_settings().hybrid_enabled is True


def test_missing_key_is_not_fatal(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    s = load_settings()
    assert s.ollama_api_key == ""
    assert s.llm_api_key == ""
