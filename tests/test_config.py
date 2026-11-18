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
        "OPENROUTER_API_KEY",
        "RATE_LIMIT_ENABLED",
        "RAG_TOP_K",
        "RAG_MODEL",
        "RAG_CHEAP_MODEL",
        "RAG_FALLBACK_MODEL",
        "RAG_FALLBACK_MODELS",
        "RAG_THINK",
        "RAG_RERANK",
        "RAG_RERANK_MODEL",
        "RAG_RERANK_CANDIDATES",
        "RAG_HYBRID",
        "PER_IP_PER_HOUR",
        "DAILY_CAP",
        "MAX_QUESTION_CHARS",
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
    assert s.openrouter_api_key == ""
    assert s.model == "glm-5.3-flash"
    assert s.cheap_model == "glm-5.3-flash"
    assert s.fallback_chain == []
    assert s.llm_think is False
    assert s.llm_api_key == ""
    assert s.llm_key_env == "OLLAMA_API_KEY"


def test_default_fallback_chain_when_both_keys_are_present(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.delenv("RAG_FALLBACK_MODELS", raising=False)
    monkeypatch.delenv("RAG_FALLBACK_MODEL", raising=False)

    assert load_settings().fallback_chain == [
        ("ollama", "deepseek-v4.1-flash:1017"),
        ("openrouter", "inclusionai/ling-3.2-flash:free"),
        ("openrouter", "nvidia/nemotron-3.5-super-vl:free"),
    ]


def test_anthropic_provider_uses_claude_defaults(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("RAG_MODEL", raising=False)
    monkeypatch.delenv("RAG_CHEAP_MODEL", raising=False)
    monkeypatch.delenv("RAG_FALLBACK_MODEL", raising=False)
    monkeypatch.delenv("RAG_FALLBACK_MODELS", raising=False)
    s = load_settings()
    assert s.llm_provider == "anthropic"
    assert s.model == "claude-sonnet-5-5"
    assert s.cheap_model == "claude-haiku-5-5"
    assert s.fallback_chain == []
    assert s.llm_key_env == "ANTHROPIC_API_KEY"


def test_openrouter_provider_uses_the_free_ling_defaults(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.delenv("RAG_MODEL", raising=False)
    monkeypatch.delenv("RAG_CHEAP_MODEL", raising=False)
    monkeypatch.delenv("RAG_FALLBACK_MODEL", raising=False)
    monkeypatch.delenv("RAG_FALLBACK_MODELS", raising=False)
    s = load_settings()
    assert s.llm_provider == "openrouter"
    assert s.model == "inclusionai/ling-3.2-flash:free"
    assert s.cheap_model == "inclusionai/ling-3.2-flash:free"
    assert s.llm_key_env == "OPENROUTER_API_KEY"


def test_openrouter_provider_reads_the_openrouter_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-key")
    s = load_settings()
    assert s.llm_api_key == "openrouter-key"


def test_openrouter_default_fallback_chain(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-key")
    monkeypatch.delenv("RAG_FALLBACK_MODELS", raising=False)
    monkeypatch.delenv("RAG_FALLBACK_MODEL", raising=False)

    assert load_settings().fallback_chain == [
        ("openrouter", "nvidia/nemotron-3.5-super-vl:free"),
        ("ollama", "glm-5.3-flash"),
    ]


def test_openrouter_default_chain_drops_ollama_without_a_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("RAG_FALLBACK_MODELS", raising=False)
    monkeypatch.delenv("RAG_FALLBACK_MODEL", raising=False)

    assert load_settings().fallback_chain == [
        ("openrouter", "nvidia/nemotron-3.5-super-vl:free")
    ]


def test_fallback_chain_splits_on_the_first_colon(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv(
        "RAG_FALLBACK_MODELS",
        "ollama:deepseek-v4.1-flash:1017,"
        "openrouter:inclusionai/ling-3.2-flash:free",
    )

    assert load_settings().fallback_chain == [
        ("ollama", "deepseek-v4.1-flash:1017"),
        ("openrouter", "inclusionai/ling-3.2-flash:free"),
    ]


def test_legacy_fallback_model_uses_the_active_provider(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-key")
    monkeypatch.delenv("RAG_FALLBACK_MODELS", raising=False)
    monkeypatch.setenv("RAG_FALLBACK_MODEL", "backup-model:1")

    assert load_settings().fallback_chain == [("ollama", "backup-model:1")]


def test_think_env_override(monkeypatch):
    monkeypatch.setenv("RAG_THINK", "yes")
    assert load_settings().llm_think is True


@pytest.mark.parametrize("value", ["none", "NONE", "off", "OFF"])
def test_fallback_chain_can_be_disabled(monkeypatch, value):
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("RAG_FALLBACK_MODELS", value)
    assert load_settings().fallback_chain == []


@pytest.mark.parametrize("value", ["none", "NONE", "off", "OFF"])
def test_legacy_fallback_model_can_be_disabled(monkeypatch, value):
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-key")
    monkeypatch.delenv("RAG_FALLBACK_MODELS", raising=False)
    monkeypatch.setenv("RAG_FALLBACK_MODEL", value)
    assert load_settings().fallback_chain == []


def test_fallback_entries_without_a_key_are_dropped(monkeypatch):
    monkeypatch.setenv(
        "RAG_FALLBACK_MODELS",
        "ollama:deepseek-v4.1-flash:1017,"
        "openrouter:inclusionai/ling-3.2-flash:free",
    )
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert load_settings().fallback_chain == []

    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-key")
    assert load_settings().fallback_chain == [
        ("ollama", "deepseek-v4.1-flash:1017")
    ]

    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    assert load_settings().fallback_chain == [
        ("openrouter", "inclusionai/ling-3.2-flash:free")
    ]


def test_unknown_fallback_provider_raises(monkeypatch):
    monkeypatch.setenv("RAG_FALLBACK_MODELS", "openai:gpt-4o")

    with pytest.raises(ValueError, match="unknown fallback provider"):
        load_settings()


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
    s = load_settings()
    assert s.database_url == "postgresql://x/y"
    assert s.top_k == 9
    assert s.rerank_enabled is True
    assert s.rerank_model == "Xenova/ms-marco-MiniLM-L-6-v2"
    assert s.rerank_candidates == 30
    assert s.hybrid_enabled is True
    assert s.rate_limit_enabled is True
    assert s.per_ip_per_hour == 3


def test_demo_lane_uses_the_database(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "1")
    assert load_settings().rate_limit_enabled is True


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
def test_think_flag_accepts_truthy_values(monkeypatch, value):
    monkeypatch.setenv("RAG_THINK", value)
    assert load_settings().llm_think is True


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
