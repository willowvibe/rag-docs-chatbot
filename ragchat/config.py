"""Every environment read in the project happens here."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


DEFAULT_MODELS = {
    "ollama": ("glm-5.3-flash", "glm-5.3-flash"),
    "anthropic": ("claude-sonnet-5-5", "claude-haiku-5-5"),
}

DEFAULT_FALLBACK_CHAIN = {
    "ollama": [
        ("ollama", "deepseek-v4.1-flash:1017"),
        ("openrouter", "inclusionai/ling-3.2-flash:free"),
        ("openrouter", "nvidia/nemotron-3.5-super-vl:free"),
    ],
    "anthropic": [],
}

FALLBACK_PROVIDERS = {"ollama", "openrouter", "anthropic"}

DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-12-v2"


@dataclass(frozen=True)
class Settings:
    database_url: str
    llm_provider: str
    ollama_api_key: str
    ollama_host: str
    anthropic_api_key: str
    openrouter_api_key: str
    model: str
    cheap_model: str
    fallback_chain: list[tuple[str, str]]
    llm_think: bool
    rate_limit_enabled: bool
    per_ip_per_hour: int
    daily_cap: int
    top_k: int
    rerank_enabled: bool
    rerank_model: str
    rerank_candidates: int
    hybrid_enabled: bool
    max_question_chars: int

    @property
    def llm_api_key(self) -> str:
        if self.llm_provider == "ollama":
            return self.ollama_api_key
        if self.llm_provider == "openrouter":
            return self.openrouter_api_key
        return self.anthropic_api_key

    @property
    def llm_key_env(self) -> str:
        if self.llm_provider == "ollama":
            return "OLLAMA_API_KEY"
        if self.llm_provider == "openrouter":
            return "OPENROUTER_API_KEY"
        return "ANTHROPIC_API_KEY"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from None


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _parse_fallback_chain(raw: str) -> list[tuple[str, str]]:
    """Parse ``provider:model`` entries, splitting on the first colon only."""
    chain: list[tuple[str, str]] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        provider, separator, model = entry.partition(":")
        provider = provider.strip().lower()
        model = model.strip()
        if not separator or not model:
            raise ValueError(
                f"RAG_FALLBACK_MODELS entries must be provider:model, got {entry!r}"
            )
        if provider not in FALLBACK_PROVIDERS:
            raise ValueError(
                f"unknown fallback provider {provider!r} in RAG_FALLBACK_MODELS"
            )
        chain.append((provider, model))
    return chain


def _drop_entries_without_keys(
    chain: list[tuple[str, str]],
    *,
    ollama_api_key: str,
    openrouter_api_key: str,
    anthropic_api_key: str,
) -> list[tuple[str, str]]:
    keys = {
        "ollama": ollama_api_key,
        "openrouter": openrouter_api_key,
        "anthropic": anthropic_api_key,
    }
    return [(provider, model) for provider, model in chain if keys.get(provider)]


def load_settings() -> Settings:
    load_dotenv()
    llm_provider = os.environ.get("LLM_PROVIDER", "").strip().lower() or "ollama"
    if llm_provider not in DEFAULT_MODELS:
        raise RuntimeError(
            f"LLM_PROVIDER must be one of {sorted(DEFAULT_MODELS)}, got {llm_provider!r}"
        )
    default_model, default_cheap_model = DEFAULT_MODELS[llm_provider]
    ollama_api_key = os.environ.get("OLLAMA_API_KEY", "").strip()
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()

    fallback_chain_raw = os.environ.get("RAG_FALLBACK_MODELS", "").strip()
    legacy_fallback_raw = os.environ.get("RAG_FALLBACK_MODEL", "").strip()
    if fallback_chain_raw:
        if fallback_chain_raw.lower() in {"none", "off"}:
            fallback_chain = []
        else:
            fallback_chain = _parse_fallback_chain(fallback_chain_raw)
    elif legacy_fallback_raw:
        if legacy_fallback_raw.lower() in {"none", "off"}:
            fallback_chain = []
        else:
            fallback_chain = [(llm_provider, legacy_fallback_raw)]
    else:
        fallback_chain = list(DEFAULT_FALLBACK_CHAIN[llm_provider])
    fallback_chain = _drop_entries_without_keys(
        fallback_chain,
        ollama_api_key=ollama_api_key,
        openrouter_api_key=openrouter_api_key,
        anthropic_api_key=anthropic_api_key,
    )

    rate_limit_enabled = _env_flag("RATE_LIMIT_ENABLED", False)
    top_k = _env_int("RAG_TOP_K", 5)
    rerank_candidates = _env_int("RAG_RERANK_CANDIDATES", 15)
    if rerank_candidates < top_k:
        raise RuntimeError(
            "RAG_RERANK_CANDIDATES must be >= RAG_TOP_K, "
            f"got {rerank_candidates} and {top_k}"
        )
    return Settings(
        database_url=os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag"),
        llm_provider=llm_provider,
        ollama_api_key=ollama_api_key,
        ollama_host=(
            os.environ.get("OLLAMA_HOST", "").strip() or "https://ollama.com"
        ).rstrip("/"),
        anthropic_api_key=anthropic_api_key,
        openrouter_api_key=openrouter_api_key,
        model=os.environ.get("RAG_MODEL", "").strip() or default_model,
        cheap_model=os.environ.get("RAG_CHEAP_MODEL", "").strip() or default_cheap_model,
        fallback_chain=fallback_chain,
        llm_think=_env_flag("RAG_THINK", False),
        rate_limit_enabled=rate_limit_enabled,
        per_ip_per_hour=_env_int("PER_IP_PER_HOUR", 20),
        daily_cap=_env_int("DAILY_CAP", 500),
        top_k=top_k,
        rerank_enabled=_env_flag("RAG_RERANK", False),
        rerank_model=(
            os.environ.get("RAG_RERANK_MODEL", "").strip() or DEFAULT_RERANK_MODEL
        ),
        rerank_candidates=rerank_candidates,
        hybrid_enabled=_env_flag("RAG_HYBRID", False),
        max_question_chars=_env_int("MAX_QUESTION_CHARS", 500),
    )
