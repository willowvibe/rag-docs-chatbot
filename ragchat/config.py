"""Every environment read in the project happens here."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


DEFAULT_MODELS = {
    "ollama": ("glm-5.3-flash", "glm-5.3-flash"),
    "anthropic": ("claude-sonnet-5-5", "claude-haiku-5-5"),
}

DEFAULT_FALLBACK_MODELS = {
    "ollama": "deepseek-v4.1-flash:1017",
    "anthropic": "",
}


@dataclass(frozen=True)
class Settings:
    database_url: str
    llm_provider: str
    ollama_api_key: str
    ollama_host: str
    anthropic_api_key: str
    model: str
    cheap_model: str
    fallback_model: str
    rate_limit_enabled: bool
    per_ip_per_hour: int
    daily_cap: int
    top_k: int
    max_question_chars: int
    upstash_redis_rest_url: str
    upstash_redis_rest_token: str

    @property
    def llm_api_key(self) -> str:
        if self.llm_provider == "ollama":
            return self.ollama_api_key
        return self.anthropic_api_key

    @property
    def llm_key_env(self) -> str:
        if self.llm_provider == "ollama":
            return "OLLAMA_API_KEY"
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


def load_settings() -> Settings:
    load_dotenv()
    llm_provider = os.environ.get("LLM_PROVIDER", "").strip().lower() or "ollama"
    if llm_provider not in DEFAULT_MODELS:
        raise RuntimeError(
            f"LLM_PROVIDER must be one of {sorted(DEFAULT_MODELS)}, got {llm_provider!r}"
        )
    default_model, default_cheap_model = DEFAULT_MODELS[llm_provider]
    fallback_raw = os.environ.get("RAG_FALLBACK_MODEL", "").strip()
    if fallback_raw.lower() in {"none", "off"}:
        fallback_model = ""
    else:
        fallback_model = fallback_raw or DEFAULT_FALLBACK_MODELS[llm_provider]
    rate_limit_enabled = _env_flag("RATE_LIMIT_ENABLED", False)
    upstash_url = os.environ.get("UPSTASH_REDIS_REST_URL", "").strip()
    upstash_token = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "").strip()
    if rate_limit_enabled and not upstash_url:
        raise RuntimeError("RATE_LIMIT_ENABLED=1 requires UPSTASH_REDIS_REST_URL to be set.")
    return Settings(
        database_url=os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag"),
        llm_provider=llm_provider,
        ollama_api_key=os.environ.get("OLLAMA_API_KEY", "").strip(),
        ollama_host=(
            os.environ.get("OLLAMA_HOST", "").strip() or "https://ollama.com"
        ).rstrip("/"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", "").strip(),
        model=os.environ.get("RAG_MODEL", "").strip() or default_model,
        cheap_model=os.environ.get("RAG_CHEAP_MODEL", "").strip() or default_cheap_model,
        fallback_model=fallback_model,
        rate_limit_enabled=rate_limit_enabled,
        per_ip_per_hour=_env_int("PER_IP_PER_HOUR", 20),
        daily_cap=_env_int("DAILY_CAP", 500),
        top_k=_env_int("RAG_TOP_K", 5),
        max_question_chars=_env_int("MAX_QUESTION_CHARS", 500),
        upstash_redis_rest_url=upstash_url,
        upstash_redis_rest_token=upstash_token,
    )
