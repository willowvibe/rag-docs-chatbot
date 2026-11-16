"""Every environment read in the project happens here."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    database_url: str
    anthropic_api_key: str
    model: str
    cheap_model: str
    rate_limit_enabled: bool
    per_ip_per_hour: int
    daily_cap: int
    top_k: int
    max_question_chars: int
    upstash_redis_rest_url: str
    upstash_redis_rest_token: str


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        sys.exit(f"{name} must be an integer, got {raw!r}")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    load_dotenv()
    rate_limit_enabled = _env_flag("RATE_LIMIT_ENABLED", False)
    upstash_url = os.environ.get("UPSTASH_REDIS_REST_URL", "").strip()
    upstash_token = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "").strip()
    if rate_limit_enabled and not upstash_url:
        sys.exit("RATE_LIMIT_ENABLED=1 requires UPSTASH_REDIS_REST_URL to be set.")
    return Settings(
        database_url=os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", "").strip(),
        model=os.environ.get("RAG_MODEL", "claude-sonnet-5-5"),
        cheap_model=os.environ.get("RAG_CHEAP_MODEL", "claude-haiku-5-5"),
        rate_limit_enabled=rate_limit_enabled,
        per_ip_per_hour=_env_int("PER_IP_PER_HOUR", 20),
        daily_cap=_env_int("DAILY_CAP", 500),
        top_k=_env_int("RAG_TOP_K", 5),
        max_question_chars=_env_int("MAX_QUESTION_CHARS", 500),
        upstash_redis_rest_url=upstash_url,
        upstash_redis_rest_token=upstash_token,
    )
