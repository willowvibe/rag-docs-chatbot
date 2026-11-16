"""Per-IP and global counters in Upstash Redis (REST). No-op unless RATE_LIMIT_ENABLED=1."""

from __future__ import annotations

import datetime as dt
import time

import httpx

from ragchat.config import load_settings

_redis: "_Redis | None" = None


class _Redis:
    """Minimal Upstash REST client: INCR and EXPIRE only."""

    def __init__(self, url: str, token: str) -> None:
        self._url = url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"}

    def _post(self, command: str, *args) -> int:
        path = "/".join([command, *[str(a) for a in args]])
        response = httpx.post(f"{self._url}/{path}", headers=self._headers, timeout=3.0)
        response.raise_for_status()
        return int(response.json()["result"])

    def incr(self, key: str) -> int:
        return self._post("incr", key)

    def expire(self, key: str, seconds: int) -> None:
        self._post("expire", key, seconds)


def _client() -> _Redis:
    global _redis
    if _redis is None:
        settings = load_settings()
        _redis = _Redis(settings.upstash_redis_rest_url, settings.upstash_redis_rest_token)
    return _redis


def check_and_consume(ip: str) -> tuple[bool, int]:
    """Returns (allowed, remaining_for_this_ip). remaining is -1 when limiting is off."""
    settings = load_settings()
    if not settings.rate_limit_enabled:
        return True, -1
    redis = _client()
    hour = int(time.time() // 3600)
    ip_key = f"rag:ip:{ip}:{hour}"
    day_key = f"rag:day:{dt.date.today().isoformat()}"

    # Set a TTL on each fresh counter so Redis does not accumulate one key per
    # hour/day forever. `_Redis` documents EXPIRE as part of its interface, so
    # test doubles must implement it too.
    used = redis.incr(ip_key)
    if used == 1:
        redis.expire(ip_key, 3700)
    spent_today = redis.incr(day_key)
    if spent_today == 1:
        redis.expire(day_key, 90_000)

    if spent_today > settings.daily_cap:
        return False, 0
    if used > settings.per_ip_per_hour:
        return False, 0
    return True, settings.per_ip_per_hour - used
