import pytest

from ragchat import ratelimit


class FakeRedis:
    def __init__(self):
        self.counts = {}

    def incr(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]


@pytest.fixture(autouse=True)
def _reset_redis():
    """`ratelimit._redis` is module state; reset it so counters cannot leak between tests."""
    yield
    ratelimit._redis = None


def test_disabled_limiter_always_allows(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "0")
    ok, remaining = ratelimit.check_and_consume("1.2.3.4")
    assert ok is True
    assert remaining == -1


def test_per_ip_limit_blocks_the_21st_call(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "1")
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "https://example.invalid")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "t")
    monkeypatch.setenv("PER_IP_PER_HOUR", "20")
    monkeypatch.setenv("DAILY_CAP", "1000")
    ratelimit._redis = FakeRedis()
    results = [ratelimit.check_and_consume("9.9.9.9")[0] for _ in range(21)]
    assert results[:20] == [True] * 20
    assert results[20] is False


def test_daily_cap_blocks_everyone(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "1")
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "https://example.invalid")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "t")
    monkeypatch.setenv("PER_IP_PER_HOUR", "1000")
    monkeypatch.setenv("DAILY_CAP", "3")
    ratelimit._redis = FakeRedis()
    assert [ratelimit.check_and_consume(f"10.0.0.{i}")[0] for i in range(3)] == [True] * 3
    assert ratelimit.check_and_consume("10.0.0.99")[0] is False
