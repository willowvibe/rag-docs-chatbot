import pytest

from ragchat import ratelimit


class FakeRedis:
    def __init__(self):
        self.counts = {}
        self.expiries = {}

    def incr(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    def expire(self, key, seconds):
        self.expiries[key] = seconds


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
    fake = FakeRedis()
    ratelimit._redis = fake
    results = [ratelimit.check_and_consume("9.9.9.9")[0] for _ in range(21)]
    assert results[:20] == [True] * 20
    assert results[20] is False

    ip_keys = [key for key in fake.expiries if key.startswith("rag:ip:")]
    assert ip_keys, "the per-IP counter must get a TTL"
    assert all(fake.expiries[key] > 3600 for key in ip_keys)


def test_per_ip_limit_is_keyed_by_ip(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "1")
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "https://example.invalid")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "t")
    monkeypatch.setenv("PER_IP_PER_HOUR", "20")
    monkeypatch.setenv("DAILY_CAP", "1000")
    ratelimit._redis = FakeRedis()

    # Exhaust IP A ...
    results = [ratelimit.check_and_consume("203.0.113.7")[0] for _ in range(21)]
    assert results[:20] == [True] * 20
    assert results[20] is False

    # ... a different IP still has its own budget, so keying by hour alone fails.
    assert ratelimit.check_and_consume("198.51.100.9")[0] is True


def test_daily_cap_blocks_everyone(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "1")
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "https://example.invalid")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "t")
    monkeypatch.setenv("PER_IP_PER_HOUR", "1000")
    monkeypatch.setenv("DAILY_CAP", "3")
    ratelimit._redis = FakeRedis()
    assert [ratelimit.check_and_consume(f"10.0.0.{i}")[0] for i in range(3)] == [True] * 3
    assert ratelimit.check_and_consume("10.0.0.99")[0] is False
