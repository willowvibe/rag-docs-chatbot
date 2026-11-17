import contextlib
import hashlib
import os
import time

import psycopg
import pytest

from ragchat import ratelimit

requires_db = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"), reason="needs a Postgres"
)


class FakeCursor:
    def __init__(self, conn):
        self._conn = conn
        self._row = None

    def execute(self, sql, params=None):
        self._conn.executed.append((sql, params))
        self._row = None
        if "INSERT INTO rate_counters" in sql:
            key, ttl = params
            self._conn.counts[key] = self._conn.counts.get(key, 0) + 1
            self._conn.ttls[key] = ttl
            self._row = (self._conn.counts[key],)

    def fetchone(self):
        return self._row

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConnection:
    def __init__(self):
        self.counts = {}
        self.ttls = {}
        self.executed = []
        self.closed = False

    def cursor(self):
        return FakeCursor(self)

    @contextlib.contextmanager
    def transaction(self):
        yield

    def close(self):
        self.closed = True


def _reset_limiter():
    conn = ratelimit._conn
    ratelimit._conn = None
    ratelimit._schema_ready = False
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


@pytest.fixture(autouse=True)
def _clean_limiter_state():
    """Connection and schema flags are module state; reset them between tests."""
    _reset_limiter()
    yield
    _reset_limiter()


def _enable(monkeypatch, *, per_ip=20, daily=1000):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "1")
    monkeypatch.setenv("PER_IP_PER_HOUR", str(per_ip))
    monkeypatch.setenv("DAILY_CAP", str(daily))
    fake = FakeConnection()
    ratelimit._conn = fake
    return fake


def _db():
    return psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)


def _delete_keys(ip):
    digest = hashlib.sha256(ip.encode()).hexdigest()[:32]
    with _db() as conn:
        # A fresh database (CI) has no table until the limiter's first call.
        conn.execute(ratelimit._SCHEMA)
        conn.execute(
            "DELETE FROM rate_counters WHERE key LIKE %s OR key = %s",
            (f"rag:ip:{digest}:%", ratelimit._day_key()),
        )


def test_disabled_limiter_always_allows(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "0")
    monkeypatch.setattr(ratelimit, "_connection", lambda: pytest.fail("no DB call"))
    ok, remaining, reason = ratelimit.check_and_consume("1.2.3.4")
    assert ok is True
    assert remaining == -1
    assert reason is None


def test_under_limit_returns_remaining(monkeypatch):
    fake = _enable(monkeypatch, per_ip=5, daily=100)
    assert ratelimit.check_and_consume("1.2.3.4") == (True, 4, None)
    assert ratelimit.check_and_consume("1.2.3.4") == (True, 3, None)
    assert fake.ttls[ratelimit._ip_key("1.2.3.4")] == 3700
    assert fake.ttls[ratelimit._day_key()] == 90000


def test_per_ip_limit_blocks_the_21st_call(monkeypatch):
    _enable(monkeypatch, per_ip=20, daily=1000)
    results = [ratelimit.check_and_consume("9.9.9.9") for _ in range(21)]
    assert [ok for ok, _, _ in results[:20]] == [True] * 20
    assert results[20] == (False, 0, "per_ip")


def test_per_ip_limit_is_keyed_by_ip(monkeypatch):
    _enable(monkeypatch, per_ip=20, daily=1000)
    results = [ratelimit.check_and_consume("203.0.113.7")[0] for _ in range(21)]
    assert results[:20] == [True] * 20
    assert results[20] is False
    assert ratelimit.check_and_consume("198.51.100.9")[0] is True


def test_daily_cap_blocks_everyone(monkeypatch):
    _enable(monkeypatch, per_ip=1000, daily=3)
    assert [ratelimit.check_and_consume(f"10.0.0.{i}")[0] for i in range(3)] == [
        True
    ] * 3
    assert ratelimit.check_and_consume("10.0.0.99") == (False, 0, "daily")


def test_ip_key_is_hashed(monkeypatch):
    fake = _enable(monkeypatch, per_ip=20, daily=1000)
    ip = "203.0.113.7"
    ratelimit.check_and_consume(ip)

    digest = hashlib.sha256(ip.encode()).hexdigest()[:32]
    expected = f"rag:ip:{digest}:{int(time.time() // 3600)}"
    assert expected in fake.counts
    assert all(ip not in key for key in fake.counts)
    params = [param for _, values in fake.executed if values for param in values]
    assert ip not in params


@requires_db
def test_db_increments_count_1_2_3(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "1")
    monkeypatch.setenv("PER_IP_PER_HOUR", "10")
    monkeypatch.setenv("DAILY_CAP", "1000")
    ip = "192.0.2.201"
    _delete_keys(ip)
    try:
        counts = []
        for _ in range(3):
            allowed, _, _ = ratelimit.check_and_consume(ip)
            assert allowed is True
            with _db() as conn:
                counts.append(
                    conn.execute(
                        "SELECT n FROM rate_counters WHERE key = %s",
                        (ratelimit._ip_key(ip),),
                    ).fetchone()[0]
                )
        assert counts == [1, 2, 3]
    finally:
        _delete_keys(ip)


@requires_db
def test_db_expired_row_resets_to_one(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "1")
    monkeypatch.setenv("PER_IP_PER_HOUR", "5")
    monkeypatch.setenv("DAILY_CAP", "1000")
    ip = "192.0.2.202"
    _delete_keys(ip)
    try:
        ratelimit.check_and_consume(ip)
        _delete_keys(ip)

        key = ratelimit._ip_key(ip)
        with _db() as conn:
            conn.execute(
                "INSERT INTO rate_counters (key, n, expires_at)"
                " VALUES (%s, %s, now() - interval '1 second')",
                (key, 99),
            )

        allowed, remaining, reason = ratelimit.check_and_consume(ip)

        assert allowed is True
        assert remaining == 4
        assert reason is None
        with _db() as conn:
            assert (
                conn.execute(
                    "SELECT n FROM rate_counters WHERE key = %s", (key,)
                ).fetchone()[0]
                == 1
            )
    finally:
        _delete_keys(ip)
