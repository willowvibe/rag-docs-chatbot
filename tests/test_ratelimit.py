import contextlib
import datetime as dt
import hashlib
import math
import os
from concurrent.futures import ThreadPoolExecutor

import psycopg
import pytest

from ragchat import ratelimit

requires_db = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"), reason="needs a Postgres"
)

UTC = dt.timezone.utc
WINDOW = dt.timedelta(minutes=60)


def _digest(ip):
    return hashlib.sha256(ip.encode()).hexdigest()[:32]


class FakeCursor:
    def __init__(self, conn):
        self._conn = conn
        self._row = None

    def execute(self, sql, params=None):
        self._conn.executed.append((sql, params))
        self._row = None
        if sql.lstrip().startswith("CREATE "):
            return
        if "pg_advisory_xact_lock" in sql:
            return
        if "DELETE FROM rate_counters" in sql:
            return
        if "DELETE FROM rate_events" in sql:
            self._delete_events(params)
            return
        if "SELECT count(*), min(at)" in sql:
            events = self._conn.events_for(params[0])
            self._row = (len(events), min(events) if events else None)
            return
        if "GREATEST" in sql:
            events = self._conn.events_for(params[0])
            if not events:
                self._row = (1,)
            else:
                remaining = (min(events) + WINDOW - self._conn.now).total_seconds()
                self._row = (max(1, math.ceil(remaining)),)
            return
        if "INSERT INTO rate_events" in sql:
            self._conn.add_event(params[0], self._conn.now)
            return
        if "INSERT INTO rate_counters" in sql:
            key, ttl = params
            self._conn.counts[key] = self._conn.counts.get(key, 0) + 1
            self._conn.ttls[key] = ttl
            self._row = (self._conn.counts[key],)
            return
        raise AssertionError(f"unexpected SQL: {sql}")

    def _delete_events(self, params):
        cutoff = self._conn.now - WINDOW
        if params:
            ip_hash = params[0]
            self._conn.events[ip_hash] = [
                at for at in self._conn.events.get(ip_hash, []) if at >= cutoff
            ]
            return
        for ip_hash in list(self._conn.events):
            self._conn.events[ip_hash] = [
                at for at in self._conn.events[ip_hash] if at >= cutoff
            ]

    def fetchone(self):
        return self._row

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConnection:
    def __init__(self, now=None):
        self.counts = {}
        self.ttls = {}
        self.events = {}
        self.executed = []
        self.closed = False
        self.now = now or dt.datetime.now(UTC)

    def events_for(self, ip_hash):
        cutoff = self.now - WINDOW
        return [at for at in self.events.get(ip_hash, []) if at > cutoff]

    def add_event(self, ip_hash, at):
        self.events.setdefault(ip_hash, []).append(at)

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


def _enable(monkeypatch, *, per_ip=20, daily=1000, now=None):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "1")
    monkeypatch.setenv("PER_IP_PER_HOUR", str(per_ip))
    monkeypatch.setenv("DAILY_CAP", str(daily))
    fake = FakeConnection(now)
    ratelimit._conn = fake
    return fake


def _db():
    return psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)


def _clean(ip):
    with _db() as conn:
        # A fresh database (CI) has no tables until the limiter's first call.
        conn.execute(ratelimit._SCHEMA)
        conn.execute("DELETE FROM rate_events WHERE ip_hash = %s", (_digest(ip),))
        conn.execute(
            "DELETE FROM rate_counters WHERE key = %s", (ratelimit._day_key(),)
        )


def test_disabled_limiter_always_allows(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "0")
    monkeypatch.setattr(ratelimit, "_connection", lambda: pytest.fail("no DB call"))
    assert ratelimit.check_and_consume("1.2.3.4") == (True, -1, None, 0)


def test_under_limit_returns_remaining_and_retry(monkeypatch):
    fake = _enable(monkeypatch, per_ip=5, daily=100)
    assert ratelimit.check_and_consume("1.2.3.4") == (True, 4, None, 3600)
    assert ratelimit.check_and_consume("1.2.3.4") == (True, 3, None, 3600)
    assert fake.ttls[ratelimit._day_key()] == 90000
    assert len(fake.events[_digest("1.2.3.4")]) == 2


def test_per_ip_block_does_not_insert_or_spend_daily(monkeypatch):
    fake = _enable(monkeypatch, per_ip=3, daily=1000)
    ip = "9.9.9.9"

    results = [ratelimit.check_and_consume(ip) for _ in range(4)]

    assert [ok for ok, _, _, _ in results] == [True, True, True, False]
    assert results[3] == (False, 0, "per_ip", 3600)
    assert len(fake.events[_digest(ip)]) == 3
    assert fake.counts[ratelimit._day_key()] == 3


def test_per_ip_limit_is_keyed_by_ip(monkeypatch):
    _enable(monkeypatch, per_ip=2, daily=1000)
    assert [ratelimit.check_and_consume("203.0.113.7")[0] for _ in range(2)] == [
        True,
        True,
    ]
    assert ratelimit.check_and_consume("203.0.113.7")[0] is False
    assert ratelimit.check_and_consume("198.51.100.9")[0] is True


def test_daily_cap_blocks_without_inserting_an_event(monkeypatch):
    fake = _enable(monkeypatch, per_ip=1000, daily=3)
    assert [ratelimit.check_and_consume(f"10.0.0.{i}")[0] for i in range(3)] == [
        True
    ] * 3

    blocked = ratelimit.check_and_consume("10.0.0.99")

    assert blocked[0] is False
    assert blocked[1] == 0
    assert blocked[2] == "daily"
    assert 1 <= blocked[3] <= 86400
    assert fake.events.get(_digest("10.0.0.99"), []) == []
    assert fake.counts[ratelimit._day_key()] == 4


def test_retry_after_is_the_oldest_event_expiry(monkeypatch):
    fake = _enable(monkeypatch, per_ip=3, daily=1000)
    ip = "203.0.113.8"
    now = fake.now
    fake.events[_digest(ip)] = [now - dt.timedelta(minutes=50), now - dt.timedelta(minutes=5)]

    allowed = ratelimit.check_and_consume(ip)
    blocked = ratelimit.check_and_consume(ip)

    assert allowed == (True, 0, None, 600)
    assert blocked == (False, 0, "per_ip", 600)


def test_ip_is_stored_only_as_a_hash(monkeypatch):
    fake = _enable(monkeypatch, per_ip=20, daily=1000)
    ip = "203.0.113.7"
    ratelimit.check_and_consume(ip)

    assert _digest(ip) in fake.events
    params = [param for _, values in fake.executed if values for param in values]
    assert ip not in params


@requires_db
def test_db_rolling_window_allows_three_then_blocks(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "1")
    monkeypatch.setenv("PER_IP_PER_HOUR", "3")
    monkeypatch.setenv("DAILY_CAP", "1000")
    ip = "192.0.2.201"
    _clean(ip)
    try:
        results = [ratelimit.check_and_consume(ip) for _ in range(4)]

        assert [ok for ok, _, _, _ in results] == [True, True, True, False]
        assert [remaining for _, remaining, _, _ in results[:3]] == [2, 1, 0]
        assert results[3][2] == "per_ip"
        assert 1 <= results[3][3] <= 3600
        with _db() as conn:
            assert (
                conn.execute(
                    "SELECT count(*) FROM rate_events WHERE ip_hash = %s",
                    (_digest(ip),),
                ).fetchone()[0]
                == 3
            )
    finally:
        _clean(ip)


@requires_db
def test_db_events_older_than_the_window_do_not_count(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "1")
    monkeypatch.setenv("PER_IP_PER_HOUR", "5")
    monkeypatch.setenv("DAILY_CAP", "1000")
    ip = "192.0.2.202"
    _clean(ip)
    try:
        with _db() as conn:
            conn.execute(
                "INSERT INTO rate_events (ip_hash, at)"
                " VALUES (%s, now() - interval '61 minutes')",
                (_digest(ip),),
            )

        allowed, remaining, reason, retry = ratelimit.check_and_consume(ip)

        assert allowed is True
        assert remaining == 4
        assert reason is None
        assert 1 <= retry <= 3600
        with _db() as conn:
            assert (
                conn.execute(
                    "SELECT count(*) FROM rate_events WHERE ip_hash = %s",
                    (_digest(ip),),
                ).fetchone()[0]
                == 1
            )
    finally:
        _clean(ip)


@requires_db
def test_db_boundary_case_spans_a_clock_hour(monkeypatch):
    """Two events from 50 minutes ago plus one from 5 minutes ago still block."""
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "1")
    monkeypatch.setenv("PER_IP_PER_HOUR", "3")
    monkeypatch.setenv("DAILY_CAP", "1000")
    ip = "192.0.2.203"
    _clean(ip)
    try:
        with _db() as conn:
            conn.execute(
                "INSERT INTO rate_events (ip_hash, at) VALUES"
                " (%s, now() - interval '50 minutes'),"
                " (%s, now() - interval '50 minutes'),"
                " (%s, now() - interval '5 minutes')",
                (_digest(ip),) * 3,
            )

        blocked = ratelimit.check_and_consume(ip)

        assert blocked[0] is False
        assert blocked[2] == "per_ip"
        assert 540 <= blocked[3] <= 600
        with _db() as conn:
            assert (
                conn.execute(
                    "SELECT count(*) FROM rate_events WHERE ip_hash = %s",
                    (_digest(ip),),
                ).fetchone()[0]
                == 3
            )
    finally:
        _clean(ip)


@requires_db
def test_db_daily_cap_blocks_without_inserting_an_event(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "1")
    monkeypatch.setenv("PER_IP_PER_HOUR", "1000")
    monkeypatch.setenv("DAILY_CAP", "1")
    first = "192.0.2.204"
    second = "192.0.2.205"
    _clean(first)
    _clean(second)
    try:
        assert ratelimit.check_and_consume(first)[0] is True

        blocked = ratelimit.check_and_consume(second)

        assert blocked[0] is False
        assert blocked[2] == "daily"
        assert 1 <= blocked[3] <= 86400
        with _db() as conn:
            assert (
                conn.execute(
                    "SELECT count(*) FROM rate_events WHERE ip_hash = %s",
                    (_digest(second),),
                ).fetchone()[0]
                == 0
            )
    finally:
        _clean(first)
        _clean(second)


@requires_db
def test_db_concurrent_calls_allow_exactly_the_limit(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "1")
    monkeypatch.setenv("PER_IP_PER_HOUR", "8")
    monkeypatch.setenv("DAILY_CAP", "1000")
    ip = "192.0.2.206"
    _clean(ip)
    try:
        with ThreadPoolExecutor(max_workers=20) as pool:
            results = list(
                pool.map(lambda _: ratelimit.check_and_consume(ip), range(20))
            )

        assert sum(1 for allowed, _, _, _ in results if allowed) == 8
        assert sum(1 for _, _, reason, _ in results if reason == "per_ip") == 12
        with _db() as conn:
            assert (
                conn.execute(
                    "SELECT count(*) FROM rate_events WHERE ip_hash = %s",
                    (_digest(ip),),
                ).fetchone()[0]
                == 8
            )
    finally:
        _clean(ip)
