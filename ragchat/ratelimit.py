"""Per-IP rolling-window and global daily counters in Postgres.

No-op unless RATE_LIMIT_ENABLED=1.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import threading
import time

import psycopg

from ragchat.config import load_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rate_counters (
    key        TEXT PRIMARY KEY,
    n          INTEGER NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS rate_counters_expires_at
    ON rate_counters (expires_at);
CREATE TABLE IF NOT EXISTS rate_events (
    ip_hash TEXT NOT NULL,
    at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS rate_events_ip_hash_at
    ON rate_events (ip_hash, at);
"""

_INCR = """
INSERT INTO rate_counters (key, n, expires_at)
VALUES (%s, 1, now() + make_interval(secs => %s))
ON CONFLICT (key) DO UPDATE SET
    n = CASE
        WHEN rate_counters.expires_at < now() THEN 1
        ELSE rate_counters.n + 1
    END,
    expires_at = CASE
        WHEN rate_counters.expires_at < now() THEN EXCLUDED.expires_at
        ELSE rate_counters.expires_at
    END
RETURNING n
"""

_CLEANUP = "DELETE FROM rate_counters WHERE expires_at < now()"

_ADVISORY_LOCK = "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))"
_DELETE_IP_OLD = (
    "DELETE FROM rate_events"
    " WHERE ip_hash = %s AND at < now() - interval '60 minutes'"
)
_DELETE_ALL_OLD = "DELETE FROM rate_events WHERE at < now() - interval '60 minutes'"
_COUNT_WINDOW = """
SELECT count(*), min(at)
FROM rate_events
WHERE ip_hash = %s AND at > now() - interval '60 minutes'
"""
_OLDEST_EXPIRY = """
SELECT GREATEST(
    1,
    ceil(extract(epoch FROM (min(at) + interval '60 minutes' - now())))
)::int
FROM rate_events
WHERE ip_hash = %s AND at > now() - interval '60 minutes'
"""
_INSERT_EVENT = "INSERT INTO rate_events (ip_hash) VALUES (%s)"

_DAY_TTL = 90000

_conn: psycopg.Connection | None = None
_schema_ready = False
# FastAPI runs sync handlers in a thread pool and Vercel Fluid compute serves
# concurrent requests per instance; serialise use of the shared connection so one
# request's statements never run inside another's transaction.
_lock = threading.Lock()


def _ip_hash(ip: str) -> str:
    return hashlib.sha256(ip.encode()).hexdigest()[:32]


def _day_key() -> str:
    return f"rag:day:{dt.date.today().isoformat()}"


def _seconds_until_utc_midnight() -> int:
    return 86400 - (int(time.time()) % 86400)


def _ensure_schema(conn: psycopg.Connection) -> None:
    global _schema_ready
    if _schema_ready:
        return
    with conn.cursor() as cur:
        cur.execute(_SCHEMA)
    _schema_ready = True


def _connect() -> psycopg.Connection:
    return psycopg.connect(load_settings().database_url, autocommit=True)


def _connection() -> psycopg.Connection:
    global _conn
    if _conn is None or _conn.closed:
        _conn = _connect()
    return _conn


def _discard_connection() -> None:
    global _conn
    conn, _conn = _conn, None
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


def _consume(
    conn: psycopg.Connection,
    ip_hash: str,
    day_key: str,
    per_ip: int,
    daily_cap: int,
) -> tuple[bool, int, str | None, int]:
    with conn.transaction():
        with conn.cursor() as cur:
            # Serialise concurrent requests from the same IP for this transaction.
            cur.execute(_ADVISORY_LOCK, (ip_hash,))
            cur.execute(_DELETE_IP_OLD, (ip_hash,))
            cur.execute(_DELETE_ALL_OLD)
            cur.execute(_CLEANUP)

            cur.execute(_COUNT_WINDOW, (ip_hash,))
            count, _oldest = cur.fetchone()
            count = int(count)

            if count >= per_ip:
                cur.execute(_OLDEST_EXPIRY, (ip_hash,))
                retry_after = int(cur.fetchone()[0])
                return False, 0, "per_ip", retry_after

            cur.execute(_INCR, (day_key, _DAY_TTL))
            spent_today = int(cur.fetchone()[0])
            if spent_today > daily_cap:
                return False, 0, "daily", _seconds_until_utc_midnight()

            cur.execute(_INSERT_EVENT, (ip_hash,))
            cur.execute(_OLDEST_EXPIRY, (ip_hash,))
            retry_after = int(cur.fetchone()[0])

    return True, per_ip - (count + 1), None, retry_after


def check_and_consume(ip: str) -> tuple[bool, int, str | None, int]:
    """Returns (allowed, remaining_for_this_ip, reason, retry_after_seconds).

    remaining is -1 and reason is None when limiting is off. When blocked, reason
    is "daily" for the global cap or "per_ip" for the per-IP cap. retry_after is
    the seconds until the oldest event in the rolling window expires (per-IP) or
    until UTC midnight (daily).
    """
    settings = load_settings()
    if not settings.rate_limit_enabled:
        return True, -1, None, 0

    ip_hash = _ip_hash(ip)
    day_key = _day_key()

    with _lock:
        for attempt in range(2):
            try:
                conn = _connection()
                _ensure_schema(conn)
                result = _consume(
                    conn,
                    ip_hash,
                    day_key,
                    settings.per_ip_per_hour,
                    settings.daily_cap,
                )
                break
            except (psycopg.OperationalError, psycopg.InterfaceError):
                _discard_connection()
                if attempt == 1:
                    raise

    return result
