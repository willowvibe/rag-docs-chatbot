"""Per-IP and global counters in Postgres. No-op unless RATE_LIMIT_ENABLED=1."""

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
_IP_TTL = 3700
_DAY_TTL = 90000

_conn: psycopg.Connection | None = None
_schema_ready = False
# FastAPI runs sync handlers in a thread pool and Vercel Fluid compute serves
# concurrent requests per instance; serialise use of the shared connection so one
# request's statements never run inside another's transaction.
_lock = threading.Lock()


def _ip_key(ip: str, hour: int | None = None) -> str:
    if hour is None:
        hour = int(time.time() // 3600)
    digest = hashlib.sha256(ip.encode()).hexdigest()[:32]
    return f"rag:ip:{digest}:{hour}"


def _day_key() -> str:
    return f"rag:day:{dt.date.today().isoformat()}"


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


def _consume(conn: psycopg.Connection, ip_key: str, day_key: str) -> tuple[int, int]:
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(_INCR, (ip_key, _IP_TTL))
            used = int(cur.fetchone()[0])
            cur.execute(_INCR, (day_key, _DAY_TTL))
            spent_today = int(cur.fetchone()[0])
            cur.execute(_CLEANUP)
    return used, spent_today


def check_and_consume(ip: str) -> tuple[bool, int, str | None]:
    """Returns (allowed, remaining_for_this_ip, reason).

    remaining is -1 and reason is None when limiting is off. When blocked, reason
    is "daily" for the global cap or "per_ip" for the per-IP cap.
    """
    settings = load_settings()
    if not settings.rate_limit_enabled:
        return True, -1, None

    ip_key = _ip_key(ip)
    day_key = _day_key()

    with _lock:
        for attempt in range(2):
            try:
                conn = _connection()
                _ensure_schema(conn)
                used, spent_today = _consume(conn, ip_key, day_key)
                break
            except (psycopg.OperationalError, psycopg.InterfaceError):
                _discard_connection()
                if attempt == 1:
                    raise

    if spent_today > settings.daily_cap:
        return False, 0, "daily"
    if used > settings.per_ip_per_hour:
        return False, 0, "per_ip"
    return True, settings.per_ip_per_hour - used, None
