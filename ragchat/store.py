"""Postgres + pgvector. The only persistence layer in the project."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, TypeVar

import psycopg
from pgvector.psycopg import register_vector
from pgvector.utils import Vector

from ragchat.ingest.chunker import Chunk

_T = TypeVar("_T")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    doc_id      TEXT    NOT NULL,
    file        TEXT    NOT NULL,
    page        INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT    NOT NULL,
    tokens      INTEGER NOT NULL,
    embedding   vector(384) NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS chunks_doc_position
    ON chunks (doc_id, page, chunk_index);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_text_fts
    ON chunks USING GIN (to_tsvector('english', text));
CREATE TABLE IF NOT EXISTS rate_counters (
    key        TEXT PRIMARY KEY,
    n          INTEGER NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS rate_counters_expires_at
    ON rate_counters (expires_at);
"""

_HYBRID_SEARCH = """
WITH semantic AS (
    SELECT id, RANK() OVER (ORDER BY embedding <=> %s) AS r
    FROM chunks
    ORDER BY embedding <=> %s
    LIMIT %s
),
keyword AS (
    SELECT id, RANK() OVER (
        ORDER BY ts_rank_cd(to_tsvector('english', text), query) DESC
    ) AS r
    FROM chunks,
         (
             SELECT NULLIF(
                 replace(plainto_tsquery('english', %s)::text, '&', '|'), ''
             )::tsquery AS query
         ) AS q
    WHERE to_tsvector('english', text) @@ query
    ORDER BY ts_rank_cd(to_tsvector('english', text), query) DESC
    LIMIT %s
)
SELECT c.file, c.page, c.chunk_index, c.text,
       COALESCE(1.0 / (60 + semantic.r), 0)
       + COALESCE(1.0 / (60 + keyword.r), 0) AS score
FROM semantic
FULL OUTER JOIN keyword ON semantic.id = keyword.id
JOIN chunks c ON c.id = COALESCE(semantic.id, keyword.id)
ORDER BY score DESC
LIMIT %s
"""


@dataclass(frozen=True)
class Hit:
    file: str
    page: int
    chunk_index: int
    text: str
    score: float


class Store:
    def __init__(self, database_url: str) -> None:
        self._url = database_url
        self._conn: psycopg.Connection | None = None
        # FastAPI runs sync handlers in a thread pool and Vercel Fluid compute serves
        # concurrent requests per instance; serialise use of the shared connection so
        # one request's statements never run inside another's transaction.
        self._lock = threading.Lock()

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self._url, autocommit=True)
        register_vector(conn)
        return conn

    def _connection(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = self._connect()
        return self._conn

    def _discard_connection(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def _run(self, operation: Callable[[psycopg.Connection], _T]) -> _T:
        with self._lock:
            for attempt in range(2):
                try:
                    return operation(self._connection())
                except (psycopg.OperationalError, psycopg.InterfaceError):
                    self._discard_connection()
                    if attempt == 1:
                        raise
        raise AssertionError("unreachable")

    def ensure_schema(self) -> None:
        # On a fresh database the vector type does not exist yet, so the
        # adapter cannot be registered until after CREATE EXTENSION runs.
        conn = psycopg.connect(self._url, autocommit=True)
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(conn)
            conn.execute(_SCHEMA)
        finally:
            conn.close()

    def replace_doc(self, doc_id: str, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        with self._connect() as conn, conn.transaction():
            conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (doc_id, file, page, chunk_index, text, tokens, embedding)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    [
                        (c.doc_id, c.file, c.page, c.chunk_index, c.text, c.tokens, Vector(v))
                        for c, v in zip(chunks, vectors)
                    ],
                )
        return len(chunks)

    def search(
        self,
        vector: list[float],
        k: int = 5,
        query: str | None = None,
        hybrid: bool = False,
    ) -> list[Hit]:
        def run(conn: psycopg.Connection):
            if hybrid and query:
                depth = max(2 * k, 40)
                return conn.execute(
                    _HYBRID_SEARCH,
                    (
                        Vector(vector),
                        Vector(vector),
                        depth,
                        query,
                        depth,
                        k,
                    ),
                ).fetchall()
            return conn.execute(
                "SELECT file, page, chunk_index, text, 1 - (embedding <=> %s) AS score"
                " FROM chunks ORDER BY embedding <=> %s LIMIT %s",
                (Vector(vector), Vector(vector), k),
            ).fetchall()

        rows = self._run(run)
        return [
            Hit(file=r[0], page=r[1], chunk_index=r[2], text=r[3], score=float(r[4]))
            for r in rows
        ]

    def count(self, doc_id: str | None = None) -> int:
        def run(conn: psycopg.Connection) -> int:
            if doc_id is None:
                return int(conn.execute("SELECT count(*) FROM chunks").fetchone()[0])
            return int(
                conn.execute(
                    "SELECT count(*) FROM chunks WHERE doc_id = %s", (doc_id,)
                ).fetchone()[0]
            )

        return self._run(run)

    def count_for_file(self, name: str) -> int:
        """Count chunks whose stored path has this basename (files may live in subdirs)."""

        def run(conn: psycopg.Connection) -> int:
            return int(
                conn.execute(
                    "SELECT count(*) FROM chunks"
                    " WHERE regexp_replace(file, '^.*/', '') = %s",
                    (name,),
                ).fetchone()[0]
            )

        return self._run(run)

    def list_documents(self) -> list[dict]:
        """Summarise stored documents by basename, ignoring directory prefixes."""

        def run(conn: psycopg.Connection):
            return conn.execute(
                "SELECT regexp_replace(file, '^.*/', '') AS name,"
                " max(page) AS pages, count(*) AS chunks"
                " FROM chunks GROUP BY name ORDER BY name"
            ).fetchall()

        rows = self._run(run)
        return [
            {"file": row[0], "pages": int(row[1]), "chunks": int(row[2])}
            for row in rows
        ]
