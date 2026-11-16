"""Postgres + pgvector. The only persistence layer in the project."""

from __future__ import annotations

from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector
from pgvector.utils import Vector

from ragchat.ingest.chunker import Chunk

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

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self._url, autocommit=True)
        register_vector(conn)
        return conn

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

    def search(self, vector: list[float], k: int = 5) -> list[Hit]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT file, page, chunk_index, text, 1 - (embedding <=> %s) AS score"
                " FROM chunks ORDER BY embedding <=> %s LIMIT %s",
                (Vector(vector), Vector(vector), k),
            ).fetchall()
        return [
            Hit(file=r[0], page=r[1], chunk_index=r[2], text=r[3], score=float(r[4]))
            for r in rows
        ]

    def count(self, doc_id: str | None = None) -> int:
        with self._connect() as conn:
            if doc_id is None:
                return int(conn.execute("SELECT count(*) FROM chunks").fetchone()[0])
            return int(
                conn.execute(
                    "SELECT count(*) FROM chunks WHERE doc_id = %s", (doc_id,)
                ).fetchone()[0]
            )

    def count_for_file(self, name: str) -> int:
        """Count chunks whose stored path has this basename (files may live in subdirs)."""
        with self._connect() as conn:
            return int(
                conn.execute(
                    "SELECT count(*) FROM chunks"
                    " WHERE regexp_replace(file, '^.*/', '') = %s",
                    (name,),
                ).fetchone()[0]
            )
