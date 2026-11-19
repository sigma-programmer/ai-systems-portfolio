"""SQLite persistence (stdlib sqlite3) for all three tiers.

Writes are write-through and committed immediately, so a second process (or a
new `MemoryManager` on the same file) sees everything a previous one wrote.
Embeddings are *not* stored: `HashingEmbedder` is deterministic, so vectors
are recomputed on load — the schema stays portable and small.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .models import Episode, Fact, WorkingItem

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, started REAL NOT NULL, ended REAL
);
CREATE TABLE IF NOT EXISTS working (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL, text TEXT NOT NULL,
    created REAL NOT NULL, last_access REAL NOT NULL, tokens INTEGER NOT NULL,
    salience REAL NOT NULL, access_count INTEGER NOT NULL DEFAULT 0, seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, start REAL NOT NULL, end REAL NOT NULL,
    summary TEXT NOT NULL, raw_turns TEXT NOT NULL, raw_tokens INTEGER NOT NULL,
    summary_tokens INTEGER NOT NULL, salience REAL NOT NULL, last_access REAL NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0, consolidated INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY, subject TEXT NOT NULL, predicate TEXT NOT NULL, value TEXT NOT NULL,
    text TEXT NOT NULL, created REAL NOT NULL, last_access REAL NOT NULL, source_episode TEXT,
    confidence REAL NOT NULL, salience REAL NOT NULL, access_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL, superseded_by TEXT, meta TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS facts_key ON facts(subject, predicate, status);
CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
"""


class SQLiteStore:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._seq = self.conn.execute("SELECT COALESCE(MAX(seq), 0) FROM working").fetchone()[0]

    def close(self) -> None:
        self.conn.close()

    # sessions -----------------------------------------------------------------------
    def start_session(self, sid: str, now: float) -> None:
        self.conn.execute("INSERT OR IGNORE INTO sessions(id, started) VALUES (?, ?)", (sid, now))
        self.conn.commit()

    def end_session(self, sid: str, now: float) -> None:
        self.conn.execute("UPDATE sessions SET ended=? WHERE id=?", (now, sid))
        self.conn.commit()

    def sessions(self) -> list[tuple[str, float, float | None]]:
        return [tuple(r) for r in self.conn.execute(
            "SELECT id, started, ended FROM sessions ORDER BY started")]

    # working ------------------------------------------------------------------------
    def save_working(self, item: WorkingItem) -> None:
        self._seq += 1
        self.conn.execute(
            "INSERT OR REPLACE INTO working VALUES (?,?,?,?,?,?,?,?,?,?)",
            (item.id, item.session_id, item.role, item.text, item.created, item.last_access,
             item.tokens, item.salience, item.access_count, self._seq))
        self.conn.commit()

    def delete_working(self, ids: list[str]) -> None:
        self.conn.executemany("DELETE FROM working WHERE id=?", [(i,) for i in ids])
        self.conn.commit()

    def load_working(self) -> list[WorkingItem]:
        rows = self.conn.execute("SELECT * FROM working ORDER BY seq").fetchall()
        return [WorkingItem(r["id"], r["session_id"], r["role"], r["text"], r["created"],
                            r["last_access"], r["tokens"], r["salience"], r["access_count"])
                for r in rows]

    # episodes -----------------------------------------------------------------------
    def save_episode(self, e: Episode) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (e.id, e.session_id, e.start, e.end, e.summary, json.dumps(e.raw_turns),
             e.raw_tokens, e.summary_tokens, e.salience, e.last_access, e.access_count,
             int(e.consolidated)))
        self.conn.commit()

    def delete_episodes(self, ids: list[str]) -> None:
        self.conn.executemany("DELETE FROM episodes WHERE id=?", [(i,) for i in ids])
        self.conn.commit()

    def load_episodes(self) -> list[Episode]:
        rows = self.conn.execute("SELECT * FROM episodes ORDER BY start").fetchall()
        return [Episode(r["id"], r["session_id"], r["start"], r["end"], r["summary"],
                        json.loads(r["raw_turns"]), r["raw_tokens"], r["summary_tokens"],
                        r["salience"], r["last_access"], r["access_count"],
                        bool(r["consolidated"])) for r in rows]

    # facts --------------------------------------------------------------------------
    def save_fact(self, f: Fact) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (f.id, f.subject, f.predicate, f.value, f.text, f.created, f.last_access,
             f.source_episode, f.confidence, f.salience, f.access_count, f.status,
             f.superseded_by, json.dumps(f.meta, sort_keys=True)))
        self.conn.commit()

    def delete_facts(self, ids: list[str]) -> None:
        self.conn.executemany("DELETE FROM facts WHERE id=?", [(i,) for i in ids])
        self.conn.commit()

    def load_facts(self) -> list[Fact]:
        rows = self.conn.execute("SELECT * FROM facts ORDER BY created").fetchall()
        return [Fact(r["id"], r["subject"], r["predicate"], r["value"], r["text"], r["created"],
                     r["last_access"], r["source_episode"], r["confidence"], r["salience"],
                     r["access_count"], r["status"], r["superseded_by"], json.loads(r["meta"]))
                for r in rows]

    # counters -----------------------------------------------------------------------
    def bump(self, name: str, by: int = 1) -> None:
        if by:
            self.conn.execute(
                "INSERT INTO counters VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET "
                "value = value + excluded.value", (name, by))
            self.conn.commit()

    def counters(self) -> dict[str, int]:
        return {r["name"]: r["value"] for r in self.conn.execute("SELECT * FROM counters")}
