"""SQLite persistence: run snapshots, an append-only event log, idempotency
records and signals.

Every state change happens inside one ``BEGIN IMMEDIATE`` transaction, so a
step's completion event, its new snapshot and its idempotency records commit
atomically or not at all. The database runs in WAL mode with
``synchronous=FULL`` by default, so a committed checkpoint survives a process
crash or power loss.

The event log is append-only by construction: the store exposes no update or
delete for events, and ``(run_id, seq)`` is unique.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    workflow         TEXT NOT NULL,
    fingerprint      TEXT NOT NULL,
    status           TEXT NOT NULL,
    current_step     TEXT,
    input_json       TEXT NOT NULL,
    state_json       TEXT NOT NULL,
    attempt          INTEGER NOT NULL DEFAULT 0,
    next_attempt_at  REAL NOT NULL DEFAULT 0,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    cancel_reason    TEXT,
    error            TEXT,
    owner            TEXT,
    lease_expires    REAL NOT NULL DEFAULT 0,
    approval_json    TEXT,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL,
    version          INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS runs_status ON runs(status);
CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    seq          INTEGER NOT NULL,
    ts           REAL NOT NULL,
    type         TEXT NOT NULL,
    step         TEXT,
    attempt      INTEGER,
    payload_json TEXT NOT NULL,
    UNIQUE(run_id, seq)
);
CREATE TABLE IF NOT EXISTS idempotency (
    key         TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL,
    step        TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS signals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    name         TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    ts           REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS signals_run ON signals(run_id, name);
"""

TERMINAL = frozenset({"COMPLETED", "FAILED", "CANCELLED"})


def dumps(value: Any) -> str:
    """Canonical JSON (raises TypeError for non-serializable values)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class Event:
    """One immutable entry of a run's history."""

    seq: int
    ts: float
    type: str
    step: str | None
    attempt: int | None
    payload: dict[str, Any]


class SQLiteStore:
    """Thread-safe SQLite store shared by an engine and its step threads.

    Args:
        path: database file (``":memory:"`` for an ephemeral store; note an
            in-memory store cannot be shared across processes).
        synchronous: SQLite ``synchronous`` pragma (``FULL`` = durable on
            power loss, ``NORMAL`` = durable on process crash in WAL mode).
    """

    def __init__(self, path: str | Path = ":memory:", synchronous: str = "FULL") -> None:
        if synchronous.upper() not in {"OFF", "NORMAL", "FULL", "EXTRA"}:
            raise ValueError(f"bad synchronous mode {synchronous!r}")
        self.path = str(path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(f"PRAGMA synchronous={synchronous.upper()}")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)
        self._depth = 0
        self.txn_seconds = 0.0
        self.txn_count = 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------ transactions
    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """``BEGIN IMMEDIATE`` ... ``COMMIT`` (rollback on any exception).

        Re-entrant: nested calls join the outer transaction.
        """
        with self._lock:
            if self._depth:
                self._depth += 1
                try:
                    yield self._conn
                finally:
                    self._depth -= 1
                return
            t0 = time.perf_counter()
            self._conn.execute("BEGIN IMMEDIATE")
            self._depth = 1
            try:
                yield self._conn
            except BaseException:
                self._depth = 0
                self._conn.execute("ROLLBACK")
                raise
            self._depth = 0
            self._conn.execute("COMMIT")
            self.txn_seconds += time.perf_counter() - t0
            self.txn_count += 1

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # ------------------------------------------------------------ runs
    def insert_run(self, conn: sqlite3.Connection, row: dict[str, Any]) -> None:
        cols = ",".join(row)
        conn.execute(f"INSERT INTO runs ({cols}) VALUES ({','.join('?' * len(row))})", tuple(row.values()))

    def get_run(self, run_id: str, conn: sqlite3.Connection | None = None) -> dict[str, Any] | None:
        with self._lock:
            c = conn or self._conn
            r = c.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            return dict(r) if r else None

    def update_run(self, conn: sqlite3.Connection, run_id: str, now: float, **fields: Any) -> None:
        sets = ",".join(f"{k}=?" for k in fields)
        conn.execute(
            f"UPDATE runs SET {sets}, updated_at=?, version=version+1 WHERE run_id=?",
            (*fields.values(), now, run_id),
        )

    def list_runs(self, status: str | None = None, workflow: str | None = None) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM runs WHERE 1=1", []
        if status is not None:
            sql += " AND status=?"
            params.append(status)
        if workflow is not None:
            sql += " AND workflow=?"
            params.append(workflow)
        sql += " ORDER BY created_at, run_id"
        return [dict(r) for r in self.query(sql, tuple(params))]

    # ------------------------------------------------------------ events
    def append_event(
        self,
        conn: sqlite3.Connection,
        run_id: str,
        ts: float,
        type: str,
        step: str | None = None,
        attempt: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> int:
        seq = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM events WHERE run_id=?", (run_id,)).fetchone()[0]
        conn.execute(
            "INSERT INTO events (run_id, seq, ts, type, step, attempt, payload_json) VALUES (?,?,?,?,?,?,?)",
            (run_id, seq, ts, type, step, attempt, dumps(payload or {})),
        )
        return seq

    def events(self, run_id: str) -> list[Event]:
        rows = self.query("SELECT seq, ts, type, step, attempt, payload_json FROM events WHERE run_id=? ORDER BY seq", (run_id,))
        return [Event(r["seq"], r["ts"], r["type"], r["step"], r["attempt"], json.loads(r["payload_json"])) for r in rows]

    def last_event(
        self, run_id: str, conn: sqlite3.Connection | None = None, types: tuple[str, ...] | None = None
    ) -> Event | None:
        """Most recent event of the run (optionally restricted to ``types``)."""
        sql = "SELECT seq, ts, type, step, attempt, payload_json FROM events WHERE run_id=?"
        params: tuple = (run_id,)
        if types:
            sql += f" AND type IN ({','.join('?' * len(types))})"
            params += tuple(types)
        sql += " ORDER BY seq DESC LIMIT 1"
        with self._lock:
            c = conn or self._conn
            r = c.execute(sql, params).fetchone()
        return Event(r["seq"], r["ts"], r["type"], r["step"], r["attempt"], json.loads(r["payload_json"])) if r else None

    # ------------------------------------------------------------ idempotency
    def idem_get(self, key: str, conn: sqlite3.Connection | None = None) -> tuple[bool, Any]:
        with self._lock:
            c = conn or self._conn
            r = c.execute("SELECT result_json FROM idempotency WHERE key=?", (key,)).fetchone()
        return (True, json.loads(r[0])) if r else (False, None)

    def idem_put(self, conn: sqlite3.Connection, key: str, run_id: str, step: str, result: Any, now: float) -> None:
        conn.execute(
            "INSERT INTO idempotency (key, run_id, step, result_json, created_at) VALUES (?,?,?,?,?)",
            (key, run_id, step, dumps(result), now),
        )

    # ------------------------------------------------------------ signals
    def add_signal(self, conn: sqlite3.Connection, run_id: str, name: str, payload: Any, now: float) -> None:
        conn.execute(
            "INSERT INTO signals (run_id, name, payload_json, ts) VALUES (?,?,?,?)",
            (run_id, name, dumps(payload), now),
        )

    def signals(self, run_id: str, name: str | None = None) -> list[dict[str, Any]]:
        if name is None:
            rows = self.query("SELECT name, payload_json, ts FROM signals WHERE run_id=? ORDER BY id", (run_id,))
        else:
            rows = self.query("SELECT name, payload_json, ts FROM signals WHERE run_id=? AND name=? ORDER BY id", (run_id, name))
        return [{"name": r["name"], "payload": json.loads(r["payload_json"]), "ts": r["ts"]} for r in rows]
