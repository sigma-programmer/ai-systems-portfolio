"""Tamper-evident, hash-chained JSONL audit log.

Each record is ``{"seq", "ts", "event", "data", "prev", "hash"}`` where
``hash = H(prev || canonical_json(record without hash))``. ``H`` is SHA-256,
or HMAC-SHA256 when a key is supplied (so an attacker who can rewrite the
file cannot simply recompute the chain).

:meth:`AuditLog.verify` recomputes the chain and reports the first broken
record: modified fields, deleted/reordered/inserted lines and garbage all
break the chain. Tail truncation cannot be detected from the file alone, so
callers can anchor the latest :attr:`AuditLog.head` externally and pass it
as ``expected_head``.

The proxy only ever logs *masked* content, so the audit trail itself never
becomes a PII store.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

__all__ = ["AuditLog", "VerifyResult", "GENESIS"]

GENESIS = "0" * 64


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    records: int
    first_bad_seq: int | None = None
    reason: str = ""


class AuditLog:
    """Append-only hash-chained log, in memory or backed by a JSONL file.

    Args:
        path: JSONL file to append to (created if missing; an existing file
            is resumed after its chain is verified).
        key: optional HMAC key.
        clock: time source (injectable for deterministic tests).
    """

    def __init__(
        self,
        path: str | os.PathLike | None = None,
        key: bytes | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self._key = key
        self._clock = clock
        self._lock = threading.Lock()
        self._memory: list[str] = []
        self.head = GENESIS
        self.seq = 0
        if self.path is not None and self.path.exists():
            res = self.verify()
            if not res.ok:
                raise ValueError(f"refusing to append to corrupted audit log: {res.reason}")
            for rec in self.records():
                self.head = rec["hash"]
                self.seq = rec["seq"] + 1

    def _digest(self, prev: str, body: dict[str, Any]) -> str:
        payload = (prev + _canonical(body)).encode()
        if self._key is not None:
            return hmac.new(self._key, payload, hashlib.sha256).hexdigest()
        return hashlib.sha256(payload).hexdigest()

    def append(self, event: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Append one event and return the full stored record."""
        with self._lock:
            body = {
                "seq": self.seq,
                "ts": round(self._clock(), 6),
                "event": event,
                "data": data or {},
                "prev": self.head,
            }
            # Round-trip through JSON so the hashed body equals what verify() reads back.
            body = json.loads(_canonical(body))
            record = {**body, "hash": self._digest(self.head, body)}
            line = _canonical(record)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
            else:
                self._memory.append(line)
            self.head = record["hash"]
            self.seq += 1
            return record

    def _lines(self) -> list[str]:
        if self.path is not None:
            if not self.path.exists():
                return []
            return [ln for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return list(self._memory)

    def records(self) -> Iterator[dict[str, Any]]:
        for ln in self._lines():
            yield json.loads(ln)

    def __len__(self) -> int:
        return len(self._lines())

    def verify(self, expected_head: str | None = None) -> VerifyResult:
        """Recompute the chain; optionally check it ends at ``expected_head``."""
        prev = GENESIS
        n = 0
        for i, ln in enumerate(self._lines()):
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                return VerifyResult(False, n, i, f"line {i} is not valid JSON")
            if not isinstance(rec, dict) or "hash" not in rec:
                return VerifyResult(False, n, i, f"line {i} is not an audit record")
            stored = rec.pop("hash")
            if rec.get("seq") != i:
                return VerifyResult(False, n, i, f"sequence gap: expected {i}, found {rec.get('seq')}")
            if rec.get("prev") != prev:
                return VerifyResult(False, n, i, f"record {i} does not link to previous hash")
            if not hmac.compare_digest(self._digest(prev, rec), stored):
                return VerifyResult(False, n, i, f"record {i} content hash mismatch")
            prev = stored
            n += 1
        if expected_head is not None and prev != expected_head:
            return VerifyResult(False, n, None, "chain head does not match expected anchor (truncated?)")
        return VerifyResult(True, n)

    # Mainly for tests/demos: simulate an attacker editing the backing store.
    def _tamper(self, index: int, mutate: Callable[[dict[str, Any]], None]) -> None:
        lines = self._lines()
        rec = json.loads(lines[index])
        mutate(rec)
        lines[index] = _canonical(rec)
        if self.path is not None:
            self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            self._memory = lines
