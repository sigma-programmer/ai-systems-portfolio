"""Token authentication, scope checks and per-client rate limiting."""
from __future__ import annotations

import hashlib
import hmac
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

KNOWN_SCOPES = frozenset({"fs:read", "fs:write", "exec:python", "net"})


@dataclass(frozen=True)
class Principal:
    """An authenticated client identity and the scopes it was granted."""

    client_id: str
    scopes: frozenset[str] = frozenset()

    def has_scope(self, scope: str) -> bool:
        """Exact match, ``<family>:*`` wildcard, or the global ``*`` grant."""
        if scope in self.scopes or "*" in self.scopes:
            return True
        family = scope.split(":", 1)[0]
        return f"{family}:*" in self.scopes

    def missing(self, required: Iterable[str]) -> list[str]:
        """Scopes from ``required`` this principal lacks (sorted, for messages)."""
        return sorted(s for s in required if not self.has_scope(s))


ANONYMOUS = Principal("anonymous", frozenset())


def _digest(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


class TokenAuthenticator:
    """Maps bearer tokens to principals.

    Only SHA-256 digests of tokens are kept in memory, and comparison uses
    :func:`hmac.compare_digest` across *all* entries so lookup time does not
    depend on which (or whether a) token matched.
    """

    def __init__(self, tokens: Mapping[str, Principal] | None = None):
        self._entries: list[tuple[bytes, Principal]] = [(_digest(t), p) for t, p in (tokens or {}).items()]

    @classmethod
    def from_config(cls, config: Mapping[str, Mapping[str, Any]]) -> "TokenAuthenticator":
        """Build from ``{token: {"client_id": str, "scopes": [str, ...]}}``."""
        tokens: dict[str, Principal] = {}
        for token, entry in config.items():
            if not isinstance(token, str) or len(token) < 8:
                raise ValueError("tokens must be strings of at least 8 characters")
            scopes = frozenset(entry.get("scopes", []))
            tokens[token] = Principal(str(entry["client_id"]), scopes)
        return cls(tokens)

    def authenticate(self, token: str | None) -> Principal | None:
        """Return the principal for ``token`` or ``None`` if it is unknown."""
        if not token:
            return None
        probe = _digest(token)
        found: Principal | None = None
        for digest, principal in self._entries:
            if hmac.compare_digest(digest, probe):
                found = principal
        return found

    def __len__(self) -> int:
        return len(self._entries)


@dataclass
class _Bucket:
    tokens: float
    updated: float


@dataclass
class RateLimiter:
    """Token-bucket limiter keyed by client id.

    ``capacity`` requests may burst; tokens refill at ``refill_per_sec``.
    ``clock`` is injectable so tests are deterministic.
    """

    capacity: float = 50.0
    refill_per_sec: float = 20.0
    clock: Callable[[], float] = time.monotonic
    _buckets: dict[str, _Bucket] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def allow(self, key: str, cost: float = 1.0) -> bool:
        now = self.clock()
        with self._lock:
            b = self._buckets.get(key)
            if b is None:
                b = self._buckets[key] = _Bucket(self.capacity, now)
            b.tokens = min(self.capacity, b.tokens + (now - b.updated) * self.refill_per_sec)
            b.updated = now
            if b.tokens >= cost:
                b.tokens -= cost
                return True
            return False

    def retry_after(self, key: str, cost: float = 1.0) -> float:
        """Seconds until ``cost`` tokens are available for ``key``."""
        with self._lock:
            b = self._buckets.get(key)
            if b is None or b.tokens >= cost or self.refill_per_sec <= 0:
                return 0.0
            return (cost - b.tokens) / self.refill_per_sec
