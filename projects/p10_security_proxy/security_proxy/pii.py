"""PII detection and reversible, session-scoped UUID masking.

Detection is a set of typed regex detectors with validators (Luhn for card
numbers, mod-97 for IBANs, octet ranges for IPv4, area rules for SSNs) plus
a light-weight heuristic for person names. Overlapping matches are resolved
by detector priority so that e.g. a Luhn-valid card number is never also
reported as a phone number.

Masking replaces every detected value with a token of the form
``<<PII:TYPE:uuid>>``. Tokens are derived with HMAC-SHA256 from a per-session
secret, so:

* the same value maps to the same token within a session (agents can still
  reason about "the same email appeared twice");
* tokens are unlinkable across sessions (different secret);
* the token reveals nothing about the value without the vault.

Unmasking is done by the :class:`PIIVault` and only for the PII types the
caller is authorised for (the proxy derives those from recipient scopes).
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import re
import secrets
import uuid
from dataclasses import dataclass, field
from typing import Callable, Iterable

__all__ = [
    "PIIType",
    "PIIMatch",
    "PIIDetector",
    "PIIVault",
    "luhn_valid",
    "iban_valid",
    "TOKEN_RE",
]


class PIIType:
    """String constants for supported PII types (also used in scopes)."""

    EMAIL = "EMAIL"
    PHONE = "PHONE"
    CREDIT_CARD = "CREDIT_CARD"
    SSN = "SSN"
    IP_ADDRESS = "IP_ADDRESS"
    IBAN = "IBAN"
    API_KEY = "API_KEY"
    PERSON = "PERSON"

    ALL: tuple[str, ...] = (
        "EMAIL", "PHONE", "CREDIT_CARD", "SSN", "IP_ADDRESS", "IBAN", "API_KEY", "PERSON",
    )


TOKEN_RE = re.compile(r"<<PII:([A-Z_]+):([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})>>")


@dataclass(frozen=True)
class PIIMatch:
    """A detected PII span in a text."""

    type: str
    value: str
    start: int
    end: int


# --------------------------------------------------------------------------- validators


def luhn_valid(number: str) -> bool:
    """Return True if the digits of ``number`` pass the Luhn checksum."""
    digits = [int(c) for c in number if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19 or len(set(digits)) == 1:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def iban_valid(iban: str) -> bool:
    """Validate an IBAN via the ISO 13616 mod-97 check."""
    s = re.sub(r"\s+", "", iban).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", s):
        return False
    rearranged = s[4:] + s[:4]
    numeric = "".join(str(int(c, 36)) for c in rearranged)
    return int(numeric) % 97 == 1


def _ip_valid(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    # Version-like strings such as "1.2.3.4" are technically valid; we keep
    # them — over-masking an IP is cheaper than leaking one.
    return not (ip.version == 6 and value.count(":") < 2)


def _phone_valid(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    return 10 <= len(digits) <= 15 and len(set(digits)) > 2


# --------------------------------------------------------------------------- name heuristics

_FIRST_NAMES = frozenset(
    """james john robert michael william david richard joseph thomas charles christopher daniel
    matthew anthony mark donald steven paul andrew joshua kenneth kevin brian george timothy
    mary patricia jennifer linda elizabeth barbara susan jessica sarah karen lisa nancy betty
    margaret sandra ashley emily donna michelle carol amanda melissa deborah stephanie rebecca
    laura sharon cynthia kathleen amy angela helen anna brenda pamela nicole emma olivia sophia
    alice bob carol dave eve frank grace heidi ivan judy mallory oscar peggy trent victor walter
    priya raj arjun ananya rahul wei li chen mohammed ahmed fatima omar yuki hiro sofia lucas
    maria jose juan carlos ana luis diego elena ivan olga hans klaus ingrid lars sven""".split()
)

_NAME_STOP = frozenset(
    """street st avenue ave road rd lane boulevard inc llc ltd corp company university
    hospital bank group team monday tuesday wednesday thursday friday saturday sunday january
    february march april may june july august september october november december the and""".split()
)

_CAP = r"[A-Z](?:[a-z]+(?:-[A-Z][a-z]+)?|[a-z]*'[A-Z][a-z]+)"
_HONORIFIC_RE = re.compile(rf"\b(?:Mr|Mrs|Ms|Miss|Mx|Dr|Prof|Sir|Madam)\.?\s+({_CAP}(?:\s+{_CAP})?)")
_CUE_RE = re.compile(
    rf"(?i:\b(?:my name is|name\s*:|signed,?|regards,?|sincerely,?|cheers,?|thanks,?|"
    rf"contact|from\s*:|to\s*:|patient|customer|employee|applicant|cardholder|account holder)"
    rf")\s+({_CAP}\s+{_CAP}(?:\s+{_CAP})?)"
)
# Lookahead so overlapping pairs are all considered ("Charge Alice Smith").
_FIRST_LAST_RE = re.compile(rf"\b(?=({_CAP})\s+({_CAP})\b)")


def _detect_names(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for rx in (_HONORIFIC_RE, _CUE_RE):
        for m in rx.finditer(text):
            words = m.group(1).split()
            if any(w.lower() in _NAME_STOP for w in words):
                continue
            spans.append((m.start(1), m.end(1)))
    for m in _FIRST_LAST_RE.finditer(text):
        first, last = m.group(1), m.group(2)
        if first.lower() in _FIRST_NAMES and last.lower() not in _NAME_STOP \
                and last.lower() not in _FIRST_NAMES | {"is", "was"}:
            spans.append((m.start(1), m.end(2)))
    return spans


# --------------------------------------------------------------------------- detectors


@dataclass(frozen=True)
class _Detector:
    type: str
    priority: int
    pattern: re.Pattern | None = None
    validator: Callable[[str], bool] | None = None
    group: int = 0
    custom: Callable[[str], list[tuple[int, int]]] | None = None

    def find(self, text: str) -> Iterable[tuple[int, int]]:
        if self.custom is not None:
            yield from self.custom(text)
            return
        assert self.pattern is not None
        for m in self.pattern.finditer(text):
            value = m.group(self.group)
            if self.validator is None or self.validator(value):
                yield m.start(self.group), m.end(self.group)


_API_KEY_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"sk-(?:ant-|proj-|live-|test-)?[A-Za-z0-9_\-]{20,}"   # OpenAI / Anthropic / Stripe-style
    r"|(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}"         # Stripe
    r"|(?:AKIA|ASIA)[0-9A-Z]{16}"                          # AWS access key id
    r"|gh[pousr]_[A-Za-z0-9]{36,}"                         # GitHub tokens
    r"|xox[abposr]-[A-Za-z0-9\-]{10,}"                     # Slack tokens
    r"|AIza[0-9A-Za-z_\-]{35}"                             # Google API key
    r"|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"  # JWT
    r")"
)
_AWS_SECRET_RE = re.compile(
    r"(?i:[a-z0-9_]*(?:secret|_key|api_?key|access_?key|password|passwd|token))\s*[=:]\s*[\"']?"
    r"([A-Za-z0-9/+=_\-]{16,})"
)

_DETECTORS: tuple[_Detector, ...] = (
    _Detector(PIIType.API_KEY, 100, _API_KEY_RE),
    _Detector(PIIType.API_KEY, 99, _AWS_SECRET_RE, group=1),
    _Detector(
        PIIType.IBAN, 90,
        re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?\b"),
        iban_valid,
    ),
    _Detector(
        PIIType.CREDIT_CARD, 80,
        re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])"),
        luhn_valid,
    ),
    _Detector(
        PIIType.SSN, 75,
        re.compile(r"(?<![\d-])(?!000|666|9\d\d)\d{3}[- ](?!00)\d{2}[- ](?!0000)\d{4}(?![\d-])"),
    ),
    _Detector(
        PIIType.EMAIL, 70,
        re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"),
    ),
    _Detector(
        PIIType.IP_ADDRESS, 60,
        re.compile(
            r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?!\.?\d)"
            r"|(?<![\w:])(?:[0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}(?![\w:])"
        ),
        _ip_valid,
    ),
    _Detector(
        PIIType.PHONE, 50,
        re.compile(
            r"(?<![\w+])(?<!\d[\s.-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{1,4}\)[\s.-]?\d{2,4}|\d{1,4})"
            r"(?:[\s.-]?\d{2,4}){1,4}(?!\w)(?![\s.-]?\d)"
        ),
        _phone_valid,
    ),
    _Detector(PIIType.PERSON, 10, custom=_detect_names),
)


class PIIDetector:
    """Find PII spans in text.

    Args:
        types: restrict detection to these PII types (default: all).
    """

    def __init__(self, types: Iterable[str] | None = None) -> None:
        wanted = set(types) if types is not None else set(PIIType.ALL)
        unknown = wanted - set(PIIType.ALL)
        if unknown:
            raise ValueError(f"unknown PII types: {sorted(unknown)}")
        self._detectors = [d for d in _DETECTORS if d.type in wanted]

    def detect(self, text: str) -> list[PIIMatch]:
        """Return non-overlapping matches sorted by position."""
        candidates: list[tuple[int, int, int, str]] = []
        for det in self._detectors:
            for start, end in det.find(text):
                # Never re-detect inside an existing mask token.
                candidates.append((start, end, det.priority, det.type))
        token_spans = [(m.start(), m.end()) for m in TOKEN_RE.finditer(text)]
        # Highest priority first, then longest.
        candidates.sort(key=lambda c: (-c[2], -(c[1] - c[0]), c[0]))
        taken: list[tuple[int, int]] = list(token_spans)
        chosen: list[PIIMatch] = []
        for start, end, _prio, typ in candidates:
            if any(start < te and ts < end for ts, te in taken):
                continue
            taken.append((start, end))
            chosen.append(PIIMatch(typ, text[start:end], start, end))
        chosen.sort(key=lambda m: m.start)
        return chosen


# --------------------------------------------------------------------------- vault


@dataclass
class _SessionVault:
    secret: bytes
    by_token: dict[str, tuple[str, str]] = field(default_factory=dict)


@dataclass
class MaskResult:
    """Output of :meth:`PIIVault.mask`."""

    text: str
    matches: list[PIIMatch]
    tokens: dict[str, str]  # token -> PII type

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for m in self.matches:
            out[m.type] = out.get(m.type, 0) + 1
        return out


class PIIVault:
    """Session-scoped reversible PII tokenisation.

    Args:
        detector: detector to use (default: all types).
        master_key: optional key from which per-session secrets are derived,
            making tokens reproducible across process restarts. If omitted a
            random key is generated (tokens then live only in memory).
    """

    def __init__(self, detector: PIIDetector | None = None, master_key: bytes | None = None) -> None:
        self.detector = detector or PIIDetector()
        self._master = master_key if master_key is not None else secrets.token_bytes(32)
        self._sessions: dict[str, _SessionVault] = {}

    def _session(self, session_id: str) -> _SessionVault:
        sv = self._sessions.get(session_id)
        if sv is None:
            secret = hmac.new(self._master, f"session:{session_id}".encode(), hashlib.sha256).digest()
            sv = _SessionVault(secret)
            self._sessions[session_id] = sv
        return sv

    def token_for(self, session_id: str, pii_type: str, value: str) -> str:
        """Deterministic token for ``value`` within ``session_id`` (and record it)."""
        sv = self._session(session_id)
        digest = hmac.new(sv.secret, f"{pii_type}\x00{value}".encode(), hashlib.sha256).digest()
        tok = f"<<PII:{pii_type}:{uuid.UUID(bytes=digest[:16], version=4)}>>"
        sv.by_token[tok] = (pii_type, value)
        return tok

    def mask(self, session_id: str, text: str) -> MaskResult:
        """Replace every PII value in ``text`` with its session token."""
        matches = self.detector.detect(text)
        parts: list[str] = []
        tokens: dict[str, str] = {}
        cursor = 0
        for m in matches:
            tok = self.token_for(session_id, m.type, m.value)
            tokens[tok] = m.type
            parts.append(text[cursor:m.start])
            parts.append(tok)
            cursor = m.end
        parts.append(text[cursor:])
        return MaskResult("".join(parts), matches, tokens)

    def unmask(self, session_id: str, text: str, allowed_types: Iterable[str] | str = ()) -> str:
        """Restore values for tokens whose type is in ``allowed_types``.

        ``allowed_types`` may be ``"*"`` for all types. Tokens of other types,
        and tokens unknown to this session (e.g. forged or from another
        session), are left untouched.
        """
        sv = self._sessions.get(session_id)
        if sv is None:
            return text
        allow_all = allowed_types == "*"
        allowed = set() if allow_all else set(allowed_types)
        if "*" in allowed:
            allow_all = True

        def repl(m: re.Match) -> str:
            entry = sv.by_token.get(m.group(0))
            if entry is None:
                return m.group(0)
            typ, value = entry
            return value if (allow_all or typ in allowed) else m.group(0)

        return TOKEN_RE.sub(repl, text)

    def size(self, session_id: str) -> int:
        """Number of distinct values recorded for a session."""
        sv = self._sessions.get(session_id)
        return len(sv.by_token) if sv else 0

    def forget(self, session_id: str) -> None:
        """Drop a session mapping; previously issued tokens can no longer be unmasked."""
        self._sessions.pop(session_id, None)
