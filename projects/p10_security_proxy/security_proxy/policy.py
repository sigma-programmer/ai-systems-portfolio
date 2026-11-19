"""Least-privilege policy engine with capability attenuation.

Model
-----
* An :class:`AgentGrant` lists the agent's :class:`Capability` objects (one per
  tool, each with per-argument :class:`ArgConstraint` objects), which
  recipients it may message, and which PII types it may see unmasked.
* Everything is **deny-by-default**: an unknown agent, an unlisted tool, an
  argument without a constraint (unless ``allow_extra_args``) or a recipient
  not in ``can_message`` is denied.
* :meth:`Policy.delegate` derives a child grant from a parent. The request is
  validated to be a *subset* of the parent (tools, recipients, PII types,
  constraint ranges); the child's constraints are then the *conjunction* of
  parent and requested constraints, so delegation can only narrow scope.
  Delegation depth is bounded.

Configuration is a plain dict / JSON document (see :meth:`Policy.from_dict`).
"""
from __future__ import annotations

import fnmatch
import json
import re
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "ArgConstraint",
    "Capability",
    "AgentGrant",
    "Decision",
    "Policy",
    "AttenuationError",
]


_GLOB_CHARS = set("*?[]")


def _pattern_within(child: str, parent: str) -> bool:
    """True if every id matched by glob ``child`` is also matched by ``parent``.

    Conservative: exact equality, a literal child matched by the parent, or a
    child that extends a pure-prefix parent pattern (``"research-*"``).
    """
    if child == parent:
        return True
    if not (set(child) & _GLOB_CHARS):
        return fnmatch.fnmatchcase(child, parent)
    if parent.endswith("*") and not (set(parent[:-1]) & _GLOB_CHARS):
        return child.startswith(parent[:-1])
    return False


class AttenuationError(ValueError):
    """Raised when a delegation request would broaden the parent's authority."""


class ArgConstraint(BaseModel):
    """Constraint on a single tool argument. All set fields must hold."""

    model_config = ConfigDict(extra="forbid")

    allowed: list[Any] | None = None
    patterns: list[str] = Field(default_factory=list)  # every regex must fullmatch
    prefix: str | None = None                            # e.g. path jail
    min_value: float | None = None
    max_value: float | None = None
    max_length: int | None = None
    required: bool = False

    @field_validator("patterns")
    @classmethod
    def _compile_check(cls, v: list[str]) -> list[str]:
        for p in v:
            try:
                re.compile(p)
            except re.error as exc:  # not a ValueError, so pydantic would not wrap it
                raise ValueError(f"invalid regex {p!r}: {exc}") from exc
        return v

    def check(self, value: Any) -> str | None:
        """Return ``None`` if ``value`` satisfies the constraint, else a reason."""
        if self.allowed is not None and value not in self.allowed:
            return f"value {value!r} not in allowlist"
        if self.patterns or self.prefix is not None or self.max_length is not None:
            if not isinstance(value, str):
                return "expected string"
            for p in self.patterns:
                if not re.fullmatch(p, value):
                    return f"value does not match pattern {p!r}"
            if self.prefix is not None:
                if ".." in value.replace("\\", "/").split("/") or not value.startswith(self.prefix):
                    return f"value must start with {self.prefix!r} (no '..')"
            if self.max_length is not None and len(value) > self.max_length:
                return f"value longer than {self.max_length}"
        if self.min_value is not None or self.max_value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return "expected number"
            if self.min_value is not None and value < self.min_value:
                return f"value below minimum {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return f"value above maximum {self.max_value}"
        return None

    def narrows(self, parent: "ArgConstraint") -> str | None:
        """Return ``None`` if ``self`` is at least as strict as ``parent``."""
        if parent.allowed is not None:
            if self.allowed is None or any(v not in parent.allowed for v in self.allowed):
                return "allowlist must be a subset of the parent's"
        if parent.prefix is not None and (self.prefix is None or not self.prefix.startswith(parent.prefix)):
            return "prefix must extend the parent's prefix"
        if parent.max_value is not None and (self.max_value is None or self.max_value > parent.max_value):
            return "max_value must not exceed the parent's"
        if parent.min_value is not None and (self.min_value is None or self.min_value < parent.min_value):
            return "min_value must not be below the parent's"
        if parent.max_length is not None and (self.max_length is None or self.max_length > parent.max_length):
            return "max_length must not exceed the parent's"
        if parent.required and not self.required:
            return "cannot drop 'required'"
        return None

    def intersect(self, other: "ArgConstraint") -> "ArgConstraint":
        """Conjunction of two constraints (strictest of each field)."""
        def tighter(a, b, fn):
            if a is None:
                return b
            if b is None:
                return a
            return fn(a, b)

        allowed = tighter(self.allowed, other.allowed, lambda a, b: [v for v in a if v in b])
        prefix = tighter(self.prefix, other.prefix, lambda a, b: a if len(a) >= len(b) else b)
        return ArgConstraint(
            allowed=allowed,
            patterns=list(dict.fromkeys(self.patterns + other.patterns)),
            prefix=prefix,
            min_value=tighter(self.min_value, other.min_value, max),
            max_value=tighter(self.max_value, other.max_value, min),
            max_length=tighter(self.max_length, other.max_length, min),
            required=self.required or other.required,
        )


class Capability(BaseModel):
    """Permission to call one tool, subject to argument constraints."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    constraints: dict[str, ArgConstraint] = Field(default_factory=dict)
    allow_extra_args: bool = False
    max_calls_per_message: int | None = None


class AgentGrant(BaseModel):
    """Everything an agent is permitted to do."""

    model_config = ConfigDict(extra="forbid")

    agent_id: str
    capabilities: list[Capability] = Field(default_factory=list)
    can_message: list[str] = Field(default_factory=list)  # glob patterns of recipient ids
    pii_unmask: list[str] = Field(default_factory=list)   # PII types or "*"
    trusted: bool = False                                  # trusted agents' content skips wrapping
    parent: str | None = None
    depth: int = 0

    def capability(self, tool: str) -> Capability | None:
        for cap in self.capabilities:
            if cap.tool == tool:
                return cap
        return None


class Decision(BaseModel):
    """Outcome of a policy check."""

    allowed: bool
    reason: str
    rule: str = ""


class Policy:
    """Collection of agent grants with deny-by-default evaluation."""

    def __init__(self, grants: Iterable[AgentGrant] = (), max_delegation_depth: int = 3) -> None:
        self.grants: dict[str, AgentGrant] = {}
        self.max_delegation_depth = max_delegation_depth
        for g in grants:
            self.add(g)

    # ---------------------------------------------------------------- config
    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "Policy":
        """Build from ``{"max_delegation_depth": 3, "agents": {id: {...grant fields}}}``."""
        agents = config.get("agents", {})
        grants = [AgentGrant(agent_id=aid, **spec) for aid, spec in agents.items()]
        return cls(grants, max_delegation_depth=int(config.get("max_delegation_depth", 3)))

    @classmethod
    def from_json(cls, text: str) -> "Policy":
        return cls.from_dict(json.loads(text))

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_delegation_depth": self.max_delegation_depth,
            "agents": {
                aid: g.model_dump(exclude={"agent_id"}, exclude_none=True) for aid, g in self.grants.items()
            },
        }

    def add(self, grant: AgentGrant) -> None:
        self.grants[grant.agent_id] = grant

    def get(self, agent_id: str) -> AgentGrant | None:
        return self.grants.get(agent_id)

    # ---------------------------------------------------------------- checks
    def check_message(self, sender: str, recipient: str) -> Decision:
        grant = self.grants.get(sender)
        if grant is None:
            return Decision(allowed=False, reason=f"unknown sender {sender!r}", rule="deny_default")
        if recipient not in self.grants:
            return Decision(allowed=False, reason=f"unknown recipient {recipient!r}", rule="deny_default")
        for pat in grant.can_message:
            if fnmatch.fnmatchcase(recipient, pat):
                return Decision(allowed=True, reason=f"recipient matches {pat!r}", rule="can_message")
        return Decision(allowed=False, reason=f"{sender} may not message {recipient}", rule="can_message")

    def check_tool_call(self, agent_id: str, tool: str, args: dict[str, Any]) -> Decision:
        grant = self.grants.get(agent_id)
        if grant is None:
            return Decision(allowed=False, reason=f"unknown agent {agent_id!r}", rule="deny_default")
        cap = grant.capability(tool)
        if cap is None:
            return Decision(allowed=False, reason=f"tool {tool!r} not granted to {agent_id}", rule="deny_default")
        for name, constraint in cap.constraints.items():
            if name not in args:
                if constraint.required:
                    return Decision(allowed=False, reason=f"missing required argument {name!r}", rule="constraint")
                continue
            why = constraint.check(args[name])
            if why:
                return Decision(allowed=False, reason=f"argument {name!r}: {why}", rule="constraint")
        if not cap.allow_extra_args:
            extra = sorted(set(args) - set(cap.constraints))
            if extra:
                return Decision(allowed=False, reason=f"unconstrained arguments not allowed: {extra}", rule="extra_args")
        return Decision(allowed=True, reason=f"granted by capability {tool!r}", rule="capability")

    def unmask_types(self, agent_id: str) -> set[str]:
        grant = self.grants.get(agent_id)
        return set(grant.pii_unmask) if grant else set()

    # ---------------------------------------------------------------- delegation
    def delegate(
        self,
        parent_id: str,
        child_id: str,
        capabilities: list[Capability | dict] | None = None,
        can_message: list[str] | None = None,
        pii_unmask: list[str] | None = None,
    ) -> AgentGrant:
        """Create ``child_id`` with a strictly attenuated subset of ``parent_id``'s grant.

        ``None`` for a field means "inherit nothing" for capabilities/PII and
        "inherit nothing" for messaging — the child must ask for what it needs.
        Raises :class:`AttenuationError` if anything requested exceeds the parent.
        """
        parent = self.grants.get(parent_id)
        if parent is None:
            raise AttenuationError(f"unknown parent {parent_id!r}")
        if child_id in self.grants:
            raise AttenuationError(f"agent {child_id!r} already exists")
        if parent.depth + 1 > self.max_delegation_depth:
            raise AttenuationError("maximum delegation depth exceeded")

        child_caps: list[Capability] = []
        for raw in capabilities or []:
            req = raw if isinstance(raw, Capability) else Capability(**raw)
            pcap = parent.capability(req.tool)
            if pcap is None:
                raise AttenuationError(f"parent lacks tool {req.tool!r}")
            if req.allow_extra_args and not pcap.allow_extra_args:
                raise AttenuationError(f"{req.tool}: cannot enable allow_extra_args")
            if not pcap.allow_extra_args:
                extra = set(req.constraints) - set(pcap.constraints)
                if extra:
                    raise AttenuationError(f"{req.tool}: arguments {sorted(extra)} not permitted by parent")
            if pcap.max_calls_per_message is not None and (
                req.max_calls_per_message is None or req.max_calls_per_message > pcap.max_calls_per_message
            ):
                raise AttenuationError(f"{req.tool}: max_calls_per_message must not exceed parent's")
            merged: dict[str, ArgConstraint] = {}
            for name, pcon in pcap.constraints.items():
                rcon = req.constraints.get(name)
                if rcon is None:
                    merged[name] = pcon
                    continue
                why = rcon.narrows(pcon)
                if why:
                    raise AttenuationError(f"{req.tool}.{name}: {why}")
                merged[name] = pcon.intersect(rcon)
            for name, rcon in req.constraints.items():
                merged.setdefault(name, rcon)
            child_caps.append(
                Capability(
                    tool=req.tool,
                    constraints=merged,
                    allow_extra_args=req.allow_extra_args and pcap.allow_extra_args,
                    max_calls_per_message=req.max_calls_per_message,
                )
            )

        msg = list(can_message or [])
        for pat in msg:
            if not any(_pattern_within(pat, ppat) for ppat in parent.can_message):
                raise AttenuationError(f"recipient pattern {pat!r} exceeds parent's messaging rights")
        pii = list(pii_unmask or [])
        parent_pii = set(parent.pii_unmask)
        for t in pii:
            if "*" not in parent_pii and t not in parent_pii:
                raise AttenuationError(f"PII type {t!r} not unmaskable by parent")
            if t == "*" and "*" not in parent_pii:
                raise AttenuationError("cannot request '*' PII unmask")

        child = AgentGrant(
            agent_id=child_id,
            capabilities=child_caps,
            can_message=msg,
            pii_unmask=pii,
            trusted=False,  # trust is never inherited
            parent=parent_id,
            depth=parent.depth + 1,
        )
        self.add(child)
        return child
