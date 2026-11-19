"""The inter-agent security proxy.

Every message between agents passes through :meth:`SecurityProxy.route`::

     sender agent                                                recipient agent
         │  Envelope(sender, recipient, content, tool_calls)            ▲
         ▼                                                              │
    ┌─────────────────────────── SecurityProxy ─────────────────────────┴──┐
    │ 1 policy: may sender message recipient?            (deny-by-default) │
    │ 2 injection scan of content + every tool-call argument               │
    │ 3 sanitize: strip invisibles, neutralize spans / quarantine, defang  │
    │ 4 PII mask with session UUID tokens (vault)                          │
    │ 5 tool calls: taint check + capability/argument policy per call      │
    │ 6 unmask only PII types the recipient is scoped for                  │
    │ 7 wrap untrusted content in nonce-tagged data delimiters             │
    │ 8 hash-chained audit record (masked content only)                    │
    └──────────────────────────────────────────────────────────────────────┘

Tool results flowing back into an agent are filtered the same way through
:meth:`SecurityProxy.filter_tool_result`.
"""
from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .audit import AuditLog
from .injection import InjectionDetector, InjectionReport
from .pii import PIIVault
from .policy import Decision, Policy
from .sanitize import Action, Sanitizer, wrap_untrusted

__all__ = ["ToolCall", "Envelope", "ToolDecision", "ProxyResult", "SecurityProxy"]


class ToolCall(BaseModel):
    """A tool invocation requested inside a message."""

    model_config = ConfigDict(extra="forbid")

    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class Envelope(BaseModel):
    """An inter-agent message."""

    model_config = ConfigDict(extra="forbid")

    sender: str
    recipient: str
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    session_id: str = "default"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolDecision(BaseModel):
    tool: str
    allowed: bool
    reason: str


class ProxyResult(BaseModel):
    """What the proxy did with an envelope."""

    delivered: bool
    envelope: Envelope | None = None
    action: str = Action.ALLOW.value
    reason: str = ""
    injection: dict[str, Any] = Field(default_factory=dict)
    pii_counts: dict[str, int] = Field(default_factory=dict)
    tool_decisions: list[ToolDecision] = Field(default_factory=list)
    quarantine_id: str | None = None
    audit_seq: int | None = None
    latency_ms: float = 0.0


def _map_strings(value: Any, fn) -> Any:
    """Apply ``fn`` to every string inside a JSON-like value."""
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, list):
        return [_map_strings(v, fn) for v in value]
    if isinstance(value, dict):
        return {k: _map_strings(v, fn) for k, v in value.items()}
    return value


class SecurityProxy:
    """Mediates all inter-agent traffic.

    Args:
        policy: least-privilege :class:`Policy`.
        detector: injection detector (default rule model, no LLM).
        vault: PII vault (default: fresh random-key vault).
        sanitizer: sanitizer (default thresholds 0.5 / 0.97).
        audit: audit log (default: in-memory).
        wrap: wrap content from untrusted senders in data delimiters.
    """

    def __init__(
        self,
        policy: Policy,
        detector: InjectionDetector | None = None,
        vault: PIIVault | None = None,
        sanitizer: Sanitizer | None = None,
        audit: AuditLog | None = None,
        wrap: bool = True,
    ) -> None:
        self.policy = policy
        self.detector = detector or InjectionDetector()
        self.vault = vault or PIIVault()
        self.sanitizer = sanitizer or Sanitizer()
        self.audit = audit if audit is not None else AuditLog()
        self.wrap = wrap

    # ------------------------------------------------------------------ helpers
    def _mask(self, session: str, text: str) -> tuple[str, dict[str, int]]:
        res = self.vault.mask(session, text)
        return res.text, res.counts

    def _mask_value(self, session: str, value: Any, counts: dict[str, int]) -> Any:
        def fn(s: str) -> str:
            masked, c = self._mask(session, s)
            for k, v in c.items():
                counts[k] = counts.get(k, 0) + v
            return masked

        return _map_strings(value, fn)

    def _deny(self, env: Envelope, reason: str, t0: float, **extra: Any) -> ProxyResult:
        masked, counts = self._mask(env.session_id, env.content)
        rec = self.audit.append(
            "message.blocked",
            {
                "sender": env.sender,
                "recipient": env.recipient,
                "session": env.session_id,
                "reason": reason,
                "content_masked": masked[:2000],
                **extra,
            },
        )
        return ProxyResult(
            delivered=False,
            action="block",
            reason=reason,
            pii_counts=counts,
            audit_seq=rec["seq"],
            latency_ms=(time.perf_counter() - t0) * 1000,
            **{k: v for k, v in extra.items() if k in {"injection", "tool_decisions"}},
        )

    # ------------------------------------------------------------------ main API
    def route(self, env: Envelope) -> ProxyResult:
        """Inspect, transform and (maybe) deliver one envelope."""
        t0 = time.perf_counter()
        sender_grant = self.policy.get(env.sender)

        # 1. messaging permission
        msg_decision = self.policy.check_message(env.sender, env.recipient)
        if not msg_decision.allowed:
            return self._deny(env, msg_decision.reason, t0)

        # 2. injection scan on content
        report: InjectionReport = self.detector.scan(env.content)
        injection_info = report.to_dict()

        # 3. sanitize (quarantined copies are stored PII-masked)
        masked_raw, _ = self._mask(env.session_id, env.content)
        san = self.sanitizer.sanitize(
            env.content,
            report,
            meta={"sender": env.sender, "recipient": env.recipient, "session": env.session_id},
            store_text=masked_raw,
        )

        # 4. PII masking of the sanitized content
        masked_content, pii_counts = self._mask(env.session_id, san.text)

        # 5. tool calls: taint + policy
        tool_decisions: list[ToolDecision] = []
        approved: list[ToolCall] = []
        tainted = san.action is not Action.ALLOW
        per_tool_count: dict[str, int] = {}
        for call in env.tool_calls:
            decision = self._check_tool_call(env.sender, call, tainted, per_tool_count)
            tool_decisions.append(ToolDecision(tool=call.name, allowed=decision.allowed, reason=decision.reason))
            if decision.allowed:
                approved.append(ToolCall(name=call.name, args=self._mask_value(env.session_id, call.args, pii_counts)))

        # 6. unmask for recipient scopes
        allowed_types = self.policy.unmask_types(env.recipient)
        delivered_content = self.vault.unmask(env.session_id, masked_content, allowed_types)
        delivered_calls = [
            ToolCall(
                name=c.name,
                args=_map_strings(c.args, lambda s: self.vault.unmask(env.session_id, s, allowed_types)),
            )
            for c in approved
        ]

        # 7. wrap untrusted content
        trusted = bool(sender_grant and sender_grant.trusted) and san.action is Action.ALLOW
        if self.wrap and not trusted and delivered_content:
            delivered_content = wrap_untrusted(delivered_content, f"agent:{env.sender}")

        out = Envelope(
            sender=env.sender,
            recipient=env.recipient,
            content=delivered_content,
            tool_calls=delivered_calls,
            session_id=env.session_id,
            metadata={**env.metadata, "proxy": {"action": san.action.value, "score": round(report.score, 4)}},
        )

        # 8. audit (masked only)
        rec = self.audit.append(
            "message.routed",
            {
                "sender": env.sender,
                "recipient": env.recipient,
                "session": env.session_id,
                "action": san.action.value,
                "injection": injection_info,
                "pii_counts": pii_counts,
                "content_masked": masked_content[:2000],
                "tools": [d.model_dump() for d in tool_decisions],
                "unmasked_types": sorted(allowed_types),
                "quarantine_id": san.quarantine_id,
            },
        )
        return ProxyResult(
            delivered=True,
            envelope=out,
            action=san.action.value,
            reason=msg_decision.reason,
            injection=injection_info,
            pii_counts=pii_counts,
            tool_decisions=tool_decisions,
            quarantine_id=san.quarantine_id,
            audit_seq=rec["seq"],
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def _check_tool_call(
        self, agent: str, call: ToolCall, tainted: bool, per_tool_count: dict[str, int]
    ) -> Decision:
        if tainted:
            return Decision(allowed=False, reason="message content flagged as injection; tool calls dropped", rule="taint")
        arg_report = self.detector.scan_value(call.args)
        if arg_report.is_injection:
            return Decision(
                allowed=False,
                reason=f"injection detected in arguments ({','.join(arg_report.categories)})",
                rule="arg_injection",
            )
        decision = self.policy.check_tool_call(agent, call.name, call.args)
        if not decision.allowed:
            return decision
        grant = self.policy.get(agent)
        cap = grant.capability(call.name) if grant else None
        per_tool_count[call.name] = per_tool_count.get(call.name, 0) + 1
        if cap and cap.max_calls_per_message is not None and per_tool_count[call.name] > cap.max_calls_per_message:
            return Decision(allowed=False, reason=f"more than {cap.max_calls_per_message} {call.name} calls in one message", rule="rate")
        return decision

    def filter_tool_result(self, agent: str, tool: str, result: str, session_id: str = "default") -> ProxyResult:
        """Filter untrusted tool output (web pages, files, API responses) before an agent sees it."""
        t0 = time.perf_counter()
        report = self.detector.scan(result)
        masked_raw, _ = self._mask(session_id, result)
        san = self.sanitizer.sanitize(result, report, meta={"tool": tool, "agent": agent}, store_text=masked_raw)
        masked, counts = self._mask(session_id, san.text)
        allowed_types = self.policy.unmask_types(agent)
        content = self.vault.unmask(session_id, masked, allowed_types)
        if self.wrap:
            content = wrap_untrusted(content, f"tool:{tool}")
        rec = self.audit.append(
            "tool_result.filtered",
            {
                "agent": agent,
                "tool": tool,
                "session": session_id,
                "action": san.action.value,
                "injection": report.to_dict(),
                "pii_counts": counts,
                "content_masked": masked[:2000],
            },
        )
        return ProxyResult(
            delivered=True,
            envelope=Envelope(sender=f"tool:{tool}", recipient=agent, content=content, session_id=session_id),
            action=san.action.value,
            injection=report.to_dict(),
            pii_counts=counts,
            quarantine_id=san.quarantine_id,
            audit_seq=rec["seq"],
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def unmask_for(self, agent: str, text: str, session_id: str = "default") -> str:
        """Explicitly unmask ``text`` for ``agent`` (scoped by policy), audited."""
        types = self.policy.unmask_types(agent)
        self.audit.append("pii.unmask", {"agent": agent, "session": session_id, "types": sorted(types)})
        return self.vault.unmask(session_id, text, types)
