"""security_proxy: a security layer that sits between cooperating AI agents.

Components:
    pii        – PII detection + reversible session-scoped UUID masking (vault)
    injection  – layered prompt-injection detection with a logistic scoring model
    sanitize   – neutralize / quarantine / defang / wrap untrusted content
    policy     – deny-by-default least-privilege grants with capability attenuation
    audit      – hash-chained tamper-evident JSONL audit log
    proxy      – the SecurityProxy that composes all of the above
"""
from .audit import AuditLog, VerifyResult
from .injection import InjectionDetector, InjectionReport, LLMInjectionClassifier, clean_text
from .pii import PIIDetector, PIIMatch, PIIType, PIIVault, iban_valid, luhn_valid
from .policy import AgentGrant, ArgConstraint, AttenuationError, Capability, Decision, Policy
from .proxy import Envelope, ProxyResult, SecurityProxy, ToolCall, ToolDecision
from .sanitize import Action, QuarantineStore, Sanitizer, wrap_untrusted

__all__ = [
    "AuditLog", "VerifyResult",
    "InjectionDetector", "InjectionReport", "LLMInjectionClassifier", "clean_text",
    "PIIDetector", "PIIMatch", "PIIType", "PIIVault", "iban_valid", "luhn_valid",
    "AgentGrant", "ArgConstraint", "AttenuationError", "Capability", "Decision", "Policy",
    "Envelope", "ProxyResult", "SecurityProxy", "ToolCall", "ToolDecision",
    "Action", "QuarantineStore", "Sanitizer", "wrap_untrusted",
]
