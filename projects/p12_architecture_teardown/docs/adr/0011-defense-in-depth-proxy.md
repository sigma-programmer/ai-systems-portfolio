# ADR-0011: A mediating security proxy between agents, deny-by-default
- Status: Accepted
- Date: 2026-11-19
- Projects: p10

## Context
In multi-agent systems, one agent's output is another's input. A compromised or manipulated agent (e.g. via a poisoned web page) can inject instructions, exfiltrate PII, or request tools it should never use.

## Decision
All inter-agent traffic passes a proxy that: masks PII with stable per-session UUID tokens (reversible only for authorized recipients); detects prompt injection with layered rules, decoding of obfuscations (base64, hex, zero-width, unicode tags) and a scoring model, optionally an LLM classifier; wraps untrusted content in data delimiters; enforces least-privilege tool grants where delegation can only attenuate capabilities; and writes a hash-chained, tamper-evident audit log.

## Consequences
Measurable precision/recall on bundled attack and benign corpora; every decision is auditable. Pattern-based detection cannot catch every novel attack — the proxy is one layer, combined with capability limits that bound the blast radius when detection fails.

## Alternatives Considered
Relying on model refusals alone (not a security boundary). Per-agent ad hoc filtering (inconsistent, unauditable).
