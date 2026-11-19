# ADR-0007: Implement MCP on raw JSON-RPC with Pydantic-generated schemas
- Status: Accepted
- Date: 2026-11-19
- Projects: p06

## Context
SDKs hide protocol details that matter in production: the difference between a JSON-RPC protocol error and a tool execution error, capability negotiation, pagination, and where validation happens. Tool input schemas written by hand drift from the code that consumes them.

## Decision
Implement JSON-RPC 2.0 and the MCP lifecycle directly (stdio and HTTP transports). Tools are registered with a Pydantic v2 model (`extra="forbid"`) whose JSON Schema becomes `inputSchema`; malformed params map to -32602, while failures inside a tool return a result with `isError: true` so the model can self-correct. Tools declare required scopes; `tools/list` only shows what the caller may call. Filesystem tools are jailed to a root; code execution runs in an isolated subprocess with rlimits, timeout, empty env and output caps.

## Consequences
Full control and auditability of the wire protocol with zero SDK dependency. We own spec conformance and must track protocol revisions ourselves.

## Alternatives Considered
Official MCP SDK (faster to build, less instructive, extra dependency). Hand-written JSON Schemas (drift risk).
