# ADR-0009: Typed handoffs and execution evidence over opinion in swarm consensus
- Status: Accepted
- Date: 2026-11-19
- Projects: p08

## Context
Multi-agent systems fail silently at handoffs (a malformed message propagates) and in review (agents agreeing with each other without evidence). A single faulty reviewer can flip a decision.

## Decision
All handoffs are Pydantic models validated at the boundary, and the topology graph is type-checked when built (an edge whose output type doesn't match the consumer's input is rejected). Invalid outputs trigger structured repair/retry. Reviewers execute generated code against tests in a sandboxed subprocess; consensus supports majority, weighted and quorum rules, and conflicts are broken by evidence strength — test execution beats opinion — before escalating to the supervisor.

## Consequences
Wiring errors surface at build time, not mid-run; consensus stays correct with one adversarial reviewer. Extra reviewers cost tokens and wall time, quantified in the benchmark.

## Alternatives Considered
Free-text message passing (fragile). Single reviewer (single point of failure). Pure LLM-vote consensus (correlated errors).
