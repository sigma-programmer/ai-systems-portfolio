# ADR-0006: Event-sourced durable execution on SQLite
- Status: Accepted
- Date: 2026-11-19
- Projects: p05

## Context
Long-running agent workflows (minutes to days, waiting on humans) will see process crashes, deploys and transient API failures. Restarting from scratch duplicates side effects (emails sent twice, tickets filed twice) and wastes tokens.

## Decision
Every step transition is appended to an event log and the state snapshot is written in the same SQLite transaction. On resume, completed steps replay their recorded outputs instead of re-executing. Side-effecting steps carry idempotency keys. Retries use exponential backoff with full jitter and a retryable/fatal error taxonomy; exhausted runs go to a dead-letter state. Human approval gates suspend the run durably so any engine instance can resume it.

## Consequences
Crash recovery is verified by killing a real subprocess mid-workflow; side effects execute exactly once. SQLite limits throughput to a single writer — fine for an agent control plane, not for high-fan-out job queues (swap the store interface for Postgres there).

## Alternatives Considered
Temporal / a workflow engine (excellent, but heavy infra for a portfolio and hides the mechanics being demonstrated). In-memory state with periodic snapshots (loses work between snapshots).
