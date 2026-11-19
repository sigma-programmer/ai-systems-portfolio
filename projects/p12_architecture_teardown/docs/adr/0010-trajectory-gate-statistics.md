# ADR-0010: Gate PRs on trajectory DAG alignment with statistical regression tests
- Status: Accepted
- Date: 2026-11-19
- Projects: p09

## Context
Final-answer evals miss agents that get the right answer the wrong way (skipping an authorization check, calling a forbidden tool, looping). Agent runs are noisy, so naive threshold comparisons block good PRs and pass bad ones.

## Decision
Golden behaviours are DAGs of required, optional and forbidden steps with partial order and argument matchers. Candidate trajectories are aligned to the DAG and graded step-by-step (rules + LLM-as-judge). The judge is calibrated against human labels (agreement and Cohen's kappa reported before/after). The CI gate compares candidate vs. baseline suites with bootstrap/permutation tests and blocks the PR (non-zero exit + markdown comment) only on statistically significant regressions or hard-policy violations.

## Consequences
Process regressions are caught even when answers still look right; noise-only changes don't block merges. Golden DAGs must be maintained as the agent legitimately evolves.

## Alternatives Considered
Exact trajectory matching (too brittle). Answer-only evals (miss process failures). Uncalibrated LLM judges (unknown error rate).
