# P08 — Typed Multi-Agent Swarm

> You orchestrate swarms, not fragile single agents.

A Researcher / Coder / Reviewer swarm that turns a natural-language function spec into working, verified
Python. Every hand-off is a typed Pydantic v2 message validated at the boundary; the topology is a
networkx DAG type-checked at build time; a supervisor plans, routes, runs coder↔reviewer revise loops
under token/step/wall budgets, and settles reviewer disagreements with evidence-weighted consensus and,
when needed, by re-executing the tests itself.

## Problem

A single LLM agent writing code has nobody checking its work. When one reviewer is added, that reviewer
becomes a single point of failure: if it is wrong, lazy or adversarial, bad code ships. Production swarms need:

* **typed contracts** between agents, so malformed output is repaired instead of crashing the run or
  quietly corrupting later steps,
* **topology checks** before any tokens are spent,
* **verifiable evidence** (code that was actually executed) rather than opinions,
* **consensus that survives a faulty member**, plus budgets and clear termination.

## Architecture

```
                         ┌────────────────────── Supervisor ──────────────────────┐
                         │ plan(task) -> subgoals + routed steps (topological)    │
                         │ budgets: steps / tokens / wall / max_revisions         │
                         │ arbitrate(): re-execute trusted suite on disputes      │
                         │ event trace -> transcript()                            │
                         └────────────────────────────────────────────────────────┘
 TaskSpec                     ResearchFindings          CodeArtifact          ReviewVerdict
 ───────▶ [Researcher] ──────────────────▶ [Coder] ──┬──▶ [exec reviewer] ──┐
          HashingEmbedder retrieval         CEGIS     ├──▶ [sampling rev.] ──┼──▶ [ConsensusPanel] ──▶ ConsensusDecision
          over knowledge base,              search    ├──▶ [static rev.]   ──┤    majority | weighted | quorum
          parses REQ:/EDGE: lines           (library  └──▶ [faulty rev.]?  ──┘    + evidence-tier conflict
                                            + AST                                   resolution + escalation
                                            mutants)          ▲                               │
                                                              └────── RevisionRequest ◀───────┘
                                                                  (feedback edge ~~>, typed;
                                                                   counterexamples + rejected hashes)

  Sandbox: AST static gate -> `python -I -S` subprocess, per-test SIGALRM timeout, rlimits,
           deep-copied args, memoised (source, test) results shared by all reviewers.
```

| Module | Responsibility |
|---|---|
| `messages.py` | `TaskSpec`, `ResearchFindings`, `CodeArtifact`, `ReviewVerdict`, `Evidence`, `ConsensusDecision`, `RevisionRequest`, `SwarmState`, `Event`. Validators check meaning, not only shape. |
| `agents.py` | `Agent` base class (role, `input_types`, `output_types`, optional `LLM`) with `handle()` = type check → act → validate → repair → fallback. `Researcher`, `Coder`, `ExecutionReviewer`, `SamplingReviewer`, `StaticReviewer`, `LLMReviewer`, `FaultyReviewer`. |
| `topology.py` | Typed `Topology` graph with build-time validation. |
| `consensus.py` | `ConsensusPanel`, `Rule`, `ReviewerLedger`. |
| `supervisor.py` | `Supervisor` (planning, routing, loops, budgets, arbitration), `Budget`, `transcript()`. |
| `sandbox.py` | `static_check` (AST) and `Sandbox` (subprocess execution with memoisation). |
| `synthesis.py` | Candidate pool (library + single-point AST mutants) and the ranking prior. |
| `library.py` | 15 bundled spec tasks with hidden tests, a 17-snippet knowledge base, and the candidate program library. |
| `factory.py`, `bench.py` | Ready-made configurations, hidden-test scoring, and the benchmark. |

## Key design decisions and tradeoffs

* **Validation encodes meaning.**
  * `CodeArtifact` must parse and define the requested function.
  * `ReviewVerdict` cannot be `approved=True` while its own test evidence lists failures.
  * `Evidence` of kind test-execution must report `tests_run > 0`.
  * When an LLM agent breaks a contract, it gets the exact Pydantic error back in a repair prompt, up to
    `max_repairs` times. After that it falls back to the algorithmic default. A `HandoffError` reaches the
    supervisor only if both fail, and the supervisor ends the run with status `handoff_failed` rather than
    crashing.
* **The topology is type-checked before it runs.** Every edge carries a payload type that the source must
  produce and the target must accept. Forward edges must form a DAG. Feedback edges (revise loops) are
  declared separately and must point to an upstream agent. Entry agents must accept `TaskSpec`. Wiring the
  coder directly into the panel is rejected with
  `CodeArtifact is not accepted by panel (consensus); it accepts ['ReviewVerdict']`.
* **The offline coder is CEGIS** (counterexample-guided inductive synthesis), not canned output. The search
  space is the library implementations plus single-point AST mutants: comparison flips, integer ±1, and
  `and`/`or` swaps. Candidates are ranked by an Occam prior (simpler first) plus a bonus for tags that
  match the researcher's requirements. The coder submits the best-ranked candidate consistent with every
  example and every counterexample collected so far. Reviewers supply the counterexamples, so each
  rejection shrinks the search space. Example: the binary-search template has an off-by-one loop
  (`lo < hi`), and the shipped fix (`lo <= hi`) was found by mutation.
* **Three sources of truth, kept apart.**
  * The spec is visible to every agent.
  * The knowledge base is visible only through the Researcher.
  * The hidden tests are visible to nobody and are used only for scoring.
  * Knowledge coverage is deliberately imperfect: the leap-year note omits the 400-year rule. As a result
    the swarm confidently ships `year % 4 == 0 and year % 100 != 0`, which counts as one false approval in
    every swarm configuration. That is an honest limit of research-driven review.
* **Evidence tiers drive conflict resolution:** `TEST_EXECUTION > STATIC_ANALYSIS > OPINION`.
  * If the losing side of a vote holds stronger trusted evidence, the evidence wins.
  * If both sides hold test-execution evidence, the disagreement can be checked, so the supervisor
    **escalates**: it re-runs the trusted suite (spec examples plus researched edge cases) and its result is
    final. This catches a reviewer that fabricates test results.
* **Weighted voting uses each reviewer's history.**
  * `ReviewerLedger` keeps a Laplace-smoothed record of how often each reviewer agreed with final decisions.
  * After `min_history` decisions, a reviewer whose accuracy is below `trust_floor` still votes, but its
    evidence can no longer force an escalation.
  * This is why `weighted` needs 4 escalations against an adversary where `majority` and `quorum` need 18.
    The cost is a short learning period.
* **Token accounting.**
  * With a real LLM, actual usage is counted.
  * Offline, each agent is charged `count_tokens(inputs) + count_tokens(output)`, the context an LLM agent
    would have used. This keeps the comparison between topologies meaningful.
* **Sandbox design: a subprocess, not `exec` in the parent.**
  * Tradeoff: about 25 ms of startup per batch. In exchange, a crash, hang or `sys.exit` in a candidate
    cannot take down the orchestrator.
  * Many candidates run in one subprocess, with a per-test SIGALRM. A mutant that loops forever simply
    counts as inconsistent.
  * Memoisation means several reviewers checking the same artifact pay for execution once.

## How to run

```bash
AIKIT_OFFLINE=1 .venv/bin/python projects/p08_agent_swarm/demo.py         # full walkthrough
.venv/bin/python -m pytest projects/p08_agent_swarm                        # 91 tests
cd projects/p08_agent_swarm && PYTHONPATH=../../common:. ../../.venv/bin/python -m agent_swarm.bench
```

The demo shows:

1. the typed topology and the rejection of an ill-typed one,
2. a full transcript of a consensus run with a revise cycle,
3. the single agent failing the same task,
4. an adversarial reviewer caught by escalation,
5. an LLM coder (MockLLM) whose malformed output is repaired twice through validation feedback,
6. a token budget stopping a run,
7. the benchmark table.

**Using Claude.** Pass `llm=aikit.get_llm()` (or `ClaudeLLM()`) to `Researcher`, `Coder` or `LLMReviewer`,
or to `build_topology(..., llm=...)`. The same validation, repair and fallback path applies.

## Benchmark (measured, 15 bundled tasks, Apple Silicon, `benchmark()` ≈ 9 s total)

**Success** means the final artifact passes every hidden test. **Verdict accuracy** means the panel's decisions
match a re-execution of the full trusted suite. The three main configurations each start with a cold sandbox, so their wall times are comparable. The resilience runs share a warm sandbox.

| Configuration | Success | Verdict acc. | Avg revisions | Avg tokens | Escalations | Wall (s) |
|---|---|---|---|---|---|---|
| single_agent (coder only) | 0.000 | – | 0.00 | 330 | 0 | 0.68 |
| swarm_review (1 exec reviewer) | **0.933** | 1.000 | 0.20 | 2344 | 0 | 1.31 |
| swarm_consensus (exec + sampling + static, weighted) | **0.933** | 1.000 | 0.20 | 3522 | 1 | 1.42 |

Resilience: a `FaultyReviewer` inverts the honest verdict and fabricates test evidence that passes schema
validation, at 0.97 confidence.

| Configuration | Success | Verdict acc. | Escalations | Notes |
|---|---|---|---|---|
| single_reviewer_adversarial | 0.000 | 0.000 | 0 | The only reviewer lies, and all 15 tasks ship wrong code (15 false approvals). |
| naive_majority_adversarial (no conflict resolution) | 0.933 | 0.941 | 0 | Head count alone gives 1 wrong verdict in 17. |
| majority_adversarial | 0.933 | **1.000** | 18 | Every contested test claim is re-verified. |
| weighted_adversarial | 0.933 | **1.000** | **4** | The ledger drops the liar's accuracy to 0.05, so its evidence stops forcing escalations. |
| quorum_adversarial (2/3) | 0.933 | **1.000** | 18 | |
| weighted_always_approve (rubber stamp) | 0.933 | **1.000** | 3 | |

Reading the numbers:

* **Single agent scores 0/15.** Without research it ranks the simplest candidate first, and that candidate
  passes the two spec examples but misses edge cases in every task (naive palindrome, FizzBuzz checked in the
  wrong order, no subtractive Roman numerals, and so on). That is the "fragile single agent" failure mode,
  reproduced deterministically.
* **One false approval in every swarm row** is the leap-year knowledge gap. Review can only be as good as the
  tests it knows about. A unit test shows that a reviewer which does know the 400-year rule drives the coder
  to the correct candidate.
* **Revisions are low (0.2 per task)** because the researcher's requirements usually rank the right candidate
  first. Revise loops happen where the requirement tags are not enough: merge_intervals, second_largest and
  leap_year each need one counterexample round.
* **Token cost:** consensus costs about 1.5× swarm_review in tokens and buys tolerance of a faulty member.
  Weighted voting is the cheapest rule that keeps verdicts correct.
