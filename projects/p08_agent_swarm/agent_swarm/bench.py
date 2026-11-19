"""Benchmark: single agent vs swarm-with-review vs swarm-with-consensus, plus fault resilience.

Metrics per configuration over the bundled spec tasks:

* ``success_rate``   — final artifact passes every *hidden* test (never shown to agents)
* ``approved_rate``  — the swarm itself approved its final artifact
* ``false_approvals``— approved but fails hidden tests (research coverage gaps)
* ``avg_revisions``, ``avg_tokens``, ``avg_steps``, ``wall_s``
* ``verdict_accuracy`` — fraction of consensus decisions that match ground
  truth, where ground truth = re-executing the full trusted suite (spec
  examples + researched edge cases) on the reviewed artifact.
* ``escalations``    — disputes the supervisor had to settle by re-execution.

Resilience: a `FaultyReviewer` (adversarial: inverts the honest verdict and
fabricates schema-valid test evidence) is injected. With a single reviewer the
swarm is helpless; with consensus + evidence-based conflict resolution the
verdicts should stay correct.
"""
from __future__ import annotations

import json
import time
from typing import Any

from .consensus import ReviewerLedger, Rule
from .factory import build_supervisor, passes_hidden
from .library import TASKS, TaskDef
from .messages import SwarmState
from .sandbox import Sandbox


def _decision_accuracy(states: list[SwarmState], sandbox: Sandbox) -> tuple[int, int]:
    correct = total = 0
    for st in states:
        if st.findings is None:
            continue
        suite = st.findings.test_suite()
        by_hash = {a.hash: a for a in st.artifacts}
        for d in st.decisions:
            art = by_hash[d.artifact_hash]
            truth = sandbox.run(art.source, st.task.func_name, suite, arity=len(st.task.params)).passed
            correct += int(d.approved == truth)
            total += 1
    return correct, total


def run_config(mode: str, tasks: list[TaskDef], scorer: Sandbox, sandbox: Sandbox | None = None,
               **kw: Any) -> dict[str, Any]:
    """Run one configuration over all tasks.

    By default each configuration gets a fresh swarm sandbox (cold cache, so wall
    times are comparable); pass ``sandbox`` to share a warm one. One ledger per
    configuration (reviewer history accumulates across tasks). ``scorer`` is a
    separate sandbox used only for ground-truth scoring.
    """
    ledger = ReviewerLedger()
    sandbox = sandbox if sandbox is not None else Sandbox()
    spawned_before = sandbox.executions
    states: list[SwarmState] = []
    success = approved = false_appr = 0
    t0 = time.perf_counter()
    for td in tasks:
        st = build_supervisor(mode, sandbox=sandbox, ledger=ledger, **kw).run(td.spec)
        states.append(st)
    wall = time.perf_counter() - t0
    for td, st in zip(tasks, states):
        ok = passes_hidden(st.final_artifact, td, scorer)
        success += ok
        approved += st.approved
        false_appr += st.approved and not ok
    n = len(tasks)
    correct, total = _decision_accuracy(states, scorer)
    return {
        "success_rate": round(success / n, 3),
        "approved_rate": round(approved / n, 3),
        "false_approvals": false_appr,
        "avg_revisions": round(sum(s.revisions for s in states) / n, 3),
        "avg_tokens": round(sum(s.tokens for s in states) / n, 1),
        "avg_steps": round(sum(s.steps for s in states) / n, 2),
        "escalations": sum(s.escalations for s in states),
        "verdict_accuracy": round(correct / total, 3) if total else None,
        "decisions": total,
        "wall_s": round(wall, 3),
        "subprocesses": sandbox.executions - spawned_before,
        "statuses": {k: sum(s.status == k for s in states) for k in sorted({s.status for s in states})},
        "ledger": ledger.snapshot() if mode != "single_agent" else {},
    }


def benchmark(tasks: list[TaskDef] | None = None) -> dict[str, Any]:
    """Deterministic benchmark (wall times aside). Runs in a few seconds."""
    tasks = tasks if tasks is not None else TASKS
    t0 = time.perf_counter()
    sb = Sandbox()
    modes = {
        "single_agent": run_config("single_agent", tasks, sb),
        "swarm_review": run_config("swarm_review", tasks, sb),
        "swarm_consensus": run_config("swarm_consensus", tasks, sb, rule=Rule.WEIGHTED),
    }
    # Resilience runs share one warm sandbox (their wall times are not compared; verdicts are).
    warm = Sandbox()
    resilience = {
        "single_reviewer_adversarial": run_config("swarm_review", tasks, sb, warm, faulty="adversarial"),
        "naive_majority_adversarial": run_config("swarm_consensus", tasks, sb, warm, rule=Rule.MAJORITY,
                                                 faulty="adversarial", resolve_conflicts=False),
        "majority_adversarial": run_config("swarm_consensus", tasks, sb, warm, rule=Rule.MAJORITY, faulty="adversarial"),
        "weighted_adversarial": run_config("swarm_consensus", tasks, sb, warm, rule=Rule.WEIGHTED, faulty="adversarial"),
        "quorum_adversarial": run_config("swarm_consensus", tasks, sb, warm, rule=Rule.QUORUM, faulty="adversarial"),
        "weighted_always_approve": run_config("swarm_consensus", tasks, sb, warm, rule=Rule.WEIGHTED,
                                              faulty="always_approve"),
    }
    return {
        "n_tasks": len(tasks),
        "modes": modes,
        "resilience": resilience,
        "total_wall_s": round(time.perf_counter() - t0, 2),
    }


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
