"""Ready-made swarm configurations and hidden-test scoring used by demo and bench."""
from __future__ import annotations

from aikit import LLM

from .agents import Coder, ExecutionReviewer, FaultyReviewer, Researcher, Reviewer, SamplingReviewer, StaticReviewer
from .consensus import ConsensusPanel, ReviewerLedger, Rule
from .library import TaskDef
from .messages import CodeArtifact
from .sandbox import Sandbox
from .supervisor import Budget, Supervisor
from .topology import Topology, review_topology, single_agent_topology

MODES = ("single_agent", "swarm_review", "swarm_consensus")


def honest_reviewers() -> list[Reviewer]:
    """Three diverse reviewers: full execution, seeded sampling, static analysis."""
    return [ExecutionReviewer("exec"), SamplingReviewer("sampler", fraction=0.5, seed=7), StaticReviewer("static")]


def build_topology(mode: str, rule: Rule | str = Rule.WEIGHTED, faulty: str | None = None,
                   ledger: ReviewerLedger | None = None, resolve_conflicts: bool = True,
                   llm: LLM | None = None, reviewers: list[Reviewer] | None = None) -> Topology:
    """Build one of the benchmark topologies.

    * ``single_agent``    — coder only, one shot, no research or review.
    * ``swarm_review``    — researcher -> coder <-> one execution reviewer.
    * ``swarm_consensus`` — researcher -> coder <-> 3 diverse reviewers + consensus panel.

    ``faulty`` injects a FaultyReviewer of that mode (it *replaces* the single
    reviewer in ``swarm_review``; it is *added* to the panel in ``swarm_consensus``).
    """
    coder = Coder("coder", llm=llm)
    if mode == "single_agent":
        return single_agent_topology(coder)
    researcher = Researcher("researcher", llm=llm)
    if mode == "swarm_review":
        revs: list[Reviewer] = [FaultyReviewer("rogue", mode=faulty)] if faulty else [ExecutionReviewer("exec")]
        panel = ConsensusPanel("panel", rule=Rule.MAJORITY, ledger=ledger)
    elif mode == "swarm_consensus":
        revs = list(reviewers) if reviewers is not None else honest_reviewers()
        if faulty:
            revs.append(FaultyReviewer("rogue", mode=faulty))
        panel = ConsensusPanel("panel", rule=rule, ledger=ledger, resolve_conflicts=resolve_conflicts)
    else:
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
    return review_topology(researcher, coder, revs, panel)


def build_supervisor(mode: str, sandbox: Sandbox | None = None, budget: Budget | None = None, **kw) -> Supervisor:
    return Supervisor(build_topology(mode, **kw), budget=budget, sandbox=sandbox)


def passes_hidden(artifact: CodeArtifact | None, task: TaskDef, sandbox: Sandbox) -> bool:
    """Ground truth: does the artifact pass every hidden test (never shown to agents)?"""
    if artifact is None:
        return False
    rep = sandbox.run(artifact.source, task.spec.func_name, task.hidden, arity=len(task.spec.params))
    return rep.passed
