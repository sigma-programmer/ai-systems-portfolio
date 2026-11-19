"""Supervisor orchestration end-to-end: loops, budgets, termination, trace."""
from __future__ import annotations

import json

import pytest

from aikit import MockLLM
from agent_swarm import (
    TASKS_BY_NAME, Budget, Coder, ConsensusPanel, ExecutionReviewer, Researcher, Rule, Supervisor, Topology,
    build_supervisor, passes_hidden, transcript,
)
from agent_swarm.bench import benchmark, run_config


def test_swarm_review_revises_and_succeeds(sandbox):
    td = TASKS_BY_NAME["merge_intervals"]
    st = build_supervisor("swarm_review", sandbox=sandbox).run(td.spec)
    assert st.status == "approved" and st.revisions == 1 and len(st.artifacts) == 2
    assert st.counterexamples and passes_hidden(st.final_artifact, td, sandbox)
    kinds = [e.kind for e in st.events]
    assert kinds[0] == "plan" and kinds[-1] == "terminate" and "revise" in kinds
    assert st.plan is not None and [s.node for s in st.plan.steps] == ["researcher", "coder", "exec", "panel"]


def test_single_agent_is_unreviewed(sandbox):
    td = TASKS_BY_NAME["roman"]
    st = build_supervisor("single_agent", sandbox=sandbox).run(td.spec)
    assert st.status == "completed_unreviewed" and st.steps == 1 and not st.decisions
    assert not passes_hidden(st.final_artifact, td, sandbox)


def test_consensus_with_adversary_still_succeeds(sandbox):
    td = TASKS_BY_NAME["second_largest"]
    st = build_supervisor("swarm_consensus", sandbox=sandbox, rule=Rule.MAJORITY, faulty="adversarial").run(td.spec)
    assert st.status == "approved" and st.escalations >= 1
    assert passes_hidden(st.final_artifact, td, sandbox)
    assert any(e.kind == "escalation" and "rogue" in e.detail for e in st.events)


def test_single_adversarial_reviewer_breaks_swarm(sandbox):
    td = TASKS_BY_NAME["second_largest"]
    st = build_supervisor("swarm_review", sandbox=sandbox, faulty="adversarial").run(td.spec)
    assert not passes_hidden(st.final_artifact, td, sandbox)


def test_max_revisions_termination(sandbox):
    td = TASKS_BY_NAME["merge_intervals"]
    st = build_supervisor("swarm_review", sandbox=sandbox, budget=Budget(max_revisions=0)).run(td.spec)
    assert st.status == "max_revisions" and st.revisions == 0 and not st.approved


@pytest.mark.parametrize("budget,reason", [(Budget(max_tokens=500), "tokens"), (Budget(max_steps=2), "steps"),
                                           (Budget(max_wall_s=0.0), "wall_time")])
def test_budget_termination(sandbox, budget, reason):
    st = build_supervisor("swarm_consensus", sandbox=sandbox, budget=budget).run(TASKS_BY_NAME["palindrome"].spec)
    assert st.status == f"budget_exhausted:{reason}"


def test_handoff_failure_is_contained(sandbox):
    llm = MockLLM(default="garbage")

    class NoFallbackCoder(Coder):
        def act_default(self, inputs, ctx):
            return {"source": "nope"}

    topo = Topology().add(Researcher(), NoFallbackCoder("coder", llm=llm, max_repairs=1), ExecutionReviewer("exec"),
                          ConsensusPanel("panel"))
    topo.connect("researcher", "coder").connect("coder", "exec").connect("exec", "panel")
    topo.connect("panel", "coder", feedback=True)
    st = Supervisor(topo, sandbox=sandbox).run(TASKS_BY_NAME["median"].spec)
    assert st.status == "handoff_failed" and st.repairs >= 2
    assert any(e.kind == "error" for e in st.events)


def test_stronger_review_closes_research_gap(sandbox):
    td = TASKS_BY_NAME["leap_year"]

    class TinyCoder(Coder):
        def __init__(self):
            super().__init__("coder", use_mutants=False)

    # The knowledge base omits the 400-year rule; a reviewer that knows it drives the coder past the gap.
    from agent_swarm.messages import TestCase

    class StrictReviewer(ExecutionReviewer):
        def select_tests(self, suite, art):
            return suite + [TestCase(id="x2000", args=[2000], expected=True, source="research")]

    topo = Topology().add(Researcher(), TinyCoder(), StrictReviewer("exec"), ConsensusPanel("panel"))
    topo.connect("researcher", "coder").connect("coder", "exec").connect("exec", "panel")
    topo.connect("panel", "coder", feedback=True)
    st = Supervisor(topo, budget=Budget(max_revisions=10), sandbox=sandbox).run(td.spec)
    assert st.status == "approved" and st.revisions >= 1
    assert passes_hidden(st.final_artifact, td, sandbox)


def test_unvalidated_topology_is_validated_by_supervisor(sandbox):
    topo = Topology().add(Coder())
    sup = Supervisor(topo, sandbox=sandbox)
    assert topo.validated and sup.plan(TASKS_BY_NAME["chunk"].spec).subgoals


def test_transcript_and_state_serialisable(sandbox):
    st = build_supervisor("swarm_consensus", sandbox=sandbox).run(TASKS_BY_NAME["fizzbuzz"].spec)
    text = transcript(st)
    assert "terminate" in text and "panel" in text
    json.loads(st.model_dump_json())
    assert st.tokens == sum(e.tokens for e in st.events)


def test_run_config_and_benchmark_subset(sandbox):
    tasks = [TASKS_BY_NAME["roman"], TASKS_BY_NAME["leap_year"]]
    r = run_config("swarm_consensus", tasks, sandbox, rule=Rule.WEIGHTED)
    assert r["success_rate"] == 0.5 and r["false_approvals"] == 1 and r["verdict_accuracy"] == 1.0
    b = benchmark(tasks)
    json.dumps(b)
    assert b["modes"]["single_agent"]["success_rate"] <= b["modes"]["swarm_consensus"]["success_rate"]
    assert b["resilience"]["weighted_adversarial"]["verdict_accuracy"] == 1.0
    assert b["resilience"]["single_reviewer_adversarial"]["verdict_accuracy"] == 0.0
