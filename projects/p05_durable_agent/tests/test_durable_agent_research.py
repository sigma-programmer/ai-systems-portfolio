"""The sample research-report agent and the benchmark contract."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from aikit import MockLLM

from durable_agent import CrashAt, Engine, FakeClock, SimulatedCrash, Status
from durable_agent.bench import chaos, checkpoint_overhead
from durable_agent.samples import FlakySearch, Ledger, research_mock_llm, research_workflow, search_corpus


def make(tmp_path: Path, **kw) -> tuple[Engine, Ledger]:
    ledger = Ledger(tmp_path / "cms.db")
    eng = Engine(
        tmp_path / "agent.db",
        [research_workflow(ledger)],
        clock=kw.pop("clock", FakeClock()),
        llm=kw.pop("llm", research_mock_llm()),
        services={"search": kw.pop("search", FlakySearch(0.0))},
        **kw,
    )
    return eng, ledger


def test_search_tool() -> None:
    hits = search_corpus("idempotency keys retry")
    assert hits and hits[0]["title"] == "idempotency"
    assert search_corpus("zzzz") == []


def test_full_agent_run_with_revision_and_publish(tmp_path: Path) -> None:
    eng, cms = make(tmp_path, search=FlakySearch(0.3, seed=4))
    info = eng.execute("research_report", {"topic": "durable agents", "notify": True})
    assert info.status == Status.WAITING_APPROVAL
    assert info.path == ["plan", "search", "draft"]
    assert len(info.outputs["plan"]["queries"]) == 3
    assert info.outputs["search"]["results"]
    assert info.pending_approval["request"]["title"] == "Report: durable agents"

    eng2, _ = make(tmp_path)  # reviewer acts from a different engine instance
    r = eng2.reject(info.run_id, "editor", "add a conclusion")
    assert r.status == Status.WAITING_APPROVAL
    assert "Revision note" in r.outputs["draft"]["body"] and r.outputs["draft"]["revision"] == 2
    done = eng2.approve(info.run_id, "editor", "great")
    assert done.status == Status.COMPLETED
    assert done.path[-2:] == ["publish", "notify"]
    assert cms.count("publish", info.run_id) == 1
    outbox = eng2.store.query("SELECT key, topic FROM outbox")
    assert [tuple(r) for r in outbox] == [(f"{info.run_id}/notify/1/notify", "reports")]


def test_only_editor_can_approve_until_escalation(tmp_path: Path) -> None:
    clock = FakeClock()
    eng, _ = make(tmp_path, clock=clock)
    rid = eng.execute("research_report", {"topic": "x"}).run_id
    from durable_agent import ApprovalError

    with pytest.raises(ApprovalError):
        eng.approve(rid, "managing_editor")
    clock.advance(24 * 3600 + 1)
    eng.check_timeouts()
    assert eng.approve(rid, "managing_editor").status == Status.COMPLETED


def test_crash_during_publish_publishes_once(tmp_path: Path) -> None:
    clock = FakeClock()
    eng, cms = make(tmp_path, clock=clock, faults=CrashAt("after_effect", "publish"))
    rid = eng.execute("research_report", {"topic": "x"}).run_id
    with pytest.raises(SimulatedCrash):
        eng.approve(rid, "editor")
    clock.advance(60)
    eng2, _ = make(tmp_path, clock=clock)
    llm_calls_before = len(eng2.llm.calls)
    assert eng2.run(rid).status == Status.COMPLETED
    assert cms.count("publish", rid) == 1 and cms.duplicates() == 0
    assert len(eng2.llm.calls) == llm_calls_before  # plan/draft were replayed, not re-asked


def test_bad_llm_plan_is_retried_then_dead_lettered(tmp_path: Path) -> None:
    llm = MockLLM(default="I cannot produce JSON")
    eng, _ = make(tmp_path, llm=llm)
    info = eng.execute("research_report", {"topic": "x"})
    assert info.status == Status.DEAD_LETTER
    assert len(llm.calls) == 4 and "retries exhausted" in info.error


def test_search_outage_dead_letters_then_recovers(tmp_path: Path) -> None:
    search = FlakySearch(1.0)
    eng, _ = make(tmp_path, search=search)
    info = eng.execute("research_report", {"topic": "x"})
    assert info.status == Status.DEAD_LETTER and "ConnectionError" in info.error
    search.fail_rate = 0.0
    assert eng.retry_dead_letter(info.run_id).status == Status.WAITING_APPROVAL


def test_chaos_benchmark_small() -> None:
    m = chaos(n_workflows=30, seed=1)
    json.dumps(m)
    assert m["completion_rate"] == 1.0
    assert m["duplicate_side_effects"] == 0
    assert m["side_effects_applied"] == m["side_effects_expected"] == 90
    assert m["crashes_injected"] > 0 and m["transient_failures_injected"] > 0


def test_chaos_is_deterministic() -> None:
    a, b = chaos(n_workflows=20, seed=5), chaos(n_workflows=20, seed=5)
    keys = ["completion_rate", "crashes_injected", "transient_failures_injected", "mean_retries_per_workflow", "effect_reinvocations_deduped_by_key"]
    assert {k: a[k] for k in keys} == {k: b[k] for k in keys}


def test_checkpoint_overhead_shape() -> None:
    m = checkpoint_overhead(n_runs=10, n_steps=3)
    assert m["steps_measured"] == 30 and m["file_sync_full_ms_per_step"] > 0
