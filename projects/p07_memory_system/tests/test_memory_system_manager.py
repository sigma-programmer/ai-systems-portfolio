"""Integration tests: MemoryManager flows, recall, TTL eviction, persistence, bench."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aikit import MockLLM
from memory_system import (DAY, HOUR, LLMSummarizer, ManualClock, MemoryConfig,
                           MemoryManager)
from memory_system.bench import benchmark
from memory_system.manager import recall_text

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT.parent.parent / "common"


@pytest.fixture
def clock():
    return ManualClock()


def chat(mm: MemoryManager, clock: ManualClock, turns: list[str]) -> None:
    for t in turns:
        clock.advance(30)
        mm.observe(t)


def test_session_flush_creates_episode_and_facts(clock):
    mm = MemoryManager(clock=clock)
    mm.start_session("s1")
    chat(mm, clock, ["Hi!", "My name is Ada Lovelace.", "I live in London.", "ok"])
    ep = mm.end_session()
    assert ep is not None and ep.consolidated and ep.raw_turns == []
    assert mm.profile() == {"name": "Ada Lovelace", "location": "London"}
    assert mm.working.items == [] and mm.session_id is None
    assert mm.end_session() is None


def test_overflow_promotes_to_episodic_mid_session(clock):
    cfg = MemoryConfig(token_budget=25, episode_max_turns=2)
    mm = MemoryManager(clock=clock, config=cfg)
    chat(mm, clock, [f"sentence {i} about the quarterly roadmap planning" for i in range(8)])
    assert mm.session_id is not None  # auto-started
    assert mm.working.tokens <= 25
    assert len(mm.episodes) >= 2
    assert mm.stats()["counters"]["working_evictions"] >= 4


def test_recall_prefers_current_fact_and_hides_superseded(clock):
    mm = MemoryManager(clock=clock)
    mm.start_session()
    chat(mm, clock, ["I live in Berlin."])
    mm.end_session()
    clock.advance(days=5)
    mm.start_session()
    chat(mm, clock, ["I moved to Lisbon recently."])
    mm.end_session()
    facts = [h for h in mm.recall("Which city does the user live in?", 5)
             if h.tier == "semantic"]
    assert "Lisbon" in facts[0].text and not any("Berlin" in h.text for h in facts)
    with_old = mm.recall("Which city does the user live in?", 10, tiers=["semantic"],
                         include_superseded=True)
    assert any("Berlin" in h.text for h in with_old)


def test_recall_reinforces_and_edge_cases(clock):
    mm = MemoryManager(clock=clock)
    chat(mm, clock, ["My favorite book is Dune."])
    mm.end_session()
    hit = mm.recall("favorite book", 1)[0]
    fact = mm.semantic.facts[hit.id]
    assert fact.access_count == 1
    mm.recall("favorite book", 1, reinforce=False)
    assert fact.access_count == 1
    assert mm.recall("   ") == []
    assert MemoryManager(clock=clock).recall("anything") == []


def test_recall_combines_tiers_including_working(clock):
    mm = MemoryManager(clock=clock)
    mm.start_session()
    chat(mm, clock, ["We are debugging the payment webhook retries."])
    hits = mm.recall("payment webhook", 3)
    assert hits[0].tier == "working"
    assert 0 < hits[0].score <= 1 and hits[0].recency == pytest.approx(1.0)
    ctx = mm.context("payment webhook", token_budget=50)
    assert ctx.startswith("- [working]")


def test_recency_and_importance_affect_ranking(clock):
    mm = MemoryManager(clock=clock, config=MemoryConfig(token_budget=1000))
    mm.start_session()
    mm.observe("project alpha deadline discussion", salience=0.1)
    clock.advance(hours=3)
    mm.observe("project alpha deadline discussion again", salience=0.9)
    hits = mm.recall("project alpha deadline", 2, tiers=["working"])
    assert "again" in hits[0].text


def test_ttl_eviction_per_tier(clock):
    mm = MemoryManager(clock=clock)
    mm.start_session()
    chat(mm, clock, ["random small talk about lunch", "I live in Berlin."])
    clock.advance(hours=3)
    out = mm.evict()
    assert out["working_expired"] == 2 and len(mm.episodes) == 1
    mm.end_session()
    chat(mm, clock, ["I moved to Lisbon."])
    mm.end_session()
    clock.advance(days=40)
    out = mm.evict()
    assert out["facts_purged"] == 1  # superseded Berlin fact
    assert out["episodes_expired"] >= 1
    assert mm.profile()["location"] == "Lisbon"  # active facts never expire


def test_high_salience_episode_survives_ttl(clock):
    mm = MemoryManager(clock=clock)
    chat(mm, clock, ["Remember that I am allergic to penicillin, this is important."])
    mm.end_session()
    chat(mm, clock, ["lol"])
    mm.end_session()
    clock.advance(days=60)
    mm.evict()
    assert len(mm.episodes) == 1
    assert "penicillin" in next(iter(mm.episodes.values())).summary


def test_cross_instance_persistence(tmp_path, clock):
    db = tmp_path / "mem.db"
    a = MemoryManager(db, clock=clock)
    a.start_session("monday")
    chat(a, clock, ["My name is Grace.", "I work at Hopper Labs.", "I prefer vim over emacs."])
    a.end_session()
    a.close()
    b = MemoryManager(db, clock=clock)
    assert b.profile()["employer"] == "Hopper Labs"
    assert "Hopper Labs" in b.recall("where does the user work?", 1)[0].text
    assert [s[0] for s in b.store.sessions()] == ["monday"]
    assert len(b.episodes) == 1


def test_new_process_recalls_prior_session(tmp_path):
    db = tmp_path / "mem.db"
    mm = MemoryManager(db, clock=ManualClock())
    chat(mm, ManualClock(), ["My dog is called Biscuit.", "I live in Oslo."])
    mm.end_session()
    mm.close()
    code = (
        "import sys, json; from memory_system import MemoryManager;"
        f"m = MemoryManager({str(db)!r});"
        "print(json.dumps({'profile': m.profile(),"
        " 'top': m.recall('name of the dog', 1)[0].text}))"
    )
    env_path = f"{ROOT}:{COMMON}"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={"PYTHONPATH": env_path, "AIKIT_OFFLINE": "1"}, check=True)
    data = json.loads(out.stdout)
    assert data["profile"]["pet_name"] == "Biscuit"
    assert "Biscuit" in data["top"]


def test_crash_recovery_promotes_orphaned_working_items(tmp_path, clock):
    db = tmp_path / "mem.db"
    a = MemoryManager(db, clock=clock)
    a.start_session("crashy")
    chat(a, clock, ["My name is Linus.", "I work at Penguin Systems."])
    a.close()  # no end_session: simulated crash
    b = MemoryManager(db, clock=clock)
    assert len(b.working.items) == 2
    b.start_session("next")
    assert b.working.items == []
    assert b.profile()["employer"] == "Penguin Systems"
    assert b.stats()["counters"]["recovered_working_items"] == 2


def test_deferred_consolidation(clock):
    mm = MemoryManager(clock=clock, config=MemoryConfig(consolidate_on_close=False))
    chat(mm, clock, ["My name is Alan."])
    mm.end_session()
    assert mm.profile() == {}
    assert mm.consolidate() == {"inserted": 1}
    assert mm.profile() == {"name": "Alan"}
    assert mm.consolidate() == {}


def test_llm_summarizer_plugs_in(clock):
    llm = MockLLM().on("SUMMARIZE EPISODE", "User introduced themselves as Ada.")
    mm = MemoryManager(clock=clock, summarizer=LLMSummarizer(llm))
    chat(mm, clock, ["My name is Ada."])
    ep = mm.end_session()
    assert ep.summary == "User introduced themselves as Ada."


def test_start_session_ends_previous(clock):
    mm = MemoryManager(clock=clock)
    mm.start_session("a")
    chat(mm, clock, ["I live in Rome."])
    mm.start_session("b")
    assert mm.profile()["location"] == "Rome" and mm.session_id == "b"


def test_recall_text_strips_boilerplate():
    assert recall_text("The user's name is Ada") == "name is Ada"
    assert recall_text("user: my dog") == "dog"


def test_benchmark_metrics():
    m = benchmark()
    json.dumps(m)
    assert m["memory_system"]["recall@3"] >= 0.9
    assert m["memory_system"]["precision@1"] > m["baseline_flat_vector_store"]["precision@1"]
    assert m["memory_system"]["stale_outranks_current"].startswith("0/")
    assert m["profile_location"] == "Lisbon" and m["profile_employer"] == "Helix Robotics"
    assert 0 < m["compression_ratio_episodic"] < 1
