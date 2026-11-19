"""Unit tests: importance scoring, working memory, summarizer, fact extraction, belief revision."""
from __future__ import annotations

import pytest

from aikit import MockLLM
from memory_system import (DAY, ExtractiveSummarizer, Extracted, ImportanceScorer,
                           LLMFactExtractor, LLMSummarizer, ManualClock, PatternFactExtractor,
                           SemanticMemory, WorkingMemory, salience_of)
from memory_system.importance import frequency, recency
from memory_system.semantic import is_single_valued, normalise, render


# ---------------------------------------------------------------- importance
def test_salience_cues_are_additive_and_bounded():
    base = salience_of("the build passed")
    assert salience_of("Remember that the build passed") > base
    assert salience_of("I love this build") > base
    assert salience_of("ok") < base
    assert salience_of("Remember: my name is X, I love it, 2027") <= 1.0


def test_recency_halves_each_half_life_and_frequency_saturates():
    assert recency(0, 10) == 1.0
    assert recency(10, 10) == pytest.approx(0.5)
    assert recency(5, 0) == 0.0
    assert frequency(0) == 0.0 and 0.6 < frequency(3) < frequency(10) < 1.0


def test_importance_formula_weights():
    s = ImportanceScorer()
    fresh = s.score(0.5, last_access=100, access_count=0, now=100, half_life=10)
    assert fresh == pytest.approx(0.45 * 0.5 + 0.30)
    stale = s.score(0.5, last_access=0, access_count=0, now=1000, half_life=10)
    assert stale < fresh
    assert s.static(1.0, 100) == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------- working memory
def test_working_memory_respects_budget_and_promotes_evictions():
    clock = ManualClock()
    wm = WorkingMemory(clock, token_budget=20)
    evicted_all = []
    for i in range(10):
        clock.advance(10)
        _, ev = wm.add(f"turn number {i} with a few words", "s")
        evicted_all.extend(ev)
    assert wm.tokens <= 20
    assert len(evicted_all) == wm.evictions > 0
    assert wm.items[-1].text.startswith("turn number 9")


def test_working_memory_keeps_salient_old_item():
    clock = ManualClock()
    wm = WorkingMemory(clock, token_budget=30, fifo_window=1.0)
    wm.add("Remember that my flight is on Friday at 9", "s")
    for i in range(6):
        clock.advance(5)
        wm.add(f"chit chat {i} about nothing much", "s")
    assert any("flight" in i.text for i in wm.items)


def test_working_memory_never_evicts_the_new_item_and_validates():
    wm = WorkingMemory(ManualClock(), token_budget=2)
    item, ev = wm.add("a very long sentence that alone exceeds the budget", "s")
    assert wm.items == [item] and ev == []
    with pytest.raises(ValueError):
        WorkingMemory(ManualClock(), token_budget=0)
    with pytest.raises(ValueError):
        WorkingMemory(ManualClock(), fifo_window=0)


def test_working_memory_ttl_and_touch():
    clock = ManualClock()
    wm = WorkingMemory(clock, token_budget=100)
    a, _ = wm.add("old", "s")
    clock.advance(100)
    b, _ = wm.add("new", "s")
    wm.touch(b)
    assert b.access_count == 1
    assert wm.expire(50) == [a] and wm.items == [b]
    assert wm.render() == "user: new"
    assert wm.drain() == [b] and wm.items == []


# ---------------------------------------------------------------- summarizer
def test_extractive_summary_compresses_and_keeps_salient_facts():
    turns = ["user: Hi!", "assistant: Hello, how can I help?",
             "user: Remember that my passport expires in March 2027.",
             "user: The weather is grey. It rained all day. The bus was late again.",
             "assistant: Sorry to hear that. Grey days are tough for everyone in the city.",
             "user: ok"]
    summ = ExtractiveSummarizer(ratio=0.4, min_tokens=12).summarize(turns)
    assert "passport" in summ
    assert "Hi!" not in summ and "user: ok" not in summ
    from aikit import count_tokens
    assert count_tokens(summ) < 0.6 * sum(count_tokens(t) for t in turns)
    assert ExtractiveSummarizer().summarize(["user: ok"]) == ""


def test_llm_summarizer_uses_protocol():
    llm = MockLLM().on("SUMMARIZE EPISODE", "User planned a trip.")
    assert LLMSummarizer(llm).summarize(["user: trip to Rome"]) == "User planned a trip."
    assert "trip to Rome" in llm.calls[0]["prompt"]


# ---------------------------------------------------------------- extraction
@pytest.mark.parametrize("text,pred,value", [
    ("My name is Priya Raman and I work at Nimbus.", "name", "Priya Raman"),
    ("I work at Nimbus Analytics.", "employer", "Nimbus Analytics"),
    ("I moved to Lisbon last week.", "location", "Lisbon"),
    ("I prefer tea over coffee.", "prefers", "tea over coffee"),
    ("I'm allergic to peanuts.", "allergy", "peanuts"),
    ("My dog is called Biscuit.", "pet_name", "Biscuit"),
    ("My favorite book is Dune.", "favorite_book", "Dune"),
    ("I love hiking in the mountains.", "likes", "hiking"),
    ("I am vegetarian.", "diet", "vegetarian"),
    ("My birthday is on June 5.", "birthday", "June 5"),
    ("I am a data engineer.", "occupation", "data engineer"),
    ("Remember that the demo is on Friday.", "note", "the demo is on Friday"),
])
def test_pattern_extraction(text, pred, value):
    got = {(e.predicate, e.value) for e in PatternFactExtractor().extract(text)}
    assert (pred, value) in got


def test_extraction_negatives_and_retractions():
    ex = PatternFactExtractor()
    assert ex.extract("I work remotely most days.") == []
    assert ex.extract("I am a big fan of rain.") == []  # not a job
    r = ex.extract("I no longer work at Nimbus Analytics.")
    assert r == [Extracted("employer", "Nimbus Analytics", retract=True)]
    assert ex.extract("I don't like coffee anymore.")[0].predicate == "dislikes"


def test_llm_fact_extractor():
    llm = MockLLM().on("EXTRACT FACTS", '[{"predicate": "name", "value": "Ada"}, {"bad": 1}]')
    assert LLMFactExtractor(llm).extract("I'm Ada") == [Extracted("name", "Ada")]
    assert LLMFactExtractor(MockLLM(default="no json")).extract("x") == []
    llm2 = MockLLM(default='{"facts": [{"predicate": "likes", "value": "tea"}]}')
    assert LLMFactExtractor(llm2).extract("x")[0].value == "tea"


# ---------------------------------------------------------------- belief revision
def test_semantic_dedup_supersede_contradiction_retract():
    sm = SemanticMemory()
    a, fa, _ = sm.upsert(Extracted("location", "Berlin"), 1.0)
    assert a == "inserted"
    a, f2, _ = sm.upsert(Extracted("location", "berlin."), 2.0)
    assert a == "duplicate" and f2 is fa and fa.confidence > 0.7
    a, fb, changed = sm.upsert(Extracted("location", "Lisbon"), 3.0)
    assert a == "superseded" and changed == [fa]
    assert fa.status == "superseded" and fa.superseded_by == fb.id
    assert sm.profile()["location"] == "Lisbon"

    sm.upsert(Extracted("likes", "coffee"), 4.0)
    sm.upsert(Extracted("likes", "tea"), 4.0)
    a, _, changed = sm.upsert(Extracted("dislikes", "coffee"), 5.0)
    assert a == "superseded" and changed[0].value == "coffee"
    assert sm.profile()["likes"] == ["tea"] and sm.profile()["dislikes"] == ["coffee"]

    sm.upsert(Extracted("employer", "Nimbus"), 6.0)
    a, _, changed = sm.upsert(Extracted("employer", "Nimbus", retract=True), 7.0)
    assert a == "retracted" and "employer" not in sm.profile()
    assert sm.upsert(Extracted("employer", "X", retract=True), 8.0)[0] == "noop"


def test_helpers():
    assert is_single_valued("favorite_color") and is_single_valued("cat_name")
    assert not is_single_valued("likes")
    assert normalise("The  Big Apple!") == "big apple"
    assert "Biscuit" in render("user", "pet_name", "Biscuit")
    assert render("user", "custom_thing", "v").endswith("custom thing: v.")
