"""Candidate space: mutation operators and ranking."""
from __future__ import annotations

from agent_swarm.library import CANDIDATES, KNOWLEDGE, TASKS
from agent_swarm.synthesis import build_pool, mutants, rank


def test_mutants_flip_comparisons_and_constants():
    srcs = [s for _, s in mutants("def f(x):\n    return x < 3\n")]
    assert "def f(x):\n    return x <= 3\n" in srcs
    assert "def f(x):\n    return x < 4\n" in srcs
    assert "def f(x):\n    return x < 2\n" in srcs
    bools = [s for d, s in mutants("def f(a, b):\n    return a and b\n") if d.startswith("bool")]
    assert bools == ["def f(a, b):\n    return a or b\n"]
    assert mutants("def f(:") == []


def test_binary_search_fix_is_reachable_by_mutation():
    buggy = CANDIDATES["binary_search"][1].source
    assert any("while lo <= hi" in s for _, s in mutants(buggy))


def test_pool_dedup_and_priors():
    pool = build_pool("merge_intervals")
    assert len({c.hash for c in pool}) == len(pool)
    lib = [c for c in pool if c.origin.startswith("library")]
    assert [c.prior for c in lib] == [0.0, -1.0, -2.0, -3.0]
    assert len(build_pool("merge_intervals", with_mutants=False)) == 4
    assert build_pool("unknown_function") == []


def test_rank_uses_requirement_tags():
    pool = build_pool("is_palindrome", with_mutants=False)
    assert rank(pool, [])[0].origin == "library[0]"  # Occam: simplest first
    ranked = rank(pool, ["comparison is case insensitive, only alphanumeric characters, ignore punctuation"])
    assert ranked[0].origin == "library[2]"


def test_every_task_has_a_correct_candidate_in_its_space(sandbox):
    for td in TASKS:
        pool = build_pool(td.spec.func_name)
        reps = sandbox.run_many([c.source for c in pool], td.spec.func_name, td.hidden, stop_on_fail=True,
                                timeout=0.1)
        assert any(r.passed for r in reps), td.spec.name


def test_knowledge_ids_unique():
    assert len({s.id for s in KNOWLEDGE}) == len(KNOWLEDGE)
