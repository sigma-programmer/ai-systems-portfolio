import json

import pytest

from aikit import MockLLM
from trajectory_eval.graders import (ArgHygieneGrader, ErrorRecoveryGrader, LatencyBudgetGrader, SafetyGrader,
                                     TokenBudgetGrader, run_graders)
from trajectory_eval.judge import HeuristicJudge, JudgeOutput, LLMJudge, offline_judge_llm, render_judge_prompt, DEFAULT_RUBRIC
from trajectory_eval.schema import Step, Trajectory


def traj(steps, answer="", task="refund order"):
    return Trajectory(run_id="r", task_id="t", task=task, steps=steps, final_answer=answer)


def test_rule_graders():
    t = traj([
        Step(type="tool_call", tool="shell", args={"cmd": "rm -rf /"}, latency_ms=9000, tokens=5000),
        Step(type="observation", error=True),
        Step(type="tool_call", tool="search", args={"q": "TODO", "x": None}),
    ])
    assert LatencyBudgetGrader(5000).grade(t)[0].step_index == 0
    assert TokenBudgetGrader(2000).grade(t)[0].score == pytest.approx(0.4)
    hyg = ArgHygieneGrader().grade(t)
    assert hyg[0].passed and not hyg[1].passed and "q" in hyg[1].message
    assert not SafetyGrader().grade(t)[0].passed
    assert ErrorRecoveryGrader().grade(t)[0].passed
    checks = run_graders(t)
    assert [c.step_index for c in checks] == sorted(c.step_index for c in checks)


def test_error_not_recovered():
    t = traj([Step(type="tool_call", tool="a"), Step(type="observation", error=True)])
    assert not ErrorRecoveryGrader().grade(t)[0].passed


def test_heuristic_judge_prefers_good_trajectory():
    good = traj([Step(type="tool_call", tool="get_order", args={"order_id": "refund order 1"}),
                 Step(type="observation", result={"status": "refunded", "amount": 10})],
                answer="The order was refunded, amount 10.")
    bad = traj([Step(type="tool_call", tool="delete_account", args={"id": 1})] * 4, answer="pizza")
    j = HeuristicJudge()
    g, b = j.judge(good), j.judge(bad)
    assert g.score > b.score
    assert b.criteria["safety"] == 0.0 and b.criteria["efficiency"] < 0.5
    assert 0 <= b.score <= 1


def test_llm_judge_with_offline_model_matches_heuristic():
    t = traj([Step(type="tool_call", tool="x", args={"a": "refund"})], answer="refund done")
    res = LLMJudge(offline_judge_llm()).judge(t)
    assert not res.fallback
    assert res.score == pytest.approx(HeuristicJudge().judge(t).score, abs=1e-6)


def test_llm_judge_parses_custom_llm_and_prompt_contents():
    llm = MockLLM().on("RUBRIC", json.dumps({"criteria": {c.name: 0.5 for c in DEFAULT_RUBRIC}, "rationale": "meh"}))
    res = LLMJudge(llm).judge(traj([]))
    assert res.score == pytest.approx(0.5) and res.rationale == "meh"
    assert "task_relevance" in llm.calls[0]["prompt"] and llm.calls[0]["json_mode"]
    assert "TRAJECTORY_JSON" in render_judge_prompt(traj([]), DEFAULT_RUBRIC)


def test_llm_judge_repairs_then_falls_back():
    calls = {"n": 0}

    def flaky(prompt, system):
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json"
        return json.dumps({"criteria": {c.name: 1.0 for c in DEFAULT_RUBRIC}})

    res = LLMJudge(MockLLM().on("RUBRIC", flaky)).judge(traj([]))
    assert calls["n"] == 2 and res.score == 1.0 and not res.fallback

    junk = MockLLM().on("RUBRIC", json.dumps({"criteria": {"safety": 7}}))
    res2 = LLMJudge(junk).judge(traj([]))
    assert res2.fallback and len(junk.calls) == 2


def test_judge_output_validation():
    with pytest.raises(Exception):
        JudgeOutput(criteria={"a": 1.5})
