import numpy as np
import pytest

from trajectory_eval.datagen import BASELINE, DATA_DIR, REGRESSED, simulate_suite
from trajectory_eval.gate import Bound, Policy, evaluate_gate, render_markdown, run_gate
from trajectory_eval.golden import load_goldens
from trajectory_eval.regression import MetricRule, compare_suites, grade_suite
from trajectory_eval.schema import TrajectorySuite
from trajectory_eval.stats import stratified_diff


@pytest.fixture(scope="module")
def goldens():
    return load_goldens(DATA_DIR / "golden")


@pytest.fixture(scope="module")
def suites():
    load = lambda n: TrajectorySuite.load(DATA_DIR / "suites" / f"{n}.json")  # noqa: E731
    return load("baseline"), load("candidate_ok"), load("candidate_regressed")


def test_stratified_diff_detects_shift_and_ignores_noise():
    rng = np.random.default_rng(0)
    strata = np.repeat(["a", "b"], 50)
    base = rng.normal(0, 1, 100) + (strata == "b") * 5
    shifted = base - 1.0
    r = stratified_diff(base, strata, shifted, strata, higher_is_better=True, n_resamples=1000)
    assert r.delta == pytest.approx(-1.0) and r.p_one_sided < 0.01 and r.ci_high < 0
    noise = rng.normal(0, 1, 100) + (strata == "b") * 5
    r2 = stratified_diff(base, strata, noise, strata, n_resamples=1000)
    assert r2.p_one_sided > 0.05 or r2.ci_high >= 0
    # lower-is-better direction
    r3 = stratified_diff(base, strata, base + 1, strata, higher_is_better=False, n_resamples=1000)
    assert r3.p_one_sided < 0.01


def test_stratified_diff_needs_common_strata():
    with pytest.raises(ValueError):
        stratified_diff(np.ones(3), np.array(["a"] * 3), np.ones(3), np.array(["b"] * 3))


def test_stratification_ignores_task_mix_change():
    """Candidate runs more of the easy task: pooled mean rises but no per-task change."""
    base = np.r_[np.zeros(30), np.ones(10)]
    bs = np.r_[["hard"] * 30, ["easy"] * 10]
    cand = np.r_[np.zeros(10), np.ones(30)]
    cs = np.r_[["hard"] * 10, ["easy"] * 30]
    assert cand.mean() - base.mean() == pytest.approx(0.5)  # naive pooled delta is large
    r = stratified_diff(base, bs, cand, cs, n_resamples=500)
    assert r.delta == pytest.approx(0.0)
    assert r.p_one_sided >= 0.5


def test_grade_suite_records(goldens, suites):
    res = grade_suite(suites[0], goldens)
    assert len(res.records) == len(suites[0].trajectories)
    agg = res.aggregates()
    assert 0.6 < agg["pass_rate"] <= 1.0 and "judge_score" not in agg
    assert set(res.per_task()) == set(goldens)


def test_unknown_task_blocks(goldens, suites):
    base = grade_suite(suites[0], goldens)
    cand_suite = suites[1].model_copy(deep=True)
    cand_suite.trajectories[0].task_id = "mystery"
    cand = grade_suite(cand_suite, goldens)
    assert cand.unknown_tasks == ["mystery"]
    res = evaluate_gate(base, cand, Policy())
    assert not res.passed and "mystery" in res.violations[0]


def test_gate_passes_noise_blocks_regression(goldens, suites):
    policy = Policy.load(DATA_DIR / "policy.json")
    ok = run_gate(suites[0], suites[1], goldens, policy)
    assert ok.passed, ok.violations
    bad = run_gate(suites[0], suites[2], goldens, policy)
    assert not bad.passed
    regressed = {c.metric for c in bad.comparisons if c.regression}
    assert {"pass_rate", "forbidden_rate", "tokens"} <= regressed
    md = render_markdown(bad)
    assert "BLOCKED" in md and "REGRESSION" in md and "| pass_rate |" in md
    assert "PASSED" in render_markdown(ok)


def test_absolute_bounds(goldens, suites):
    base, cand = grade_suite(suites[0], goldens), grade_suite(suites[1], goldens)
    res = evaluate_gate(base, cand, Policy(absolute={"pass_rate": Bound(min=0.99)}))
    assert not res.passed and not res.absolute[0].passed
    res2 = evaluate_gate(base, cand, Policy(absolute={"nonexistent": Bound(max=1)}))
    assert "unknown metric" in res2.violations[0]


def test_min_effect_prevents_tiny_regressions(goldens):
    base = grade_suite(simulate_suite("b", BASELINE, 10, seed=5)[0], goldens)
    cand = grade_suite(simulate_suite("c", REGRESSED, 10, seed=6)[0], goldens)
    strict = compare_suites(base, cand, {"pass_rate": MetricRule()}, n_resamples=500)
    lax = compare_suites(base, cand, {"pass_rate": MetricRule(min_effect=0.9)}, n_resamples=500)
    assert strict[0].regression and not lax[0].regression and lax[0].significant


def test_holm_correction_is_stricter(goldens):
    base = grade_suite(simulate_suite("b", BASELINE, 12, seed=1)[0], goldens)
    rules = {m: MetricRule() for m in ["pass_rate", "score", "required_recall", "answer_accuracy", "tokens",
                                       "latency_ms", "steps", "precision", "order_score", "loop_rate"]}
    n_none = n_holm = 0
    for k in range(6):
        cand = grade_suite(simulate_suite(f"c{k}", BASELINE, 12, seed=50 + k)[0], goldens)
        n_none += sum(c.regression for c in compare_suites(base, cand, rules, n_resamples=500, correction="none"))
        n_holm += sum(c.regression for c in compare_suites(base, cand, rules, n_resamples=500, correction="holm"))
    assert n_holm <= n_none
