"""Suite grading and baseline-vs-candidate regression detection."""
from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from .align import AlignmentReport, align
from .golden import GoldenDAG
from .graders import StepCheck, StepGrader, run_graders
from .judge import Judge, JudgeResult
from .schema import TrajectorySuite
from .stats import stratified_diff

METRICS: dict[str, Literal["higher", "lower"]] = {
    "pass_rate": "higher",
    "score": "higher",
    "required_recall": "higher",
    "precision": "higher",
    "order_score": "higher",
    "answer_accuracy": "higher",
    "forbidden_rate": "lower",
    "loop_rate": "lower",
    "step_check_failures": "lower",
    "tokens": "lower",
    "latency_ms": "lower",
    "steps": "lower",
    "judge_score": "higher",
}


class RunRecord(BaseModel):
    """Flat per-run metric vector used for statistics."""

    run_id: str
    task_id: str
    values: dict[str, float]


class SuiteResult(BaseModel):
    name: str
    reports: list[AlignmentReport]
    checks: dict[str, list[StepCheck]]
    judgements: dict[str, JudgeResult] = Field(default_factory=dict)
    records: list[RunRecord]
    unknown_tasks: list[str]

    def aggregates(self) -> dict[str, float]:
        if not self.records:
            return {}
        keys = self.records[0].values.keys()
        return {k: float(np.mean([r.values[k] for r in self.records])) for k in keys}

    def per_task(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for t in sorted({r.task_id for r in self.records}):
            rs = [r for r in self.records if r.task_id == t]
            out[t] = {k: float(np.mean([r.values[k] for r in rs])) for k in rs[0].values}
        return out

    def array(self, metric: str) -> tuple[np.ndarray, np.ndarray]:
        return (np.array([r.values[metric] for r in self.records], float),
                np.array([r.task_id for r in self.records]))


def grade_suite(
    suite: TrajectorySuite,
    goldens: dict[str, GoldenDAG],
    judge: Judge | None = None,
    graders: list[StepGrader] | None = None,
) -> SuiteResult:
    """Align every trajectory with its golden DAG, run step graders and (optionally) the judge."""
    reports: list[AlignmentReport] = []
    checks: dict[str, list[StepCheck]] = {}
    judgements: dict[str, JudgeResult] = {}
    records: list[RunRecord] = []
    unknown: set[str] = set()
    for t in suite.trajectories:
        g = goldens.get(t.task_id)
        if g is None:
            unknown.add(t.task_id)
            continue
        rep = align(t, g)
        chk = run_graders(t, graders)
        reports.append(rep)
        checks[t.run_id] = chk
        m = rep.metrics
        vals = {
            "pass_rate": float(rep.passed),
            "score": m.score,
            "required_recall": m.required_recall,
            "precision": m.precision,
            "order_score": m.order_score,
            "answer_accuracy": float(m.answer_correct) if m.answer_correct is not None else 1.0,
            "forbidden_rate": float(m.forbidden > 0),
            "loop_rate": float(m.loops > 0),
            "step_check_failures": float(sum(not c.passed for c in chk)),
            "tokens": float(m.tokens),
            "latency_ms": m.latency_ms,
            "steps": float(m.steps),
        }
        if judge is not None:
            jr = judge.judge(t)
            judgements[t.run_id] = jr
            vals["judge_score"] = jr.score
        records.append(RunRecord(run_id=t.run_id, task_id=t.task_id, values=vals))
    return SuiteResult(name=suite.name, reports=reports, checks=checks, judgements=judgements,
                       records=records, unknown_tasks=sorted(unknown))


class MetricRule(BaseModel):
    """When does a metric change count as a regression?"""

    direction: Literal["higher", "lower"] | None = None  # defaults from METRICS
    min_effect: float = 0.0  # absolute delta in the bad direction
    min_effect_rel: float = 0.0  # relative delta vs baseline mean
    alpha: float | None = None  # overrides policy alpha


class MetricComparison(BaseModel):
    metric: str
    direction: Literal["higher", "lower"]
    baseline: float
    candidate: float
    delta: float
    rel_delta: float
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool
    material: bool
    regression: bool
    improvement: bool


def compare_suites(
    base: SuiteResult,
    cand: SuiteResult,
    rules: dict[str, MetricRule],
    alpha: float = 0.05,
    n_resamples: int = 2000,
    seed: int = 0,
    correction: Literal["holm", "none"] = "holm",
) -> list[MetricComparison]:
    """Compare candidate to baseline on every metric that has a rule.

    A regression requires all three: the delta is in the bad direction, it is
    *material* (exceeds the absolute/relative minimum effect) and it is
    *significant* (one-sided stratified permutation p below the Holm-adjusted
    alpha and the bootstrap CI excludes zero). Holm-Bonferroni controls the
    family-wise error rate across the metric family, so checking ten metrics
    does not mean ten chances for noise to block a PR.
    """
    out: list[MetricComparison] = []
    for i, (metric, rule) in enumerate(sorted(rules.items())):
        if not base.records or not cand.records or metric not in base.records[0].values \
                or metric not in cand.records[0].values:
            continue
        direction = rule.direction or METRICS.get(metric, "higher")
        hib = direction == "higher"
        bv, bs = base.array(metric)
        cv, cs = cand.array(metric)
        d = stratified_diff(bv, bs, cv, cs, higher_is_better=hib, n_resamples=n_resamples, seed=seed + i)
        bmean = float(bv[np.isin(bs, cs)].mean())
        cmean = bmean + d.delta  # task-mix-adjusted candidate mean
        rel = d.delta / abs(bmean) if bmean else (0.0 if d.delta == 0 else float("inf"))
        bad = -d.delta if hib else d.delta
        bad_rel = -rel if hib else rel
        a = rule.alpha if rule.alpha is not None else alpha
        ci_excludes = d.ci_high < 0 if hib else d.ci_low > 0
        ci_good = d.ci_low > 0 if hib else d.ci_high < 0
        significant = d.p_one_sided < a and ci_excludes  # refined by Holm below
        material = bad > 0 and bad >= rule.min_effect and bad_rel >= rule.min_effect_rel
        out.append(MetricComparison(
            metric=metric, direction=direction, baseline=bmean, candidate=cmean,
            delta=d.delta, rel_delta=rel, ci_low=d.ci_low, ci_high=d.ci_high,
            p_value=d.p_one_sided, significant=significant, material=material,
            regression=significant and material, improvement=ci_good and d.p_two_sided < a,
        ))
    if correction == "holm" and out:
        _holm(out, {c.metric: (rules[c.metric].alpha if rules[c.metric].alpha is not None else alpha) for c in out})
    return out


def _holm(comps: list[MetricComparison], alphas: dict[str, float]) -> None:
    """Holm step-down: the k-th smallest p must beat alpha/(m-k); stop at the first failure."""
    m = len(comps)
    rejected = True
    for k, c in enumerate(sorted(comps, key=lambda c: c.p_value)):
        rejected = rejected and c.p_value < alphas[c.metric] / (m - k)
        c.significant = c.significant and rejected
        c.regression = c.significant and c.material
