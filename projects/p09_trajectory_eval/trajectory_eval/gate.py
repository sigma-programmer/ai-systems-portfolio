"""CI gate: policy evaluation and the markdown PR comment."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .golden import GoldenDAG, load_goldens
from .judge import HeuristicJudge, Judge
from .regression import MetricComparison, MetricRule, SuiteResult, compare_suites, grade_suite
from .schema import TrajectorySuite


class Bound(BaseModel):
    min: float | None = None
    max: float | None = None


class Policy(BaseModel):
    """Gate policy (loaded from policy.json)."""

    absolute: dict[str, Bound] = Field(default_factory=dict)
    regression: dict[str, MetricRule] = Field(default_factory=dict)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    correction: Literal["holm", "none"] = "holm"
    n_resamples: int = Field(default=2000, ge=100)
    seed: int = 0
    use_judge: bool = True
    fail_on_unknown_task: bool = True
    min_runs_per_task: int = Field(default=1, ge=1)

    @classmethod
    def load(cls, path: str | Path) -> "Policy":
        return cls.model_validate_json(Path(path).read_text())


class AbsoluteCheck(BaseModel):
    metric: str
    value: float
    bound: Bound
    passed: bool


class GateResult(BaseModel):
    passed: bool
    violations: list[str]
    absolute: list[AbsoluteCheck]
    comparisons: list[MetricComparison]
    baseline_name: str
    candidate_name: str
    candidate_aggregates: dict[str, float]
    baseline_aggregates: dict[str, float]
    per_task_pass: dict[str, tuple[float, float]]
    worst_runs: list[dict]


def evaluate_gate(base: SuiteResult, cand: SuiteResult, policy: Policy) -> GateResult:
    """Apply the policy to two graded suites."""
    violations: list[str] = []
    if policy.fail_on_unknown_task and cand.unknown_tasks:
        violations.append(f"candidate has runs for tasks without a golden DAG: {cand.unknown_tasks}")
    cand_tasks = {r.task_id for r in cand.records}
    base_tasks = {r.task_id for r in base.records}
    missing = sorted(base_tasks - cand_tasks)
    if missing:
        violations.append(f"candidate is missing tasks covered by baseline: {missing}")
    for t in sorted(cand_tasks):
        n = sum(r.task_id == t for r in cand.records)
        if n < policy.min_runs_per_task:
            violations.append(f"task {t!r}: {n} run(s) < min_runs_per_task={policy.min_runs_per_task}")

    agg_c, agg_b = cand.aggregates(), base.aggregates()
    absolute: list[AbsoluteCheck] = []
    for metric, bound in sorted(policy.absolute.items()):
        if metric not in agg_c:
            violations.append(f"policy references unknown metric {metric!r}")
            continue
        v = agg_c[metric]
        ok = (bound.min is None or v >= bound.min) and (bound.max is None or v <= bound.max)
        absolute.append(AbsoluteCheck(metric=metric, value=v, bound=bound, passed=ok))
        if not ok:
            violations.append(f"{metric}={v:.4g} violates bound (min={bound.min}, max={bound.max})")

    comps = compare_suites(base, cand, policy.regression, alpha=policy.alpha,
                           n_resamples=policy.n_resamples, seed=policy.seed,
                           correction=policy.correction) if cand.records and base.records else []
    for c in comps:
        if c.regression:
            violations.append(
                f"significant regression in {c.metric}: {c.baseline:.4g} -> {c.candidate:.4g} "
                f"(delta {c.delta:+.4g}, 95% CI [{c.ci_low:+.4g}, {c.ci_high:+.4g}], p={c.p_value:.4f})"
            )

    bt, ct = base.per_task(), cand.per_task()
    per_task = {t: (bt.get(t, {}).get("pass_rate", float("nan")), ct[t]["pass_rate"]) for t in ct}
    worst = sorted(cand.reports, key=lambda r: r.metrics.score)[:5]
    worst_runs = [{"run_id": r.run_id, "task_id": r.task_id, "score": r.metrics.score,
                   "reasons": r.failure_reasons} for r in worst if not r.passed]
    return GateResult(passed=not violations, violations=violations, absolute=absolute, comparisons=comps,
                      baseline_name=base.name, candidate_name=cand.name, candidate_aggregates=agg_c,
                      baseline_aggregates=agg_b, per_task_pass=per_task, worst_runs=worst_runs)


def run_gate(
    baseline: TrajectorySuite,
    candidate: TrajectorySuite,
    goldens: dict[str, GoldenDAG],
    policy: Policy,
    judge: Judge | None = None,
) -> GateResult:
    """Grade both suites and evaluate the gate."""
    j = judge if judge is not None else (HeuristicJudge() if policy.use_judge else None)
    return evaluate_gate(grade_suite(baseline, goldens, j), grade_suite(candidate, goldens, j), policy)


def run_gate_paths(baseline: str | Path, candidate: str | Path, goldens_dir: str | Path,
                   policy: str | Path) -> GateResult:
    return run_gate(TrajectorySuite.load(baseline), TrajectorySuite.load(candidate),
                    load_goldens(goldens_dir), Policy.load(policy))


def _fmt(x: float) -> str:
    if x != x:  # NaN
        return "n/a"
    return f"{x:.0f}" if abs(x) >= 100 else f"{x:.3f}"


def render_markdown(res: GateResult) -> str:
    """PR comment body."""
    status = "PASSED" if res.passed else "BLOCKED"
    icon = ":white_check_mark:" if res.passed else ":x:"
    lines = [
        f"## {icon} Trajectory Eval Gate: **{status}**",
        "",
        f"Baseline `{res.baseline_name}` vs candidate `{res.candidate_name}`.",
        "",
    ]
    if res.violations:
        lines += ["### Violations", ""] + [f"- {v}" for v in res.violations] + [""]
    if res.comparisons:
        lines += [
            "### Metric comparison (stratified bootstrap 95% CI, one-sided permutation p, Holm-corrected)",
            "",
            "| metric | better | baseline | candidate | delta | 95% CI | p | verdict |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for c in res.comparisons:
            verdict = ("REGRESSION" if c.regression else "improved" if c.improvement
                       else "noise" if not c.significant and c.material else "ok")
            lines.append(
                f"| {c.metric} | {c.direction} | {_fmt(c.baseline)} | {_fmt(c.candidate)} | "
                f"{c.delta:+.3g} | [{c.ci_low:+.3g}, {c.ci_high:+.3g}] | {c.p_value:.3f} | {verdict} |"
            )
        lines.append("")
    if res.absolute:
        lines += ["### Absolute thresholds", "", "| metric | value | min | max | ok |", "|---|---|---|---|---|"]
        for a in res.absolute:
            lines.append(f"| {a.metric} | {_fmt(a.value)} | {a.bound.min} | {a.bound.max} | "
                         f"{'yes' if a.passed else '**no**'} |")
        lines.append("")
    if res.per_task_pass:
        lines += ["### Per-task pass rate", "", "| task | baseline | candidate |", "|---|---|---|"]
        for t, (b, c) in sorted(res.per_task_pass.items()):
            lines.append(f"| {t} | {_fmt(b)} | {_fmt(c)} |")
        lines.append("")
    if res.worst_runs:
        lines += ["<details><summary>Worst candidate runs</summary>", ""]
        for w in res.worst_runs:
            lines.append(f"- `{w['run_id']}` ({w['task_id']}, score {w['score']:.3f}): {'; '.join(w['reasons'])}")
        lines += ["", "</details>", ""]
    lines.append("_Generated by trajectory_eval: grade the journey, not just the answer._")
    return "\n".join(lines)
