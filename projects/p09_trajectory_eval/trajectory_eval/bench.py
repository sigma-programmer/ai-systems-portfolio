"""Benchmark: grading throughput, judge calibration, regression-detection quality.

``benchmark()`` is deterministic in every metric except the wall-clock
throughput numbers.
"""
from __future__ import annotations

import json
import time
from dataclasses import replace

import numpy as np

from .align import align
from .calibration import calibrate_judge
from .datagen import BASELINE, DATA_DIR, MIXED, human_labels, simulate_suite
from .gate import Policy, evaluate_gate
from .golden import load_goldens
from .judge import HeuristicJudge, LLMJudge, offline_judge_llm
from .regression import grade_suite

INJECTIONS = {
    "drop_required_step": replace(BASELINE, p_skip=0.30),
    "forbidden_action": replace(BASELINE, p_forbidden=0.15),
    "token_bloat": replace(BASELINE, llm_tokens=420.0),
    "reordering": replace(BASELINE, p_swap=0.35),
}


def _naive_blocks(base, cand, policy: Policy) -> bool:
    """Strawman gate: block whenever a mean moves the wrong way by min_effect, no statistics."""
    ab, ac = base.aggregates(), cand.aggregates()
    for m, rule in policy.regression.items():
        if m not in ab:
            continue
        hib = (rule.direction or ("lower" if m in {"forbidden_rate", "loop_rate", "tokens", "latency_ms"} else "higher")) == "higher"
        bad = (ab[m] - ac[m]) if hib else (ac[m] - ab[m])
        rel = bad / abs(ab[m]) if ab[m] else 0.0
        if bad > 0 and bad >= rule.min_effect and rel >= rule.min_effect_rel:
            return True
    return False


def benchmark(n_noise: int = 32, n_per_injection: int = 8, runs_per_task: int = 20) -> dict:
    goldens = load_goldens(DATA_DIR / "golden")

    # 1. Grading throughput.
    suite, sims = simulate_suite("thr", MIXED, runs_per_task=50, seed=11)
    n_steps = sum(len(t.steps) for t in suite.trajectories)
    t0 = time.perf_counter()
    reports = [align(t, goldens[t.task_id]) for t in suite.trajectories]
    dt = time.perf_counter() - t0
    agree = float(np.mean([r.passed == s.good for r, s in zip(reports, sims)]))

    # 2. Judge calibration on a fresh labelled set (heuristic and via the LLM-protocol path).
    lab_suite, lab_sims = simulate_suite("lab", MIXED, runs_per_task=40, seed=404)
    labels = {d["run_id"]: d["label"] for d in human_labels(lab_sims, noise=0.06, seed=405)}
    y = np.array([labels[t.run_id] for t in lab_suite.trajectories])
    hj = HeuristicJudge()
    scores = np.array([hj.judge(t).score for t in lab_suite.trajectories])
    llm_judge = LLMJudge(offline_judge_llm())
    t0 = time.perf_counter()
    llm_scores = np.array([llm_judge.judge(t).score for t in lab_suite.trajectories[:40]])
    llm_dt = time.perf_counter() - t0
    cal = {}
    for method in ("platt", "isotonic"):
        rep, _, _ = calibrate_judge(scores, y, method=method)
        cal[method] = {"kappa_pre": round(rep.pre.kappa, 4), "kappa_post_cv": round(rep.post.kappa, 4),
                       "acc_pre": round(rep.pre.accuracy, 4), "acc_post_cv": round(rep.post.accuracy, 4),
                       "ece_pre": round(rep.pre.ece, 4), "ece_post_cv": round(rep.post.ece, 4),
                       "brier_pre": round(rep.pre.brier, 4), "brier_post_cv": round(rep.post.brier, 4)}

    # 3. Regression detection: noise-only vs injected regressions.
    policy = Policy.load(DATA_DIR / "policy.json").model_copy(update={"n_resamples": 1000, "absolute": {}})
    base_res = grade_suite(simulate_suite("base", BASELINE, runs_per_task, seed=1)[0], goldens, hj)
    outcomes: list[tuple[str, bool, bool, bool]] = []  # (kind, is_regression, gate, naive)
    for k in range(n_noise):
        cand = grade_suite(simulate_suite(f"noise{k}", BASELINE, runs_per_task, seed=1000 + k)[0], goldens, hj)
        outcomes.append(("noise", False, not evaluate_gate(base_res, cand, policy).passed,
                         _naive_blocks(base_res, cand, policy)))
    for name, prof in INJECTIONS.items():
        for k in range(n_per_injection):
            cand = grade_suite(simulate_suite(f"{name}{k}", prof, runs_per_task, seed=2000 + 37 * k)[0], goldens, hj)
            outcomes.append((name, True, not evaluate_gate(base_res, cand, policy).passed,
                             _naive_blocks(base_res, cand, policy)))

    def _det(idx: int) -> dict:
        tp = sum(o[1] and o[idx] for o in outcomes)
        fp = sum((not o[1]) and o[idx] for o in outcomes)
        fn = sum(o[1] and not o[idx] for o in outcomes)
        tn = sum((not o[1]) and not o[idx] for o in outcomes)
        return {"precision": round(tp / (tp + fp), 4) if tp + fp else 1.0,
                "recall": round(tp / (tp + fn), 4) if tp + fn else 1.0,
                "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    per_injection = {name: sum(o[2] for o in outcomes if o[0] == name) / n_per_injection for name in INJECTIONS}
    return {
        "grading": {
            "trajectories": len(suite.trajectories), "steps": n_steps,
            "trajectories_per_sec": round(len(suite.trajectories) / dt, 1),
            "steps_per_sec": round(n_steps / dt, 1),
            "alignment_vs_ground_truth_agreement": round(agree, 4),
        },
        "judge": {
            "n_labels": int(len(y)), "positive_rate": round(float(y.mean()), 4),
            "raw_score_mean_pos": round(float(scores[y == 1].mean()), 4),
            "raw_score_mean_neg": round(float(scores[y == 0].mean()), 4),
            "llm_protocol_path_matches_heuristic": bool(np.allclose(llm_scores, scores[:40], atol=1e-6)),
            "llm_protocol_judgements_per_sec": round(40 / llm_dt, 1),
            **cal,
        },
        "regression_detection": {
            "scenarios": len(outcomes), "noise_only": n_noise,
            "injected": n_per_injection * len(INJECTIONS), "runs_per_suite": runs_per_task * len(goldens),
            "stat_gate": _det(2), "naive_mean_delta_gate": _det(3),
            "stat_gate_recall_by_injection": per_injection,
        },
    }


if __name__ == "__main__":
    t = time.perf_counter()
    out = benchmark()
    out["wall_s"] = round(time.perf_counter() - t, 2)
    print(json.dumps(out, indent=2))
