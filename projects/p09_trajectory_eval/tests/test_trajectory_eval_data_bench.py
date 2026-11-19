import json

import numpy as np

from trajectory_eval.align import align
from trajectory_eval.bench import benchmark
from trajectory_eval.datagen import BASELINE, DATA_DIR, MIXED, simulate_suite, task_library
from trajectory_eval.golden import load_goldens


def test_simulator_deterministic():
    a, _ = simulate_suite("s", MIXED, 3, seed=9)
    b, _ = simulate_suite("s", MIXED, 3, seed=9)
    assert a == b


def test_alignment_agrees_with_injected_ground_truth():
    goldens = load_goldens(DATA_DIR / "golden")
    _, sims = simulate_suite("s", MIXED, 25, seed=3)
    agree = np.mean([align(s.trajectory, goldens[s.trajectory.task_id]).passed == s.good for s in sims])
    assert agree >= 0.97


def test_clean_profile_always_passes():
    from dataclasses import replace
    clean = replace(BASELINE, p_skip=0, p_swap=0, p_forbidden=0, p_loop=0, p_wrong_arg=0, p_wrong_answer=0,
                    p_extra=0, p_error=0.5)
    goldens = load_goldens(DATA_DIR / "golden")
    suite, _ = simulate_suite("c", clean, 10, seed=1)
    assert all(align(t, goldens[t.task_id]).passed for t in suite.trajectories)


def test_bundled_goldens_match_library():
    goldens = load_goldens(DATA_DIR / "golden")
    for tid, spec in task_library().items():
        assert goldens[tid].model_dump() == spec.golden.model_dump()


def test_bundled_labels_cover_labeled_suite():
    labels = json.loads((DATA_DIR / "judge_labels.json").read_text())
    suite = json.loads((DATA_DIR / "suites/labeled.json").read_text())
    assert {d["run_id"] for d in labels} == {t["run_id"] for t in suite["trajectories"]}


def test_benchmark_small():
    out = benchmark(n_noise=4, n_per_injection=1, runs_per_task=10)
    json.dumps(out)
    assert out["grading"]["alignment_vs_ground_truth_agreement"] >= 0.97
    assert out["judge"]["platt"]["kappa_post_cv"] > out["judge"]["platt"]["kappa_pre"]
    assert out["judge"]["llm_protocol_path_matches_heuristic"]
    assert out["regression_detection"]["scenarios"] == 8
