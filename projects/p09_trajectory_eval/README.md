# P09: Trajectory Eval CI Gate

> You grade the journey, not just the answer.

## Problem

An agent can reach the right answer by doing the wrong thing. It might issue a refund before it checks the refund policy, `git push --force` on its way to green tests, or loop on the same tool call six times. If an eval only checks the final answer, it passes all three. Grading full trajectories has two problems of its own:

1. **Agent runs are nondeterministic.** If you block a PR every time a mean moves, you block constantly. Reviewers learn to ignore the gate.
2. **LLM judges are not calibrated.** A raw judge score of 0.8 may mean nothing about what a human would accept.

`trajectory_eval` handles both. It grades each step against a golden partial-order DAG and calibrates the judge against human labels. It only blocks a PR on regressions that are both statistically significant and large enough to matter.

## Architecture

```
                 golden DAGs (networkx partial order)          policy.json
                 required / optional / forbidden nodes            |
                 arg matchers: exact|regex|schema|semantic         |
                               |                                   |
 baseline.json --+             v                                   v
                 |   +-------------------+   +-----------+   +-------------+   exit 0 / 1 / 2
 candidate.json -+-->|  align (per run)  |-->| RunRecord |-->| compare     |--> gate_report.md
                     |  tiered greedy    |   | metrics   |   | stratified  |    (PR comment)
                     |  matching, order  |   | per run   |   | bootstrap + |--> gate_result.json
                     |  closure, loops   |   +-----------+   | permutation,|
                     +-------------------+        ^          | Holm, min-  |
                     | step graders      |--------+          | effect,     |
                     | (rule-based)      |        |          | abs. bounds |
                     +-------------------+        |          +-------------+
                     | judge (LLM proto- |--------+
                     | col | heuristic)  |<-- calibration (Platt / isotonic, CV kappa)
                     +-------------------+        ^
                                                  |
                                       human labels (judge_labels.json)
```

| module | role |
|---|---|
| `schema.py` | Pydantic `Step` (tool_call / llm / observation, tool, args, result, tokens, latency, error), `Trajectory`, `TrajectorySuite` |
| `matchers.py` | `ArgMatcher`: exact, regex, JSON-schema subset, semantic (HashingEmbedder cosine), any |
| `golden.py` | `GoldenNode` / `GoldenDAG`, validated at load time (acyclic, known nodes, forbidden nodes have no edges) |
| `align.py` | trajectory-to-DAG alignment, which produces a per-step `StepGrade` plus metrics |
| `graders.py` | rule-based step graders: latency, token, argument hygiene, safety, error recovery |
| `judge.py` | `LLMJudge` (any `aikit.LLM`, rubric, schema-validated, repair then fallback), `HeuristicJudge`, `offline_judge_llm()` |
| `calibration.py` | kappa, ECE, Brier, Platt (Newton/IRLS), isotonic (PAV), kappa-optimal threshold, cross-validated report |
| `stats.py` | stratified bootstrap CI and stratified permutation test |
| `regression.py` | suite grading and baseline-vs-candidate comparison with Holm correction |
| `gate.py`, `cli.py` | policy, gate verdict, markdown PR comment, CLI |
| `datagen.py` | deterministic agent simulator plus a 4-task library, used to build the bundled data |

## Key design decisions and tradeoffs

- **Goldens are partial orders, not sequences.** Edge `u -> v` means "u before v". Unrelated nodes can run in any order or in parallel. The order check runs over the **transitive closure**, so if an optional middle node is skipped, the order constraint across it still holds. `order_score = 1 - violations / comparable pairs`.
- **Tiered greedy alignment.** Each graded step goes to: (0) an unmatched node whose required ancestors are all satisfied, then (1) an allowed repeat, then (2) an unmatched node that isn't ready yet (graded `out_of_order`), then (3) a repeat past `max_repeats` (graded `redundant`). This handles goldens where the same tool appears more than once. For example, `run_tests` before and after `edit_file` in `code_fix` are assigned by readiness. Forbidden patterns are checked first, so `issue_refund(amount=900)` is graded `forbidden` rather than a match. A retry right after an errored observation is graded `repeat`, not `redundant`. Greedy matching is O(steps x nodes) and fast. The tradeoff is that it isn't a globally optimal assignment. On the simulator it agrees 100% with the injected ground truth.
- **Loop detection** looks for back-to-back repeats (3 or more) of blocks with period 1 to 3, using canonical tool+args signatures.
- **The judge doesn't see the golden.** An LLM judge in production also only sees the task and the transcript. The offline `HeuristicJudge` uses the same rubric: relevance, groundedness, efficiency, error handling, safety. `offline_judge_llm()` is a `MockLLM` that parses the trajectory JSON in the prompt and grades it, so the full `LLMJudge` path runs offline: prompt, JSON parse, Pydantic validation, repair, fallback. Swap in `ClaudeLLM` with no code changes.
- **Calibration numbers are cross-validated.** The post-calibration kappa comes from 5-fold CV, with the calibrator and threshold fitted on the training folds and scored on held-out data. That keeps the improvement from being in-sample fitting. scikit-learn is **not installed** in this environment, so Platt (Newton's method with Platt's smoothed targets) and isotonic regression (pool-adjacent-violators) are implemented in numpy.
- **Noise can't block a PR on its own.** A metric is flagged as a regression only if all of these hold:
  - it moved in the bad direction;
  - the change is **material**, meaning it clears the policy's `min_effect` or `min_effect_rel`;
  - it is **significant**: the one-sided stratified permutation p-value beats the Holm-adjusted alpha *and* the stratified bootstrap 95% CI excludes 0.

  Runs are stratified by task, and the statistic is the stratum-weighted difference using the baseline's task mix. Running the easy task more often in the candidate therefore cannot look like an improvement or hide a regression (covered by a test). Holm-Bonferroni controls family-wise error across the ~10 gated metrics. In the benchmark, this took the false-positive rate on noise-only runs from 15.6% to 0%.
- **Absolute bounds sit next to relative ones.** For example `pass_rate >= 0.6` and `forbidden_rate <= 0.1`. A slow drift that never shows up as a significant step can't slip under the floor.
- **Exit codes:** `0` passed, `1` blocked by policy, `2` bad input or config (missing file, invalid suite or policy JSON, argparse usage error).

## Bundled data (`trajectory_eval/data/`)

- `golden/`: 4 tasks.
  - `refund_request`: chain with an optional confirmation email; forbidden `delete_account` and refunds over $500.
  - `trip_planner`: 3 parallel searches, then an LLM compare node, then optional booking; forbidden `charge_card`.
  - `code_fix`: the same tool in two nodes; forbidden destructive shell matched by regex.
  - `research_summary`: semantic query matcher and an LLM summary node; forbidden social post.
- `suites/baseline.json`, `candidate_ok.json` (same agent, different seed, noise only), `candidate_regressed.json` (deliberately regressed: skips steps, takes forbidden actions, more tokens). 48 runs each.
- `suites/labeled.json` (160 runs) with `judge_labels.json`: simulated human labels with a 6% flip rate, used for calibration.
- `policy.json`: the gate policy. `ci/trajectory-gate.yml` is a ready-to-copy GitHub Actions workflow that runs the gate, uploads the artifacts, posts a sticky PR comment, and fails the job on a non-zero exit.

Regenerate the data with `PYTHONPATH=common:projects/p09_trajectory_eval .venv/bin/python -m trajectory_eval.datagen`.

## How to run

```bash
# tests (64)
.venv/bin/python -m pytest projects/p09_trajectory_eval

# end-to-end demo
AIKIT_OFFLINE=1 .venv/bin/python projects/p09_trajectory_eval/demo.py

# CI gate
export PYTHONPATH=common:projects/p09_trajectory_eval
D=projects/p09_trajectory_eval/trajectory_eval/data
.venv/bin/python -m trajectory_eval gate --baseline $D/suites/baseline.json \
    --candidate $D/suites/candidate_regressed.json --policy $D/policy.json \
    --report pr_comment.md --json gate.json; echo "exit=$?"     # exit=1
.venv/bin/python -m trajectory_eval grade --suite $D/suites/candidate_regressed.json \
    --run-id candidate_regressed-refund_request-001             # step-level report
.venv/bin/python -m trajectory_eval calibrate --suite $D/suites/labeled.json \
    --labels $D/judge_labels.json --method isotonic

# benchmark
.venv/bin/python -m trajectory_eval.bench
```

## Benchmark (measured, `python -m trajectory_eval.bench`, about 2.9 s wall)

**Grading.** 200 trajectories and 1,936 steps.

| metric | value |
|---|---|
| trajectories/s | about 11,500 (wall-clock, varies by machine) |
| steps/s | about 112,000 |
| alignment pass/fail vs injected ground-truth defects | **100%** agreement |

**Judge calibration.** 160 human labels, 44% positive, 5-fold CV. The raw judge scores every run above 0.5 (mean 0.95 on positives, 0.85 on negatives), so the naive 0.5 cut agrees with humans only by chance.

| method | kappa pre → post | accuracy pre → post | ECE pre → post | Brier pre → post |
|---|---|---|---|---|
| Platt | 0.000 → **0.534** | 0.444 → 0.763 | 0.448 → 0.114 | 0.409 → 0.191 |
| isotonic | 0.000 → **0.534** | 0.444 → 0.763 | 0.448 → 0.114 | 0.409 → 0.188 |

The `LLMJudge` path through the LLM protocol gives the same scores as the heuristic judge (about 3,600 judgements/s offline).

**Regression detection.** 64 scenarios of 80 vs 80 runs: 32 noise-only, and 32 with injected regressions (8 each of dropped required step, forbidden action, token bloat, reordering).

| gate | precision | recall | false-positive rate |
|---|---|---|---|
| **statistical gate** (stratified permutation + bootstrap, Holm, min-effect) | **1.000** | 0.938 | **0.000** |
| naive "mean moved by min_effect" gate | 0.667 | 1.000 | 0.500 |

Recall per injection type for the statistical gate:

| injection | recall |
|---|---|
| drop step | 1.00 |
| forbidden action | 0.75 |
| token bloat | 1.00 |
| reordering | 1.00 |

The two misses were forbidden-action injections at a 15% rate. Absolute bounds such as `forbidden_rate <= 0.1` exist to catch that kind of low-rate safety regression. The bound was disabled in this benchmark so it measures the statistical test on its own. Before the Holm correction was added, the same benchmark gave the statistical gate a 15.6% false-positive rate.
