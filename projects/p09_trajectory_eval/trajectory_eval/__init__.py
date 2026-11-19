"""trajectory_eval: grade the journey, not just the answer.

Step-level trajectory grading against golden partial-order DAGs, LLM-as-judge
with calibration, statistically sound regression detection and a CI gate.
"""
from .align import AlignmentReport, StepStatus, align, detect_loops
from .calibration import IsotonicCalibrator, PlattCalibrator, calibrate_judge, cohen_kappa
from .gate import GateResult, Policy, evaluate_gate, render_markdown, run_gate
from .golden import GoldenDAG, GoldenNode, load_goldens
from .graders import default_graders, run_graders
from .judge import HeuristicJudge, LLMJudge, offline_judge_llm
from .matchers import ArgMatcher
from .regression import compare_suites, grade_suite
from .schema import Step, StepType, Trajectory, TrajectorySuite

__all__ = [
    "AlignmentReport", "StepStatus", "align", "detect_loops",
    "IsotonicCalibrator", "PlattCalibrator", "calibrate_judge", "cohen_kappa",
    "GateResult", "Policy", "evaluate_gate", "render_markdown", "run_gate",
    "GoldenDAG", "GoldenNode", "load_goldens", "default_graders", "run_graders",
    "HeuristicJudge", "LLMJudge", "offline_judge_llm", "ArgMatcher",
    "compare_suites", "grade_suite", "Step", "StepType", "Trajectory", "TrajectorySuite",
]
