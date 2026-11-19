"""CLI exit codes, exercised via real subprocesses."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
DATA = PROJECT / "trajectory_eval" / "data"


def cli(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(PROJECT), str(REPO / "common")]), "AIKIT_OFFLINE": "1"}
    return subprocess.run([sys.executable, "-m", "trajectory_eval", *args], capture_output=True, text=True,
                          env=env, timeout=120)


def gate(candidate: str, tmp_path: Path, policy: Path = DATA / "policy.json") -> subprocess.CompletedProcess:
    return cli("gate", "--baseline", str(DATA / "suites/baseline.json"), "--candidate", candidate,
               "--policy", str(policy), "--report", str(tmp_path / "report.md"), "--json", str(tmp_path / "r.json"))


def test_gate_passes_exit_0(tmp_path):
    p = gate(str(DATA / "suites/candidate_ok.json"), tmp_path)
    assert p.returncode == 0, p.stderr
    assert "PASSED" in (tmp_path / "report.md").read_text()
    assert json.loads((tmp_path / "r.json").read_text())["passed"] is True


def test_gate_blocks_regression_exit_1(tmp_path):
    p = gate(str(DATA / "suites/candidate_regressed.json"), tmp_path)
    assert p.returncode == 1
    md = (tmp_path / "report.md").read_text()
    assert "BLOCKED" in md and "significant regression in pass_rate" in md
    assert "BLOCKED" in p.stdout


def test_gate_bad_input_exit_2(tmp_path):
    assert gate(str(tmp_path / "missing.json"), tmp_path).returncode == 2
    bad = tmp_path / "bad.json"
    bad.write_text('{"name": "x", "trajectories": [{"run_id": 1}]}')
    p = gate(str(bad), tmp_path)
    assert p.returncode == 2 and "error" in p.stderr
    bad_policy = tmp_path / "policy.json"
    bad_policy.write_text('{"alpha": 3}')
    assert gate(str(DATA / "suites/candidate_ok.json"), tmp_path, bad_policy).returncode == 2


def test_usage_error_exit_2():
    assert cli("gate", "--baseline", "x").returncode == 2
    assert cli("frobnicate").returncode == 2


def test_strict_policy_blocks_even_ok_candidate(tmp_path):
    policy = json.loads((DATA / "policy.json").read_text())
    policy["absolute"]["pass_rate"] = {"min": 0.99}
    pp = tmp_path / "strict.json"
    pp.write_text(json.dumps(policy))
    assert gate(str(DATA / "suites/candidate_ok.json"), tmp_path, pp).returncode == 1


def test_grade_and_calibrate_commands():
    p = cli("grade", "--suite", str(DATA / "suites/candidate_regressed.json"),
            "--run-id", "candidate_regressed-refund_request-001")
    assert p.returncode == 0 and "forbidden" in p.stdout
    p = cli("calibrate", "--suite", str(DATA / "suites/labeled.json"), "--labels", str(DATA / "judge_labels.json"),
            "--method", "isotonic")
    assert p.returncode == 0
    rep = json.loads(p.stdout)
    assert rep["post"]["kappa"] > rep["pre"]["kappa"]


@pytest.mark.parametrize("path", ["ci/trajectory-gate.yml"])
def test_ci_workflow_present(path):
    text = (PROJECT / path).read_text()
    assert "python -m trajectory_eval gate" in text and "pull_request" in text
