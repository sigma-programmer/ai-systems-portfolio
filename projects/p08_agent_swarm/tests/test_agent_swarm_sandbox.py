"""Sandbox execution and static analysis."""
from __future__ import annotations

from agent_swarm import Sandbox, TestCase, static_check

T = [TestCase(id="a", args=[2], expected=4), TestCase(id="b", args=[-1], expected=1)]


def test_pass_and_fail(sandbox):
    rep = sandbox.run("def sq(x):\n    return x * x\n", "sq", T)
    assert rep.passed and rep.n_passed == 2
    rep = sandbox.run("def sq(x):\n    return x + x\n", "sq", T)
    assert not rep.passed and rep.failed_ids == ["b"]


def test_expected_exception(sandbox):
    tests = [TestCase(id="r", args=[0], raises="ValueError")]
    assert sandbox.run("def f(x):\n    raise ValueError('no')\n", "f", tests).passed
    rep = sandbox.run("def f(x):\n    return 1\n", "f", tests)
    assert not rep.passed and "expected ValueError" in rep.results[0].error
    rep = sandbox.run("def f(x):\n    raise KeyError(x)\n", "f", tests)
    assert not rep.passed and rep.results[0].error.startswith("KeyError")


def test_infinite_loop_times_out_per_test():
    sb = Sandbox(per_test_timeout=0.2)
    src = "def f(x):\n    while x > 0:\n        x = x + 1\n    return x\n"
    rep = sb.run(src, "f", [TestCase(id="loop", args=[1], expected=0), TestCase(id="ok", args=[0], expected=0)])
    assert rep.results[0].error == "timeout"
    assert rep.results[1].passed  # the batch survives the hung test


def test_static_gate_blocks_dangerous_code(sandbox):
    rep = sandbox.run("import os\ndef f(x):\n    return os.getcwd()\n", "f", T[:1])
    assert rep.compile_error and "forbidden import 'os'" in rep.compile_error
    assert sandbox.executions >= 0


def test_static_check_rules():
    assert static_check("def f(x):\n    return x\n", "f", 1).ok
    assert "missing top-level function" in static_check("def g(x):\n    return x\n", "f").errors[0]
    assert not static_check("def f(x):\n    return eval(x)\n", "f").ok
    assert not static_check("def f(x):\n    return x.__class__\n", "f").ok
    assert not static_check("def f(x):\n    while True:\n        x += 1\n", "f").ok
    assert not static_check("def f(x, y):\n    return x\n", "f", 1).ok
    assert static_check("def f(x, y=1):\n    return x\n", "f", 1).ok
    assert not static_check("def f(:\n", "f").ok
    rep = static_check("def f(x, acc=[]):\n    try:\n        return x\n    except:\n        return 0\n", "f", 1)
    assert rep.ok and len(rep.warnings) == 2


def test_args_are_isolated_between_tests(sandbox):
    src = "def f(xs):\n    xs.append(1)\n    return len(xs)\n"
    tests = [TestCase(id="a", args=[[]], expected=1), TestCase(id="b", args=[[]], expected=1)]
    assert sandbox.run(src, "f", tests).passed


def test_bool_int_and_float_comparison(sandbox):
    assert not sandbox.run("def f():\n    return 1\n", "f", [TestCase(id="a", args=[], expected=True)]).passed
    assert sandbox.run("def f():\n    return 0.1 + 0.2\n", "f", [TestCase(id="a", args=[], expected=0.3)]).passed
    assert sandbox.run("def f():\n    return (1, 2)\n", "f", [TestCase(id="a", args=[], expected=[1, 2])]).passed


def test_runtime_compile_error_and_cache():
    sb = Sandbox()
    rep = sb.run("def f():\n    return 1\nraise RuntimeError('boom')\n", "f", T[:1])
    assert rep.compile_error and "RuntimeError" in rep.compile_error
    src = "def sq(x):\n    return x * x\n"
    sb.run(src, "sq", T)
    spawned = sb.executions
    sb.run(src, "sq", T)
    assert sb.executions == spawned and sb.cache_hits >= 2


def test_run_many_batches_in_one_subprocess():
    sb = Sandbox()
    srcs = [f"def sq(x):\n    return x * {k}\n" for k in range(5)]
    reps = sb.run_many(srcs, "sq", T)
    assert sb.executions == 1
    assert [r.passed for r in reps] == [False, False, False, False, False]
    reps = sb.run_many(["def sq(x):\n    return x * x\n"] + srcs, "sq", T, stop_on_fail=True)
    assert reps[0].passed and not any(r.passed for r in reps[1:])
    assert len(reps[1].results) == 1  # x*0 fails the first test and stops early
