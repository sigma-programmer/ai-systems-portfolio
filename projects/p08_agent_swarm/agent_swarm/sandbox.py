"""Subprocess sandbox + AST static analysis for untrusted generated code.

`static_check` rejects dangerous constructs (imports of os/subprocess/...,
eval/exec/open, dunder attribute escapes) *before* anything executes.
`Sandbox.run` then executes one or many candidate sources against a list of
`TestCase`s in a fresh ``python -I -S`` subprocess with:

* a per-test wall-clock alarm (``signal.setitimer``) so an infinite loop in one
  candidate fails that test instead of hanging the batch,
* a hard overall subprocess timeout as a second line of defence,
* best-effort CPU/memory rlimits on POSIX,
* deep-copied arguments so a mutating candidate cannot corrupt later tests.

Results are memoised per (source, test) so several reviewers inspecting the
same artifact — or the coder re-checking candidates across revisions — only
pay for execution once.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import dataclass, field

from .messages import TestCase, source_hash

FORBIDDEN_MODULES = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib", "ctypes", "builtins",
    "signal", "threading", "multiprocessing", "pickle", "marshal", "inspect", "gc", "resource", "io",
})
FORBIDDEN_CALLS = frozenset({
    "eval", "exec", "compile", "open", "__import__", "input", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "breakpoint", "exit", "quit",
})


@dataclass
class StaticReport:
    """Outcome of AST analysis. ``errors`` block execution; ``warnings`` do not."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def static_check(source: str, func_name: str, arity: int | None = None) -> StaticReport:
    """Analyse ``source`` without executing it."""
    errors: list[str] = []
    warnings: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return StaticReport(False, [f"syntax error: {e.msg} (line {e.lineno})"])
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    fn = funcs.get(func_name)
    if fn is None:
        errors.append(f"missing top-level function {func_name!r}")
    elif arity is not None:
        n_pos = len(fn.args.posonlyargs) + len(fn.args.args)
        n_required = n_pos - len(fn.args.defaults)
        if not (n_required <= arity <= n_pos or fn.args.vararg):
            errors.append(f"{func_name} takes {n_pos} positional args, spec requires {arity}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in FORBIDDEN_MODULES:
                    errors.append(f"forbidden import {a.name!r}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in FORBIDDEN_MODULES:
                errors.append(f"forbidden import from {node.module!r}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            errors.append(f"forbidden call {node.func.id}()")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            errors.append(f"dunder attribute access .{node.attr}")
        elif isinstance(node, ast.While):
            if isinstance(node.test, ast.Constant) and node.test.value and not any(
                isinstance(n, (ast.Break, ast.Return)) for n in ast.walk(node)
            ):
                errors.append("unconditional while loop with no break/return")
        elif isinstance(node, ast.ExceptHandler) and node.type is None:
            warnings.append("bare except clause")
        elif isinstance(node, ast.FunctionDef):
            for d in node.args.defaults:
                if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                    warnings.append(f"mutable default argument in {node.name}")
            if not node.body or all(isinstance(s, ast.Pass) for s in node.body):
                errors.append(f"function {node.name} has an empty body")
    return StaticReport(not errors, errors, warnings)


# The runner executes inside the child interpreter. It reads one JSON job from
# stdin and writes one JSON result to stdout.
_RUNNER = r'''
import copy, json, signal, sys
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
except Exception:
    pass

class _Timeout(BaseException):
    pass

def _alarm(signum, frame):
    raise _Timeout()

signal.signal(signal.SIGALRM, _alarm)

def _norm(x):
    return json.loads(json.dumps(x, default=repr))

def _eq(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(b)))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_eq(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b

job = json.loads(sys.stdin.read())
out = []
for src in job["sources"]:
    ns = {"__name__": "candidate"}
    res = {"compile_error": None, "results": []}
    try:
        signal.setitimer(signal.ITIMER_REAL, job["timeout"])
        exec(compile(src, "<candidate>", "exec"), ns)
        signal.setitimer(signal.ITIMER_REAL, 0)
        fn = ns[job["func"]]
    except BaseException as e:
        signal.setitimer(signal.ITIMER_REAL, 0)
        res["compile_error"] = type(e).__name__ + ": " + str(e)[:200]
        out.append(res)
        continue
    for t in job["tests"]:
        r = {"id": t["id"], "passed": False, "error": None, "got": None}
        try:
            signal.setitimer(signal.ITIMER_REAL, job["timeout"])
            got = fn(*copy.deepcopy(t["args"]))
            signal.setitimer(signal.ITIMER_REAL, 0)
            if t.get("raises"):
                r["error"] = "expected %s, returned %r" % (t["raises"], got)
            else:
                g = _norm(got)
                r["got"] = repr(got)[:200]
                r["passed"] = _eq(g, t["expected"])
        except _Timeout:
            r["error"] = "timeout"
        except BaseException as e:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if t.get("raises") and type(e).__name__ == t["raises"]:
                r["passed"] = True
            else:
                r["error"] = type(e).__name__ + ": " + str(e)[:200]
        res["results"].append(r)
        if job["stop_on_fail"] and not r["passed"]:
            break
    out.append(res)
sys.stdout.write(json.dumps(out))
'''


@dataclass
class TestResult:
    """Outcome of one test against one source."""

    __test__ = False

    id: str
    passed: bool
    error: str | None = None
    got: str | None = None


@dataclass
class RunReport:
    """Outcome of one source against a test list."""

    source_hash: str
    compile_error: str | None
    results: list[TestResult]
    static: StaticReport | None = None

    @property
    def passed(self) -> bool:
        return self.compile_error is None and bool(self.results) and all(r.passed for r in self.results)

    @property
    def failed_ids(self) -> list[str]:
        return [r.id for r in self.results if not r.passed]

    @property
    def n_passed(self) -> int:
        return sum(r.passed for r in self.results)


class Sandbox:
    """Executes candidate code in isolated subprocesses, with result memoisation."""

    def __init__(self, per_test_timeout: float = 0.5, batch_timeout: float = 20.0, python: str | None = None):
        self.per_test_timeout = per_test_timeout
        self.batch_timeout = batch_timeout
        self.python = python or sys.executable
        self.executions = 0          # subprocesses spawned
        self.cache_hits = 0          # (source, test) pairs served from cache
        self._cache: dict[tuple[str, str], TestResult] = {}
        self._compile_errors: dict[str, str] = {}

    @staticmethod
    def _test_key(t: TestCase) -> str:
        return json.dumps([t.args, t.expected, t.raises], sort_keys=True, default=repr)

    def _spawn(self, sources: list[str], func: str, tests: list[TestCase], stop_on_fail: bool,
               timeout: float) -> list[dict]:
        job = {
            "sources": sources,
            "func": func,
            "tests": [{"id": t.id, "args": t.args, "expected": t.expected, "raises": t.raises} for t in tests],
            "timeout": timeout,
            "stop_on_fail": stop_on_fail,
        }
        self.executions += 1
        budget = min(self.batch_timeout, 2.0 + timeout * len(tests) * len(sources) * 1.5)
        try:
            proc = subprocess.run(
                [self.python, "-I", "-S", "-c", _RUNNER], input=json.dumps(job, default=repr),
                capture_output=True, text=True, timeout=budget,
            )
        except subprocess.TimeoutExpired:
            return [{"compile_error": "sandbox timeout", "results": []} for _ in sources]
        if proc.returncode != 0:
            err = (proc.stderr.strip().splitlines() or ["sandbox crashed"])[-1][:200]
            return [{"compile_error": f"sandbox error: {err}", "results": []} for _ in sources]
        return json.loads(proc.stdout)

    def run_many(
        self, sources: list[str], func: str, tests: list[TestCase], stop_on_fail: bool = False,
        arity: int | None = None, timeout: float | None = None,
    ) -> list[RunReport]:
        """Run every source against every test (static-gated, memoised).

        ``timeout`` overrides the per-test alarm (the coder's search uses a short
        one: a mutant that loops is simply inconsistent).
        """
        reports: list[RunReport | None] = [None] * len(sources)
        todo: list[int] = []
        statics: list[StaticReport] = []
        for i, src in enumerate(sources):
            st = static_check(src, func, arity)
            statics.append(st)
            h = source_hash(src)
            if not st.ok:
                reports[i] = RunReport(h, "static check failed: " + "; ".join(st.errors), [], st)
            elif h in self._compile_errors:
                reports[i] = RunReport(h, self._compile_errors[h], [], st)
            elif self._servable(h, tests, stop_on_fail):
                rep = self._from_cache(h, tests, st, stop_on_fail)
                self.cache_hits += len(rep.results)
                reports[i] = rep
            else:
                todo.append(i)
        if todo:
            raw = self._spawn([sources[i] for i in todo], func, tests, stop_on_fail,
                              timeout if timeout is not None else self.per_test_timeout)
            for i, res in zip(todo, raw):
                h = source_hash(sources[i])
                if res["compile_error"]:
                    if not res["compile_error"].startswith("sandbox"):
                        self._compile_errors[h] = res["compile_error"]
                    reports[i] = RunReport(h, res["compile_error"], [], statics[i])
                    continue
                by_id = {r["id"]: r for r in res["results"]}
                for t in tests:
                    if t.id in by_id:
                        r = by_id[t.id]
                        self._cache[(h, self._test_key(t))] = TestResult(t.id, r["passed"], r["error"], r["got"])
                reports[i] = self._from_cache(h, tests, statics[i], stop_on_fail)
        return [r for r in reports if r is not None]

    def _servable(self, h: str, tests: list[TestCase], stop_on_fail: bool) -> bool:
        """True if cached results fully answer this request (a cached failure ends a stop-on-fail run)."""
        for t in tests:
            cached = self._cache.get((h, self._test_key(t)))
            if cached is None:
                return False
            if stop_on_fail and not cached.passed:
                return True
        return True

    def _from_cache(self, h: str, tests: list[TestCase], st: StaticReport, stop_on_fail: bool) -> RunReport:
        results: list[TestResult] = []
        for t in tests:
            cached = self._cache.get((h, self._test_key(t)))
            if cached is None:
                break
            results.append(TestResult(t.id, cached.passed, cached.error, cached.got))
            if stop_on_fail and not cached.passed:
                break
        return RunReport(h, None, results, st)

    def run(self, source: str, func: str, tests: list[TestCase], arity: int | None = None) -> RunReport:
        """Run one source against all tests."""
        return self.run_many([source], func, tests, arity=arity)[0]
