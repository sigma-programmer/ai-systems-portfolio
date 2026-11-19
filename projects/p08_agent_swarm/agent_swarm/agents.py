"""Agents: typed, role-scoped workers with validated hand-offs.

Every agent declares a ``role``, the message types it accepts
(``input_types``) and produces (``output_types``), and an optional dependency
on the ``aikit.LLM`` protocol. ``Agent.handle`` is the hand-off boundary:

1. reject inputs whose type the agent does not accept (a routing bug),
2. produce a raw output (LLM path if an LLM is wired in, else the algorithmic
   default),
3. validate it against the primary output model,
4. on a ``ValidationError`` ask the agent to *repair* (the LLM sees the exact
   validation errors), up to ``max_repairs`` times,
5. if the LLM path still fails, fall back to the algorithmic default; only if
   that also fails is a ``HandoffError`` raised for the supervisor to handle.
"""
from __future__ import annotations

import json
import random
import re
from ast import literal_eval
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Sequence

from pydantic import BaseModel, ValidationError

from aikit import LLM, HashingEmbedder, count_tokens, tokenize_words
from aikit.llm import extract_json

from .library import KNOWLEDGE, Snippet
from .messages import (
    CodeArtifact,
    Evidence,
    EvidenceKind,
    ResearchFindings,
    ReviewVerdict,
    RevisionRequest,
    Role,
    SwarmState,
    TaskSpec,
    TestCase,
)
from .sandbox import RunReport, Sandbox, static_check
from .synthesis import build_pool, rank


class HandoffError(RuntimeError):
    """An agent could not produce a valid output even after repair and fallback."""


class SynthesisExhausted(HandoffError):
    """The coder has no untried candidate left in its search space."""


@dataclass
class ArbiterResult:
    """What the supervisor finds when it re-executes the trusted test suite."""

    approved: bool
    failing: list[TestCase]
    contradicted: list[str] = field(default_factory=list)


@dataclass
class AgentContext:
    """Per-run services the supervisor lends to agents."""

    state: SwarmState
    sandbox: Sandbox
    emit: Callable[..., None] = lambda *a, **k: None
    arbiter: Callable[[list[ReviewVerdict]], ArbiterResult] | None = None


class Agent:
    """Base class for every swarm member."""

    role: ClassVar[Role]
    input_types: ClassVar[tuple[type[BaseModel], ...]] = ()
    output_types: ClassVar[tuple[type[BaseModel], ...]] = ()
    system_prompt: ClassVar[str] = "You are a member of a software engineering agent swarm."

    def __init__(self, name: str | None = None, llm: LLM | None = None, max_repairs: int = 2):
        self.name = name or self.role.value
        self.llm = llm
        self.max_repairs = max_repairs
        self._llm_tokens = 0

    # -- type metadata -----------------------------------------------------
    @property
    def output_type(self) -> type[BaseModel]:
        return self.output_types[0]

    def accepts(self, t: type) -> bool:
        return any(issubclass(t, it) for it in self.input_types)

    # -- behaviour to override ----------------------------------------------
    def act_default(self, inputs: Sequence[BaseModel], ctx: AgentContext) -> Any:
        raise NotImplementedError

    def act_llm(self, inputs: Sequence[BaseModel], ctx: AgentContext) -> Any:
        """LLM path; subclasses build a prompt and return parsed JSON (dict)."""
        return self.act_default(inputs, ctx)

    def to_model_input(self, raw: Any, inputs: Sequence[BaseModel], ctx: AgentContext) -> Any:
        """Hook to enrich a raw LLM dict (e.g. add ids) before validation."""
        return raw

    # -- LLM plumbing --------------------------------------------------------
    def ask(self, prompt: str) -> Any:
        assert self.llm is not None
        resp = self.llm.complete(prompt, system=self.system_prompt, json_mode=True)
        self._llm_tokens += resp.input_tokens + resp.output_tokens
        try:
            return resp.json()
        except ValueError:
            return resp.text  # unparseable: let validation reject it and trigger repair

    def repair(self, raw: Any, error: str, inputs: Sequence[BaseModel], ctx: AgentContext) -> Any | None:
        """Ask the LLM to fix its own output given the validation errors."""
        if self.llm is None:
            return None
        schema = json.dumps(self.output_type.model_json_schema())[:1500]
        prompt = (
            "Your previous output failed validation and must be repaired.\n"
            f"VALIDATION ERRORS:\n{error}\n\nPREVIOUS OUTPUT:\n{str(raw)[:2000]}\n\n"
            f"Return only corrected JSON for the {self.output_type.__name__} schema:\n{schema}"
        )
        return self.ask(prompt)

    # -- hand-off boundary ---------------------------------------------------
    def _validate(self, raw: Any, inputs: Sequence[BaseModel], ctx: AgentContext) -> BaseModel:
        if isinstance(raw, BaseModel):
            raw = raw.model_dump()
        if isinstance(raw, str):
            raise ValueError(f"expected a JSON object, got text: {raw[:120]!r}")
        if not isinstance(raw, dict):
            raise ValueError(f"expected a JSON object, got {type(raw).__name__}")
        return self.output_type.model_validate(self.to_model_input(raw, inputs, ctx))

    def handle(self, inputs: Sequence[BaseModel], ctx: AgentContext) -> BaseModel:
        """Validated hand-off: check inputs, act, validate, repair, fall back."""
        for m in inputs:
            if not self.accepts(type(m)):
                raise HandoffError(f"{self.name} does not accept {type(m).__name__}")
        self._llm_tokens = 0
        use_llm = self.llm is not None
        raw = self.act_llm(inputs, ctx) if use_llm else self.act_default(inputs, ctx)
        last_error = ""
        for attempt in range(self.max_repairs + 1):
            try:
                out = self._validate(raw, inputs, ctx)
                self._charge(inputs, out, ctx)
                return out
            except (ValidationError, ValueError) as e:
                last_error = str(e)
                ctx.state.repairs += 1
                ctx.emit("validation_failed", self.name, f"attempt {attempt + 1}: {last_error.splitlines()[0][:160]}")
                if attempt == self.max_repairs:
                    break
                fixed = self.repair(raw, last_error, inputs, ctx)
                if fixed is None:
                    break
                ctx.emit("repair", self.name, f"repair attempt {attempt + 1}")
                raw = fixed
        if use_llm:
            ctx.emit("fallback", self.name, "LLM output unrecoverable; using algorithmic default")
            try:
                out = self._validate(self.act_default(inputs, ctx), inputs, ctx)
                self._charge(inputs, out, ctx)
                return out
            except (ValidationError, ValueError) as e:
                last_error = str(e)
        raise HandoffError(f"{self.name} produced invalid {self.output_type.__name__}: {last_error[:300]}")

    def _charge(self, inputs: Sequence[BaseModel], out: BaseModel, ctx: AgentContext) -> None:
        """Token accounting: real LLM usage, or the context an LLM agent would have consumed."""
        if self._llm_tokens:
            tokens = self._llm_tokens
        else:
            tokens = sum(count_tokens(m.model_dump_json()) for m in inputs) + count_tokens(out.model_dump_json())
        ctx.state.tokens += tokens
        self.last_tokens = tokens


# ---------------------------------------------------------------------------
# Researcher
# ---------------------------------------------------------------------------
_EDGE_RE = re.compile(r"^EDGE:\s*(\w+)\((.*)\)\s*(->|raises)\s*(.+?)\s*$")


def parse_edge_line(line: str) -> tuple[str, list[Any], Any, str | None] | None:
    """Parse ``EDGE: f(args) -> expected`` / ``EDGE: f(args) raises Exc``."""
    m = _EDGE_RE.match(line.strip())
    if not m:
        return None
    fn, argstr, arrow, rhs = m.groups()
    try:
        args = list(literal_eval(f"[{argstr}]"))
    except (ValueError, SyntaxError):
        return None
    if arrow == "raises":
        return (fn, args, None, rhs) if rhs.isidentifier() else None
    try:
        return fn, args, literal_eval(rhs), None
    except (ValueError, SyntaxError):
        return None


class Researcher(Agent):
    """Retrieves knowledge snippets by embedding similarity and extracts requirements/edge cases."""

    role = Role.RESEARCHER
    input_types = (TaskSpec,)
    output_types = (ResearchFindings,)
    system_prompt = ("You are the Researcher. Identify precise requirements and tricky edge cases "
                     "for a Python function. Respond with JSON only.")

    def __init__(self, name: str | None = None, llm: LLM | None = None, knowledge: list[Snippet] | None = None,
                 top_k: int = 4, min_sim: float = 0.12, max_repairs: int = 2):
        super().__init__(name, llm, max_repairs)
        self.knowledge = knowledge if knowledge is not None else KNOWLEDGE
        self.top_k = top_k
        self.min_sim = min_sim
        self.embedder = HashingEmbedder()
        self._kb_vecs = self.embedder.embed([f"{s.title}\n{s.text}" for s in self.knowledge])

    def retrieve(self, task: TaskSpec) -> list[tuple[Snippet, float]]:
        """Top-k snippets by cosine similarity to the task description + function name."""
        if not self.knowledge:
            return []
        q = f"{task.description} {task.func_name.replace('_', ' ')} {task.name.replace('_', ' ')}"
        sims = self._kb_vecs @ self.embedder.embed_one(q)
        order = sorted(range(len(self.knowledge)), key=lambda i: -float(sims[i]))[: self.top_k]
        return [(self.knowledge[i], float(sims[i])) for i in order if sims[i] >= self.min_sim]

    def act_default(self, inputs: Sequence[BaseModel], ctx: AgentContext) -> ResearchFindings:
        task = inputs[0]
        assert isinstance(task, TaskSpec)
        known_args = {json.dumps(t.args, default=repr) for t in task.examples}
        requirements = [task.description]
        edges: list[TestCase] = []
        sources: list[str] = []
        for snip, _sim in self.retrieve(task):
            lines = snip.text.splitlines()
            parsed = [p for p in (parse_edge_line(ln) for ln in lines if ln.startswith("EDGE:")) if p]
            mine = [p for p in parsed if p[0] == task.func_name]
            # A snippet is on-topic only if it carries edge cases for *this* function.
            if not mine:
                continue
            sources.append(snip.id)
            requirements += [ln[4:].strip() for ln in lines if ln.startswith("REQ:")]
            for _fn, args, expected, raises in mine:
                key = json.dumps(args, default=repr)
                if key in known_args:
                    continue
                known_args.add(key)
                edges.append(TestCase(id=f"r{len(edges)}", args=args, expected=expected, raises=raises,
                                      source="research", note=snip.id))
        words = [w for r in requirements for w in tokenize_words(r) if len(w) > 3]
        keywords = sorted(set(words), key=words.index)[:12]
        return ResearchFindings(task=task, requirements=requirements, edge_cases=edges, sources=sources,
                                keywords=keywords)

    def act_llm(self, inputs: Sequence[BaseModel], ctx: AgentContext) -> Any:
        task = inputs[0]
        assert isinstance(task, TaskSpec)
        notes = "\n\n".join(f"[{s.id}] {s.text}" for s, _ in self.retrieve(task))
        prompt = (
            f"TASK: implement {task.func_name}({', '.join(task.params)}): {task.description}\n"
            f"EXAMPLES: {[t.describe(task.func_name) for t in task.examples]}\n\nKNOWLEDGE:\n{notes}\n\n"
            'Return JSON {"requirements": [str], "edge_cases": [{"args": [...], "expected": <value>, '
            '"raises": null | "ExceptionName"}], "sources": [snippet ids]}'
        )
        return self.ask(prompt)

    def to_model_input(self, raw: Any, inputs: Sequence[BaseModel], ctx: AgentContext) -> Any:
        if isinstance(raw, dict) and "task" not in raw:
            task = inputs[0]
            raw = dict(raw)
            raw["task"] = task.model_dump()
            raw["edge_cases"] = [
                {"id": f"r{i}", "source": "research", **{k: v for k, v in e.items() if k in ("args", "expected", "raises")}}
                if isinstance(e, dict) else e
                for i, e in enumerate(raw.get("edge_cases", []))
            ]
        return raw


# ---------------------------------------------------------------------------
# Coder
# ---------------------------------------------------------------------------
class Coder(Agent):
    """CEGIS coder: best-ranked candidate consistent with all examples and counterexamples."""

    role = Role.CODER
    input_types = (TaskSpec, ResearchFindings, RevisionRequest)
    output_types = (CodeArtifact,)
    system_prompt = ("You are the Coder. Write a correct, self-contained Python function. "
                     "Respond with JSON only.")

    def __init__(self, name: str | None = None, llm: LLM | None = None, use_mutants: bool = True,
                 chunk: int = 8, search_timeout: float = 0.1, max_repairs: int = 2):
        super().__init__(name, llm, max_repairs)
        self.use_mutants = use_mutants
        self.chunk = chunk
        self.search_timeout = search_timeout

    @staticmethod
    def _unpack(inputs: Sequence[BaseModel]) -> tuple[TaskSpec, ResearchFindings | None, RevisionRequest | None]:
        task = findings = rr = None
        for m in inputs:
            if isinstance(m, TaskSpec):
                task = m
            elif isinstance(m, ResearchFindings):
                findings, task = m, m.task
            elif isinstance(m, RevisionRequest):
                rr = m
        if task is None:
            raise HandoffError("coder received no TaskSpec or ResearchFindings")
        return task, findings, rr

    def act_default(self, inputs: Sequence[BaseModel], ctx: AgentContext) -> CodeArtifact:
        task, findings, rr = self._unpack(inputs)
        requirements = findings.requirements if findings else [task.description]
        constraints = list(task.examples) + (list(rr.counterexamples) if rr else [])
        tried = set(rr.rejected_hashes) if rr else set()
        ranked = [c for c in rank(build_pool(task.func_name, with_mutants=self.use_mutants), requirements)
                  if c.hash not in tried]
        if not ranked:
            raise SynthesisExhausted(f"no untried candidates for {task.func_name}")
        revision = rr.revision if rr else 0
        evaluated = 0
        for start in range(0, len(ranked), self.chunk):
            batch = ranked[start : start + self.chunk]
            reports = ctx.sandbox.run_many([c.source for c in batch], task.func_name, constraints,
                                           stop_on_fail=True, arity=len(task.params),
                                           timeout=self.search_timeout)
            evaluated += len(batch)
            for c, rep in zip(batch, reports):
                if rep.compile_error is None and len(rep.results) == len(constraints) and rep.passed:
                    return CodeArtifact(
                        task_name=task.name, func_name=task.func_name, source=c.source, revision=revision,
                        origin=c.origin,
                        rationale=(f"best-ranked candidate (score {c.score:.2f}) consistent with "
                                   f"{len(constraints)} examples/counterexamples; searched {evaluated}"),
                    )
        c = ranked[0]
        return CodeArtifact(task_name=task.name, func_name=task.func_name, source=c.source, revision=revision,
                            origin=c.origin, rationale="no candidate satisfies all constraints; best-ranked guess")

    def act_llm(self, inputs: Sequence[BaseModel], ctx: AgentContext) -> Any:
        task, findings, rr = self._unpack(inputs)
        parts = [f"Implement {task.func_name}({', '.join(task.params)}): {task.description}",
                 "Examples:\n" + "\n".join(t.describe(task.func_name) for t in task.examples)]
        if findings:
            parts.append("Requirements:\n- " + "\n- ".join(findings.requirements))
            parts.append("Edge cases:\n" + "\n".join(t.describe(task.func_name) for t in findings.edge_cases))
        if rr:
            parts.append("Your previous attempt was REJECTED. Failing cases:\n"
                         + "\n".join(t.describe(task.func_name) for t in rr.counterexamples))
            parts.append("Reviewer feedback:\n" + "\n".join(rr.feedback))
        parts.append('Return JSON {"source": "<complete python code defining the function>", "rationale": str}. '
                     "Do not import os, sys or subprocess.")
        return self.ask("\n\n".join(parts))

    def to_model_input(self, raw: Any, inputs: Sequence[BaseModel], ctx: AgentContext) -> Any:
        if isinstance(raw, dict) and "func_name" not in raw:
            task, _f, rr = self._unpack(inputs)
            raw = {"task_name": task.name, "func_name": task.func_name, "source": raw.get("source", ""),
                   "revision": rr.revision if rr else 0, "origin": f"llm:{getattr(self.llm, 'model', '?')}",
                   "rationale": str(raw.get("rationale", ""))}
        return raw


# ---------------------------------------------------------------------------
# Reviewers
# ---------------------------------------------------------------------------
def review_suite(ctx: AgentContext) -> list[TestCase]:
    """The trusted, shared test suite: spec examples + researched edge cases."""
    if ctx.state.findings is not None:
        return ctx.state.findings.test_suite()
    return list(ctx.state.task.examples)


def _failure_feedback(task: TaskSpec, suite: list[TestCase], rep: RunReport) -> tuple[list[str], list[TestCase]]:
    by_id = {t.id: t for t in suite}
    fb, cex = [], []
    for r in rep.results:
        if not r.passed and r.id in by_id:
            t = by_id[r.id]
            fb.append(f"FAIL {t.describe(task.func_name)}: {r.error or 'got ' + str(r.got)}")
            cex.append(t)
    return fb, cex


class Reviewer(Agent):
    """Base reviewer: CodeArtifact -> ReviewVerdict."""

    role = Role.REVIEWER
    input_types = (CodeArtifact,)
    output_types = (ReviewVerdict,)
    strategy: ClassVar[str] = "base"

    def _static_reject(self, art: CodeArtifact, task: TaskSpec) -> ReviewVerdict | None:
        st = static_check(art.source, task.func_name, len(task.params))
        if st.ok:
            return None
        return ReviewVerdict(
            reviewer=self.name, strategy=self.strategy, artifact_hash=art.hash, approved=False, confidence=0.9,
            evidence=[Evidence(kind=EvidenceKind.STATIC_ANALYSIS, detail="; ".join(st.errors), artifact_hash=art.hash)],
            feedback=[f"static: {e}" for e in st.errors],
        )


class ExecutionReviewer(Reviewer):
    """Executes the full trusted suite in the sandbox (plus static checks)."""

    strategy = "execution"

    def act_default(self, inputs: Sequence[BaseModel], ctx: AgentContext) -> ReviewVerdict:
        art = inputs[0]
        assert isinstance(art, CodeArtifact)
        task = ctx.state.task
        rejected = self._static_reject(art, task)
        if rejected:
            return rejected
        suite = self.select_tests(review_suite(ctx), art)
        rep = ctx.sandbox.run(art.source, task.func_name, suite, arity=len(task.params))
        if rep.compile_error:
            return ReviewVerdict(
                reviewer=self.name, strategy=self.strategy, artifact_hash=art.hash, approved=False, confidence=0.95,
                evidence=[Evidence(kind=EvidenceKind.TEST_EXECUTION, detail=rep.compile_error, tests_run=len(suite),
                                   failed_test_ids=[t.id for t in suite], artifact_hash=art.hash)],
                feedback=[f"code failed to load: {rep.compile_error}"], counterexamples=suite[:3],
            )
        fb, cex = _failure_feedback(task, suite, rep)
        return ReviewVerdict(
            reviewer=self.name, strategy=self.strategy, artifact_hash=art.hash, approved=not rep.failed_ids,
            confidence=self.confidence(len(suite), len(review_suite(ctx))),
            evidence=[Evidence(kind=EvidenceKind.TEST_EXECUTION, detail=f"{rep.n_passed}/{len(suite)} tests passed",
                               tests_run=len(suite), failed_test_ids=rep.failed_ids, artifact_hash=art.hash)],
            feedback=fb, counterexamples=cex,
        )

    def select_tests(self, suite: list[TestCase], art: CodeArtifact) -> list[TestCase]:
        return suite

    def confidence(self, n_run: int, n_total: int) -> float:
        return 0.95


class SamplingReviewer(ExecutionReviewer):
    """Executes a seeded random subset of the suite: cheaper, but can miss bugs."""

    strategy = "sampling"

    def __init__(self, name: str | None = None, fraction: float = 0.5, seed: int = 0, **kw: Any):
        super().__init__(name, **kw)
        self.fraction = fraction
        self.seed = seed

    def select_tests(self, suite: list[TestCase], art: CodeArtifact) -> list[TestCase]:
        rng = random.Random(f"{self.seed}:{art.hash}")
        k = max(1, round(len(suite) * self.fraction))
        return rng.sample(suite, min(k, len(suite)))

    def confidence(self, n_run: int, n_total: int) -> float:
        return round(0.5 + 0.45 * n_run / max(1, n_total), 3)


class StaticReviewer(Reviewer):
    """AST-only review: safety, signature and code-smell checks. Never executes code."""

    strategy = "static"

    def act_default(self, inputs: Sequence[BaseModel], ctx: AgentContext) -> ReviewVerdict:
        art = inputs[0]
        assert isinstance(art, CodeArtifact)
        task = ctx.state.task
        rejected = self._static_reject(art, task)
        if rejected:
            return rejected
        st = static_check(art.source, task.func_name, len(task.params))
        return ReviewVerdict(
            reviewer=self.name, strategy=self.strategy, artifact_hash=art.hash, approved=True,
            confidence=0.4 if st.warnings else 0.6,
            evidence=[Evidence(kind=EvidenceKind.STATIC_ANALYSIS, artifact_hash=art.hash,
                               detail="static checks passed" + (f"; warnings: {st.warnings}" if st.warnings else ""))],
            feedback=[f"warning: {w}" for w in st.warnings],
        )


class LLMReviewer(Reviewer):
    """Opinion review by an LLM (opinion-tier evidence). Offline default = static heuristics as opinion."""

    strategy = "llm_opinion"
    system_prompt = "You are a meticulous code reviewer. Respond with JSON only."

    def act_default(self, inputs: Sequence[BaseModel], ctx: AgentContext) -> ReviewVerdict:
        art = inputs[0]
        assert isinstance(art, CodeArtifact)
        st = static_check(art.source, ctx.state.task.func_name, len(ctx.state.task.params))
        return ReviewVerdict(
            reviewer=self.name, strategy=self.strategy, artifact_hash=art.hash, approved=st.ok, confidence=0.5,
            evidence=[Evidence(kind=EvidenceKind.OPINION, detail="heuristic opinion (no LLM)", artifact_hash=art.hash)],
            feedback=st.errors + st.warnings,
        )

    def act_llm(self, inputs: Sequence[BaseModel], ctx: AgentContext) -> Any:
        art = inputs[0]
        assert isinstance(art, CodeArtifact)
        task = ctx.state.task
        return self.ask(
            f"Review this implementation of {task.func_name}: {task.description}\n```python\n{art.source}```\n"
            'Return JSON {"approved": bool, "confidence": 0..1, "feedback": [str]}'
        )

    def to_model_input(self, raw: Any, inputs: Sequence[BaseModel], ctx: AgentContext) -> Any:
        if isinstance(raw, dict) and "reviewer" not in raw:
            art = inputs[0]
            raw = {"reviewer": self.name, "strategy": self.strategy, "artifact_hash": art.hash,
                   "approved": raw.get("approved"), "confidence": raw.get("confidence", 0.5),
                   "feedback": raw.get("feedback", []),
                   "evidence": [{"kind": int(EvidenceKind.OPINION), "detail": "LLM opinion", "artifact_hash": art.hash}]}
        return raw


class FaultyReviewer(ExecutionReviewer):
    """Fault injection for resilience tests.

    * ``adversarial``: runs the real tests, then reports the *opposite* verdict
      with fabricated, schema-valid test evidence and high confidence.
    * ``always_approve``: rubber-stamps everything, claiming all tests passed.
    * ``random``: coin-flip verdicts (seeded) with fabricated evidence.
    """

    strategy = "faulty"

    def __init__(self, name: str | None = None, mode: str = "adversarial", seed: int = 0, **kw: Any):
        super().__init__(name, **kw)
        if mode not in ("adversarial", "always_approve", "random"):
            raise ValueError(f"unknown fault mode {mode!r}")
        self.mode = mode
        self.rng = random.Random(seed)

    def act_default(self, inputs: Sequence[BaseModel], ctx: AgentContext) -> ReviewVerdict:
        honest = super().act_default(inputs, ctx)
        art = inputs[0]
        assert isinstance(art, CodeArtifact)
        suite = review_suite(ctx)
        if self.mode == "adversarial":
            approve = not honest.approved
        elif self.mode == "always_approve":
            approve = True
        else:
            approve = self.rng.random() < 0.5
        fake_failed = [] if approve else [t.id for t in suite[:2]]
        return ReviewVerdict(
            reviewer=self.name, strategy=f"faulty:{self.mode}", artifact_hash=art.hash, approved=approve,
            confidence=0.97,
            evidence=[Evidence(kind=EvidenceKind.TEST_EXECUTION, detail=f"{len(suite) - len(fake_failed)}/{len(suite)} passed",
                               tests_run=len(suite), failed_test_ids=fake_failed, artifact_hash=art.hash)],
            feedback=[] if approve else [f"FAIL {t.describe(ctx.state.task.func_name)}" for t in suite[:2]],
            counterexamples=[] if approve else suite[:2],
        )
