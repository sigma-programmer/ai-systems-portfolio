"""Trajectory-to-DAG alignment.

Maps every graded step of a trajectory onto a golden node (or classifies it
as extra / forbidden / redundant), then checks the partial order over the
transitive closure of the golden DAG.

Matching is greedy in trajectory order with a tiered preference that keeps
the alignment faithful when the same tool appears in several golden nodes:

    tier 0  unmatched node whose required ancestors are all satisfied
    tier 1  repeat of an already-matched node within its ``max_repeats``
    tier 2  unmatched node with unsatisfied ancestors   -> out_of_order
    tier 3  repeat beyond ``max_repeats``               -> redundant

Ties break on argument-match score, then golden declaration order.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .golden import GoldenDAG, GoldenNode
from .matchers import ArgMatcher
from .schema import Step, StepType, Trajectory


class StepStatus(str, Enum):
    MATCHED = "matched"
    REPEAT = "repeat"
    OUT_OF_ORDER = "out_of_order"
    REDUNDANT = "redundant"
    ARG_MISMATCH = "arg_mismatch"
    EXTRA = "extra"
    FORBIDDEN = "forbidden"
    IGNORED = "ignored"


GOOD_STATUSES = {StepStatus.MATCHED, StepStatus.REPEAT}


class StepGrade(BaseModel):
    index: int
    step_type: StepType
    tool: str | None
    status: StepStatus
    node_id: str | None = None
    arg_score: float = 0.0
    in_loop: bool = False
    notes: list[str] = Field(default_factory=list)


class OrderViolation(BaseModel):
    before: str
    after: str
    before_pos: int
    after_pos: int


class Loop(BaseModel):
    start: int  # index into the trajectory's steps
    period: int
    repeats: int
    signatures: list[str]


class TrajectoryMetrics(BaseModel):
    required_recall: float
    precision: float
    f1: float
    order_score: float
    order_violations: int
    forbidden: int
    extra: int
    redundant: int
    loops: int
    answer_correct: bool | None
    steps: int
    tokens: int
    latency_ms: float
    over_budget: bool
    score: float


class AlignmentReport(BaseModel):
    run_id: str
    task_id: str
    step_grades: list[StepGrade]
    missing_required: list[str]
    missing_optional: list[str]
    order_violations: list[OrderViolation]
    loops: list[Loop]
    metrics: TrajectoryMetrics
    passed: bool
    failure_reasons: list[str]


def detect_loops(signatures: list[tuple[int, str]], min_repeats: int = 3, max_period: int = 3) -> list[Loop]:
    """Find consecutive repetitions of a block of 1..max_period signatures.

    ``signatures`` is a list of (step_index, signature). A loop is reported
    when a block repeats at least ``min_repeats`` times back to back.
    Detected regions do not overlap; shorter periods win.
    """
    sigs = [s for _, s in signatures]
    n = len(sigs)
    loops: list[Loop] = []
    i = 0
    while i < n:
        found = False
        for p in range(1, max_period + 1):
            if i + p * min_repeats > n:
                break
            block = sigs[i : i + p]
            if p > 1 and len(set(block)) == 1:
                continue  # covered by period 1
            reps = 1
            while i + (reps + 1) * p <= n and sigs[i + reps * p : i + (reps + 1) * p] == block:
                reps += 1
            if reps >= min_repeats:
                loops.append(Loop(start=signatures[i][0], period=p, repeats=reps, signatures=block))
                i += reps * p
                found = True
                break
        if not found:
            i += 1
    return loops


def _choose(
    cands: list[tuple[GoldenNode, float, list[str]]],
    golden: GoldenDAG,
    counts: dict[str, int],
    order: dict[str, int],
    retry_allowance: dict[str, int],
) -> tuple[GoldenNode, int, float]:
    best: tuple[tuple[int, float, int], GoldenNode, int, float] | None = None
    for node, score, _ in cands:
        c = counts.get(node.id, 0)
        if c == 0:
            ready = all(
                counts.get(a, 0) > 0 or golden.node_map[a].optional for a in golden.ancestors[node.id]
            )
            tier = 0 if ready else 2
        else:
            tier = 1 if c < node.max_repeats + retry_allowance.get(node.id, 0) else 3
        key = (tier, -score, order[node.id])
        if best is None or key < best[0]:
            best = (key, node, tier, score)
    assert best is not None
    return best[1], best[2], best[3]


def align(
    traj: Trajectory,
    golden: GoldenDAG,
    loop_min_repeats: int = 3,
    weights: dict[str, float] | None = None,
) -> AlignmentReport:
    """Align a trajectory against its golden DAG and grade every step."""
    w = {"recall": 0.35, "precision": 0.2, "order": 0.2, "answer": 0.25}
    if weights:
        w.update(weights)
    graded_types = set(golden.graded_step_types) | {n.step_type for n in golden.nodes}
    order = {n.id: i for i, n in enumerate(golden.nodes)}
    counts: dict[str, int] = {}
    retry_allowance: dict[str, int] = {}
    first_pos: dict[str, int] = {}
    grades: list[StepGrade] = []
    last_node: str | None = None

    for i, step in enumerate(traj.steps):
        g = StepGrade(index=i, step_type=step.type, tool=step.tool, status=StepStatus.IGNORED)
        grades.append(g)
        if step.type not in graded_types:
            # A retry after a failed call is legitimate, not redundant.
            if step.type == StepType.OBSERVATION and step.error and last_node is not None:
                retry_allowance[last_node] = retry_allowance.get(last_node, 0) + 1
            continue
        last_node = None
        forb = [n for n in golden.forbidden_nodes if n.match(step)[0]]
        if forb:
            g.status, g.node_id = StepStatus.FORBIDDEN, forb[0].id
            g.notes.append(f"forbidden step pattern {forb[0].id!r}")
            continue
        cands = [(n, *n.match(step)) for n in golden.allowed_nodes]
        ok_cands = [(n, s, notes) for n, ok, s, notes in cands if ok]
        if not ok_cands:
            near = [(n, s, notes) for n, ok, s, notes in cands if n.tool_matches(step)]
            if near:
                n, s, notes = max(near, key=lambda x: x[1])
                g.status, g.node_id, g.arg_score = StepStatus.ARG_MISMATCH, n.id, s
                g.notes.extend(notes)
            else:
                g.status = StepStatus.EXTRA
                g.notes.append("step not in golden trajectory")
            continue
        node, tier, score = _choose(ok_cands, golden, counts, order, retry_allowance)
        last_node = node.id
        g.node_id, g.arg_score = node.id, score
        g.status = {0: StepStatus.MATCHED, 1: StepStatus.REPEAT, 2: StepStatus.OUT_OF_ORDER, 3: StepStatus.REDUNDANT}[tier]
        if tier == 2:
            missing = sorted(
                a for a in golden.ancestors[node.id]
                if counts.get(a, 0) == 0 and not golden.node_map[a].optional
            )
            g.notes.append(f"preconditions not yet satisfied: {missing}")
        if tier == 3:
            g.notes.append(f"{node.id!r} exceeded max_repeats={node.max_repeats}")
        counts[node.id] = counts.get(node.id, 0) + 1
        first_pos.setdefault(node.id, i)

    # Partial-order check over the transitive closure.
    violations = [
        OrderViolation(before=a, after=d, before_pos=first_pos[a], after_pos=first_pos[d])
        for a, d in golden.ordering_pairs()
        if a in first_pos and d in first_pos and first_pos[a] > first_pos[d]
    ]
    comparable = sum(1 for a, d in golden.ordering_pairs() if a in first_pos and d in first_pos)
    order_score = 1.0 - len(violations) / comparable if comparable else 1.0

    tool_sigs = [(i, s.signature()) for i, s in enumerate(traj.steps) if s.type == StepType.TOOL_CALL]
    loops = detect_loops(tool_sigs, min_repeats=loop_min_repeats)
    pos_of = {idx: k for k, (idx, _) in enumerate(tool_sigs)}
    for lp in loops:
        k0 = pos_of[lp.start]
        for idx, _ in tool_sigs[k0 : k0 + lp.period * lp.repeats]:
            grades[idx].in_loop = True

    required = golden.required_ids
    missing_required = [r for r in required if r not in counts]
    missing_optional = [n.id for n in golden.nodes if n.optional and n.id not in counts]
    recall = 1.0 - len(missing_required) / len(required) if required else 1.0
    graded = [g for g in grades if g.status != StepStatus.IGNORED]
    good = sum(1 for g in graded if g.status in GOOD_STATUSES)
    precision = good / len(graded) if graded else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    n_forbidden = sum(1 for g in graded if g.status == StepStatus.FORBIDDEN)
    n_extra = sum(1 for g in graded if g.status in (StepStatus.EXTRA, StepStatus.ARG_MISMATCH))
    n_redundant = sum(1 for g in graded if g.status == StepStatus.REDUNDANT)

    answer_correct: bool | None = None
    if golden.answer is not None:
        m: ArgMatcher = golden.answer
        answer_correct = m.matches({m.key: traj.final_answer})

    over_budget = bool(
        (golden.max_steps is not None and len(traj.steps) > golden.max_steps)
        or (golden.max_tokens is not None and traj.total_tokens > golden.max_tokens)
    )

    parts = {"recall": recall, "precision": precision, "order": order_score}
    if answer_correct is not None:
        parts["answer"] = 1.0 if answer_correct else 0.0
    tw = sum(w[k] for k in parts)
    score = sum(w[k] * v for k, v in parts.items()) / tw
    score *= 0.5 ** n_forbidden
    score *= 0.85 ** len(loops)
    if over_budget:
        score *= 0.9

    reasons: list[str] = []
    if missing_required:
        reasons.append(f"missing required steps: {missing_required}")
    if violations:
        reasons.append(f"{len(violations)} order violation(s)")
    if n_forbidden:
        reasons.append(f"{n_forbidden} forbidden step(s)")
    n_argmis = sum(1 for g in graded if g.status == StepStatus.ARG_MISMATCH)
    if n_argmis:
        reasons.append(f"{n_argmis} step(s) with mismatched arguments")
    if loops:
        reasons.append(f"{len(loops)} loop(s) detected")
    if answer_correct is False:
        reasons.append("final answer incorrect")
    if over_budget:
        reasons.append("step/token budget exceeded")

    metrics = TrajectoryMetrics(
        required_recall=recall, precision=precision, f1=f1, order_score=order_score,
        order_violations=len(violations), forbidden=n_forbidden, extra=n_extra,
        redundant=n_redundant, loops=len(loops), answer_correct=answer_correct,
        steps=len(traj.steps), tokens=traj.total_tokens, latency_ms=traj.total_latency_ms,
        over_budget=over_budget, score=round(score, 6),
    )
    return AlignmentReport(
        run_id=traj.run_id, task_id=traj.task_id, step_grades=grades,
        missing_required=missing_required, missing_optional=missing_optional,
        order_violations=violations, loops=loops, metrics=metrics,
        passed=not reasons, failure_reasons=reasons,
    )
