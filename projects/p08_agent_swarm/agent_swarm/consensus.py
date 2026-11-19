"""Consensus voting and conflict resolution over reviewer verdicts.

Rules
-----
* ``majority``  — one reviewer, one vote; approve iff approve share > 0.5.
* ``weighted``  — each vote weighted by the reviewer's historical accuracy
  (Laplace-smoothed, tracked in a persistent `ReviewerLedger`); reviewers whose
  accuracy falls below ``trust_floor`` (after ``min_history`` decisions) are
  *untrusted*: they still vote, but their evidence cannot trigger escalation.
* ``quorum``    — approve iff approve share >= ``quorum`` (default 2/3).

Conflict resolution (when reviewers disagree)
---------------------------------------------
1. Unanimous → done.
2. Compare the strongest *trusted* evidence tier on each side
   (test execution > static analysis > opinion).
   * loser side stronger  → the evidence wins over the head count ("evidence").
   * winner side stronger → the vote stands ("vote").
   * both sides hold test-execution evidence that contradicts → the claim is
     verifiable, so escalate: the supervisor re-executes the trusted suite
     ("escalated") and its result is final.
   * an exact tie with equal evidence → escalate.
3. After every decision the ledger records which reviewers agreed with it.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Sequence

from pydantic import BaseModel

from .agents import Agent, AgentContext, ArbiterResult, HandoffError
from .messages import (
    ConsensusDecision,
    EvidenceKind,
    ReviewVerdict,
    RevisionRequest,
    Role,
    SwarmState,
    TestCase,
    Vote,
)


class Rule(str, Enum):
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    QUORUM = "quorum"


class ReviewerLedger:
    """Historical accuracy per reviewer (Beta(1,1)-smoothed)."""

    def __init__(self) -> None:
        self.correct: dict[str, int] = {}
        self.total: dict[str, int] = {}

    def accuracy(self, name: str) -> float:
        return (self.correct.get(name, 0) + 1) / (self.total.get(name, 0) + 2)

    def history(self, name: str) -> int:
        return self.total.get(name, 0)

    def record(self, name: str, correct: bool) -> None:
        self.total[name] = self.total.get(name, 0) + 1
        self.correct[name] = self.correct.get(name, 0) + int(correct)

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {n: {"accuracy": round(self.accuracy(n), 3), "decisions": self.total[n]} for n in sorted(self.total)}


class ConsensusPanel(Agent):
    """Aggregates N ReviewVerdicts into one ConsensusDecision (and revision requests)."""

    role = Role.CONSENSUS
    input_types = (ReviewVerdict,)
    output_types = (ConsensusDecision, RevisionRequest)

    def __init__(self, name: str = "panel", rule: Rule | str = Rule.MAJORITY, quorum: float = 2 / 3,
                 ledger: ReviewerLedger | None = None, resolve_conflicts: bool = True,
                 trust_floor: float = 0.5, min_history: int = 3):
        super().__init__(name)
        self.rule = Rule(rule)
        if not 0.5 <= quorum <= 1.0:
            raise ValueError("quorum must be within [0.5, 1.0]")
        self.quorum = quorum
        self.ledger = ledger if ledger is not None else ReviewerLedger()
        self.resolve_conflicts = resolve_conflicts
        self.trust_floor = trust_floor
        self.min_history = min_history

    @property
    def label(self) -> str:
        base = self.rule.value + (f"({self.quorum:.2f})" if self.rule == Rule.QUORUM else "")
        return base if self.resolve_conflicts else base + "/no-resolution"

    def _vote(self, v: ReviewVerdict) -> Vote:
        acc = self.ledger.accuracy(v.reviewer)
        weight = acc if self.rule == Rule.WEIGHTED else 1.0
        trusted = not (self.rule == Rule.WEIGHTED and self.ledger.history(v.reviewer) >= self.min_history
                       and acc < self.trust_floor)
        return Vote(reviewer=v.reviewer, approved=v.approved, weight=round(weight, 4), strength=v.strength,
                    trusted=trusted)

    def decide(self, verdicts: Sequence[ReviewVerdict], arbiter: Any = None) -> ConsensusDecision:
        """Pure decision procedure (arbiter: callable(verdicts) -> ArbiterResult, or None)."""
        if not verdicts:
            raise HandoffError("consensus panel received no verdicts")
        hashes = {v.artifact_hash for v in verdicts}
        if len(hashes) != 1:
            raise HandoffError(f"verdicts reference different artifacts: {sorted(hashes)}")
        votes = [self._vote(v) for v in verdicts]
        total = sum(v.weight for v in votes)
        share = sum(v.weight for v in votes if v.approved) / total if total > 0 else 0.5
        if self.rule == Rule.QUORUM:
            outcome, tie = share >= self.quorum - 1e-9, False
        else:
            outcome, tie = share > 0.5, abs(share - 0.5) < 1e-9

        def tier(side: bool) -> int:
            return max((int(v.strength) for v in votes if v.approved == side and v.trusted), default=0)

        resolution = "vote"
        rationale = f"{self.label}: approve share {share:.2f}"
        escalate = False
        if all(v.approved == votes[0].approved for v in votes):
            outcome, resolution, rationale = votes[0].approved, "unanimous", f"all {len(votes)} reviewers agree"
        elif self.resolve_conflicts:
            win, lose = tier(outcome), tier(not outcome)
            if tie:
                if tier(True) != tier(False):
                    outcome = tier(True) > tier(False)
                    resolution, rationale = "evidence", "tied vote broken by stronger evidence"
                else:
                    escalate = True
            elif lose > win:
                outcome = not outcome
                resolution = "evidence"
                rationale += (f"; overridden: {EvidenceKind(lose).name} evidence beats "
                              f"{EvidenceKind(win).name if win else 'no trusted'} evidence")
            elif lose == win == int(EvidenceKind.TEST_EXECUTION):
                escalate = True
        else:
            outcome = outcome and not tie

        failing: list[TestCase] = []
        if escalate:
            resolution = "escalated"
            if arbiter is None:
                outcome, rationale = False, "conflicting test evidence and no arbiter: conservative reject"
            else:
                res: ArbiterResult = arbiter(list(verdicts))
                outcome, failing = res.approved, res.failing
                rationale = (f"conflicting test evidence escalated to supervisor; re-execution "
                             f"{'passed' if res.approved else f'failed {len(res.failing)} tests'}")
                if res.contradicted:
                    rationale += f"; contradicted claims from {', '.join(res.contradicted)}"

        if not outcome and not failing:
            seen: set[str] = set()
            for v, vote in zip(verdicts, votes):
                if v.approved or not vote.trusted:
                    continue
                for t in v.counterexamples:
                    if t.id not in seen:
                        seen.add(t.id)
                        failing.append(t)
        feedback = [] if outcome else sorted({f for v in verdicts if not v.approved for f in v.feedback})
        for vote in votes:
            self.ledger.record(vote.reviewer, vote.approved == outcome)
        return ConsensusDecision(
            artifact_hash=verdicts[0].artifact_hash, approved=outcome, rule=self.label, resolution=resolution,
            approve_share=round(min(1.0, max(0.0, share)), 4), votes=votes, counterexamples=failing,
            feedback=feedback[:10], rationale=rationale,
        )

    def act_default(self, inputs: Sequence[BaseModel], ctx: AgentContext) -> ConsensusDecision:
        verdicts = [m for m in inputs if isinstance(m, ReviewVerdict)]
        return self.decide(verdicts, ctx.arbiter)

    @staticmethod
    def to_revision(decision: ConsensusDecision, state: SwarmState) -> RevisionRequest:
        """Turn a rejection into the typed message for the coder feedback edge."""
        return RevisionRequest(
            task_name=state.task.name, revision=state.revisions,
            counterexamples=list(state.counterexamples), feedback=decision.feedback,
            rejected_hashes=[a.hash for a in state.artifacts],
        )
