"""Consensus rules, ledger, conflict resolution and escalation."""
from __future__ import annotations

import pytest

from agent_swarm import ArbiterResult, ConsensusPanel, Evidence, EvidenceKind, HandoffError, ReviewerLedger, \
    ReviewVerdict, Rule, TestCase


def V(name, approved, kind=EvidenceKind.TEST_EXECUTION, h="h", failed=None):
    if kind == EvidenceKind.TEST_EXECUTION:
        failed = failed if failed is not None else ([] if approved else ["ex0"])
        ev = Evidence(kind=kind, tests_run=3, failed_test_ids=failed)
    else:
        ev = Evidence(kind=kind)
    cex = [] if approved else [TestCase(id="ex0", args=[1], expected=1)]
    return ReviewVerdict(reviewer=name, strategy="t", artifact_hash=h, approved=approved, confidence=0.9,
                         evidence=[ev], counterexamples=cex, feedback=[] if approved else [f"{name} says no"])


OP, ST, TE = EvidenceKind.OPINION, EvidenceKind.STATIC_ANALYSIS, EvidenceKind.TEST_EXECUTION


def test_unanimous():
    d = ConsensusPanel().decide([V("a", True), V("b", True, OP)])
    assert d.approved and d.resolution == "unanimous" and d.approve_share == 1.0


def test_majority_vote_stands_when_winner_has_stronger_evidence():
    d = ConsensusPanel(rule="majority").decide([V("a", True), V("b", True), V("c", False, OP)])
    assert d.approved and d.resolution == "vote"


def test_test_evidence_beats_opinion_majority():
    d = ConsensusPanel(rule="majority").decide([V("a", False, TE), V("b", True, OP), V("c", True, ST)])
    assert not d.approved and d.resolution == "evidence"
    assert [t.id for t in d.counterexamples] == ["ex0"]
    assert d.feedback == ["a says no"]


def test_conflicting_test_evidence_escalates():
    calls = []

    def arbiter(vs):
        calls.append(len(vs))
        return ArbiterResult(approved=False, failing=[TestCase(id="r1", args=[], expected=0)], contradicted=["b"])

    d = ConsensusPanel(rule="majority").decide([V("a", False), V("b", True), V("c", True, ST)], arbiter)
    assert calls == [3] and not d.approved and d.resolution == "escalated"
    assert [t.id for t in d.counterexamples] == ["r1"] and "b" in d.rationale


def test_escalation_without_arbiter_is_conservative():
    d = ConsensusPanel().decide([V("a", False), V("b", True)])
    assert not d.approved and d.resolution == "escalated"


def test_tie_broken_by_evidence():
    d = ConsensusPanel().decide([V("a", True, TE), V("b", False, OP)])
    assert d.approved and d.resolution == "evidence"


def test_no_resolution_mode_is_pure_vote():
    p = ConsensusPanel(rule="majority", resolve_conflicts=False)
    assert p.decide([V("a", False, TE), V("b", True, OP), V("c", True, OP)]).approved
    assert not p.decide([V("a", True), V("b", False)]).approved  # tie -> conservative
    assert p.label == "majority/no-resolution"


def test_quorum_rule():
    p = ConsensusPanel(rule="quorum", quorum=0.75, resolve_conflicts=False)
    assert not p.decide([V("a", True), V("b", True), V("c", False)]).approved
    assert p.decide([V("a", True), V("b", True), V("c", True), V("d", False)]).approved
    with pytest.raises(ValueError):
        ConsensusPanel(rule="quorum", quorum=0.3)


def test_weighted_rule_uses_ledger_and_trust_floor():
    ledger = ReviewerLedger()
    for _ in range(10):
        ledger.record("good", True)
        ledger.record("liar", False)
    p = ConsensusPanel(rule=Rule.WEIGHTED, ledger=ledger)
    d = p.decide([V("good", False), V("liar", True), V("liar2", True, ST)])
    votes = {v.reviewer: v for v in d.votes}
    assert votes["liar"].trusted is False and votes["good"].weight > 0.9
    # liar's test evidence is untrusted, so 'good' (TE) beats liar2 (static): no escalation needed
    assert not d.approved and d.resolution in ("evidence", "vote")


def test_ledger_updates_after_decision():
    ledger = ReviewerLedger()
    ConsensusPanel(ledger=ledger).decide([V("a", True), V("b", True), V("c", False, OP)])
    assert ledger.history("a") == 1 and ledger.accuracy("a") == pytest.approx(2 / 3)
    assert ledger.accuracy("c") == pytest.approx(1 / 3)
    assert ledger.snapshot()["c"]["decisions"] == 1


def test_panel_input_guards():
    with pytest.raises(HandoffError):
        ConsensusPanel().decide([])
    with pytest.raises(HandoffError, match="different artifacts"):
        ConsensusPanel().decide([V("a", True, h="x"), V("b", True, h="y")])


def test_majority_with_adversary_still_correct_via_escalation(sandbox):
    # Honest exec rejects with real failing test; adversary + static approve. Majority says approve,
    # but conflicting test evidence is escalated and the arbiter's re-execution decides.
    arb = lambda vs: ArbiterResult(approved=False, failing=[TestCase(id="ex0", args=[1], expected=1)])
    d = ConsensusPanel(rule="majority").decide([V("exec", False), V("rogue", True), V("static", True, ST)], arb)
    assert not d.approved and d.resolution == "escalated"
