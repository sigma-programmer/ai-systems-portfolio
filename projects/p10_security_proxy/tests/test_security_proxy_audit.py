"""Tamper-evident hash-chained audit log."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from security_proxy import AuditLog


def clock():
    t = [1000.0]

    def now() -> float:
        t[0] += 1
        return t[0]

    return now


def filled(path: Path | None = None, key: bytes | None = None, n: int = 5) -> AuditLog:
    log = AuditLog(path, key=key, clock=clock())
    for i in range(n):
        log.append("evt", {"i": i, "msg": f"m{i}", "ü": "ñ"})
    return log


def test_chain_links_and_verifies() -> None:
    log = filled()
    recs = list(log.records())
    assert [r["seq"] for r in recs] == list(range(5))
    assert recs[0]["prev"] == "0" * 64
    for a, b in zip(recs, recs[1:]):
        assert b["prev"] == a["hash"]
    assert log.verify().ok and log.verify().records == 5
    assert log.verify(expected_head=log.head).ok


def test_modification_detected() -> None:
    log = filled()
    log._tamper(2, lambda r: r["data"].update(msg="edited"))
    res = log.verify()
    assert not res.ok and res.first_bad_seq == 2 and "hash" in res.reason


def test_recomputed_hash_breaks_next_link() -> None:
    log = filled()
    import hashlib

    from security_proxy.audit import _canonical

    def forge(r: dict) -> None:
        r["data"]["msg"] = "forged"
        body = {k: v for k, v in r.items() if k != "hash"}
        r["hash"] = hashlib.sha256((r["prev"] + _canonical(body)).encode()).hexdigest()

    log._tamper(1, forge)
    res = log.verify()
    assert not res.ok and res.first_bad_seq == 2


def test_deletion_reorder_and_garbage_detected(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    filled(p)
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines[:1] + lines[2:]) + "\n")
    log = AuditLog()  # in-memory verifier pointed at the edited file
    log.path = p
    assert not log.verify().ok
    p.write_text("\n".join([lines[1], lines[0]] + lines[2:]) + "\n")
    assert not log.verify().ok
    p.write_text("\n".join(lines + ["not json"]) + "\n")
    assert "not valid JSON" in log.verify().reason
    p.write_text("\n".join(lines + ['{"a": 1}']) + "\n")
    assert not log.verify().ok


def test_truncation_needs_anchor(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    log = filled(p)
    head = log.head
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines[:3]) + "\n")
    assert log.verify().ok  # truncation alone is undetectable
    assert not log.verify(expected_head=head).ok


def test_file_persistence_and_resume(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    filled(p, n=3)
    log2 = AuditLog(p)
    assert log2.seq == 3
    log2.append("more", {})
    assert log2.verify().ok and len(log2) == 4
    for line in p.read_text().splitlines():
        json.loads(line)


def test_refuses_to_resume_corrupted(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    log = filled(p)
    log._tamper(0, lambda r: r.update(event="x"))
    with pytest.raises(ValueError, match="corrupted"):
        AuditLog(p)


def test_hmac_key_required_to_verify() -> None:
    log = filled(key=b"secret")
    assert log.verify().ok
    other = AuditLog(key=b"wrong")
    other._memory = log._lines()
    assert not other.verify().ok


def test_empty_log_verifies() -> None:
    assert AuditLog().verify().ok
