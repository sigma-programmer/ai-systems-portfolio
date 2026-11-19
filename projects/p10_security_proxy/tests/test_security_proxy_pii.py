"""PII detection, validators and reversible session-scoped masking."""
from __future__ import annotations

import pytest

from security_proxy import PIIDetector, PIIType, PIIVault, iban_valid, luhn_valid
from security_proxy.corpus import PII_HELDOUT, PII_NEGATIVES, PII_SAMPLES
from security_proxy.pii import TOKEN_RE


def types_values(text: str) -> set[tuple[str, str]]:
    return {(m.type, m.value) for m in PIIDetector().detect(text)}


@pytest.mark.parametrize(
    "number,ok",
    [("4111 1111 1111 1111", True), ("4111-1111-1111-1112", False), ("378282246310005", True), ("0", False), ("abc", False)],
)
def test_luhn(number: str, ok: bool) -> None:
    assert luhn_valid(number) is ok


@pytest.mark.parametrize(
    "iban,ok",
    [("GB82 WEST 1234 5698 7654 32", True), ("DE89370400440532013000", True), ("GB82WEST12345698765433", False), ("XX00", False)],
)
def test_iban(iban: str, ok: bool) -> None:
    assert iban_valid(iban) is ok


@pytest.mark.parametrize(
    "text,expected",
    [
        ("mail jane.doe@example.com now", ("EMAIL", "jane.doe@example.com")),
        ("card 4111 1111 1111 1111 ok", ("CREDIT_CARD", "4111 1111 1111 1111")),
        ("ssn 123-45-6789.", ("SSN", "123-45-6789")),
        ("from 192.168.10.4 today", ("IP_ADDRESS", "192.168.10.4")),
        ("key sk-abcdefghijklmnopqrstuvwx123456 here", ("API_KEY", "sk-abcdefghijklmnopqrstuvwx123456")),
        ("aws AKIAIOSFODNN7EXAMPLE", ("API_KEY", "AKIAIOSFODNN7EXAMPLE")),
        ("pay to DE89370400440532013000 please", ("IBAN", "DE89370400440532013000")),
    ],
)
def test_detects_each_type(text: str, expected: tuple[str, str]) -> None:
    assert expected in types_values(text)


def test_invalid_card_and_ip_not_detected_as_such() -> None:
    found = types_values("card 4111 1111 1111 1112 and ip 999.1.1.1")
    assert not any(t in (PIIType.CREDIT_CARD, PIIType.IP_ADDRESS) for t, _ in found)


def test_invalid_ssn_area_rejected() -> None:
    assert ("SSN", "000-12-3456") not in types_values("ssn 000-12-3456")
    assert ("SSN", "666-12-3456") not in types_values("ssn 666-12-3456")


def test_person_name_heuristics() -> None:
    found = types_values("Please ask Dr. Alice Johnson and Mr. Smith about it.")
    persons = {v for t, v in found if t == PIIType.PERSON}
    assert any("Alice" in p for p in persons)
    assert any("Smith" in p for p in persons)


def test_card_not_double_reported_as_phone() -> None:
    matches = PIIDetector().detect("4111 1111 1111 1111")
    assert [m.type for m in matches] == [PIIType.CREDIT_CARD]


def test_matches_are_non_overlapping_and_sorted() -> None:
    text = "jane@x.io 4111111111111111 10.0.0.1 jane@x.io"
    ms = PIIDetector().detect(text)
    assert [m.start for m in ms] == sorted(m.start for m in ms)
    for a, b in zip(ms, ms[1:]):
        assert a.end <= b.start


def test_detector_type_filter_and_unknown_type() -> None:
    det = PIIDetector(types=[PIIType.EMAIL])
    assert {m.type for m in det.detect("a@b.io 10.0.0.1")} == {PIIType.EMAIL}
    with pytest.raises(ValueError):
        PIIDetector(types=["SHOE_SIZE"])


def test_corpus_recall_and_negatives() -> None:
    det = PIIDetector()
    for text, labels in PII_SAMPLES + PII_HELDOUT:
        got = {(m.type, m.value) for m in det.detect(text)}
        for lbl in labels:
            assert tuple(lbl) in got, (text, lbl)
    for neg in PII_NEGATIVES:
        assert det.detect(neg) == [], neg


# ------------------------------------------------------------------ vault


def test_mask_stable_tokens_within_session() -> None:
    v = PIIVault(master_key=b"k" * 32)
    r1 = v.mask("s1", "write to bob@corp.io and bob@corp.io")
    toks = TOKEN_RE.findall(r1.text)
    assert len(toks) == 2 and toks[0] == toks[1]
    assert "bob@corp.io" not in r1.text
    r2 = v.mask("s1", "again bob@corp.io")
    assert TOKEN_RE.findall(r2.text)[0] == toks[0]
    assert r1.counts == {"EMAIL": 2}
    assert v.size("s1") == 1


def test_tokens_unlinkable_across_sessions_and_deterministic_with_key() -> None:
    v = PIIVault(master_key=b"k" * 32)
    a = v.mask("s1", "bob@corp.io").text
    b = v.mask("s2", "bob@corp.io").text
    assert a != b
    v2 = PIIVault(master_key=b"k" * 32)
    assert v2.mask("s1", "bob@corp.io").text == a


def test_unmask_scoped_by_type() -> None:
    v = PIIVault()
    masked = v.mask("s", "bob@corp.io card 4111 1111 1111 1111").text
    only_email = v.unmask("s", masked, ["EMAIL"])
    assert "bob@corp.io" in only_email and "4111" not in only_email
    assert v.unmask("s", masked, "*") == "bob@corp.io card 4111 1111 1111 1111"
    assert v.unmask("s", masked, ["*"]) == "bob@corp.io card 4111 1111 1111 1111"
    assert v.unmask("s", masked) == masked  # nothing authorised


def test_unmask_ignores_forged_and_cross_session_tokens() -> None:
    v = PIIVault()
    masked = v.mask("s1", "bob@corp.io").text
    v.mask("s2", "x@y.io")
    assert v.unmask("s2", masked, "*") == masked
    forged = "<<PII:EMAIL:12345678-1234-4234-8234-123456789abc>>"
    assert v.unmask("s1", forged, "*") == forged
    assert v.unmask("nosuch", masked, "*") == masked


def test_mask_does_not_remask_tokens() -> None:
    v = PIIVault()
    once = v.mask("s", "bob@corp.io 10.1.2.3").text
    twice = v.mask("s", once).text
    assert once == twice


def test_forget_session() -> None:
    v = PIIVault()
    masked = v.mask("s", "bob@corp.io").text
    v.forget("s")
    assert v.unmask("s", masked, "*") == masked
    assert v.size("s") == 0


def test_empty_text() -> None:
    v = PIIVault()
    r = v.mask("s", "")
    assert r.text == "" and r.matches == []
