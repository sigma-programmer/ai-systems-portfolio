"""Semantic memory: distilled facts with dedup and contradiction handling.

`PatternFactExtractor` recognises first-person statements ("my name is",
"I work at", "I prefer X over Y", "remember that ...", "I no longer ...") and
emits `(subject, predicate, value)` triples. `SemanticMemory.upsert` then
applies the belief-revision policy:

* same key + same normalised value       → duplicate: reinforce (confidence↑, access↑)
* single-valued predicate, new value     → newer fact supersedes the old one
* likes X  vs  dislikes X (either order) → newer polarity supersedes the old one
* retraction ("I no longer work at X")   → supersede the matching active fact
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

from aikit import LLM

from .models import Fact, new_id

SINGLE_VALUED_PREFIXES = ("name", "employer", "location", "occupation", "birthday",
                          "favorite_", "pet_name", "partner_name", "child_name", "diet")
MULTI_VALUED = ("likes", "dislikes", "allergy", "note", "prefers")
_OPPOSITE = {"likes": "dislikes", "dislikes": "likes"}


def is_single_valued(predicate: str) -> bool:
    return predicate.startswith(SINGLE_VALUED_PREFIXES) or predicate.endswith("_name")


def normalise(value: str) -> str:
    v = re.sub(r"[^\w\s'-]", "", value.lower()).strip()
    return re.sub(r"^(the|a|an)\s+", "", re.sub(r"\s+", " ", v))


@dataclass(frozen=True)
class Extracted:
    predicate: str
    value: str
    retract: bool = False
    subject: str = "user"


def render(subject: str, predicate: str, value: str) -> str:
    """Natural-language rendering used for embedding-based recall."""
    who = "The user" if subject == "user" else subject
    p = predicate
    if p == "name":
        return f"{who}'s name is {value}."
    if p == "employer":
        return f"{who} works at {value} (employer, job, company)."
    if p == "location":
        return f"{who} lives in {value} (home city, location)."
    if p == "occupation":
        return f"{who} works as {value} (occupation, job)."
    if p == "birthday":
        return f"{who}'s birthday is {value}."
    if p == "likes":
        return f"{who} likes {value} (preference, hobby, enjoys)."
    if p == "dislikes":
        return f"{who} dislikes {value} (preference)."
    if p == "prefers":
        return f"{who} prefers {value} (preference)."
    if p == "allergy":
        return f"{who} is allergic to {value} (allergy, food, health)."
    if p == "diet":
        return f"{who} is {value} (diet, food)."
    if p == "note":
        return f"Remember: {value}."
    if p.startswith("favorite_"):
        return f"{who}'s favorite {p[9:].replace('_', ' ')} is {value}."
    if p == "pet_name":
        return f"{who}'s pet (dog or cat) is named {value}."
    if p.endswith("_name"):
        return f"{who}'s {p[:-5].replace('_', ' ')} is named {value}."
    return f"{who} {p.replace('_', ' ')}: {value}."


_V = r"([A-Za-z0-9][\w\s'&.-]{0,40}?)"
_END = (r"(?=\s*(?:[.,!?;]|\b(?:and|but|so|now|as|since|with|in|on|because|which|who|"
        r"these days|anymore|last|this|next|yesterday|today|recently|ago)\b|$))")

_PATTERNS: list[tuple[re.Pattern, Callable[[re.Match], Extracted | None]]] = [
    (re.compile(rf"\bremember that\s+(.{{3,120}}?)(?:[.!?]|$)", re.I),
     lambda m: Extracted("note", m.group(1).strip())),
    (re.compile(rf"\bmy name is\s+{_V}{_END}", re.I), lambda m: Extracted("name", m.group(1))),
    (re.compile(rf"\b(?:call me)\s+{_V}{_END}", re.I), lambda m: Extracted("name", m.group(1))),
    (re.compile(rf"\bi (?:no longer|don'?t|do not) work (?:at|for)\s+{_V}{_END}", re.I),
     lambda m: Extracted("employer", m.group(1), retract=True)),
    (re.compile(rf"\bi(?:'ve| have)? (?:just )?(?:joined|started (?:working )?(?:at|for)|"
                rf"work (?:at|for)|got a job (?:at|with))\s+{_V}{_END}", re.I),
     lambda m: Extracted("employer", m.group(1))),
    (re.compile(rf"\bi (?:live in|moved to|relocated to|am based in|'m based in)\s+{_V}{_END}",
                re.I),
     lambda m: Extracted("location", m.group(1))),
    (re.compile(rf"\bi(?:'m| am) allergic to\s+{_V}{_END}", re.I),
     lambda m: Extracted("allergy", m.group(1))),
    (re.compile(rf"\bi(?:'m| am) (?:a |an )?(vegan|vegetarian|pescatarian|keto)\b", re.I),
     lambda m: Extracted("diet", m.group(1))),
    (re.compile(rf"\bmy birthday is (?:on )?\s*{_V}{_END}", re.I),
     lambda m: Extracted("birthday", m.group(1))),
    (re.compile(rf"\bmy favou?rite ([a-z ]{{2,20}}?) is\s+{_V}{_END}", re.I),
     lambda m: Extracted("favorite_" + m.group(1).strip().replace(" ", "_").lower(),
                         m.group(2))),
    (re.compile(rf"\bmy (\w+)(?:'s| is) name is\s+{_V}{_END}|\bmy (\w+) is (?:called|named)\s+"
                rf"{_V}{_END}", re.I),
     lambda m: Extracted(
         ("pet" if (m.group(1) or m.group(3)).lower() in ("dog", "cat", "pet", "parrot")
          else (m.group(1) or m.group(3)).lower()) + "_name", m.group(2) or m.group(4))),
    (re.compile(rf"\bi prefer\s+{_V}\s+(?:over|to|instead of)\s+{_V}{_END}", re.I),
     lambda m: Extracted("prefers", f"{m.group(1)} over {m.group(2)}")),
    (re.compile(rf"\bi prefer\s+{_V}{_END}", re.I), lambda m: Extracted("prefers", m.group(1))),
    (re.compile(rf"\bi (?:no longer|don'?t|do not) (?:like|love|enjoy)\s+{_V}"
                rf"(?:\s+anymore)?{_END}", re.I),
     lambda m: Extracted("dislikes", re.sub(r"\s+anymore$", "", m.group(1)))),
    (re.compile(rf"\bi (?:really )?(?:like|love|enjoy|adore)\s+{_V}{_END}", re.I),
     lambda m: Extracted("likes", m.group(1))),
    (re.compile(rf"\bi (?:really )?(?:hate|dislike|can'?t stand|detest)\s+{_V}{_END}", re.I),
     lambda m: Extracted("dislikes", m.group(1))),
    (re.compile(rf"\bi(?:'m| am) (?:a|an)\s+([a-z][a-z ]{{2,30}}?)\s+(?:at|for)\s+{_V}{_END}",
                re.I),
     lambda m: Extracted("occupation", m.group(1))),
    (re.compile(rf"\bi(?:'m| am| work as) (?:a|an)\s+([a-z][a-z ]{{2,30}}?){_END}", re.I),
     lambda m: Extracted("occupation", m.group(1))),
]

_JOB_WORDS = re.compile(r"(engineer|developer|designer|teacher|doctor|nurse|manager|scientist|"
                        r"analyst|writer|student|lawyer|architect|consultant|researcher|chef|"
                        r"accountant|artist|pilot|founder)s?$", re.I)


class FactExtractor(Protocol):
    def extract(self, text: str) -> list[Extracted]: ...


class PatternFactExtractor:
    """Regex-based first-person fact extraction (offline default)."""

    def extract(self, text: str) -> list[Extracted]:
        out: list[Extracted] = []
        claimed: list[tuple[int, int]] = []
        for pat, build in _PATTERNS:
            for m in pat.finditer(text):
                if any(m.start() < e and s < m.end() for s, e in claimed):
                    continue
                ex = build(m)
                if ex is None:
                    continue
                value = ex.value.strip(" .'")
                if not value or (ex.predicate == "occupation" and not _JOB_WORDS.search(value)):
                    continue
                if ex.predicate == "employer" and value.lower() in ("home", "it", "this"):
                    continue
                out.append(Extracted(ex.predicate, value, ex.retract, ex.subject))
                claimed.append((m.start(), m.end()))
        return out


class LLMFactExtractor:
    """LLM extraction; expects JSON [{"predicate":..,"value":..,"retract":bool}]."""

    SYSTEM = ("Extract durable facts about the user from the message as a JSON list of "
              "{\"predicate\", \"value\", \"retract\"}. Predicates: name, employer, location, "
              "occupation, birthday, likes, dislikes, prefers, allergy, diet, note, "
              "favorite_<thing>, <relation>_name.")

    def __init__(self, llm: LLM):
        self.llm = llm

    def extract(self, text: str) -> list[Extracted]:
        resp = self.llm.complete(f"EXTRACT FACTS\n{text}", system=self.SYSTEM, json_mode=True)
        try:
            data = resp.json()
        except ValueError:
            return []
        if isinstance(data, dict):
            data = data.get("facts", [])
        return [Extracted(str(d["predicate"]), str(d["value"]), bool(d.get("retract", False)))
                for d in data if isinstance(d, dict) and d.get("predicate") and d.get("value")]


class SemanticMemory:
    """Active + superseded facts with the belief-revision policy in the module docstring."""

    def __init__(self) -> None:
        self.facts: dict[str, Fact] = {}

    def active(self) -> list[Fact]:
        return [f for f in self.facts.values() if f.status == "active"]

    def find_active(self, subject: str, predicate: str) -> list[Fact]:
        return [f for f in self.active() if f.subject == subject and f.predicate == predicate]

    def _supersede(self, old: Fact, new_id_: str | None, now: float) -> None:
        old.status = "superseded"
        old.superseded_by = new_id_
        old.last_access = now  # superseded-TTL counts from the moment of revision

    def upsert(self, ex: Extracted, now: float, source_episode: str | None = None,
               salience: float = 0.7) -> tuple[str, Fact | None, list[Fact]]:
        """Apply one extracted fact; returns (action, fact, facts superseded)."""
        norm = normalise(ex.value)
        changed: list[Fact] = []
        if ex.retract:
            for f in self.find_active(ex.subject, ex.predicate):
                if normalise(f.value) == norm or is_single_valued(ex.predicate):
                    self._supersede(f, None, now)
                    changed.append(f)
            return ("retracted" if changed else "noop"), None, changed
        for f in self.find_active(ex.subject, ex.predicate):
            if normalise(f.value) == norm:
                f.confidence = min(0.99, f.confidence + 0.1)
                f.access_count += 1
                f.last_access = now
                return "duplicate", f, []
        fact = Fact(new_id("fact"), ex.subject, ex.predicate, ex.value,
                    render(ex.subject, ex.predicate, ex.value), now, now, source_episode,
                    salience=salience)
        if is_single_valued(ex.predicate):
            changed.extend(self.find_active(ex.subject, ex.predicate))
        if ex.predicate in _OPPOSITE:
            changed.extend(f for f in self.find_active(ex.subject, _OPPOSITE[ex.predicate])
                           if normalise(f.value) == norm)
        for f in changed:
            self._supersede(f, fact.id, now)
        self.facts[fact.id] = fact
        return ("superseded" if changed else "inserted"), fact, changed

    def profile(self) -> dict[str, list[str] | str]:
        out: dict[str, list[str] | str] = {}
        for f in sorted(self.active(), key=lambda f: f.created):
            if is_single_valued(f.predicate):
                out[f.predicate] = f.value
            else:
                lst = out.setdefault(f.predicate, [])
                assert isinstance(lst, list)
                lst.append(f.value)
        return out

    def load(self, facts: Iterable[Fact]) -> None:
        self.facts = {f.id: f for f in facts}
