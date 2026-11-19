"""Entity and relation extraction.

Two interchangeable extractors implement the :class:`Extractor` protocol:

* :class:`HeuristicExtractor` — fully offline. Capitalised-span NER with
  gazetteer / head-noun / context typing, lightweight pronoun coreference
  ("the company", "it", "its", "he", "she"), pattern-based relation triples
  over a placeholder-rewritten sentence, alias detection ("X (Y)"), and a
  weighted co-occurrence fallback.
* :class:`LLMExtractor` — asks any :class:`aikit.LLM` for JSON entities and
  relations, validated with pydantic, falling back to the heuristic extractor
  if the model's output cannot be parsed.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from aikit import LLM

from .models import Chunk, EntityType, ExtractedRelation, ExtractionResult, Mention, RelationType
from .text import normalize_name, split_sentences

# --------------------------------------------------------------------------- lexicons

ORG_HEADS = frozenset(
    """Inc Corp Corporation Ltd LLC Co Labs Lab Systems Group Capital Ventures Robotics Analytics Freight Bio
    Energy Grid Materials Technologies Technology Partners Holdings Bank University Institute Foundation Networks
    Dynamics Logistics Pharma Pharmaceuticals Media Studios Industries Motors Airlines Software Solutions
    Semiconductor Semiconductors Therapeutics Consulting Fund College""".split()
)
LEADING_ORG_HEADS = frozenset({"University", "Bank", "Institute", "College"})
LOCATION_GAZETTEER = frozenset(
    """Port Alder|Northvale|Eastmere|Kestrel Bay|Redmoor|Lowford|London|Paris|Berlin|Tokyo|New York|San Francisco|
    Boston|Seattle|Singapore|Toronto|Sydney|Mumbai|Lagos|Nairobi|Chicago|Austin|Dublin|Zurich""".replace("\n", "")
    .replace("    ", "")
    .split("|")
)
LOCATION_HEADS = frozenset({"Bay", "City", "Valley", "Harbor", "Harbour", "Falls", "Springs", "Heights", "Port"})
PRODUCT_HEADS = frozenset(
    {"Chip", "Sensor", "Arm", "Platform", "Engine", "Suite", "Cloud", "OS", "Phone", "Drone", "Robot", "App",
     "Processor", "Battery", "Model", "Studio", "Kit"}
)
FIRST_NAMES = frozenset(
    """Alice Ben Dmitri Ingrid Marcus Priya Rafael Sofia Tomas Hana Grace Olu Ken Leila Elena Victor Erik
    James John Mary Robert Michael Linda David Sarah Emma Olivia Liam Noah Ava Mia Lucas Ethan Chloe Daniel
    Anna Maria Wei Li Yuki Omar Fatima Ahmed Ravi Anil Sanjay Carlos Lucia Pedro Ana Jean Pierre Hans Lars
    Nadia Ivan Igor Kofi Amara Chen Mei Jin Kenji Aiko Paul Peter Laura Julia Tom Sam Alex Kate Nina""".split()
)
TITLES = frozenset({"Dr", "Mr", "Mrs", "Ms", "Prof", "Sir", "Dame"})
CONNECTORS = frozenset({"of", "&", "de", "van", "von", "der"})
# Capitalised words that are never (part of) entity names.
NON_ENTITY_CAPS = frozenset(
    """CEO CTO CFO COO CMO AI ML API IT US EU UK IPO R&D Q1 Q2 Q3 Q4
    January February March April May June July August September October November December
    Monday Tuesday Wednesday Thursday Friday Saturday Sunday""".split()
)
COMMON_WORDS = frozenset(
    """the a an this that these those its it his her their our he she they we in on at after before following
    under like since during when while both today last next analysts according however meanwhile although
    though despite with without for from by as each every some many most several few other another all any
    under over also later earlier then now once yet still because if unless until where why how what which who
    there here grid-scale industry investors critics sources""".split()
)

ROLE_ALT = (
    r"(ceo|cto|cfo|chief\s+executive(?:\s+officer)?|chief\s+technology\s+officer|chief\s+financial\s+officer)"
)
_ROLE_TO_REL = {
    "ceo": RelationType.CEO_OF,
    "chief executive": RelationType.CEO_OF,
    "chief executive officer": RelationType.CEO_OF,
    "cto": RelationType.CTO_OF,
    "chief technology officer": RelationType.CTO_OF,
    "cfo": RelationType.CFO_OF,
    "chief financial officer": RelationType.CFO_OF,
}

# --------------------------------------------------------------------------- relation patterns

_E = r"<E(\d+)>"
_L = r"(<E\d+>(?:(?:\s*,\s*(?:and\s+)?|\s+and\s+)<E\d+>)*)"
_A = r"(?:\s+(?:also|later|then|subsequently|eventually|recently|formally|officially|quietly|now|still|previously))?"
_BE = r"(?:is|was|remains|remained|serves\s+as|served\s+as|became|has\s+been|is\s+now)"


@dataclass(frozen=True)
class RelationPattern:
    """A regex over placeholder text. ``subj``/``obj`` are group numbers;
    ``role`` (optional) is the group carrying a job title that picks the relation."""

    regex: re.Pattern
    relation: RelationType | None
    subj: int
    obj: int
    role: int | None = None


def _p(pattern: str, relation: RelationType | None, subj: int, obj: int, role: int | None = None) -> RelationPattern:
    return RelationPattern(re.compile(pattern, re.I), relation, subj, obj, role)


R = RelationType
RELATION_PATTERNS: list[RelationPattern] = [
    # FOUNDED
    _p(rf"{_L}{_A}\s+(?:co-)?founded\s+{_L}", R.FOUNDED, 1, 2),
    _p(rf"{_L}\s+(?:was|were|had\s+been|has\s+been)\s+(?:co-)?founded(?:\s+in\s+\d{{4}})?\s+by\s+{_L}", R.FOUNDED, 2, 1),
    _p(rf"{_E},\s+(?:the\s+|a\s+|its\s+)?(?:co-)?founder\s+of\s+{_E}", R.FOUNDED, 1, 2),
    _p(rf"{_E}\s+(?:is|was)\s+(?:the\s+|a\s+)?(?:co-)?founder\s+of\s+{_E}", R.FOUNDED, 1, 2),
    # ACQUIRED
    _p(rf"{_E}{_A}\s+(?:acquired|bought|purchased|took\s+over)\s+{_E}", R.ACQUIRED, 1, 2),
    _p(rf"{_E}\s+(?:was|had\s+been|has\s+been)\s+(?:acquired|bought|purchased)(?:\s+in\s+\d{{4}})?\s+by\s+{_E}", R.ACQUIRED, 2, 1),
    _p(rf"{_E}'s\s+acquisition\s+of\s+{_E}", R.ACQUIRED, 1, 2),
    # Executive roles
    _p(rf"{_E}(?:,?\s+who){_A}\s+{_BE}\s+(?:the\s+)?{ROLE_ALT}\s+(?:of|at)\s+{_E}", None, 1, 3, role=2),
    _p(rf"{_E}{_A}\s+{_BE}\s+(?:the\s+)?{ROLE_ALT}\s+(?:of|at)\s+{_E}", None, 1, 3, role=2),
    _p(rf"{_E},\s+(?:the\s+)?{ROLE_ALT}\s+of\s+{_E}", None, 1, 3, role=2),
    _p(rf"{_E}'s\s+{ROLE_ALT}(?:\s+is|,)\s+{_E}", None, 3, 1, role=2),
    _p(rf"{_E}(?:,?\s+who)?{_A}\s+{_BE}\s+{_E}'s\s+{ROLE_ALT}", None, 1, 2, role=3),
    _p(rf"{_E}\s+(?:named|appointed|hired)\s+{_E}\s+(?:as\s+)?(?:its\s+)?(?:new\s+)?{ROLE_ALT}", None, 2, 1, role=3),
    # LOCATED_IN
    _p(rf"{_E}[^<.]{{0,60}}?\b(?:headquartered|based|located)\s+in\s+{_E}", R.LOCATED_IN, 1, 2),
    _p(rf"{_E}'s\s+headquarters\s+(?:is|are)\s+in\s+{_E}", R.LOCATED_IN, 1, 2),
    # INVESTED_IN
    _p(rf"{_E}{_A}\s+(?:invested|invests)\s+in\s+{_E}", R.INVESTED_IN, 1, 2),
    _p(rf"{_E}\s+led\s+[^<]{{0,40}}?(?:round|investment)\s+in\s+{_E}", R.INVESTED_IN, 1, 2),
    # PARTNERED_WITH
    _p(rf"{_E}{_A}\s+(?:partnered|teamed\s+up|allied)\s+with\s+{_E}", R.PARTNERED_WITH, 1, 2),
    _p(rf"{_E}\s+and\s+{_E}\s+(?:formed|announced|signed)\s+(?:a\s+)?partnership", R.PARTNERED_WITH, 1, 2),
    # WORKS_AT
    _p(rf"{_E}{_A}\s+(?:joined|works\s+at|worked\s+at|works\s+for|worked\s+for)\s+{_E}", R.WORKS_AT, 1, 2),
    _p(rf"{_E}{_A}\s+(?:works|worked|served)\s+as\s+(?:an?\s+)?[a-z][a-z\s-]{{0,40}}?\s+(?:at|for)\s+{_E}", R.WORKS_AT, 1, 2),
    _p(rf"{_E}\s+(?:is|was)\s+(?:the|an?)\s+(?:managing\s+partner|partner|chief\s+scientist|president|chair|"
       rf"chairman|chairwoman|director|head\s+of\s+[a-z]+|engineer|researcher)\s+(?:of|at)\s+{_E}", R.WORKS_AT, 1, 2),
    _p(rf"{_E}\s+leads\s+[^<]{{0,40}}?\s+at\s+{_E}", R.WORKS_AT, 1, 2),
    # SUBSIDIARY_OF
    _p(rf"{_E}\s+(?:is|was|became)\s+(?:a|an|the)\s+(?:wholly[- ]owned\s+)?(?:subsidiary|division|unit)\s+of\s+{_E}", R.SUBSIDIARY_OF, 1, 2),
    _p(rf"{_E}\s+owns\s+{_E}", R.SUBSIDIARY_OF, 2, 1),
    # PRODUCES
    _p(rf"{_E}{_A}\s+(?:developed|launched|released|makes|built|created|introduced|produces)\s+(?:the\s+)?{_E}", R.PRODUCES, 1, 2),
    # SUPPLIES
    _p(rf"{_E}{_A}\s+(?:supplies|supplied)\s+{_E}", R.SUPPLIES, 1, 2),
    _p(rf"{_E}\s+(?:is|was|became)\s+(?:the\s+|a\s+)?(?:main\s+|primary\s+|key\s+)?(?:[a-z-]+\s+)?supplier\s+(?:to|for|of)\s+(?:the\s+)?{_E}", R.SUPPLIES, 1, 2),
    _p(rf"makes\s+{_E}\s+the\s+(?:main\s+|primary\s+|key\s+)?(?:[a-z-]+\s+)?supplier\s+(?:to|for|of)\s+(?:the\s+)?{_E}", R.SUPPLIES, 1, 2),
    # ALUMNUS_OF
    _p(rf"{_E}{_A}\s+(?:studied\s+at|graduated\s+from|attended)\s+{_E}", R.ALUMNUS_OF, 1, 2),
]
ALIAS_PATTERNS = [
    re.compile(rf"{_E}\s+\(\s*{_E}\s*\)"),
    re.compile(rf"{_E},?\s+(?:also\s+known\s+as|formerly\s+known\s+as|formerly|known\s+as)\s+{_E}", re.I),
]

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-&][A-Za-z0-9]+)*|'s|\S")
_ORG_REF = re.compile(r"\b(?:[Tt]he|[Tt]his) (?:company|firm|startup|business|conglomerate)\b")
_IT_REF = re.compile(r"^It\b")
_ITS_REF = re.compile(r"\b[Ii]ts\b")
_PERSON_REF = re.compile(r"(?:^|(?<=[,;] ))(?:He|She|he|she)\b")


# --------------------------------------------------------------------------- protocol & state


class Extractor(Protocol):
    """Anything that turns chunks into :class:`ExtractionResult` objects."""

    def extract_document(self, chunks: list[Chunk]) -> list[ExtractionResult]: ...


@dataclass
class CorefState:
    """Most recent antecedents, carried across sentences/chunks of one document."""

    last: dict[EntityType, str] = field(default_factory=dict)


@dataclass
class _Span:
    text: str
    start: int
    end: int
    type: EntityType
    coref: bool = False
    possessive: bool = False


# --------------------------------------------------------------------------- heuristic NER


def _indices(group: str) -> list[str]:
    """Placeholder indices captured by a pattern group ("3" or "<E1>, <E2> and <E4>")."""
    return [group] if group.isdigit() else re.findall(r"<E(\d+)>", group)


def _is_cap(tok: str) -> bool:
    return tok[:1].isupper() and tok not in NON_ENTITY_CAPS


def classify_span(tokens: list[str], prev_word: str | None) -> EntityType:
    """Assign an entity type to a capitalised span from its head words and left context."""
    text = " ".join(tokens)
    first, last = tokens[0], tokens[-1]
    if last in ORG_HEADS or first in LEADING_ORG_HEADS:
        return EntityType.ORG
    if text in LOCATION_GAZETTEER or (len(tokens) > 1 and last in LOCATION_HEADS):
        return EntityType.LOCATION
    if len(tokens) > 1 and last in PRODUCT_HEADS:
        return EntityType.PRODUCT
    if first in FIRST_NAMES or (prev_word or "") in TITLES:
        return EntityType.PERSON
    if (prev_word or "").lower() in {"in", "near", "outside"} and len(tokens) <= 2:
        return EntityType.LOCATION
    if len(tokens) == 1 and first.isupper() and 2 <= len(first) <= 5:
        return EntityType.ORG  # acronym
    if 2 <= len(tokens) <= 3 and all(t.isalpha() for t in tokens):
        return EntityType.PERSON
    return EntityType.OTHER


def find_entity_spans(sentence: str) -> list[_Span]:
    """Capitalised-span NER for one sentence."""
    toks = [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(sentence)]
    spans: list[_Span] = []
    i, n = 0, len(toks)
    while i < n:
        if not _is_cap(toks[i][0]):
            i += 1
            continue
        j = i + 1
        while j < n:
            if _is_cap(toks[j][0]):
                j += 1
            elif (
                toks[j][0] in CONNECTORS
                and j + 1 < n
                and _is_cap(toks[j + 1][0])
                and (toks[j][0] != "of" or toks[j - 1][0] in LEADING_ORG_HEADS)
            ):
                j += 2
            else:
                break
        group = toks[i:j]
        # strip leading function words ("After", "The", "Under", ...)
        while group and group[0][0].lower() in COMMON_WORDS:
            group = group[1:]
        if group:
            words = [g[0] for g in group]
            sentence_initial = group[0][1] == 0
            junk = sentence_initial and len(words) == 1 and re.search(r"-[a-z]", words[0])
            if not junk:
                prev_idx = toks.index(group[0]) - 1
                prev_word = toks[prev_idx][0].rstrip(".") if prev_idx >= 0 else None
                start, end = group[0][1], group[-1][2]
                spans.append(_Span(sentence[start:end], start, end, classify_span(words, prev_word)))
        i = j
    return spans


def _coref_spans(sentence: str, spans: list[_Span]) -> list[_Span]:
    """Locate pronominal references that should be bound to an antecedent."""
    occupied = [(s.start, s.end) for s in spans]

    def free(a: int, b: int) -> bool:
        return all(b <= s or a >= e for s, e in occupied)

    refs: list[_Span] = []
    for rx, etype, poss in (
        (_ORG_REF, EntityType.ORG, False),
        (_IT_REF, EntityType.ORG, False),
        (_ITS_REF, EntityType.ORG, True),
        (_PERSON_REF, EntityType.PERSON, False),
    ):
        for m in rx.finditer(sentence):
            if free(m.start(), m.end()):
                refs.append(_Span(m.group(), m.start(), m.end(), etype, coref=True, possessive=poss))
    return refs


# --------------------------------------------------------------------------- heuristic extractor


class HeuristicExtractor:
    """Offline rule-based entity + relation extractor.

    Args:
        cooccurrence: emit RELATED_TO edges for entity pairs in the same
            sentence that no typed pattern connected.
        cooccurrence_confidence: confidence (edge weight) of those edges.
    """

    def __init__(self, cooccurrence: bool = True, cooccurrence_confidence: float = 0.5):
        self.cooccurrence = cooccurrence
        self.cooccurrence_confidence = cooccurrence_confidence

    def extract_document(self, chunks: list[Chunk]) -> list[ExtractionResult]:
        state = CorefState()
        return [self.extract(c, state) for c in chunks]

    def extract(self, chunk: Chunk, state: CorefState | None = None) -> ExtractionResult:
        state = state or CorefState()
        result = ExtractionResult(chunk_id=chunk.id)
        for s_idx, sentence in enumerate(split_sentences(chunk.text)):
            self._extract_sentence(sentence, s_idx, chunk.id, state, result)
        return result

    def _extract_sentence(
        self, sentence: str, s_idx: int, chunk_id: str, state: CorefState, result: ExtractionResult
    ) -> None:
        spans = find_entity_spans(sentence)
        items = sorted(spans + _coref_spans(sentence, spans), key=lambda s: s.start)
        bound: list[_Span] = []
        for sp in items:
            if sp.coref:
                antecedent = state.last.get(sp.type)
                if antecedent is None:
                    continue
                bound.append(_Span(antecedent, sp.start, sp.end, sp.type, coref=True, possessive=sp.possessive))
            else:
                bound.append(sp)
                if sp.type in (EntityType.ORG, EntityType.PERSON):
                    state.last[sp.type] = sp.text
                result.mentions.append(
                    Mention(text=sp.text, type=sp.type, chunk_id=chunk_id, sentence_index=s_idx, start=sp.start, end=sp.end)
                )
        if not bound:
            return

        # Rewrite the sentence with placeholders so patterns see "<E0> acquired <E1>".
        parts, cursor = [], 0
        for k, sp in enumerate(bound):
            parts.append(sentence[cursor : sp.start])
            parts.append(f"<E{k}>'s" if sp.possessive else f"<E{k}>")
            cursor = sp.end
        parts.append(sentence[cursor:])
        templ = "".join(parts)

        def surface(idx: str) -> str:
            return bound[int(idx)].text

        linked: set[frozenset[str]] = set()
        for rx in ALIAS_PATTERNS:
            for m in rx.finditer(templ):
                full, alias = surface(m.group(1)), surface(m.group(2))
                if normalize_name(full) != normalize_name(alias):
                    result.aliases.append((alias, full))
                    linked.add(frozenset((normalize_name(full), normalize_name(alias))))

        # "Nimbus Freight (NF) is ..." -> "<E0> is ..." so patterns are not blocked by aliases.
        rel_templ = re.sub(r"(<E\d+>)\s+\(\s*<E\d+>\s*\)", r"\1", templ)
        seen: set[tuple[str, RelationType, str]] = set()
        for pat in RELATION_PATTERNS:
            for m in pat.regex.finditer(rel_templ):
                rel = pat.relation
                if pat.role is not None:
                    role = re.sub(r"\s+", " ", m.group(pat.role).lower())
                    rel = _ROLE_TO_REL[role]
                assert rel is not None
                subs = _indices(m.group(pat.subj))
                objs = _indices(m.group(pat.obj))
                for si, oi in itertools.product(subs, objs):
                    s, o = surface(si), surface(oi)
                    if normalize_name(s) == normalize_name(o):
                        continue
                    key = (s, rel, o)
                    if key in seen:
                        continue
                    seen.add(key)
                    linked.add(frozenset((normalize_name(s), normalize_name(o))))
                    result.relations.append(
                        ExtractedRelation(subject=s, relation=rel, object=o, chunk_id=chunk_id, evidence=sentence)
                    )

        if self.cooccurrence:
            names = list(dict.fromkeys(sp.text for sp in bound))
            for a, b in itertools.combinations(names, 2):
                pair = frozenset((normalize_name(a), normalize_name(b)))
                if len(pair) < 2 or pair in linked:
                    continue
                linked.add(pair)
                result.relations.append(
                    ExtractedRelation(
                        subject=a,
                        relation=RelationType.RELATED_TO,
                        object=b,
                        chunk_id=chunk_id,
                        evidence=sentence,
                        confidence=self.cooccurrence_confidence,
                    )
                )


# --------------------------------------------------------------------------- LLM extractor


class _LLMEntity(BaseModel):
    name: str
    type: str = "OTHER"


class _LLMRelation(BaseModel):
    source: str
    relation: str
    target: str
    evidence: str = ""


class _LLMOutput(BaseModel):
    entities: list[_LLMEntity] = Field(default_factory=list)
    relations: list[_LLMRelation] = Field(default_factory=list)


_REL_SYNONYMS = {
    "CEO": R.CEO_OF, "IS_CEO_OF": R.CEO_OF, "CHIEF_EXECUTIVE_OF": R.CEO_OF,
    "CTO": R.CTO_OF, "CFO": R.CFO_OF,
    "ACQUIRES": R.ACQUIRED, "ACQUISITION": R.ACQUIRED, "BOUGHT": R.ACQUIRED,
    "FOUNDER_OF": R.FOUNDED, "FOUNDED_BY": None, "CO_FOUNDED": R.FOUNDED,
    "HEADQUARTERED_IN": R.LOCATED_IN, "BASED_IN": R.LOCATED_IN, "LOCATED": R.LOCATED_IN,
    "INVESTS_IN": R.INVESTED_IN, "PARTNER": R.PARTNERED_WITH, "PARTNERS_WITH": R.PARTNERED_WITH,
    "EMPLOYED_BY": R.WORKS_AT, "WORKS_FOR": R.WORKS_AT, "JOINED": R.WORKS_AT,
    "PART_OF": R.SUBSIDIARY_OF, "DIVISION_OF": R.SUBSIDIARY_OF, "MAKES": R.PRODUCES, "DEVELOPED": R.PRODUCES,
    "SUPPLIER_OF": R.SUPPLIES, "STUDIED_AT": R.ALUMNUS_OF, "GRADUATED_FROM": R.ALUMNUS_OF,
}

LLM_EXTRACTION_SYSTEM = (
    "You are an information-extraction engine that builds a knowledge graph. "
    "Extract named entities and the relations between them from the text."
)
LLM_EXTRACTION_PROMPT = """Extract entities and relations from the text below.

Entity types: PERSON, ORG, LOCATION, PRODUCT, OTHER.
Relation types: {relations}.
Use RELATED_TO only when no specific type applies. Use full names as written.

Return JSON: {{"entities": [{{"name": str, "type": str}}],
"relations": [{{"source": str, "relation": str, "target": str, "evidence": str}}]}}

TEXT:
{text}"""


def parse_relation_type(name: str) -> tuple[RelationType, bool]:
    """Map a free-form relation label to the canonical vocabulary.

    Returns ``(relation, reversed)``; ``reversed`` is True for passive labels
    such as FOUNDED_BY whose subject/object must be swapped.
    """
    key = re.sub(r"[^A-Z]+", "_", name.upper()).strip("_")
    if key == "FOUNDED_BY":
        return R.FOUNDED, True
    if key == "ACQUIRED_BY":
        return R.ACQUIRED, True
    try:
        return RelationType(key), False
    except ValueError:
        pass
    mapped = _REL_SYNONYMS.get(key)
    return (mapped, False) if mapped else (R.RELATED_TO, False)


def _parse_entity_type(name: str) -> EntityType:
    key = name.upper().strip()
    aliases = {"ORGANIZATION": "ORG", "COMPANY": "ORG", "PER": "PERSON", "LOC": "LOCATION", "GPE": "LOCATION", "CITY": "LOCATION"}
    key = aliases.get(key, key)
    try:
        return EntityType(key)
    except ValueError:
        return EntityType.OTHER


class LLMExtractor:
    """Extraction via any :class:`aikit.LLM` that returns JSON.

    Args:
        llm: the model client.
        fallback: extractor used when the reply is not valid JSON of the
            expected shape (defaults to :class:`HeuristicExtractor`).
        strict: if True, malformed replies raise instead of falling back.
    """

    def __init__(
        self, llm: LLM, fallback: HeuristicExtractor | None = None, max_tokens: int = 2048, strict: bool = False
    ):
        self.llm = llm
        self.fallback = None if strict else (fallback if fallback is not None else HeuristicExtractor())
        self.max_tokens = max_tokens
        self.failures = 0

    def extract_document(self, chunks: list[Chunk]) -> list[ExtractionResult]:
        return [self.extract(c) for c in chunks]

    def extract(self, chunk: Chunk) -> ExtractionResult:
        prompt = LLM_EXTRACTION_PROMPT.format(
            relations=", ".join(r.value for r in RelationType), text=chunk.text
        )
        resp = self.llm.complete(prompt, system=LLM_EXTRACTION_SYSTEM, max_tokens=self.max_tokens, json_mode=True)
        try:
            data = resp.json()
            if isinstance(data, list):
                data = {"entities": data}
            parsed = _LLMOutput.model_validate(data)
        except (ValueError, ValidationError):
            self.failures += 1
            if self.fallback is None:
                raise
            return self.fallback.extract(chunk)
        return self._to_result(chunk, parsed)

    @staticmethod
    def _to_result(chunk: Chunk, parsed: _LLMOutput) -> ExtractionResult:
        result = ExtractionResult(chunk_id=chunk.id)
        for ent in parsed.entities:
            name = ent.name.strip()
            if not name:
                continue
            pos = chunk.text.find(name)
            result.mentions.append(
                Mention(
                    text=name,
                    type=_parse_entity_type(ent.type),
                    chunk_id=chunk.id,
                    start=max(pos, 0),
                    end=max(pos, 0) + len(name),
                )
            )
        known = {m.text for m in result.mentions}
        for rel in parsed.relations:
            rtype, rev = parse_relation_type(rel.relation)
            s, o = (rel.target, rel.source) if rev else (rel.source, rel.target)
            s, o = s.strip(), o.strip()
            if not s or not o or normalize_name(s) == normalize_name(o):
                continue
            for name in (s, o):
                if name not in known:
                    known.add(name)
                    result.mentions.append(Mention(text=name, type=EntityType.OTHER, chunk_id=chunk.id))
            result.relations.append(
                ExtractedRelation(subject=s, relation=rtype, object=o, chunk_id=chunk.id, evidence=rel.evidence)
            )
        return result
