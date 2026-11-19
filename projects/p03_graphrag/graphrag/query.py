"""Query engines.

* :class:`QuestionParser` — a small backtracking semantic parser that maps
  nested natural-language questions onto a chain of graph hops, e.g.

      "Who is the CEO of the company that acquired Quillon Systems?"
      -> subj[CEO_OF](subj[ACQUIRED](Quillon Systems))

* :class:`LocalSearch` — executes the parse against the knowledge graph
  (entity-anchored multi-hop traversal), returning answers with full path
  explanations and provenance. When a question cannot be parsed it falls
  back to scored neighbourhood expansion around the entities it mentions.
  An optional LLM can phrase the final answer from the retrieved subgraph.
* :class:`GlobalSearch` — map-reduce over community summaries for
  corpus-wide questions ("Summarize the major acquisitions").
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Union

import numpy as np
from pydantic import BaseModel, Field

from aikit import LLM, HashingEmbedder

from .community import Community, verbalize
from .graph import KnowledgeGraph
from .models import EntityType, RelationType
from .text import keyword_stems, normalize_name

R = RelationType

# --------------------------------------------------------------------------- lexicons

#: verb phrase -> (relation, inverted). Inverted verbs ("owns") flip direction.
VERBS: dict[str, tuple[RelationType, bool]] = {
    "acquired": (R.ACQUIRED, False), "bought": (R.ACQUIRED, False), "purchased": (R.ACQUIRED, False),
    "founded": (R.FOUNDED, False), "co-founded": (R.FOUNDED, False), "started": (R.FOUNDED, False),
    "invested in": (R.INVESTED_IN, False), "backed": (R.INVESTED_IN, False),
    "partnered with": (R.PARTNERED_WITH, False),
    "supplies": (R.SUPPLIES, False), "supplied": (R.SUPPLIES, False),
    "developed": (R.PRODUCES, False), "makes": (R.PRODUCES, False), "launched": (R.PRODUCES, False),
    "released": (R.PRODUCES, False), "built": (R.PRODUCES, False),
    "joined": (R.WORKS_AT, False), "works at": (R.WORKS_AT, False), "works for": (R.WORKS_AT, False),
    "studied at": (R.ALUMNUS_OF, False), "graduated from": (R.ALUMNUS_OF, False),
    "owns": (R.SUBSIDIARY_OF, True),
}
#: base verb forms used after "did" ("Which company did X acquire?").
BASE_VERBS: dict[str, tuple[RelationType, bool]] = {
    "acquire": (R.ACQUIRED, False), "buy": (R.ACQUIRED, False), "purchase": (R.ACQUIRED, False),
    "found": (R.FOUNDED, False), "start": (R.FOUNDED, False), "invest in": (R.INVESTED_IN, False),
    "back": (R.INVESTED_IN, False), "partner with": (R.PARTNERED_WITH, False), "supply": (R.SUPPLIES, False),
    "develop": (R.PRODUCES, False), "make": (R.PRODUCES, False), "launch": (R.PRODUCES, False),
    "release": (R.PRODUCES, False), "build": (R.PRODUCES, False), "join": (R.WORKS_AT, False),
    "work at": (R.WORKS_AT, False), "work for": (R.WORKS_AT, False), "study at": (R.ALUMNUS_OF, False),
    "own": (R.SUBSIDIARY_OF, True),
}
#: "the <role> of X" -> (want, relation). want="subj": answer is the subject of REL with X as object.
ROLES: dict[str, tuple[str, RelationType]] = {
    "chief executive officer": ("subj", R.CEO_OF), "chief executive": ("subj", R.CEO_OF), "ceo": ("subj", R.CEO_OF),
    "chief technology officer": ("subj", R.CTO_OF), "cto": ("subj", R.CTO_OF),
    "chief financial officer": ("subj", R.CFO_OF), "cfo": ("subj", R.CFO_OF),
    "founders": ("subj", R.FOUNDED), "founder": ("subj", R.FOUNDED), "co-founder": ("subj", R.FOUNDED),
    "parent company": ("obj", R.SUBSIDIARY_OF), "parent": ("obj", R.SUBSIDIARY_OF), "owner": ("obj", R.SUBSIDIARY_OF),
    "subsidiary": ("subj", R.SUBSIDIARY_OF), "acquirer": ("subj", R.ACQUIRED), "buyer": ("subj", R.ACQUIRED),
    "investors": ("subj", R.INVESTED_IN), "investor": ("subj", R.INVESTED_IN), "backer": ("subj", R.INVESTED_IN),
    "supplier": ("subj", R.SUPPLIES), "partners": ("obj", R.PARTNERED_WITH), "partner": ("obj", R.PARTNERED_WITH),
    "headquarters": ("obj", R.LOCATED_IN), "home city": ("obj", R.LOCATED_IN), "location": ("obj", R.LOCATED_IN),
    "products": ("obj", R.PRODUCES), "product": ("obj", R.PRODUCES), "alma mater": ("obj", R.ALUMNUS_OF),
    "employer": ("obj", R.WORKS_AT),
}
NOUN_TYPES: dict[str, EntityType | None] = {
    "company": EntityType.ORG, "firm": EntityType.ORG, "startup": EntityType.ORG, "business": EntityType.ORG,
    "organization": EntityType.ORG, "conglomerate": EntityType.ORG, "university": EntityType.ORG,
    "investor": EntityType.ORG, "person": EntityType.PERSON, "executive": EntityType.PERSON,
    "city": EntityType.LOCATION, "place": EntityType.LOCATION, "product": EntityType.PRODUCT,
    "products": EntityType.PRODUCT, "entity": None,
}
#: words that signal a relation in free-text questions (for neighbourhood fallback).
RELATION_CUES: dict[RelationType, set[str]] = {
    R.CEO_OF: {"ceo", "chief", "execu", "leads", "runs", "head"},
    R.CTO_OF: {"cto", "techn"}, R.CFO_OF: {"cfo", "finan"},
    R.FOUNDED: {"found", "start", "creat"}, R.ACQUIRED: {"acqui", "bough", "buy", "purch"},
    R.LOCATED_IN: {"where", "headq", "based", "locat", "city"}, R.INVESTED_IN: {"inves", "fund", "backe"},
    R.PARTNERED_WITH: {"partn"}, R.WORKS_AT: {"work", "join", "emplo"}, R.SUBSIDIARY_OF: {"paren", "subsi", "owns"},
    R.PRODUCES: {"produ", "devel", "make", "build"}, R.SUPPLIES: {"suppl"}, R.ALUMNUS_OF: {"study", "studi", "gradu"},
}


def _alt(words) -> str:
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


_VERB, _BASE, _ROLE, _NOUN = _alt(VERBS), _alt(BASE_VERBS), _alt(ROLES), _alt(NOUN_TYPES)
_LOC = r"(?:headquartered|based|located)"

# --------------------------------------------------------------------------- logical forms


@dataclass(frozen=True)
class EntityRef:
    node: str
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Hop:
    """Answer set = subjects (want='subj') or objects (want='obj') of ``relation``
    edges whose other end is in the inner expression's answer set."""

    relation: RelationType
    want: str
    inner: "Expr"
    type_filter: EntityType | None = None

    def __str__(self) -> str:
        return f"{self.want}[{self.relation.value}]({self.inner})"

    @property
    def depth(self) -> int:
        return 1 + (self.inner.depth if isinstance(self.inner, Hop) else 0)


Expr = Union[EntityRef, Hop]


class QuestionParser:
    """Backtracking template parser from questions to :data:`Expr` trees."""

    def __init__(self, kg: KnowledgeGraph, fuzzy_cutoff: float = 0.88):
        self.kg = kg
        self.fuzzy_cutoff = fuzzy_cutoff

    def parse(self, question: str) -> Expr | None:
        q = re.sub(r"\s+", " ", question.strip().rstrip("?.!").strip()).lower()
        m = re.fullmatch(rf"(?:who|what|which \w+) (?:is|was|are|were) the ({_ROLE}) of (.+)", q)
        if m and (e := self._role(m.group(1), m.group(2))):
            return e
        m = re.fullmatch(rf"where (?:is|was) (.+?) {_LOC}(?: in)?", q)
        if m and (inner := self.parse_expr(m.group(1))):
            return Hop(R.LOCATED_IN, "obj", inner, EntityType.LOCATION)
        m = re.fullmatch(rf"(?:in )?(?:which|what) city (?:is|was) (.+?) {_LOC}(?: in)?", q)
        if m and (inner := self.parse_expr(m.group(1))):
            return Hop(R.LOCATED_IN, "obj", inner, EntityType.LOCATION)
        m = re.fullmatch(rf"(?:who|(?:which|what) ({_NOUN})) ({_VERB}) (.+)", q)
        if m and (inner := self.parse_expr(m.group(3))):
            rel, inv = VERBS[m.group(2)]
            return Hop(rel, "obj" if inv else "subj", inner, NOUN_TYPES.get(m.group(1) or "", None))
        m = re.fullmatch(rf"(?:what|which)(?: ({_NOUN}))? did (.+?) ({_BASE})", q)
        if m and (inner := self.parse_expr(m.group(2))):
            rel, inv = BASE_VERBS[m.group(3)]
            return Hop(rel, "subj" if inv else "obj", inner, NOUN_TYPES.get(m.group(1) or "", None))
        m = re.fullmatch(r"where (?:did|does|do) (.+?) (study|work)", q)
        if m and (inner := self.parse_expr(m.group(1))):
            return Hop(R.ALUMNUS_OF if m.group(2) == "study" else R.WORKS_AT, "obj", inner)
        return None

    # -------------------------------------------------------------- expressions
    def _role(self, role: str, rest: str) -> Hop | None:
        inner = self.parse_expr(rest)
        if inner is None:
            return None
        want, rel = ROLES[role]
        return Hop(rel, want, inner)

    def link(self, text: str) -> EntityRef | None:
        """Resolve an entity mention (exact alias, then fuzzy)."""
        key = normalize_name(text)
        node = self.kg.alias_index.get(key)
        if node is None and key:
            close = get_close_matches(key, list(self.kg.alias_index), n=1, cutoff=self.fuzzy_cutoff)
            node = self.kg.alias_index[close[0]] if close else None
        if node is None or node not in self.kg.g:
            return None
        return EntityRef(node, self.kg.name(node))

    def parse_expr(self, s: str) -> Expr | None:
        s = s.strip()
        if not s:
            return None
        if not s.startswith("the ") or len(s.split()) <= 4:
            ref = self.link(s)
            if ref is not None:
                return ref
        attempts: list[tuple[str, callable]] = [
            (rf"the ({_ROLE}) of (.+)", lambda m: self._role(m.group(1), m.group(2))),
            (rf"(.+?)'s ({_ROLE})", lambda m: self._role(m.group(2), m.group(1))),
            (rf"the ({_NOUN}) (?:that|which|who) ({_VERB}) (.+)", self._active_subject),
            (rf"the ({_NOUN}) (?:(?:that|which|who) (?:was|were|is|has been) )?({_VERB}) by (.+)", self._passive),
            (rf"the ({_NOUN}) (?:that|which|whom) (.+) ({_VERB})", self._active_object),
            (rf"the ({_NOUN}) whose ({_ROLE}) (?:is|was) (?:in )?(.+)", self._whose),
            (rf"the ({_NOUN}) (?:(?:that|which) is )?{_LOC} in (.+)", self._located),
        ]
        for pattern, build in attempts:
            m = re.fullmatch(pattern, s)
            if m:
                expr = build(m)
                if expr is not None:
                    return expr
        return self.link(s)

    def _active_subject(self, m: re.Match) -> Hop | None:
        inner = self.parse_expr(m.group(3))
        if inner is None:
            return None
        rel, inv = VERBS[m.group(2)]
        return Hop(rel, "obj" if inv else "subj", inner, NOUN_TYPES[m.group(1)])

    def _passive(self, m: re.Match) -> Hop | None:
        inner = self.parse_expr(m.group(3))
        if inner is None:
            return None
        rel, inv = VERBS[m.group(2)]
        return Hop(rel, "subj" if inv else "obj", inner, NOUN_TYPES[m.group(1)])

    def _active_object(self, m: re.Match) -> Hop | None:
        inner = self.parse_expr(m.group(2))
        if inner is None:
            return None
        rel, inv = VERBS[m.group(3)]
        return Hop(rel, "subj" if inv else "obj", inner, NOUN_TYPES[m.group(1)])

    def _whose(self, m: re.Match) -> Hop | None:
        inner = self.parse_expr(m.group(3))
        if inner is None:
            return None
        want, rel = ROLES[m.group(2)]
        return Hop(rel, "obj" if want == "subj" else "subj", inner, NOUN_TYPES[m.group(1)])

    def _located(self, m: re.Match) -> Hop | None:
        inner = self.parse_expr(m.group(2))
        if inner is None:
            return None
        return Hop(R.LOCATED_IN, "subj", inner, NOUN_TYPES[m.group(1)])


# --------------------------------------------------------------------------- results


class PathStep(BaseModel):
    """One traversed edge, with provenance."""

    subject: str
    relation: str
    object: str
    weight: float
    source_chunks: list[str]
    evidence: str = ""

    def __str__(self) -> str:
        return f"{self.subject} -[{self.relation}]-> {self.object}"


class LocalAnswer(BaseModel):
    question: str
    answer: str | None
    candidates: list[str] = Field(default_factory=list)
    paths: list[list[PathStep]] = Field(default_factory=list)
    logical_form: str | None = None
    method: str = "graph-traversal"
    supporting_chunks: list[str] = Field(default_factory=list)

    @property
    def explanation(self) -> str:
        if not self.paths:
            return "no supporting path found"
        return " ; ".join(str(s) for s in self.paths[0]) + f"  => {self.answer}"


class GlobalPoint(BaseModel):
    community: str
    text: str
    score: float


class GlobalAnswer(BaseModel):
    question: str
    answer: str
    points: list[GlobalPoint] = Field(default_factory=list)
    communities_used: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- local search

ANSWER_SYSTEM = "Answer the question using only the knowledge-graph facts and source passages provided. Be brief."
ANSWER_PROMPT = """Question: {question}

Knowledge-graph paths:
{paths}

Source passages:
{passages}

Answer with the entity name only."""


class LocalSearch:
    """Entity-anchored multi-hop traversal with path explanations."""

    def __init__(self, kg: KnowledgeGraph, llm: LLM | None = None, max_hops: int = 2, max_paths: int = 5):
        self.kg = kg
        self.llm = llm
        self.parser = QuestionParser(kg)
        self.max_hops = max_hops
        self.max_paths = max_paths

    # -------------------------------------------------------------- execution
    def execute(self, expr: Expr) -> list[tuple[str, list[PathStep]]]:
        """Evaluate a logical form; returns (node, path) pairs, best path first."""
        if isinstance(expr, EntityRef):
            return [(expr.node, [])]
        best: dict[str, list[PathStep]] = {}
        for node, path in self.execute(expr.inner):
            visited = {node} | {self.kg.find(s.subject) for s in path} | {self.kg.find(s.object) for s in path}
            edges = self.kg.edges_to(node, expr.relation) if expr.want == "subj" else self.kg.edges_from(node, expr.relation)
            for u, v, d in edges:
                target = u if expr.want == "subj" else v
                if target in visited:
                    continue
                if expr.type_filter is not None and self.kg.type(target) not in (expr.type_filter, EntityType.OTHER):
                    continue
                step = PathStep(
                    subject=self.kg.name(u), relation=expr.relation.value, object=self.kg.name(v),
                    weight=float(d["weight"]), source_chunks=list(d["source_chunks"]),
                    evidence=d["evidence"][0] if d["evidence"] else "",
                )
                new_path = path + [step]
                if target not in best or self._path_score(new_path) > self._path_score(best[target]):
                    best[target] = new_path
        return sorted(best.items(), key=lambda kv: (-self._path_score(kv[1]), self.kg.name(kv[0])))

    @staticmethod
    def _path_score(path: list[PathStep]) -> float:
        return min((s.weight for s in path), default=0.0)

    def search(self, question: str) -> LocalAnswer:
        expr = self.parser.parse(question)
        answer: LocalAnswer | None = None
        if expr is not None:
            results = self.execute(expr)
            if results:
                answer = LocalAnswer(
                    question=question,
                    answer=self.kg.name(results[0][0]),
                    candidates=[self.kg.name(n) for n, _ in results],
                    paths=[p for _, p in results[: self.max_paths]],
                    logical_form=str(expr),
                    method="graph-traversal",
                )
        if answer is None:
            answer = self.neighborhood_search(question)
            if expr is not None:
                answer.logical_form = str(expr)
        answer.supporting_chunks = sorted({c for p in answer.paths for s in p for c in s.source_chunks})
        if self.llm is not None:
            answer = self._llm_answer(answer)
        return answer

    # -------------------------------------------------------------- fallback
    def link_mentions(self, question: str) -> list[str]:
        """Entities mentioned in free text (see :func:`link_mentions`)."""
        return link_mentions(self.kg, question)

    def neighborhood_search(self, question: str) -> LocalAnswer:
        """Score edges within ``max_hops`` of the question's entities by cue overlap."""
        anchors = self.link_mentions(question)
        if not anchors:
            return LocalAnswer(question=question, answer=None, method="neighborhood")
        stems = keyword_stems(question, prefix=5) | set(re.findall(r"\b(ceo|cto|cfo)\b", question.lower()))
        want_type = _expected_type(question)
        frontier = [(a, []) for a in anchors]
        seen = set(anchors)
        scored: list[tuple[float, str, list[PathStep]]] = []
        for _hop in range(self.max_hops):
            nxt = []
            for node, path in frontier:
                for u, v, d in list(self.kg.edges_from(node)) + list(self.kg.edges_to(node)):
                    other = v if u == node else u
                    if other in seen:
                        continue
                    rel = d["relation"]
                    cue = len(RELATION_CUES.get(rel, set()) & stems)
                    step = PathStep(
                        subject=self.kg.name(u), relation=rel.value, object=self.kg.name(v),
                        weight=float(d["weight"]), source_chunks=list(d["source_chunks"]),
                        evidence=d["evidence"][0] if d["evidence"] else "",
                    )
                    new_path = path + [step]
                    type_bonus = 1.0 if want_type is not None and self.kg.type(other) == want_type else 0.0
                    score = 2.0 * cue + type_bonus + 0.1 * float(d["weight"]) - 0.5 * len(path)
                    scored.append((score, other, new_path))
                    nxt.append((other, new_path))
            seen |= {n for n, _ in nxt}
            frontier = nxt
        scored.sort(key=lambda x: (-x[0], self.kg.name(x[1])))
        ranked = list(dict.fromkeys(n for _, n, _ in scored))
        return LocalAnswer(
            question=question,
            answer=self.kg.name(ranked[0]) if ranked else None,
            candidates=[self.kg.name(n) for n in ranked[:5]],
            paths=[p for _, _, p in scored[: self.max_paths]],
            method="neighborhood",
        )

    def _llm_answer(self, answer: LocalAnswer) -> LocalAnswer:
        assert self.llm is not None
        paths = "\n".join(f"- {' ; '.join(str(s) for s in p)}" for p in answer.paths) or "- (none)"
        passages = "\n".join(
            f"[{c}] {self.kg.chunk_text.get(c, '')}" for c in answer.supporting_chunks
        ) or "(none)"
        resp = self.llm.complete(
            ANSWER_PROMPT.format(question=answer.question, paths=paths, passages=passages),
            system=ANSWER_SYSTEM, max_tokens=300,
        )
        return answer.model_copy(update={"answer": resp.text.strip(), "method": answer.method + "+llm"})


def link_mentions(kg: KnowledgeGraph, text: str) -> list[str]:
    """Entity ids mentioned in free text: longest alias matches on word boundaries."""
    q = " " + normalize_name(text) + " "
    found: list[tuple[int, int, str]] = []
    for alias, node in kg.alias_index.items():
        if len(alias) < 3 or node not in kg.g:
            continue
        for m in re.finditer(rf"(?<= ){re.escape(alias)}(?= )", q):
            found.append((m.start(), m.end(), node))
    found.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
    chosen: list[tuple[int, int, str]] = []
    for a, b, node in found:
        if all(b <= c or a >= d for c, d, _ in chosen):
            chosen.append((a, b, node))
    return list(dict.fromkeys(n for _, _, n in sorted(chosen)))


def _expected_type(question: str) -> EntityType | None:
    q = question.lower()
    if q.startswith("where") or "which city" in q or "what city" in q:
        return EntityType.LOCATION
    if q.startswith("who"):
        return EntityType.PERSON
    if "product" in q:
        return EntityType.PRODUCT
    if "company" in q or "firm" in q:
        return EntityType.ORG
    return None


# --------------------------------------------------------------------------- global search

MAP_SYSTEM = "You extract the points of a community report that help answer a question."
MAP_PROMPT = """Question: {question}

Community report:
{summary}

Return JSON {{"points": [{{"text": str, "score": int 0-100}}]}} with the points relevant to the question
(empty list if none)."""
REDUCE_SYSTEM = "You synthesize analyst notes into a direct, well-organized answer."
REDUCE_PROMPT = """Question: {question}

Notes from community reports, most relevant first:
{points}

Write the final answer."""


class GlobalSearch:
    """Map-reduce question answering over community summaries."""

    def __init__(
        self,
        kg: KnowledgeGraph,
        communities: list[Community],
        level: int = 0,
        llm: LLM | None = None,
        embedder: HashingEmbedder | None = None,
        top_points: int = 8,
        min_score: float = 0.12,
        relative_cutoff: float = 0.6,
    ):
        self.kg = kg
        self.communities = [c for c in communities if c.level == level] or communities
        self.llm = llm
        self.embedder = embedder or HashingEmbedder()
        self.top_points = top_points
        self.min_score = min_score
        self.relative_cutoff = relative_cutoff

    def _map_offline(self, question: str, community: Community) -> list[GlobalPoint]:
        """Score the community's facts and source sentences against the question.

        score = 0.4 * embedding similarity + 0.4 * keyword-stem overlap
                + 0.2 if the question cues the relation the point expresses
                + 0.2 if the point names an entity the question mentions
        """
        candidates = self._candidates(community)
        if not candidates:
            return []
        texts = [t for t, _ in candidates]
        q_vec = self.embedder.embed([question])[0]
        sims = self.embedder.embed(texts) @ q_vec
        q_stems = keyword_stems(question)
        cued = {rel for rel, cues in RELATION_CUES.items() if cues & q_stems}
        anchors = [self.kg.name(n) for n in link_mentions(self.kg, question)]
        points = []
        for (text, rel), sim in zip(candidates, sims):
            overlap = len(q_stems & keyword_stems(text)) / max(1, len(q_stems))
            score = 0.4 * float(sim) + 0.4 * overlap + (0.2 if rel in cued else 0.0)
            if any(a in text for a in anchors):
                score += 0.2
            if score >= self.min_score:
                points.append(GlobalPoint(community=community.id, text=text, score=round(score, 4)))
        return points

    def _candidates(self, community: Community) -> list[tuple[str, RelationType]]:
        """(text, relation) points: verbalised facts and evidence sentences of owned edges.

        A community owns the edges whose subject it contains. A verbalised fact
        fully covered by its own source sentence is dropped in favour of the
        sentence, which carries more context.
        """
        members = set(community.members)
        out: list[tuple[str, RelationType]] = []
        for u, v, d in sorted(self.kg.g.edges(data=True), key=lambda e: (-e[2]["weight"], e[0], e[1])):
            rel = d["relation"]
            if u not in members or (v not in members and rel == RelationType.RELATED_TO):
                continue
            sentences = [ev for ev in d["evidence"] if " " in ev]
            fact = verbalize(self.kg.name(u), rel, self.kg.name(v))
            if rel != RelationType.RELATED_TO and not any(keyword_stems(fact) <= keyword_stems(s) for s in sentences):
                out.append((fact, rel))
            out.extend((s, rel) for s in sentences)
        seen: set[str] = set()
        return [(t, r) for t, r in out if not (t in seen or seen.add(t))]

    def _map_llm(self, question: str, community: Community) -> list[GlobalPoint]:
        assert self.llm is not None
        resp = self.llm.complete(
            MAP_PROMPT.format(question=question, summary=community.summary), system=MAP_SYSTEM, json_mode=True
        )
        try:
            data = resp.json()
        except ValueError:
            return []
        items = data.get("points", []) if isinstance(data, dict) else data
        points = []
        for it in items if isinstance(items, list) else []:
            if isinstance(it, dict) and str(it.get("text", "")).strip():
                try:
                    score = float(it.get("score", 0)) / 100.0
                except (TypeError, ValueError):
                    score = 0.0
                if score > 0:
                    points.append(GlobalPoint(community=community.id, text=str(it["text"]).strip(), score=score))
        return points

    def search(self, question: str) -> GlobalAnswer:
        mapped: list[GlobalPoint] = []
        for c in self.communities:
            mapped.extend(self._map_llm(question, c) if self.llm is not None else self._map_offline(question, c))
        mapped.sort(key=lambda p: (-p.score, p.community, p.text))
        if mapped:  # keep only points reasonably close to the best one
            floor = mapped[0].score * self.relative_cutoff
            mapped = [p for p in mapped if p.score >= floor]
        # reduce: dedupe near-identical points, keep the strongest
        kept: list[GlobalPoint] = []
        kept_vecs: list[np.ndarray] = []
        kept_stems: list[set[str]] = []
        for p in mapped:
            vec = self.embedder.embed([p.text])[0]
            stems = keyword_stems(p.text)
            # drop near-duplicates and points whose content is already covered
            if any(float(vec @ kv) > 0.92 for kv in kept_vecs) or any(stems <= ks for ks in kept_stems):
                continue
            kept.append(p)
            kept_vecs.append(vec)
            kept_stems.append(stems)
            if len(kept) >= self.top_points:
                break
        used = list(dict.fromkeys(p.community for p in kept))
        if not kept:
            return GlobalAnswer(question=question, answer="No community reports are relevant to this question.")
        if self.llm is not None:
            notes = "\n".join(f"- ({p.community}, score {p.score:.2f}) {p.text}" for p in kept)
            text = self.llm.complete(REDUCE_PROMPT.format(question=question, points=notes), system=REDUCE_SYSTEM).text.strip()
        else:
            titles = {c.id: c.title for c in self.communities}
            lines = [f"- {p.text} [{titles.get(p.community, p.community)}]" for p in kept]
            text = f"Drawing on {len(used)} of {len(self.communities)} communities:\n" + "\n".join(lines)
        return GlobalAnswer(question=question, answer=text, points=kept, communities_used=used)


__all__ = [
    "EntityRef", "Hop", "QuestionParser", "LocalSearch", "LocalAnswer", "PathStep",
    "GlobalSearch", "GlobalAnswer", "GlobalPoint", "verbalize",
]
