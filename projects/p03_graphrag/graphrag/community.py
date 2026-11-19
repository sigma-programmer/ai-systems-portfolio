"""Hierarchical community detection and community summaries.

Louvain (``networkx.community.louvain_partitions``) yields successively
coarser partitions; each becomes a level of the hierarchy (level 0 = finest).
Every community is summarised either extractively (offline: central entities
by weighted degree + its strongest typed facts, verbalised) or by an LLM.
Higher-level summaries also reference their child communities.
"""
from __future__ import annotations

from collections import Counter

import networkx as nx
from pydantic import BaseModel, Field

from aikit import LLM

from .graph import KnowledgeGraph
from .models import RelationType

_VERBAL = {
    RelationType.FOUNDED: "{s} founded {o}",
    RelationType.ACQUIRED: "{s} acquired {o}",
    RelationType.CEO_OF: "{s} is the CEO of {o}",
    RelationType.CTO_OF: "{s} is the CTO of {o}",
    RelationType.CFO_OF: "{s} is the CFO of {o}",
    RelationType.LOCATED_IN: "{s} is headquartered in {o}",
    RelationType.INVESTED_IN: "{s} invested in {o}",
    RelationType.PARTNERED_WITH: "{s} partnered with {o}",
    RelationType.WORKS_AT: "{s} works at {o}",
    RelationType.SUBSIDIARY_OF: "{s} is a subsidiary of {o}",
    RelationType.PRODUCES: "{s} makes the {o}",
    RelationType.SUPPLIES: "{s} supplies {o}",
    RelationType.ALUMNUS_OF: "{s} studied at {o}",
    RelationType.RELATED_TO: "{s} is mentioned with {o}",
}


def verbalize(subject: str, relation: RelationType, obj: str) -> str:
    """Render a triple as an English sentence fragment."""
    return _VERBAL[relation].format(s=subject, o=obj)


class Community(BaseModel):
    """One community at one hierarchy level."""

    id: str
    level: int
    members: list[str]
    parent: str | None = None
    children: list[str] = Field(default_factory=list)
    title: str = ""
    summary: str = ""
    key_entities: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)


def detect_communities(kg: KnowledgeGraph, seed: int = 42, resolution: float = 1.0) -> list[Community]:
    """Run Louvain and return communities for every hierarchy level."""
    ug = kg.to_undirected_weighted()
    if ug.number_of_nodes() == 0:
        return []
    levels: list[list[set[str]]] = []
    for partition in nx.community.louvain_partitions(ug, weight="weight", resolution=resolution, seed=seed):
        part = [set(c) for c in partition]
        if not levels or {frozenset(c) for c in part} != {frozenset(c) for c in levels[-1]}:
            levels.append(part)
    communities: list[Community] = []
    by_level: list[list[Community]] = []
    for lvl, part in enumerate(levels):
        ordered = sorted(part, key=lambda c: (-len(c), min(c)))
        comms = [Community(id=f"L{lvl}-C{i}", level=lvl, members=sorted(c)) for i, c in enumerate(ordered)]
        by_level.append(comms)
        communities.extend(comms)
    for lvl in range(len(by_level) - 1):
        owner = {m: c for c in by_level[lvl + 1] for m in c.members}
        for c in by_level[lvl]:
            parent = owner[c.members[0]]
            c.parent = parent.id
            parent.children.append(c.id)
    return communities


SUMMARY_SYSTEM = "You write concise, factual summaries of clusters in a knowledge graph."
SUMMARY_PROMPT = """Summarize this community of related entities in 2-4 sentences. Mention the most
important entities and how they are connected. Use only the facts given.

Entities: {entities}
Facts:
{facts}"""


class CommunitySummarizer:
    """Summarise communities extractively (default) or with an LLM."""

    def __init__(self, llm: LLM | None = None, max_facts: int = 10, max_key_entities: int = 5):
        self.llm = llm
        self.max_facts = max_facts
        self.max_key_entities = max_key_entities

    def summarize_all(self, kg: KnowledgeGraph, communities: list[Community]) -> list[Community]:
        index = {c.id: c for c in communities}
        ug = kg.to_undirected_weighted()
        for c in sorted(communities, key=lambda c: c.level):
            self.summarize(kg, c, [index[ch] for ch in c.children], ug=ug)
        return communities

    def summarize(
        self,
        kg: KnowledgeGraph,
        community: Community,
        children: list[Community] | None = None,
        ug: nx.Graph | None = None,
    ) -> Community:
        members = set(community.members)
        sub = (ug if ug is not None else kg.to_undirected_weighted()).subgraph(members)
        strength = dict(sub.degree(weight="weight"))
        key = sorted(members, key=lambda n: (-strength.get(n, 0.0), -kg.g.nodes[n]["mention_count"], kg.name(n)))
        community.key_entities = [kg.name(n) for n in key[: self.max_key_entities]]

        facts: list[tuple[float, str]] = []
        cooc: list[tuple[float, str]] = []
        # A community owns every edge whose subject it contains, so boundary facts
        # ("Halvorsen Group acquired Zentrix Labs") appear in exactly one report.
        for u, v, k, d in kg.g.edges(keys=True, data=True):
            if u in members and (v in members or k != RelationType.RELATED_TO.value):
                rel = RelationType(k)
                item = (d["weight"], verbalize(kg.name(u), rel, kg.name(v)))
                (cooc if rel == RelationType.RELATED_TO else facts).append(item)
        facts.sort(key=lambda x: (-x[0], x[1]))
        cooc.sort(key=lambda x: (-x[0], x[1]))
        chosen = [f for _, f in facts[: self.max_facts]] or [f for _, f in cooc[:3]]
        community.facts = chosen

        types = Counter(kg.type(n).value for n in members)
        top = community.key_entities[:2]
        community.title = " & ".join(top) if top else community.id

        if self.llm is not None:
            prompt = SUMMARY_PROMPT.format(entities=", ".join(community.key_entities), facts="\n".join(f"- {f}" for f in chosen))
            community.summary = self.llm.complete(prompt, system=SUMMARY_SYSTEM, max_tokens=400).text.strip()
            return community

        type_desc = ", ".join(f"{n} {t.lower()}" for t, n in sorted(types.items(), key=lambda x: (-x[1], x[0])))
        parts = [
            f"Community '{community.title}' groups {len(members)} entities ({type_desc}).",
            f"Central entities: {', '.join(community.key_entities)}.",
        ]
        if chosen:
            parts.append("Key facts: " + "; ".join(chosen) + ".")
        if children:
            parts.append("Sub-communities: " + "; ".join(ch.title for ch in children) + ".")
        community.summary = " ".join(parts)
        return community
