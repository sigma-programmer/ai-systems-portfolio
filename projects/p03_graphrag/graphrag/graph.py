"""Knowledge graph on networkx with full provenance on nodes and edges."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import networkx as nx

from .models import RELATION_SIGNATURES, SYMMETRIC_RELATIONS, Entity, EntityType, ExtractionResult, RelationType
from .resolution import Resolution
from .text import normalize_name


def _signature_ok(rel: RelationType, s_type: EntityType, o_type: EntityType) -> bool:
    subj_ok, obj_ok = RELATION_SIGNATURES[rel]
    if subj_ok is not None and s_type != EntityType.OTHER and s_type not in subj_ok:
        return False
    if obj_ok is not None and o_type != EntityType.OTHER and o_type not in obj_ok:
        return False
    return True


class KnowledgeGraph:
    """A typed multi-relational graph.

    Nodes are entity ids with ``name``, ``type``, ``aliases``,
    ``mention_count`` and ``source_chunks``. Edges are keyed by relation type
    (``MultiDiGraph`` key) and carry ``weight``, ``source_chunks`` and the
    ``evidence`` sentences they were extracted from. Re-extracting the same
    sentence (e.g. from overlapping chunks) never double-counts weight.
    """

    def __init__(self) -> None:
        self.g: nx.MultiDiGraph = nx.MultiDiGraph()
        self.alias_index: dict[str, str] = {}
        self.chunk_text: dict[str, str] = {}
        self.rejected: list[tuple[str, str, str]] = []

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        results: list[ExtractionResult],
        resolution: Resolution,
        chunk_text: dict[str, str] | None = None,
    ) -> "KnowledgeGraph":
        kg = cls()
        kg.alias_index = dict(resolution.alias_index)
        kg.chunk_text = dict(chunk_text or {})
        for ent in resolution.entities.values():
            kg.add_entity(ent)
        for res in results:
            for rel in res.relations:
                s, o = resolution.lookup(rel.subject), resolution.lookup(rel.object)
                if s is None or o is None or s == o:
                    continue
                relation = rel.relation
                if not _signature_ok(relation, kg.g.nodes[s]["type"], kg.g.nodes[o]["type"]):
                    kg.rejected.append((kg.name(s), relation.value, kg.name(o)))
                    relation = RelationType.RELATED_TO
                kg.add_relation(s, relation, o, rel.chunk_id, rel.evidence, rel.confidence)
        return kg

    def add_entity(self, ent: Entity) -> None:
        self.g.add_node(
            ent.id,
            name=ent.name,
            type=ent.type,
            aliases=list(ent.aliases),
            mention_count=ent.mention_count,
            source_chunks=list(ent.source_chunks),
        )
        self.alias_index.setdefault(normalize_name(ent.name), ent.id)
        for a in ent.aliases:
            self.alias_index.setdefault(normalize_name(a), ent.id)

    def add_relation(
        self,
        src: str,
        relation: RelationType,
        dst: str,
        chunk_id: str,
        evidence: str = "",
        weight: float = 1.0,
    ) -> None:
        if relation in SYMMETRIC_RELATIONS and src > dst:
            src, dst = dst, src  # one canonical direction for symmetric relations
        key = relation.value
        if self.g.has_edge(src, dst, key):
            data = self.g.edges[src, dst, key]
            ev_key = evidence or chunk_id
            if ev_key in data["evidence"] or (not evidence and chunk_id in data["source_chunks"]):
                return
            data["evidence"].append(ev_key)
            data["weight"] += weight
            if chunk_id not in data["source_chunks"]:
                data["source_chunks"].append(chunk_id)
        else:
            self.g.add_edge(
                src, dst, key=key, relation=relation, weight=weight,
                source_chunks=[chunk_id], evidence=[evidence or chunk_id],
            )

    # ------------------------------------------------------------------ access
    def name(self, node: str) -> str:
        return self.g.nodes[node]["name"]

    def type(self, node: str) -> EntityType:
        return self.g.nodes[node]["type"]

    def find(self, surface: str) -> str | None:
        """Entity id for a surface form (any alias, case/suffix-insensitive)."""
        return self.alias_index.get(normalize_name(surface))

    def edges_from(self, node: str, relation: RelationType | None = None) -> Iterator[tuple[str, str, dict]]:
        """Edges where ``node`` is the subject (plus symmetric edges in either direction)."""
        for u, v, k, d in self.g.out_edges(node, keys=True, data=True):
            if relation is None or k == relation.value:
                yield u, v, d
        for u, v, k, d in self.g.in_edges(node, keys=True, data=True):
            if d["relation"] in SYMMETRIC_RELATIONS and (relation is None or k == relation.value):
                yield v, u, d

    def edges_to(self, node: str, relation: RelationType | None = None) -> Iterator[tuple[str, str, dict]]:
        """Edges where ``node`` is the object (plus symmetric edges in either direction)."""
        for u, v, k, d in self.g.in_edges(node, keys=True, data=True):
            if relation is None or k == relation.value:
                yield u, v, d
        for u, v, k, d in self.g.out_edges(node, keys=True, data=True):
            if d["relation"] in SYMMETRIC_RELATIONS and (relation is None or k == relation.value):
                yield v, u, d

    def triples(self, typed_only: bool = True) -> set[tuple[str, str, str]]:
        """(subject name, relation, object name) for every edge."""
        out = set()
        for u, v, k in self.g.edges(keys=True):
            if typed_only and k == RelationType.RELATED_TO.value:
                continue
            out.add((self.name(u), k, self.name(v)))
        return out

    def to_undirected_weighted(self, typed_boost: float = 2.0) -> nx.Graph:
        """Collapse to a simple weighted graph for community detection.

        Typed edges count ``typed_boost`` times their weight: an explicit
        "X acquired Y" is stronger evidence of relatedness than co-occurrence.
        """
        ug = nx.Graph()
        ug.add_nodes_from(self.g.nodes)
        for u, v, k, d in self.g.edges(keys=True, data=True):
            w = d["weight"] * (1.0 if k == RelationType.RELATED_TO.value else typed_boost)
            if ug.has_edge(u, v):
                ug[u][v]["weight"] += w
            else:
                ug.add_edge(u, v, weight=w)
        return ug

    def stats(self) -> dict[str, Any]:
        typed = sum(1 for _, _, k in self.g.edges(keys=True) if k != RelationType.RELATED_TO.value)
        types: dict[str, int] = {}
        for _, d in self.g.nodes(data=True):
            types[d["type"].value] = types.get(d["type"].value, 0) + 1
        return {
            "nodes": self.g.number_of_nodes(),
            "edges": self.g.number_of_edges(),
            "typed_edges": typed,
            "cooccurrence_edges": self.g.number_of_edges() - typed,
            "node_types": dict(sorted(types.items())),
        }

    # ------------------------------------------------------------------ (de)serialisation
    def to_dict(self) -> dict[str, Any]:
        nodes = [
            {"id": n, **{k: (v.value if isinstance(v, EntityType) else v) for k, v in d.items()}}
            for n, d in sorted(self.g.nodes(data=True))
        ]
        edges = [
            {
                "source": u, "target": v, "relation": k, "weight": round(d["weight"], 4),
                "source_chunks": list(d["source_chunks"]), "evidence": list(d["evidence"]),
            }
            for u, v, k, d in sorted(self.g.edges(keys=True, data=True), key=lambda e: (e[0], e[1], e[2]))
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "alias_index": dict(sorted(self.alias_index.items())),
            "chunks": dict(sorted(self.chunk_text.items())),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeGraph":
        kg = cls()
        for n in data["nodes"]:
            attrs = {k: v for k, v in n.items() if k != "id"}
            attrs["type"] = EntityType(attrs["type"])
            kg.g.add_node(n["id"], **attrs)
        for e in data["edges"]:
            kg.g.add_edge(
                e["source"], e["target"], key=e["relation"], relation=RelationType(e["relation"]),
                weight=e["weight"], source_chunks=list(e["source_chunks"]), evidence=list(e["evidence"]),
            )
        kg.alias_index = dict(data.get("alias_index", {}))
        kg.chunk_text = dict(data.get("chunks", {}))
        return kg

    def save_json(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        payload = self.to_dict()
        if extra:
            payload.update(extra)
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load_json(cls, path: str | Path) -> "KnowledgeGraph":
        return cls.from_dict(json.loads(Path(path).read_text()))
