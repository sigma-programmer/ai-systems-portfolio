"""Topology diagrams: networkx graph -> Mermaid / ASCII."""
from __future__ import annotations

import re

import networkx as nx


def _nid(name: str) -> str:
    return re.sub(r"\W+", "_", name).strip("_") or "n"


def to_mermaid(g: nx.DiGraph, direction: str = "LR") -> str:
    """Render a DiGraph as a Mermaid flowchart.

    Node attrs: label, group (rendered as subgraph), shape ('db' | 'round' | 'box').
    Edge attrs: label.
    """
    lines = [f"flowchart {direction}"]
    groups: dict[str | None, list[str]] = {}
    for n, d in g.nodes(data=True):
        groups.setdefault(d.get("group"), []).append(n)

    def node_decl(n):
        d = g.nodes[n]
        label = d.get("label", n).replace('"', "'")
        shape = d.get("shape", "box")
        if shape == "db":
            return f'{_nid(n)}[("{label}")]'
        if shape == "round":
            return f'{_nid(n)}(["{label}"])'
        return f'{_nid(n)}["{label}"]'

    for grp, nodes in groups.items():
        if grp is None:
            lines += [f"    {node_decl(n)}" for n in nodes]
        else:
            lines.append(f'    subgraph {_nid(grp)}["{grp}"]')
            lines += [f"        {node_decl(n)}" for n in nodes]
            lines.append("    end")
    for u, v, d in g.edges(data=True):
        lbl = d.get("label")
        arrow = f' -->|"{lbl}"| ' if lbl else " --> "
        lines.append(f"    {_nid(u)}{arrow}{_nid(v)}")
    return "\n".join(lines)


def to_ascii(g: nx.DiGraph) -> str:
    """Layered ASCII rendering (topological generations) for terminals/READMEs."""
    if not nx.is_directed_acyclic_graph(g):
        layers = [list(g.nodes)]
    else:
        layers = [sorted(gen) for gen in nx.topological_generations(g)]
    out = []
    for i, layer in enumerate(layers):
        out.append("  ".join(f"[{g.nodes[n].get('label', n)}]" for n in layer))
        if i < len(layers) - 1:
            out.append("        |")
            out.append("        v")
    edges = [f"  {g.nodes[u].get('label', u)} -> {g.nodes[v].get('label', v)}"
             + (f"  ({d['label']})" if d.get("label") else "")
             for u, v, d in g.edges(data=True)]
    return "\n".join(out + ["", "edges:"] + edges)


def platform_graph() -> nx.DiGraph:
    """How the 12 projects compose into one reference agent platform."""
    g = nx.DiGraph()
    nodes = [
        ("client", "Client / Agent", "Edge", "round"),
        ("gateway", "Semantic Gateway (p11)", "Edge", "box"),
        ("proxy", "Security Proxy (p10)", "Edge", "box"),
        ("supervisor", "Swarm Supervisor (p08)", "Orchestration", "box"),
        ("durable", "Durable State Machine (p05)", "Orchestration", "box"),
        ("assembler", "Context Assembler (p01)", "Context", "box"),
        ("memory", "3-Tier Memory (p07)", "Context", "db"),
        ("agentic", "Agentic RAG (p04)", "Knowledge", "box"),
        ("hybrid", "Hybrid Retrieval (p02)", "Knowledge", "box"),
        ("graph", "GraphRAG (p03)", "Knowledge", "db"),
        ("mcp", "MCP Tool Server (p06)", "Tools", "box"),
        ("llm", "Claude models", "Inference", "round"),
        ("evalgate", "Trajectory CI Gate (p09)", "Quality", "box"),
    ]
    for n, label, grp, shape in nodes:
        g.add_node(n, label=label, group=grp, shape=shape)
    edges = [
        ("client", "proxy", "request"), ("proxy", "supervisor", "sanitized msg"),
        ("supervisor", "durable", "checkpointed steps"), ("durable", "assembler", "step context"),
        ("assembler", "memory", "recall"), ("assembler", "agentic", "evidence"),
        ("agentic", "hybrid", "retrieve"), ("agentic", "graph", "multi-hop"),
        ("durable", "mcp", "tools/call"), ("assembler", "gateway", "prompt"),
        ("gateway", "llm", "routed call"), ("durable", "evalgate", "trajectory"),
    ]
    for u, v, lbl in edges:
        g.add_edge(u, v, label=lbl)
    return g


def swarm_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    for n, label in [("sup", "Supervisor"), ("res", "Researcher"), ("cod", "Coder"),
                     ("rv1", "Reviewer A"), ("rv2", "Reviewer B"), ("rv3", "Reviewer C"),
                     ("vote", "Consensus vote")]:
        g.add_node(n, label=label, group="Swarm" if n != "sup" else None)
    for u, v, l in [("sup", "res", "TaskSpec"), ("res", "cod", "ResearchFindings"),
                    ("cod", "rv1", "CodeArtifact"), ("cod", "rv2", "CodeArtifact"),
                    ("cod", "rv3", "CodeArtifact"), ("rv1", "vote", "ReviewVerdict"),
                    ("rv2", "vote", "ReviewVerdict"), ("rv3", "vote", "ReviewVerdict"),
                    ("vote", "sup", "accept / revise")]:
        g.add_edge(u, v, label=l)
    return g
