"""Registry of the portfolio's projects (single source of truth for tooling)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # repo root
PROJECT_DIR = ROOT / "projects"


@dataclass(frozen=True)
class Project:
    id: str
    dirname: str
    package: str
    title: str
    shows: str
    base: Path | None = None  # override projects/ dir (tests, external repos)

    @property
    def path(self) -> Path:
        return (self.base or PROJECT_DIR) / self.dirname


PROJECTS: list[Project] = [
    Project("p01", "p01_context_assembler", "context_assembler", "Context Assembler Engine",
            "You master the context window, not fight it"),
    Project("p02", "p02_hybrid_retrieval", "hybrid_retrieval", "Hybrid Reranking Retrieval Stack",
            "You build retrieval that actually finds things"),
    Project("p03", "p03_graphrag", "graphrag", "GraphRAG Knowledge Pipeline",
            "You can make AI reason across connected data"),
    Project("p04", "p04_agentic_rag", "agentic_rag", "Self-Correcting Agentic RAG",
            "You can stop RAG from confidently lying"),
    Project("p05", "p05_durable_agent", "durable_agent", "Durable Agent State Machine",
            "You build agents that survive production failures"),
    Project("p06", "p06_mcp_server", "mcp_server", "Raw MCP Tool Server",
            "You understand tooling at the protocol level"),
    Project("p07", "p07_memory_system", "memory_system", "3-Tier Memory System",
            "You can build agents that truly remember"),
    Project("p08", "p08_agent_swarm", "agent_swarm", "Typed Multi-Agent Swarm",
            "You orchestrate swarms, not fragile single agents"),
    Project("p09", "p09_trajectory_eval", "trajectory_eval", "Trajectory Eval CI Gate",
            "You grade the journey, not just the answer"),
    Project("p10", "p10_security_proxy", "security_proxy", "Inter-Agent Security Proxy",
            "You can secure agents against each other"),
    Project("p11", "p11_semantic_gateway", "semantic_gateway", "Semantic Cache + Routing Gateway",
            "You can slash inference costs by 40-60%"),
]
