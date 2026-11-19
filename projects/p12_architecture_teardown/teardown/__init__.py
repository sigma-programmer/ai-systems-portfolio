"""Public Architecture Teardown toolkit.

Reproducible benchmarks, cost-per-task analysis, topology diagrams,
ADR validation and a static deep-dive site for the whole portfolio.
"""
from .registry import PROJECTS, Project, ROOT
from .cost import PRICING, ModelPrice, LLMCall, TaskProfile, cost_of_call, compare_strategies
from .diagrams import to_mermaid, to_ascii, platform_graph
from .adr import ADR, load_adrs, validate_adr
from .runner import run_benchmark, run_all, stable_fingerprint

__all__ = [
    "PROJECTS", "Project", "ROOT",
    "PRICING", "ModelPrice", "LLMCall", "TaskProfile", "cost_of_call", "compare_strategies",
    "to_mermaid", "to_ascii", "platform_graph",
    "ADR", "load_adrs", "validate_adr",
    "run_benchmark", "run_all", "stable_fingerprint",
]
