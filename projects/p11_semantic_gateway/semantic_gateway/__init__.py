"""Semantic cache + complexity-routing LLM gateway with cost kill-switches and fallbacks."""
from .backends import (BackendError, BackendTimeout, BackendUnavailable, RateLimited,
                       SimulatedBackend, default_backends)
from .budget import CircuitBreaker, CostTracker, SpendRateBreaker, TenantBudget
from .cache import NullCache, SemanticCache, guard_reason
from .clock import ManualClock, SystemClock
from .complexity import ComplexityAnalyzer, ComplexityReport
from .gateway import Gateway, GatewayRequest, GatewayResponse, Router
from .pricing import HAIKU, MODELS, OPUS, SONNET, ModelSpec, cost_of

__all__ = [
    "BackendError", "BackendTimeout", "BackendUnavailable", "RateLimited", "SimulatedBackend",
    "default_backends", "CircuitBreaker", "CostTracker", "SpendRateBreaker", "TenantBudget",
    "NullCache", "SemanticCache", "guard_reason", "ManualClock", "SystemClock", "ComplexityAnalyzer",
    "ComplexityReport", "Gateway", "GatewayRequest", "GatewayResponse", "Router", "HAIKU",
    "MODELS", "OPUS", "SONNET", "ModelSpec", "cost_of",
]
