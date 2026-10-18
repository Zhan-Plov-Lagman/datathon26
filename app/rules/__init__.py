"""Local, deterministic evaluation; no model calls or dataset lookup."""
from .engine import RuleInputError, evaluate, load_registry
from .policy import POLICY_VERSION, finalize_detection

__all__ = ["RuleInputError", "evaluate", "load_registry", "POLICY_VERSION", "finalize_detection"]
