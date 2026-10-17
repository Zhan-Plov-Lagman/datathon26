"""Local, deterministic evaluation; no model calls or dataset lookup."""
from .engine import RuleInputError, evaluate, load_registry

__all__ = ["RuleInputError", "evaluate", "load_registry"]
