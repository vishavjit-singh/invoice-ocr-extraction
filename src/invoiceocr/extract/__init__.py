"""Extractors behind one interface: ``rules`` (default), ``llm`` and ``hybrid``."""
from __future__ import annotations

from .base import ExtractionResult, Extractor
from .hybrid import HybridExtractor
from .llm import LLMError, LLMExtractor
from .rules import RuleExtractor

NAMES = ("rules", "hybrid", "llm")


def get_extractor(name: str = "rules", params: dict | None = None) -> Extractor:
    params = params or {}
    llm_p = params.get("llm", {})
    if name == "rules":
        return RuleExtractor()
    llm = LLMExtractor(model=llm_p.get("model", "gpt-4o-mini"), timeout=llm_p.get("timeout_s", 60))
    if name == "llm":
        return llm
    if name == "hybrid":
        thr = params.get("routing", {}).get("llm_fallback_confidence", 0.7)
        return HybridExtractor(RuleExtractor(), llm, thr)
    raise ValueError(f"unknown extractor {name!r}; choose from {NAMES}")


__all__ = ["ExtractionResult", "Extractor", "HybridExtractor", "LLMError", "LLMExtractor", "RuleExtractor", "NAMES",
           "get_extractor"]
