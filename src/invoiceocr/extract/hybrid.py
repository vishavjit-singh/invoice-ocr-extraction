"""Rules first, LLM only for what the rules could not read well.

The rule extractor runs on every page (free, fast, explainable). Fields that are missing, below the confidence
threshold, or part of a failed arithmetic check are the *weak* fields. Only if there are weak fields is the LLM
called, and only those fields are taken from its answer. Without an API key the hybrid returns the rule result
and lists the weak fields in ``notes`` so the caller can see what an LLM call would have been asked for.
"""
from __future__ import annotations

from ..ocr import OcrPage
from ..schema import SCALAR_FIELDS, cross_field_problems
from .base import ExtractionResult
from .llm import LLMError, LLMExtractor
from .rules import RuleExtractor

ARITHMETIC_FIELDS = {"line_amount_mismatch": ["line_items"], "line_sum_mismatch": ["line_items", "subtotal"],
                     "total_mismatch": ["subtotal", "tax", "total"]}


def weak_fields(result: ExtractionResult, threshold: float) -> list[str]:
    d = result.draft
    weak = []
    for f in SCALAR_FIELDS:
        if f == "due_date" and getattr(d, f) is None and result.confidence.get(f, 0) > 0:
            continue                                     # "Due on receipt": read fine, there is no date
        if getattr(d, f) in (None, "") and f != "due_date" or result.confidence.get(f, 0.0) < threshold:
            weak.append(f)
    items_bad = not d.line_items or any(None in (li.quantity, li.unit_price, li.amount) for li in d.line_items)
    if items_bad or result.confidence.get("line_items", 0.0) < threshold:
        weak.append("line_items")
    for code, _ in cross_field_problems(d.model_dump()):
        for f in ARITHMETIC_FIELDS.get(code, []):
            if f not in weak:
                weak.append(f)
    return weak


class HybridExtractor:
    name = "hybrid"

    def __init__(self, rules: RuleExtractor | None = None, llm: LLMExtractor | None = None, threshold: float = 0.7):
        self.rules = rules or RuleExtractor()
        self.llm = llm or LLMExtractor()
        self.threshold = threshold

    def extract(self, page: OcrPage) -> ExtractionResult:
        base = self.rules.extract(page)
        base.extractor = self.name
        weak = weak_fields(base, self.threshold)
        if not weak:
            return base
        if not self.llm.available:
            base.notes.append("llm_skipped:no_api_key:" + ",".join(weak))
            return base
        try:
            llm = self.llm.extract(page)
        except LLMError as e:
            base.notes.append(f"llm_failed:{e}")
            return base
        values = base.draft.model_dump()
        for f in weak:
            new = getattr(llm.draft, f)
            if new in (None, [], ""):
                continue
            values[f] = [li.model_dump() for li in new] if f == "line_items" else new
            base.confidence[f] = llm.confidence.get(f, 0.0)
            base.llm_fields.append(f)
            base.evidence.pop(f, None)
        base.draft = type(base.draft).model_validate(values)
        return base
