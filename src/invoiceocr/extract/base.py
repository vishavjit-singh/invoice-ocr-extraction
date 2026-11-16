"""The extractor interface: OCR page in, draft invoice with per-field confidence out."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..ocr import OcrPage
from ..schema import InvoiceDraft


@dataclass
class ExtractionResult:
    draft: InvoiceDraft
    confidence: dict[str, float]                       # 0-1 per field, plus "line_items"
    extractor: str
    evidence: dict[str, list[tuple]] = field(default_factory=dict)   # field → word boxes the value came from
    llm_fields: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class Extractor(Protocol):
    name: str

    def extract(self, page: OcrPage) -> ExtractionResult: ...
