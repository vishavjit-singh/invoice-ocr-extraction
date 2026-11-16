"""LLM extractor: OpenAI Chat Completions with a strict JSON schema generated from the pydantic model.

The model sees the OCR text with the layout kept (segments separated by `|`) and must answer in the exact shape
of ``InvoiceDraft``. The call uses the standard library (urllib), so there is no SDK dependency; ``transport`` can
be swapped for a fake in tests. It only runs when ``OPENAI_API_KEY`` is set: `make demo` and CI never call it.

An LLM gives no calibrated confidence, so each returned value gets a *grounding* score instead: a value that can
be found in the OCR text scores 0.9, one that cannot (possibly invented) scores 0.4 and will be sent to review.
"""
from __future__ import annotations

import copy
import json
import os
import re
import urllib.error
import urllib.request
from typing import Callable

from ..layout import build_lines, layout_text
from ..normalize import norm_text
from ..ocr import OcrPage
from ..schema import SCALAR_FIELDS, InvoiceDraft
from .base import ExtractionResult

API_URL = "https://api.openai.com/v1/chat/completions"
SYSTEM_PROMPT = (
    "You extract fields from the OCR text of one invoice. The text keeps the page layout: one printed line per "
    "line, and ' | ' between blocks printed far apart on the same line. Copy values exactly as printed; do not "
    "compute or guess missing ones, use null instead. vendor_name and vendor_tax_id are the seller's (not the "
    "buyer's). Dates as YYYY-MM-DD: Indian (GST/INR) invoices print day first, US invoices month first. Amounts as "
    "plain numbers without currency symbols or thousands separators. tax is the sum of all tax lines (CGST + SGST, "
    "IGST, sales tax). currency is INR or USD."
)

Transport = Callable[[str, dict, dict, float], dict]


class LLMError(RuntimeError):
    pass


def strict_json_schema(model=InvoiceDraft) -> dict:
    """Pydantic JSON schema → OpenAI strict structured-output schema.

    Strict mode needs every property listed in ``required`` (optional ones stay nullable), ``additionalProperties:
    false`` on every object, and no ``default``/``title`` keywords.
    """
    schema = copy.deepcopy(model.model_json_schema())

    def fix(node):
        if isinstance(node, dict):
            node.pop("default", None)
            node.pop("title", None)
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            for v in node.values():
                fix(v)
        elif isinstance(node, list):
            for v in node:
                fix(v)

    fix(schema)
    return schema


def build_request(ocr_text: str, model: str) -> dict:
    return {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": f"OCR text:\n\n{ocr_text}"}],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "invoice", "strict": True, "schema": strict_json_schema()}},
    }


def urllib_transport(url: str, headers: dict, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise LLMError(f"OpenAI API returned HTTP {e.code}: {e.read()[:300]!r}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise LLMError(f"OpenAI API unreachable: {e}") from e


def parse_response(resp: dict) -> InvoiceDraft:
    try:
        msg = resp["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError("unexpected response shape") from e
    if msg.get("refusal"):
        raise LLMError(f"model refused: {msg['refusal']}")
    try:
        return InvoiceDraft.model_validate_json(msg["content"])
    except Exception as e:                      # pydantic ValidationError or bad JSON
        raise LLMError(f"response did not match the schema: {e}") from e


def _number_forms(x: float) -> set[str]:
    whole = f"{x:.2f}"
    return {whole, f"{x:,.2f}", whole.rstrip("0").rstrip("."), f"{x:g}"}


def grounding(field: str, value, ocr_text: str) -> float:
    """0.9 if the value can be found in the OCR text, else 0.4."""
    if value is None:
        return 0.0
    flat = ocr_text.replace(" | ", " ")
    if isinstance(value, (int, float)):
        compact = re.sub(r"(?<=\d),(?=\d)", "", flat)
        return 0.9 if any(f in flat or f.replace(",", "") in compact for f in _number_forms(float(value))) else 0.4
    if field in ("issue_date", "due_date", "currency"):
        return 0.8                               # normalised by the model; cannot be found verbatim
    return 0.9 if norm_text(str(value)) and norm_text(str(value)) in norm_text(flat) else 0.4


class LLMExtractor:
    name = "llm"

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini", timeout: float = 60.0,
                 transport: Transport | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.timeout = timeout
        self.transport = transport or urllib_transport

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def extract(self, page: OcrPage) -> ExtractionResult:
        if not self.available:
            raise LLMError("OPENAI_API_KEY is not set")
        text = layout_text(build_lines(page.words))
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        draft = parse_response(self.transport(API_URL, headers, build_request(text, self.model), self.timeout))
        conf = {f: grounding(f, getattr(draft, f), text) for f in SCALAR_FIELDS}
        items = draft.line_items
        conf["line_items"] = min((grounding("amount", li.amount, text) for li in items), default=0.0)
        return ExtractionResult(draft, conf, self.name, llm_fields=list(SCALAR_FIELDS) + ["line_items"])
