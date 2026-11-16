"""The LLM path, tested offline with a fake transport and a hand-written response in the Chat Completions shape.

No test here (and nothing in `make demo` or CI) calls the OpenAI API.
"""
import json
from pathlib import Path

import pytest

from invoiceocr.extract.hybrid import HybridExtractor, weak_fields
from invoiceocr.extract.llm import LLMError, LLMExtractor, build_request, parse_response, strict_json_schema
from invoiceocr.extract.rules import RuleExtractor
from invoiceocr.ocr import OcrPage, Word

FIX = Path(__file__).parent / "fixtures"
RESPONSE = json.loads((FIX / "chat_completion_invoice.json").read_text())


def page_from_text(text: str, low_conf_words=()) -> OcrPage:
    """Lay out 'a | b | c' lines as word boxes: segments at fixed columns, 40 px per line."""
    words, cols = [], [70, 620, 900, 1100]
    for i, line in enumerate(text.strip().splitlines()):
        for col, seg in zip(cols, line.split("  |  ")):
            x = col
            for tok in seg.split():
                w = 11 * len(tok)
                words.append(Word(tok, x, 100 + 40 * i, x + w, 120 + 40 * i, 40.0 if tok in low_conf_words else 95.0))
                x += w + 8
    return OcrPage(words, 1240, 1754, "tesseract", 0.5)


class FakeTransport:
    def __init__(self, response=RESPONSE, error=None):
        self.response, self.error, self.calls = response, error, []

    def __call__(self, url, headers, body, timeout):
        self.calls.append(body)
        if self.error:
            raise self.error
        return self.response


def test_strict_schema_meets_structured_output_rules():
    schema = strict_json_schema()

    def objects(node):
        if isinstance(node, dict):
            if "properties" in node:
                yield node
            for v in node.values():
                yield from objects(v)
        elif isinstance(node, list):
            for v in node:
                yield from objects(v)

    objs = list(objects(schema))
    assert len(objs) == 2                                            # InvoiceDraft and LineItemDraft
    for o in objs:
        assert o["additionalProperties"] is False and set(o["required"]) == set(o["properties"])
    assert "default" not in json.dumps(schema)
    req = build_request("text", "gpt-4o-mini")
    assert req["response_format"]["json_schema"]["strict"] is True and req["temperature"] == 0


def test_llm_extractor_parses_and_grounds_values():
    text = (FIX / "ocr_lines_harborview.txt").read_text()
    fake = FakeTransport()
    r = LLMExtractor(api_key="test-key", transport=fake).extract(page_from_text(text))
    assert r.draft.invoice_number == "HA-07874" and r.draft.line_items[0].quantity == 22.5
    assert r.confidence["invoice_number"] == 0.9 and r.confidence["total"] == 0.9
    assert "HA-07874" in fake.calls[0]["messages"][1]["content"]
    # a value that is nowhere in the OCR text (possibly invented) gets a low grounding score
    invented = json.loads(json.dumps(RESPONSE))
    content = json.loads(invented["choices"][0]["message"]["content"])
    invented["choices"][0]["message"]["content"] = json.dumps({**content, "invoice_number": "HA-99999"})
    r2 = LLMExtractor(api_key="k", transport=FakeTransport(invented)).extract(page_from_text(text))
    assert r2.confidence["invoice_number"] == 0.4


def test_llm_errors_are_explicit():
    with pytest.raises(LLMError):
        LLMExtractor(api_key=None).extract(page_from_text("x"))
    with pytest.raises(LLMError, match="refused"):
        parse_response({"choices": [{"message": {"refusal": "no", "content": None}}]})
    with pytest.raises(LLMError, match="schema"):
        parse_response({"choices": [{"message": {"content": '{"total": "lots"}'}}]})


def test_hybrid_asks_the_llm_only_for_weak_fields(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    text = (FIX / "ocr_lines_harborview.txt").read_text()
    page = page_from_text(text, low_conf_words={"HA-07874"})
    rules = RuleExtractor().extract(page)
    assert weak_fields(rules, 0.7) == ["invoice_number"]
    fake = FakeTransport()
    r = HybridExtractor(llm=LLMExtractor(api_key="k", transport=fake)).extract(page)
    assert r.llm_fields == ["invoice_number"] and len(fake.calls) == 1 and r.draft.total == 4280.85
    # without a key, or when the call fails, the rule result comes back with a note saying why
    r = HybridExtractor(llm=LLMExtractor(api_key=None)).extract(page)
    assert r.llm_fields == [] and r.notes[-1] == "llm_skipped:no_api_key:invoice_number"
    r = HybridExtractor(llm=LLMExtractor(api_key="k", transport=FakeTransport(error=LLMError("HTTP 500")))).extract(
        page)
    assert r.notes[-1].startswith("llm_failed")
    clean = page_from_text(text)
    fake = FakeTransport()
    HybridExtractor(llm=LLMExtractor(api_key="k", transport=fake)).extract(clean)
    assert fake.calls == []                                           # nothing weak: no LLM call at all
