"""Validate an extracted draft and decide: approve automatically, or send to a person with reasons.

A field can be trusted for one of two reasons. Either a cross-check covers it (the arithmetic ties line items,
subtotal, tax and total together; the GSTIN check character validates the tax id), or OCR read it with high
confidence. Fields that nothing can cross-check (vendor name, invoice number, dates, an EIN) need confidence.
Any missing required field, broken rule or unconfident unchecked field sends the invoice to review.
"""
from __future__ import annotations

from pydantic import ValidationError

from .extract.base import ExtractionResult
from .schema import GSTIN_RE, REQUIRED_FIELDS, Decision, Invoice, Reason, cross_field_problems, gstin_valid

ARITHMETIC_CODES = {"line_amount_mismatch", "line_sum_mismatch", "total_mismatch"}
UNCHECKED_FIELDS = ("vendor_name", "invoice_number", "issue_date", "due_date", "currency")


def decide(result: ExtractionResult, min_confidence: float = 0.7) -> Decision:
    draft = result.draft
    values = draft.model_dump()
    reasons: list[Reason] = []
    missing = [f for f in REQUIRED_FIELDS if values.get(f) in (None, "")]
    for f in missing:
        reasons.append(Reason(code="missing_field", field=f, message=f"{f} was not found"))
    if not draft.line_items:
        reasons.append(Reason(code="missing_field", field="line_items", message="no line items were found"))
    else:
        for i, li in enumerate(draft.line_items):
            gaps = [k for k in ("description", "quantity", "unit_price", "amount") if getattr(li, k) is None]
            if gaps:
                reasons.append(Reason(code="incomplete_line_item", field="line_items",
                                      message=f"line {i + 1} is missing {', '.join(gaps)}"))
                break

    # field-level rules from the strict model (format, checksum, ranges), skipping what is already reported missing
    try:
        Invoice.model_validate(values)
    except ValidationError as e:
        for err in e.errors():
            if err["type"] == "cross_field":
                continue                               # reported below, rule by rule
            loc = str(err["loc"][0]) if err["loc"] else None
            if loc in missing or loc == "line_items":
                continue
            reasons.append(Reason(code=err["type"], field=loc, message=err["msg"]))

    for note in result.notes:
        if note.startswith("unparsed:"):
            _, f, raw = note.split(":", 2)
            reasons.append(Reason(code="unparsed_field", field=f, message=f"could not read {f} from {raw!r}"))
        elif note.startswith("unassigned:"):
            _, kind, raw = note.split(":", 2)
            reasons.append(Reason(code="unassigned_value", field=None,
                                  message=f"a {kind} was printed but no field label matched it: {raw!r}"))

    problems = cross_field_problems(values)
    for code, msg in problems:
        reasons.append(Reason(code=code, field=None, message=msg))

    # which fields are vouched for by a passing cross-check
    codes = {c for c, _ in problems}
    verified: set[str] = set()
    numbers_present = all(values.get(f) is not None for f in ("subtotal", "tax", "total")) and draft.line_items and \
        all(li.amount is not None and li.quantity is not None and li.unit_price is not None for li in draft.line_items)
    if numbers_present and not (codes & ARITHMETIC_CODES):
        verified |= {"subtotal", "tax", "total", "line_items"}
    tax_id = (values.get("vendor_tax_id") or "").upper()
    if GSTIN_RE.match(tax_id) and gstin_valid(tax_id):
        verified.add("vendor_tax_id")
    for f, c in result.confidence.items():
        if f in verified or f in missing or (f == "line_items" and not draft.line_items):
            continue
        if f == "due_date" and values.get("due_date") is None and c == 0.0:
            continue                                   # no due date printed: nothing to check
        if c < min_confidence:
            reasons.append(Reason(code="low_confidence", field=f, message=f"{f} read with confidence {c:.2f}"))
    status = "auto_approved" if not reasons else "needs_review"
    return Decision(status=status, reasons=reasons, invoice=draft, confidence=result.confidence,
                    extractor=result.extractor, llm_fields=result.llm_fields)
