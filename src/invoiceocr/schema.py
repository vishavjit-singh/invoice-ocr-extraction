"""The invoice contract.

``InvoiceDraft`` is what an extractor produces: every field optional, nothing checked. ``Invoice`` is the strict
model an invoice must satisfy to be approved without a person looking at it: required fields, tax-id format and
GSTIN checksum, ISO dates, and cross-field arithmetic (line items, subtotal + tax = total, due date after issue).
"""
from __future__ import annotations

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

GSTIN_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GSTIN_RE = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
EIN_RE = re.compile(r"^[0-9]{2}-[0-9]{7}$")

SCALAR_FIELDS = ("vendor_name", "vendor_tax_id", "invoice_number", "issue_date", "due_date", "currency",
                 "subtotal", "tax", "total")
REQUIRED_FIELDS = ("vendor_name", "vendor_tax_id", "invoice_number", "issue_date", "currency", "subtotal", "tax",
                   "total")
MAX_PAYMENT_TERMS_DAYS = 180   # longer terms are almost always a misread year or month
MONEY_TOLERANCE = 0.011          # amounts are printed to 2 decimals, so anything above a paisa/cent is a mismatch


def gstin_checksum(first14: str) -> str:
    """Check character of a GSTIN (mod-36 over the first 14 characters, weights alternating 1, 2)."""
    total = 0
    for i, ch in enumerate(first14):
        product = GSTIN_CHARS.index(ch) * (1 if i % 2 == 0 else 2)
        total += product // 36 + product % 36
    return GSTIN_CHARS[(36 - total % 36) % 36]


def gstin_valid(value: str) -> bool:
    return bool(GSTIN_RE.match(value)) and gstin_checksum(value[:14]) == value[14]


class LineItemDraft(BaseModel):
    description: str | None = None
    quantity: float | None = None
    unit_price: float | None = None
    amount: float | None = None


class InvoiceDraft(BaseModel):
    """Extractor output. Dates are ISO strings once an extractor has normalised them."""
    model_config = ConfigDict(extra="forbid")
    vendor_name: str | None = None
    vendor_tax_id: str | None = None
    invoice_number: str | None = None
    issue_date: str | None = None
    due_date: str | None = None
    currency: str | None = None
    line_items: list[LineItemDraft] = Field(default_factory=list)
    subtotal: float | None = None
    tax: float | None = None
    total: float | None = None


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = Field(min_length=1)
    quantity: float = Field(gt=0)
    unit_price: float = Field(ge=0)
    amount: float = Field(ge=0)


class Invoice(BaseModel):
    """A validated invoice. Construction fails with one error per broken rule."""
    model_config = ConfigDict(extra="forbid")
    vendor_name: str = Field(min_length=2, max_length=120)
    vendor_tax_id: str
    invoice_number: str = Field(min_length=2, max_length=40)
    issue_date: date
    due_date: date | None = None
    currency: Literal["INR", "USD"]
    line_items: list[LineItem] = Field(min_length=1)
    subtotal: float = Field(ge=0)
    tax: float = Field(ge=0)
    total: float = Field(gt=0)

    @field_validator("vendor_tax_id")
    @classmethod
    def _tax_id(cls, v: str) -> str:
        v = v.strip().upper().replace(" ", "")
        if GSTIN_RE.match(v):
            if not gstin_valid(v):
                raise PydanticCustomError("gstin_checksum", "GSTIN check character does not match")
            return v
        if EIN_RE.match(v):
            return v
        raise PydanticCustomError("tax_id_format", "tax id is neither a GSTIN nor an EIN (NN-NNNNNNN)")

    @model_validator(mode="after")
    def _cross_field(self) -> "Invoice":
        problems = cross_field_problems(self.model_dump())
        if problems:
            codes = [c for c, _ in problems]
            raise PydanticCustomError("cross_field", "; ".join(m for _, m in problems), {"codes": codes})
        return self


def _as_date(v) -> date | None:
    if v is None or isinstance(v, date):
        return v
    try:
        return date.fromisoformat(str(v))
    except ValueError:
        return None


def cross_field_problems(v: dict) -> list[tuple[str, str]]:
    """Arithmetic and consistency rules. Works on a draft too: a rule is skipped when its inputs are missing."""
    out: list[tuple[str, str]] = []
    items = v.get("line_items") or []
    for i, li in enumerate(items):
        q, p, a = li.get("quantity"), li.get("unit_price"), li.get("amount")
        if None not in (q, p, a) and abs(q * p - a) > MONEY_TOLERANCE:
            out.append(("line_amount_mismatch", f"line {i + 1}: {q:g} × {p:.2f} ≠ {a:.2f}"))
            break
    sub, tax, total = v.get("subtotal"), v.get("tax"), v.get("total")
    amounts = [li.get("amount") for li in items]
    if items and None not in amounts and sub is not None:
        lines = round(sum(amounts), 2)
        if abs(lines - sub) > MONEY_TOLERANCE:
            out.append(("line_sum_mismatch", f"line items sum to {lines:.2f}, subtotal is {sub:.2f}"))
    if None not in (sub, tax, total) and abs(sub + tax - total) > MONEY_TOLERANCE:
        out.append(("total_mismatch", f"subtotal {sub:.2f} + tax {tax:.2f} ≠ total {total:.2f}"))
    issue, due = _as_date(v.get("issue_date")), _as_date(v.get("due_date"))
    if issue and due and due < issue:
        out.append(("due_before_issue", f"due {due} is before issue {issue}"))
    elif issue and due and (due - issue).days > MAX_PAYMENT_TERMS_DAYS:
        out.append(("due_date_implausible", f"due {due} is {(due - issue).days} days after issue {issue}"))
    tax_id, cur = v.get("vendor_tax_id"), v.get("currency")
    if tax_id and cur in ("INR", "USD"):
        is_gstin = bool(GSTIN_RE.match(str(tax_id).upper().replace(" ", "")))
        if (cur == "INR") != is_gstin:
            out.append(("tax_id_currency_mismatch", f"{cur} invoice with tax id {tax_id}"))
    return out


class Reason(BaseModel):
    code: str
    field: str | None = None
    message: str


class Decision(BaseModel):
    """What the pipeline returns for one document."""
    status: Literal["auto_approved", "needs_review"]
    reasons: list[Reason] = Field(default_factory=list)
    invoice: InvoiceDraft
    confidence: dict[str, float] = Field(default_factory=dict)
    extractor: str
    llm_fields: list[str] = Field(default_factory=list)
