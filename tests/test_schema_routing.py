import numpy as np
import pytest
from pydantic import ValidationError

from invoiceocr.extract.base import ExtractionResult
from invoiceocr.generate import make_gstin
from invoiceocr.routing import decide
from invoiceocr.schema import Invoice, InvoiceDraft, cross_field_problems, gstin_checksum, gstin_valid

GOOD = {
    "vendor_name": "Kestrel Agro Supplies", "vendor_tax_id": "", "invoice_number": "KAS-02462",
    "issue_date": "2026-09-11", "due_date": "2026-10-11", "currency": "INR",
    "line_items": [{"description": "Hybrid wheat seed, 40 kg bag", "quantity": 7, "unit_price": 2339.0,
                    "amount": 16373.0},
                   {"description": "Knapsack sprayer, 16 L", "quantity": 2.5, "unit_price": 100.0, "amount": 250.0}],
    "subtotal": 16623.0, "tax": 1994.76, "total": 18617.76,
}


@pytest.fixture
def good():
    d = dict(GOOD)
    d["vendor_tax_id"] = make_gstin(np.random.default_rng(1), "03", "Kestrel")
    return d


def result(values: dict, conf: float = 0.95, notes=None) -> ExtractionResult:
    draft = InvoiceDraft(**values)
    fields = list(InvoiceDraft.model_fields)
    return ExtractionResult(draft, {f: conf for f in fields}, "rules", notes=notes or [])


def test_gstin_checksum_and_generator_agree():
    rng = np.random.default_rng(0)
    for _ in range(50):
        g = make_gstin(rng, "27", "Neelgiri")
        assert gstin_valid(g) and g[-1] == gstin_checksum(g[:14])
        assert not gstin_valid(make_gstin(rng, "27", "Neelgiri", valid=False))


def test_valid_invoice_passes_strict_model(good):
    inv = Invoice.model_validate(good)
    assert inv.issue_date.isoformat() == "2026-09-11" and inv.total == 18617.76


@pytest.mark.parametrize("change, code", [
    ({"total": 18600.00}, "total_mismatch"),
    ({"subtotal": 16000.00, "total": 17994.76}, "line_sum_mismatch"),
    ({"due_date": "2026-09-01"}, "due_before_issue"),
    ({"due_date": "2028-10-11"}, "due_date_implausible"),
    ({"currency": "USD"}, "tax_id_currency_mismatch"),
])
def test_cross_field_rules(good, change, code):
    values = {**good, **change}
    assert code in [c for c, _ in cross_field_problems(values)]
    with pytest.raises(ValidationError) as e:
        Invoice.model_validate(values)
    assert e.value.errors()[0]["type"] == "cross_field" and code in e.value.errors()[0]["ctx"]["codes"]


def test_bad_gstin_check_character_is_its_own_error(good):
    g = good["vendor_tax_id"]
    bad = g[:14] + ("A" if g[14] != "A" else "B")
    with pytest.raises(ValidationError) as e:
        Invoice.model_validate({**good, "vendor_tax_id": bad})
    assert e.value.errors()[0]["type"] == "gstin_checksum"


def test_rules_are_skipped_when_inputs_are_missing():
    assert cross_field_problems({"subtotal": None, "tax": 5.0, "total": 10.0, "line_items": []}) == []


def test_routing_approves_a_clean_confident_invoice(good):
    d = decide(result(good))
    assert d.status == "auto_approved" and d.reasons == []


def test_low_confidence_matters_only_for_fields_nothing_cross_checks(good):
    r = result(good)
    r.confidence.update({"subtotal": 0.3, "tax": 0.3, "total": 0.3, "vendor_tax_id": 0.3})
    assert decide(r).status == "auto_approved"            # arithmetic + GSTIN checksum vouch for these
    r.confidence["invoice_number"] = 0.5
    d = decide(r)
    assert d.status == "needs_review" and [(x.code, x.field) for x in d.reasons] == [("low_confidence",
                                                                                         "invoice_number")]


def test_routing_reasons_for_missing_and_broken(good):
    d = decide(result({**good, "invoice_number": None, "total": 1.0}))
    codes = {x.code for x in d.reasons}
    assert d.status == "needs_review" and {"missing_field", "total_mismatch"} <= codes


def test_unassigned_date_note_sends_to_review(good):
    d = decide(result(good, notes=["unassigned:date:07/27/2025"]))
    assert d.status == "needs_review" and d.reasons[0].code == "unassigned_value"
