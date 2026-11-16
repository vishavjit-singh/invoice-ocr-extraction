import numpy as np
import pytest
from conftest import needs_tesseract

from invoiceocr.evaluate import field_correct, match_items
from invoiceocr.extract.rules import RuleExtractor, derive_missing, repair_quantity
from invoiceocr.generate import VENDOR_BY_KEY, degrade, layout, make_spec, render_image
from invoiceocr.ocr import estimate_noise, estimate_skew, ocr_document
from invoiceocr.schema import SCALAR_FIELDS


def test_text_layer_pdfs_extract_exactly_for_every_layout(pdf_per_template):
    """No OCR noise: every field and every line item of all 11 vendors / 4 layouts must come out right."""
    ex = RuleExtractor()
    for vendor, path, truth in pdf_per_template:
        page, _ = ocr_document(path)
        assert page.source == "pdf_text" and page.mean_conf == 100.0
        got = ex.extract(page).draft.model_dump()
        wrong = [f for f in SCALAR_FIELDS if not field_correct(f, truth[f], got[f])]
        assert not wrong, (vendor.key, wrong, {f: (truth[f], got[f]) for f in wrong})
        assert match_items(truth["line_items"], got["line_items"]) == len(truth["line_items"]) == \
            len(got["line_items"]), vendor.key


def test_vendor_gstin_is_not_confused_with_the_buyer_gstin(pdf_per_template):
    vendor, path, truth = next(d for d in pdf_per_template if d[0].template == "gst_classic")
    page, _ = ocr_document(path)
    assert RuleExtractor().extract(page).draft.vendor_tax_id == truth["vendor_tax_id"]


def test_rows_are_read_by_position_when_the_header_is_unreadable(pdf_per_template):
    vendor, path, truth = next(d for d in pdf_per_template if d[0].template == "us_simple")
    page, _ = ocr_document(path)
    page.words = [w for w in page.words if w.text not in ("Description", "Qty", "Unit", "Price", "Amount")]
    r = RuleExtractor().extract(page)
    assert "table header not found: rows read by position" in r.notes
    assert match_items(truth["line_items"], r.draft.model_dump()["line_items"]) == len(truth["line_items"])


def test_arithmetic_repairs_are_narrow():
    assert repair_quantity("175", 869.08, 15208.9) == 17.5                 # dropped decimal point
    assert derive_missing(None, 0.0, 1724.66, 8623.3)[:2] == (5.0, 1724.66)  # unreadable quantity
    assert derive_missing(4.0, 0.96, 2210.0, 11050.0)[2] is None           # confident read that disagrees: flag it
    assert derive_missing(None, 0.0, 3.7, 10.0)[2] is None                # not a clean quantity: do not invent


def test_deskew_recovers_a_known_rotation():
    rng = np.random.default_rng(0)
    img = render_image(layout(make_spec(rng, 0, VENDOR_BY_KEY["saffron"], None)))
    spec = {"rotation": 2.4, "noise": 0, "blur": 0.0, "jpeg": None, "contrast": 1.0, "downscale": 1.0}
    rotated, applied = degrade(img, spec, rng)
    correction = estimate_skew(np.asarray(rotated))
    assert abs(applied["rotation_deg"] + correction) < 0.3
    assert estimate_noise(np.asarray(img.convert("L"))) < 2 < estimate_noise(np.asarray(degrade(
        img, {**spec, "rotation": 0, "noise": 12}, rng)[0]))


@needs_tesseract
def test_tesseract_reads_a_clean_scan_and_caches_it(tmp_path):
    rng = np.random.default_rng(2)
    spec = make_spec(rng, 0, VENDOR_BY_KEY["northgate"], None)
    path = tmp_path / "clean.png"
    render_image(layout(spec)).save(path)
    page, _ = ocr_document(path, cache_dir=tmp_path / "cache")
    assert page.source == "tesseract" and page.mean_conf > 80
    r = RuleExtractor().extract(page).draft
    assert r.invoice_number == spec.invoice_number and r.total == pytest.approx(spec.total)
    again, _ = ocr_document(path, cache_dir=tmp_path / "cache")
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1 and again.seconds == page.seconds
