import json

import pytest
from asgi import call, multipart
from conftest import needs_tesseract

from invoiceocr.api import create_app
from invoiceocr.cli import main


@pytest.fixture(scope="module")
def app(params):
    return create_app(params)


def test_health_and_schema(app):
    r = call(app, "GET", "/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    s = call(app, "GET", "/schema").json()
    assert "vendor_tax_id" in s["properties"] and "line_items" in s["required"]


def test_extract_pdf_upload_returns_validated_json(app, pdf_per_template):
    vendor, path, truth = pdf_per_template[0]
    body, ctype = multipart("file", "invoice.pdf", path.read_bytes(), "application/pdf")
    r = call(app, "POST", "/extract", body, {"content-type": ctype})
    out = r.json()
    assert r.status_code == 200 and out["status"] == "auto_approved" and out["ocr"]["source"] == "pdf_text"
    assert out["invoice"]["invoice_number"] == truth["invoice_number"] and out["invoice"]["total"] == truth["total"]


@pytest.mark.parametrize("body, headers, query, status, code", [
    (b"", {}, {}, 400, "empty_body"),
    (b"hello, not a document", {"content-type": "text/plain"}, {}, 415, "unsupported_type"),
    (b"%PDF-1.4", {"content-type": "application/pdf"}, {"extractor": "magic"}, 400, "bad_extractor"),
    (b"%PDF-1.4", {"content-type": "application/pdf"}, {"extractor": "llm"}, 400, "llm_unavailable"),
    (b"%PDF-1.4 truncated", {"content-type": "application/pdf"}, {}, 422, "unreadable_document"),
])
def test_extract_errors(app, monkeypatch, body, headers, query, status, code):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = call(app, "POST", "/extract", body, headers, query)
    assert r.status_code == status and r.json()["error"]["code"] == code


@needs_tesseract
def test_end_to_end_on_a_small_synthetic_set(project):
    paths, m = project
    assert m["synthetic"] is True and m["n_invoices"] == 16
    assert len(list(paths.predictions.glob("inv_*.json"))) == 16 and (paths.outputs / "results.csv").exists()
    assert m["by_condition"]["pdf_text"]["all_correct"] == 1.0          # text layer: nothing to misread
    assert m["by_condition"]["L0"]["field_accuracy"] >= 0.9
    r = m["routing"]
    assert r["n_auto"] + r["n_review"] == 16 and 0 <= r["stp_rate"] <= 1
    assert r["n_auto_with_error"] == 0                                   # nothing wrong slipped through here
    assert m["document_errors"]["n"] >= 1 and m["document_errors"]["caught"] == m["document_errors"]["n"]
    assert 0 < m["ocr"]["tesseract_mean_s"] < 30


@needs_tesseract
def test_report_and_cli(project, params, capsys, tmp_path):
    from invoiceocr.report import render_all

    paths, _ = project
    images = render_all(paths, params)
    assert {p.name for p in images} == {"architecture.png", "field_accuracy.png", "accuracy_by_condition.png",
                                        "review_routing.png", "degradation_levels.png", "sample_overlay.png"}
    assert all(p.stat().st_size > 10_000 for p in images)
    pdf = next(paths.invoices.glob("*.pdf"))
    main(["--root", str(paths.root), "extract", str(pdf), "--overlay", str(tmp_path / "o.png")])
    out = json.loads(capsys.readouterr().out)
    assert out["status"] in ("auto_approved", "needs_review") and (tmp_path / "o.png").exists()
