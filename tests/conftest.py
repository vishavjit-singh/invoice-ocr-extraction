import copy
import shutil

import pytest

from invoiceocr import config
from invoiceocr.generate import VENDORS, layout, make_spec, render_pdf, spec_to_truth

needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract binary not installed")


@pytest.fixture(scope="session")
def params():
    p = copy.deepcopy(config.load_params())
    p["generate"]["n_invoices"] = 16
    p["ocr"]["workers"] = 1
    return p


@pytest.fixture(scope="session")
def project(tmp_path_factory, params):
    """A small SYNTHETIC set generated, OCR'd, extracted and scored once into a temporary project root."""
    from invoiceocr.evaluate import evaluate
    from invoiceocr.generate import generate

    root = tmp_path_factory.mktemp("project")
    paths = config.Paths(root)
    generate(paths.invoices, paths.ground_truth, params["generate"])
    metrics = evaluate(paths, params)
    return paths, metrics


@pytest.fixture(scope="session")
def pdf_per_template(tmp_path_factory):
    """One digital PDF (text layer) per vendor, with its ground truth."""
    import numpy as np

    out = tmp_path_factory.mktemp("pdfs")
    rng = np.random.default_rng(7)
    docs = []
    for i, v in enumerate(VENDORS):
        spec = make_spec(rng, i, v, None)
        path = out / f"{v.key}.pdf"
        render_pdf(layout(spec), path)
        docs.append((v, path, spec_to_truth(spec, v)))
    return docs
