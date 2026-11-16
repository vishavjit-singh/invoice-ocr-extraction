from datetime import date

import numpy as np
import pytest

from invoiceocr.generate import make_gstin
from invoiceocr.normalize import coerce_gstin, find_ein, find_gstins, label_similarity, parse_amount, parse_date


@pytest.mark.parametrize("text, value, repaired", [
    ("1,23,456.00", 123456.0, False),          # Indian grouping
    ("$1,234.50", 1234.5, False),
    ("30,71304", 30713.04, True),              # "30,713 04": decimal point lost, joined by the caller
    ("1,234,56", 1234.56, True),               # decimal point read as a comma
    ("7.03425", 7034.25, True),                # thousands comma read as a point, decimal point lost
    ("1O2.50", 102.5, True),                   # letter O for zero
    ("9%", None, False),
])
def test_parse_amount(text, value, repaired):
    assert parse_amount(text) == (value, repaired)


@pytest.mark.parametrize("text, day_first, expected", [
    ("05/03/2026", True, date(2026, 3, 5)),
    ("05/03/2026", False, date(2026, 5, 3)),
    ("23-Sep-2026", True, date(2026, 9, 23)),
    ("March 5, 2026", False, date(2026, 3, 5)),
])
def test_parse_date_formats(text, day_first, expected):
    assert parse_date(text, day_first)[0] == expected


def test_parse_date_repairs_but_says_so():
    assert parse_date("2309/2026", True) == (date(2026, 9, 23), True)      # a lost separator
    assert parse_date("Sop 12, 2026", False) == (date(2026, 9, 12), True)  # month misread
    assert parse_date("30-00-2026", True) == (None, False)                # no valid reading: do not guess


def test_gstin_coercion_by_position():
    g = make_gstin(np.random.default_rng(3), "29", "Bluefin")
    noisy = g[:1].replace("2", "Z") + g[1:7].replace("O", "0") + g[7:]   # letters/digits swapped by OCR
    noisy = noisy[:2] + noisy[2:7].translate(str.maketrans("OSB", "058")) + noisy[7:]
    assert coerce_gstin(noisy)[0] == g


def test_find_gstins_handles_common_tesseract_slips():
    g = make_gstin(np.random.default_rng(5), "07", "Marigold")
    assert find_gstins(["GSTIN", g[:7], g[7:]])[0][0] == g                # split into two tokens
    assert find_gstins([g[:13] + "2" + g[13:]])[0][0] == g                 # the fixed 'Z' read as '2Z'
    gj = g[:14] + "J"
    assert find_gstins(["GSTIN:", gj[:14] + "]"])[0][0] == gj             # ']' for J
    assert find_gstins([g])[0][2] is True


def test_find_ein():
    assert find_ein("EIN: 14-8360558") == ("14-8360558", False)
    assert find_ein("Tax ID 274300777") == ("27-4300777", True)          # hyphen lost, label present
    assert find_ein("Phone 274300777") == (None, False)                   # no label: not an EIN


def test_fuzzy_labels():
    assert label_similarity("Invowe No", "invoice no") >= 0.8
    assert label_similarity("Invoice Date", "invoice no") < 0.8
