"""Turning OCR strings into typed values: amounts, dates, tax ids, labels.

Each parser returns the value and whether it had to *repair* the text (for example a letter O read where a digit
must be, or a decimal point the OCR dropped). Repairs lower the field's confidence; they never hide a failure.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from difflib import SequenceMatcher, get_close_matches

from .schema import GSTIN_RE, gstin_valid

CURRENCY_PREFIX = re.compile(r"^(rs\.?|inr|usd|us\$|\$|₹)", re.I)
TO_DIGIT = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1", "|": "1", "i": "1",
                          "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", "G": "6", "g": "9", "?": "7"})
TO_LETTER = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G", "4": "A", "7": "T"})
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


# ---------------------------------------------------------------- amounts
def parse_amount(text: str, expect_decimals: bool = True) -> tuple[float | None, bool]:
    """'1,23,456.00', '$1,234.50', 'Rs. 980.00', '30,713 04' → float. Returns (value, repaired)."""
    s = text.strip().replace(" ", "")
    s = CURRENCY_PREFIX.sub("", s).strip(":;'\"`‘’")
    if not s or "%" in s or "@" in s:
        return None, False
    repaired = False
    if not re.fullmatch(r"[\d,.]+", s):
        letters = sum(c.isalpha() for c in s)
        if letters > max(1, len(s) // 4):
            return None, False
        s2 = s.translate(TO_DIGIT)
        if not re.fullmatch(r"[\d,.]+", s2):
            return None, False
        s, repaired = s2, True
    s = s.strip(",.")
    if not s or not any(c.isdigit() for c in s):
        return None, False
    # decimal part: a final ".dd" or ",dd" (OCR often reads the point as a comma)
    m = re.fullmatch(r"([\d,.]*?)[.,](\d{2})", s)
    if m and (m.group(1) == "" or re.fullmatch(r"\d{1,3}(,\d{2,3})*|\d+", m.group(1).replace(".", ","))):
        whole, frac = m.group(1).replace(",", "").replace(".", ""), m.group(2)
        repaired = repaired or s[-3] == ","                    # decimal point read as a comma
        return float(f"{whole or 0}.{frac}"), repaired
    digits = s.replace(",", "").replace(".", "")
    if not digits.isdigit():
        return None, False
    if expect_decimals and len(digits) >= 3:
        return float(f"{digits[:-2]}.{digits[-2:]}"), True     # the decimal point was lost
    return float(digits), repaired


def parse_quantity(text: str) -> tuple[float | None, bool]:
    s = text.strip()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s), False
    s2 = s.translate(TO_DIGIT)
    if re.fullmatch(r"\d+(\.\d+)?", s2):
        return float(s2), True
    return None, False


def is_numeric_token(text: str) -> bool:
    s = CURRENCY_PREFIX.sub("", text.strip())
    return bool(s) and sum(c.isdigit() for c in s) >= max(1, int(len(s) * 0.6))


# ---------------------------------------------------------------- dates
DAY_FIRST = ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d %b %Y", "%d-%b-%Y", "%d %B %Y", "%Y-%m-%d", "%b %d, %Y",
             "%B %d, %Y", "%b %d %Y", "%B %d %Y")
MONTH_FIRST = ("%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y", "%Y-%m-%d",
               "%d %b %Y", "%d-%b-%Y")


def _fix_month_names(s: str) -> tuple[str, bool]:
    out, fixed = [], False
    for tok in re.split(r"(\W+)", s):
        low = tok.lower()
        if tok.isalpha() and len(tok) >= 3 and low[:3] not in MONTHS:
            full = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
                    "october", "november", "december"]
            m = get_close_matches(low, MONTHS + full, n=1, cutoff=0.6)
            if m:
                tok, fixed = m[0].capitalize(), True
        out.append(tok)
    return "".join(out), fixed


def parse_date(text: str, day_first: bool = True) -> tuple[date | None, bool]:
    """Parse a printed date. Day-first for Indian invoices, month-first for US ones."""
    s = re.sub(r"\s+", " ", text.strip().strip(":.,;")).replace(" ,", ",")
    if not s:
        return None, False
    fmts = DAY_FIRST if day_first else MONTH_FIRST
    candidates = [(s, False)]
    s_digits = re.sub(r"(?<=\d)[Oo]|[Oo](?=\d)", "0", s)
    if s_digits != s:
        candidates.append((s_digits, True))
    s_month, fixed = _fix_month_names(s)
    if fixed:
        candidates.append((s_month, True))
    for cand, repaired in candidates:
        for fmt in fmts:
            try:
                d = datetime.strptime(cand, fmt).date()
            except ValueError:
                continue
            if 2000 <= d.year <= 2100:
                return d, repaired
    digits = re.sub(r"\D", "", s.translate(TO_DIGIT))
    if len(digits) == 8 and not re.search(r"[A-Za-z]{2}", s):   # separators lost: ddmmyyyy / mmddyyyy / yyyymmdd
        orders = [(0, 2, 4), (2, 0, 4)] if day_first else [(2, 0, 4), (0, 2, 4)]
        if digits[:2] in ("19", "20"):
            try:
                return date(int(digits[:4]), int(digits[4:6]), int(digits[6:])), True
            except ValueError:
                pass
        for d_i, m_i, y_i in orders:
            try:
                return date(int(digits[y_i:y_i + 4]), int(digits[m_i:m_i + 2]), int(digits[d_i:d_i + 2])), True
            except ValueError:
                continue
    return None, False


# ---------------------------------------------------------------- tax ids
GSTIN_LETTER_POS = {2, 3, 4, 5, 6, 11}
GSTIN_DIGIT_POS = {0, 1, 7, 8, 9, 10}
EIN_RE = re.compile(r"(?<![\d-])(\d{2})\s?-\s?(\d{7})(?!\d)")


def coerce_gstin(raw: str) -> tuple[str | None, bool]:
    """Force OCR text into GSTIN shape by character class per position. Returns (gstin, repaired) or (None, _)."""
    s = re.sub(r"[^0-9A-Za-z]", "", raw).upper()
    if len(s) != 15:
        return None, False
    chars = list(s)
    for i, c in enumerate(chars):
        if i in GSTIN_DIGIT_POS:
            chars[i] = c.translate(TO_DIGIT)
        elif i in GSTIN_LETTER_POS:
            chars[i] = c.translate(TO_LETTER)
        elif i == 13:
            chars[i] = "Z"                    # always Z in the current GSTIN format
    out = "".join(chars)
    if not GSTIN_RE.match(out):
        return None, False
    return out, out != s


def find_gstins(tokens: list[str]) -> list[tuple[str, bool, bool]]:
    """GSTIN-shaped tokens in reading order: (gstin, repaired, checksum_ok). Joins a token split in two by OCR."""
    found: list[tuple[str, bool, bool]] = []
    # ']', ')', '}' and '/' are how Tesseract most often misreads a J; '|' and '[' an I. The checksum still has the
    # final say, so a wrong guess here ends in review, not in a wrong approval.
    clean = [re.sub(r"[^0-9A-Za-z]", "", t.translate(str.maketrans("])}/|[", "JJJJII"))) for t in tokens]
    i = 0
    while i < len(clean):
        for span in (1, 2, 3):
            chunk = "".join(clean[i:i + span])
            if len(chunk) == 16 and chunk[13:15].upper() == "2Z":
                chunk = chunk[:13] + chunk[14:]        # Tesseract often reads the fixed 'Z' as '2Z'
            if len(chunk) == 15:
                g, rep = coerce_gstin(chunk)
                if g:
                    found.append((g, rep or span > 1, gstin_valid(g)))
                    i += span - 1
                    break
            if len(chunk) >= 16:
                break
        i += 1
    return found


def find_ein(text: str) -> tuple[str | None, bool]:
    """An EIN (NN-NNNNNNN). On a line labelled EIN / Tax ID, nine digits with the hyphen lost also count."""
    fixed = text.translate(str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1"}))
    m = EIN_RE.search(fixed)
    if m:
        return f"{m.group(1)}-{m.group(2)}", fixed[m.start():m.end()] != text[m.start():m.end()]
    if re.search(r"\b(EIN|TIN|Tax\s*ID)\b", text, re.I):
        m = re.search(r"(?<!\d)(\d{9})(?!\d)", fixed)
        if m:
            return f"{m.group(1)[:2]}-{m.group(1)[2:]}", True
    return None, False


# ---------------------------------------------------------------- labels
def norm_label(text: str) -> str:
    return re.sub(r"[^a-z#]", "", text.lower())


def label_similarity(text: str, label: str) -> float:
    a, b = norm_label(text), norm_label(label)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def norm_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (text or "").lower())).strip()
