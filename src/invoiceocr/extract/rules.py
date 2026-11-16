"""Layout-aware rule extractor.

Works on word boxes, not on a flat string:

- key/value fields: find a label segment ("Invoice No:", "Payment due", "Balance due", fuzzy-matched so OCR slips
  like "Invowe No" still count) and read the value from the rest of that segment or the next segment on the line
- vendor name: the tallest text block in the top fifth of the page that is not a title like "TAX INVOICE"
- tax id: the first GSTIN (forced into shape position by position, then checksum-tested) above the bill-to block,
  or an EIN
- line items: locate the table header, take each column's x-extent from its header word, assign every word below
  to a column by its right edge (numbers are right-aligned), and stop at the totals block
"""
from __future__ import annotations

import re

from ..layout import Line, Segment, build_lines
from ..normalize import (find_ein, find_gstins, is_numeric_token, label_similarity, norm_label, parse_amount,
                         parse_date, parse_quantity)
from ..ocr import OcrPage, Word
from ..schema import InvoiceDraft, LineItemDraft
from .base import ExtractionResult


def derive_missing(qty, qty_conf: float, price, amount) -> tuple:
    """Fill a quantity or unit price the OCR could not read, from the other two numbers on the row.

    Only for a value that is missing or read with low confidence, and only when the result is clean (a whole or
    half quantity, a price with at most two decimals that reproduces the amount exactly). A confidently read
    quantity that disagrees with the amount is left alone so the arithmetic check can flag it.
    """
    if amount is None:
        return qty, price, None
    if price and (qty is None or (qty_conf < 0.85 and abs(qty * price - amount) > 0.011)):
        q = round(amount / price * 2) / 2
        if q > 0 and abs(q * price - amount) <= 0.011:
            return q, price, "quantity"
    if price is None and qty:
        p = round(amount / qty, 2)
        if p > 0 and abs(p * qty - amount) <= 0.011:
            return qty, p, "unit_price"
    return qty, price, None


def repair_quantity(raw: str, price: float, amount: float) -> float | None:
    """'175' printed as 17.5: put back a decimal point the OCR dropped, if that makes qty × price = amount."""
    digits = re.sub(r"\D", "", raw)
    if "." in raw or len(digits) < 2 or price <= 0:
        return None
    for i in range(len(digits) - 1, 0, -1):
        q = float(f"{digits[:i]}.{digits[i:]}")
        if abs(q * price - amount) <= 0.011:
            return q
    return None

LABELS = {
    "invoice_number": ["invoice no", "invoice number", "invoice #", "invoice num", "inv no", "bill no"],
    "issue_date": ["invoice date", "date", "issued", "date of issue", "dated", "bill date"],
    "due_date": ["due date", "payment due", "due", "due by"],
    "subtotal": ["sub total", "subtotal", "taxable value", "taxable amount", "net amount"],
    "tax": ["cgst", "sgst", "igst", "utgst", "sales tax", "tax", "vat", "gst"],
    "total": ["total", "grand total", "total due", "balance due", "amount due", "invoice total", "total amount"],
    "bill_to": ["bill to", "billed to", "buyer", "customer", "ship to", "sold to"],
}
TITLES = {"invoice", "taxinvoice", "proformainvoice", "receipt"}
HEADER_TYPES = {"description": "desc", "item": "desc", "particulars": "desc", "qty": "qty", "quantity": "qty",
                "rate": "price", "price": "price", "unitprice": "price", "amount": "amount", "total": "amount",
                "linetotal": "amount", "hsnsac": "hsn", "hsn": "hsn", "sac": "hsn", "#": "idx", "sno": "idx"}
NUMERIC_COLS = ("qty", "price", "amount", "hsn", "idx")
LABEL_THRESHOLD = 0.8
DATE_LIKE = re.compile(r"\b\d{1,4}[/.-]\d{1,2}[/.-]\d{2,4}\b"
                       r"|\b[A-Za-z]{3,9}\.? \d{1,2},? \d{4}\b"
                       r"|\b\d{1,2}[ -][A-Za-z]{3}[ -]\d{4}\b")
REPAIR_PENALTY = 0.85


def _box(words: list[Word]) -> tuple:
    return (min(w.x0 for w in words), min(w.y0 for w in words), max(w.x1 for w in words), max(w.y1 for w in words))


def _conf(words: list[Word]) -> float:
    return min(w.conf for w in words) / 100 if words else 0.0


def is_title(text: str) -> bool:
    n = norm_label(text)
    if "#" in n:
        return False
    return n in TITLES or any(abs(len(n) - len(t)) <= 1 and label_similarity(n, t) >= 0.85 for t in TITLES)


def match_label(seg: Segment, fields=None) -> tuple[str | None, int, float]:
    """Best (field, words used by the label, score) for the start of a segment."""
    if is_title(seg.text):
        return None, 0, 0.0
    best = (None, 0, 0.0)
    for n in range(1, min(3, len(seg.words)) + 1):
        prefix = " ".join(w.text for w in seg.words[:n])
        for f, labels in LABELS.items():
            if fields and f not in fields:
                continue
            for lab in labels:
                s = label_similarity(prefix, lab)
                if s > best[2] or (s == best[2] and n > best[1]):
                    best = (f, n, s)
    return best if best[2] >= LABEL_THRESHOLD else (None, 0, 0.0)


class RuleExtractor:
    name = "rules"

    def extract(self, page: OcrPage) -> ExtractionResult:
        lines = build_lines(page.words)
        out: dict = {}
        conf: dict[str, float] = {}
        ev: dict[str, list[tuple]] = {}
        notes: list[str] = []

        def put(field, value, words, c):
            out[field] = value
            conf[field] = round(c, 3)
            if words:
                ev[field] = [_box(words)]

        # ------------------------------------------------ labelled key/value fields (header block)
        bill_to_y = None
        kv: dict[str, list[Word]] = {}
        used: set[int] = set()                         # ids of words consumed as labels or values
        for ln in lines:
            for i, seg in enumerate(ln.segments):
                f, n, _ = match_label(seg, ("invoice_number", "issue_date", "due_date", "bill_to"))
                if not f:
                    continue
                if f == "bill_to":
                    bill_to_y = ln.y0 if bill_to_y is None else bill_to_y
                    continue
                if f in kv:
                    continue
                value = seg.words[n:]
                value = [w for w in value if w.text not in (":", "-", "#")]
                if not value and i + 1 < len(ln.segments):
                    nxt = ln.segments[i + 1]
                    nf, nn, _ = match_label(nxt)
                    if nf is None or nn < len(nxt.words):        # skip a neighbour that is itself just a label
                        value = nxt.words
                if value:
                    kv[f] = value
                    used.update(id(w) for w in seg.words + value)

        # ------------------------------------------------ tax id and currency
        tokens_text = " ".join(w.text for w in page.words)
        gst_candidates = []
        for ln in lines:
            if bill_to_y is not None and ln.y0 >= bill_to_y:
                continue
            if "buyer" in norm_label(ln.text):
                continue
            for g, rep, ok in find_gstins([w.text for w in ln.words]):
                gst_candidates.append((g, rep, ok, ln))
        ein_words = None
        if gst_candidates:
            g, rep, ok, ln = gst_candidates[0]
            words = [w for w in ln.words if len(re.sub(r"[^0-9A-Za-z]", "", w.text)) >= 4
                     and "gstin" not in w.text.lower()] or ln.words
            c = max(_conf(words), 0.95) if ok else _conf(words) * 0.5
            put("vendor_tax_id", g, words, c * (REPAIR_PENALTY if rep and not ok else 1.0))
            if rep:
                notes.append(f"vendor_tax_id repaired to {g} (checksum {'ok' if ok else 'failed'})")
        else:
            for ln in lines:
                if bill_to_y is not None and ln.y0 >= bill_to_y:
                    continue
                ein, rep = find_ein(ln.text)
                if ein:
                    ein_words = [w for w in ln.words if re.search(r"\d", w.text)]
                    put("vendor_tax_id", ein, ein_words, _conf(ein_words) * (REPAIR_PENALTY if rep else 1.0))
                    break

        explicit = None
        if re.search(r"\bINR\b|\bRs\.?(?=\s|\d|$)|₹", tokens_text):
            explicit = "INR"
        elif re.search(r"\bUSD\b|\$", tokens_text):
            explicit = "USD"
        implied = "INR" if gst_candidates else ("USD" if ein_words else None)
        currency = explicit or implied
        if currency:
            put("currency", currency, [], 0.97 if explicit and (implied in (None, explicit)) else 0.7)
        day_first = currency != "USD"

        # ------------------------------------------------ key/value parsing
        if "invoice_number" in kv:
            words = kv["invoice_number"]
            value = "".join(w.text for w in words).strip(":.,;|").lstrip("#:")
            if 2 <= len(value) <= 40:
                put("invoice_number", value, words, _conf(words))
        for f in ("issue_date", "due_date"):
            if f not in kv:
                continue
            words = kv[f]
            raw = " ".join(w.text for w in words)
            if f == "due_date" and re.search(r"receipt|immediate", raw, re.I):
                put(f, None, words, _conf(words))
                continue
            d, rep = parse_date(raw, day_first=day_first)
            if d:
                put(f, d.isoformat(), words, _conf(words) * (REPAIR_PENALTY if rep else 1.0))
            else:
                notes.append(f"unparsed:{f}:{raw}")

        # ------------------------------------------------ vendor name: tallest text near the top
        top = [(seg, ln) for ln in lines for seg in ln.segments if seg.y1 < 0.2 * page.height]
        cands = [(seg, ln) for seg, ln in top
                 if not is_title(seg.text) and match_label(seg)[0] is None and len(seg.words) >= 2
                 and sum(c.isalpha() for c in seg.text) >= 0.6 * len(seg.text.replace(" ", ""))
                 and not any(id(w) in used for w in seg.words)]
        if cands:
            seg, _ = max(cands, key=lambda t: (round(t[0].height / 3), -t[0].y0))
            put("vendor_name", seg.text.strip(" .,:;|"), seg.words, seg.conf)

        # ------------------------------------------------ table and totals
        header_i, cols = self._find_header(lines)
        # safety net: a date printed above the table that no label claimed (e.g. a garbled "Due Date" label)
        header_end = lines[header_i].y0 if header_i is not None else 0.4 * page.height
        for ln in lines:
            for seg in ln.segments:
                if ln.y0 < header_end and DATE_LIKE.search(seg.text) and not any(id(w) in used for w in seg.words):
                    notes.append(f"unassigned:date:{seg.text}")
        items, item_conf, item_boxes, end_i = [], 1.0, [], header_i
        if header_i is not None:
            items, item_conf, item_boxes, end_i = self._read_table(lines, header_i, cols, notes)
        else:
            items, item_conf, item_boxes, end_i = self._read_rows_without_header(lines, bill_to_y, page, notes)
        if items:
            out["line_items"] = items
            conf["line_items"] = round(item_conf, 3)
            ev["line_items"] = item_boxes
        if end_i is not None:
            start = end_i + 1
        else:
            start = next((i for i, ln in enumerate(lines) if bill_to_y is not None and ln.y0 > bill_to_y), 0)
        tax_total, tax_words, tax_seen = 0.0, [], False
        for ln in lines[start:]:
            seg = ln.segments[0]
            f, n, _ = match_label(seg, ("subtotal", "tax", "total"))
            if not f:
                continue
            amount, words, rep = self._rightmost_amount(ln, seg.words[n:])
            if amount is None:
                continue
            c = _conf(words) * (REPAIR_PENALTY if rep else 1.0)
            if f == "tax":
                tax_total += amount
                tax_words += words
                tax_seen = True
                conf["tax"] = round(min(conf.get("tax", 1.0), c), 3)
            elif f == "total":
                put("total", amount, words, c)                      # the last "total" line wins
            elif "subtotal" not in out:
                put("subtotal", amount, words, c)
        if tax_seen:
            out["tax"] = round(tax_total, 2)
            ev["tax"] = [_box(tax_words)]
        elif "subtotal" in out and "total" in out and abs(out["total"] - out["subtotal"]) < 0.005:
            put("tax", 0.0, [], 0.6)                                 # no tax line printed and total == subtotal

        draft = InvoiceDraft(**{k: v for k, v in out.items() if k != "line_items"},
                             line_items=[LineItemDraft(**li) for li in out.get("line_items", [])])
        for f in ("vendor_name", "vendor_tax_id", "invoice_number", "issue_date", "due_date", "currency",
                  "subtotal", "tax", "total"):
            conf.setdefault(f, 0.0)
        conf.setdefault("line_items", 0.0)
        return ExtractionResult(draft, conf, self.name, ev, notes=notes)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _rightmost_amount(ln: Line, after: list[Word]):
        words = [w for w in ln.words if w.x0 >= (after[0].x0 if after else ln.segments[0].x1)]
        words = [w for w in words if re.search(r"\d", w.text)]
        if not words:
            return None, [], False
        # use the right-most run of numeric words (a value split in two by OCR, e.g. "30,713 04")
        run = [words[-1]]
        for w in reversed(words[:-1]):
            if run[0].x0 - w.x1 < 1.2 * max(w.h, 1) and is_numeric_token(w.text) and "%" not in w.text:
                run.insert(0, w)
            else:
                break
        value, rep = parse_amount("".join(w.text for w in run))
        if value is None and len(run) > 1:
            run = run[-1:]
            value, rep = parse_amount(run[0].text)
        return value, run, rep

    @staticmethod
    def _find_header(lines: list[Line]):
        for i, ln in enumerate(lines):
            cols: dict[str, list[float]] = {}
            ws = ln.words
            j = 0
            while j < len(ws):
                matched = False
                for span in (2, 1):
                    if j + span > len(ws):
                        continue
                    text = norm_label("".join(w.text for w in ws[j:j + span]))
                    for key, typ in HEADER_TYPES.items():
                        fuzzy = span == 1 and len(key) > 3 and label_similarity(text, key) >= 0.8
                        if text and (text == key or fuzzy):
                            if typ not in cols:
                                cols[typ] = [ws[j].x0, ws[j + span - 1].x1, ws[j].y1]
                            j += span
                            matched = True
                            break
                    if matched:
                        break
                if not matched:
                    j += 1
            if "desc" in cols and "amount" in cols and ("qty" in cols or "price" in cols):
                return i, cols
        return None, None

    def _read_rows_without_header(self, lines: list[Line], bill_to_y, page: OcrPage, notes: list[str]):
        """Fallback when the header row is unreadable (e.g. text on a dark fill): a row is a line that starts with
        a description and ends with numbers; the last three numbers are quantity, unit price and amount."""
        start_y = (bill_to_y or 0) + 1 if bill_to_y is not None else 0.25 * page.height
        items, confs, boxes, end_i = [], [], [], None
        for i, ln in enumerate(lines):
            if ln.y0 <= start_y:
                continue
            if match_label(ln.segments[0], ("subtotal", "tax", "total"))[0]:
                break
            if len(ln.segments) < 2 or sum(c.isalpha() for c in ln.segments[0].text) < 3:
                continue
            nums = [w for seg in ln.segments[1:] for w in seg.words if is_numeric_token(w.text)]
            if len(nums) < 2:
                continue
            amount, rep_a = parse_amount(nums[-1].text)
            price, rep_p = parse_amount(nums[-2].text)
            qty, rep_q = parse_quantity(nums[-3].text) if len(nums) >= 3 else (None, False)
            if amount is None or "." not in nums[-1].text and "," not in nums[-1].text:
                continue
            first = ln.segments[0].words
            if re.fullmatch(r"\d{1,2}", first[0].text) and len(first) > 1:
                first = first[1:]                              # a row number in front of the description
            qty, price, derived = derive_missing(qty, _conf(nums[-3:-2]), price, amount)
            if derived:
                notes.append(f"derived {derived} for line {len(items) + 1} from amount")
            items.append({"description": " ".join(w.text for w in first), "quantity": qty, "unit_price": price,
                          "amount": amount})
            used = first + nums[-3:]
            confs.append(_conf(used) * REPAIR_PENALTY * (REPAIR_PENALTY if (rep_a or rep_p or rep_q or derived)
                                                          else 1.0))
            boxes.append(_box(used))
            end_i = i
        if items:
            notes.append("table header not found: rows read by position")
        return items, (min(confs) if confs else 0.0), boxes, end_i

    def _read_table(self, lines: list[Line], header_i: int, cols: dict, notes: list[str]):
        numeric = {k: v for k, v in cols.items() if k in NUMERIC_COLS}
        desc_x0 = cols["desc"][0]
        right_of_desc = [v[0] for k, v in cols.items() if v[0] > desc_x0 + 5]
        desc_x1 = min(right_of_desc) - 5 if right_of_desc else 10_000
        items: list[dict] = []
        confs: list[float] = []
        boxes: list[tuple] = []
        end_i = header_i
        last_y = lines[header_i].y1
        line_h = max(lines[header_i].y1 - lines[header_i].y0, 10)
        for i in range(header_i + 1, len(lines)):
            ln = lines[i]
            if match_label(ln.segments[0], ("subtotal", "tax", "total"))[0]:
                break
            if items and ln.y0 - last_y > 4 * line_h:
                break
            cells: dict[str, list[Word]] = {}
            for w in ln.words:
                col = None
                if is_numeric_token(w.text):
                    for k, (x0, x1, _) in numeric.items():
                        if x0 - 12 <= w.x1 <= x1 + 28:
                            col = k
                            break
                if col is None and desc_x0 - 15 <= w.x0 and w.x1 <= desc_x1 + 10:
                    col = "desc"
                if col:
                    cells.setdefault(col, []).append(w)
            if not cells:
                continue
            has_numbers = any(k in cells for k in ("qty", "price", "amount"))
            if not has_numbers:
                if "desc" in cells and items:            # a wrapped description line
                    items[-1]["description"] = (items[-1]["description"] or "") + " " + " ".join(
                        w.text for w in cells["desc"])
                    end_i, last_y = i, ln.y1
                continue
            if "desc" not in cells and "qty" not in cells:
                break                                     # numbers with no description: the totals block
            amount, rep_a = (parse_amount("".join(w.text for w in cells["amount"])) if "amount" in cells
                             else (None, False))
            price, rep_p = (parse_amount("".join(w.text for w in cells["price"])) if "price" in cells
                            else (None, False))
            qty, rep_q = (parse_quantity("".join(w.text for w in cells["qty"])) if "qty" in cells else (None, False))
            if None not in (qty, price, amount) and abs(qty * price - amount) > 0.011:
                fixed = repair_quantity("".join(w.text for w in cells["qty"]), price, amount)
                if fixed is not None:
                    qty, rep_q = fixed, True
            qty, price, derived = derive_missing(qty, _conf(cells.get("qty", [])), price, amount)
            if derived:
                rep_q = True
                notes.append(f"derived {derived} for line {len(items) + 1} from amount")
            desc = " ".join(w.text for w in cells.get("desc", [])).strip(" |")
            row_words = [w for c in cells.values() for w in c]
            c = _conf(row_words) * (REPAIR_PENALTY if (rep_a or rep_p or rep_q) else 1.0)
            if amount is None:
                c *= 0.5
            items.append({"description": desc or None, "quantity": qty, "unit_price": price, "amount": amount})
            confs.append(c)
            boxes.append(_box(row_words))
            end_i, last_y = i, ln.y1
        return items, (min(confs) if confs else 0.0), boxes, end_i
