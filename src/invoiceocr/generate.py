"""SYNTHETIC invoices with ground truth.

Eleven fictional vendors, four layout templates (two Indian GST layouts, two US layouts), INR and USD, dates in
the vendor's own format, line-item tables, and a footer on every page saying the document is synthetic. Each
invoice is drawn once as a list of drawing operations, then rendered to PNG/JPEG with Pillow or to a PDF with a
real text layer with reportlab, so the image and the PDF of a layout are the same document.

Scanned copies get controlled degradation (rotation, downscale, blur, low contrast, noise, JPEG artifacts) at
levels 0-3. A share of invoices is printed with a genuine mistake (wrong total, subtotal that does not match the
lines, due date before the issue date, a GSTIN with a bad check character) so the validators have something real
to catch. Everything is seeded and deterministic.
"""
from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .schema import GSTIN_CHARS, gstin_checksum

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"   # bundled with matplotlib: same fonts everywhere
FONT_FILES = {"sans": "DejaVuSans.ttf", "sans-bold": "DejaVuSans-Bold.ttf", "serif": "DejaVuSerif.ttf",
              "serif-bold": "DejaVuSerif-Bold.ttf"}
PAGE_W, PAGE_H, DPI = 1240, 1754, 150          # A4 at 150 dpi
INK, GREY, LIGHT = (25, 25, 30), (110, 110, 118), (226, 228, 232)
FOOTER = "SYNTHETIC INVOICE - generated test data for an OCR demo. Fictional company. Not a real document."


# ---------------------------------------------------------------- drawing operations
@dataclass
class Txt:
    x: float
    y: float                      # baseline
    text: str
    size: int = 20
    font: str = "sans"
    align: str = "left"           # left | right | center
    color: tuple = INK


@dataclass
class Rule:
    x0: float
    y0: float
    x1: float
    y1: float
    width: int = 1
    color: tuple = GREY


@dataclass
class Fill:
    x0: float
    y0: float
    x1: float
    y1: float
    color: tuple = LIGHT


_FONT_CACHE: dict = {}


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    key = (name, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(str(FONT_DIR / FONT_FILES[name]), size)
    return _FONT_CACHE[key]


def render_image(ops: list) -> Image.Image:
    img = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    d = ImageDraw.Draw(img)
    for op in ops:
        if isinstance(op, Fill):
            d.rectangle([op.x0, op.y0, op.x1, op.y1], fill=op.color)
        elif isinstance(op, Rule):
            d.line([op.x0, op.y0, op.x1, op.y1], fill=op.color, width=op.width)
        else:
            anchor = {"left": "ls", "right": "rs", "center": "ms"}[op.align]
            d.text((op.x, op.y), op.text, font=_font(op.font, op.size), fill=op.color, anchor=anchor)
    return img


def render_pdf(ops: list, path: Path) -> None:
    """Same drawing operations as a PDF page with a real text layer."""
    from reportlab.lib.colors import Color
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    for name, file in FONT_FILES.items():
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / file)))
    k = 72 / DPI
    c = canvas.Canvas(str(path), pagesize=(PAGE_W * k, PAGE_H * k), invariant=1)
    c.setTitle("SYNTHETIC invoice")

    def col(rgb):
        return Color(*(v / 255 for v in rgb))

    for op in ops:
        if isinstance(op, Fill):
            c.setFillColor(col(op.color))
            c.rect(op.x0 * k, (PAGE_H - op.y1) * k, (op.x1 - op.x0) * k, (op.y1 - op.y0) * k, stroke=0, fill=1)
        elif isinstance(op, Rule):
            c.setStrokeColor(col(op.color))
            c.setLineWidth(op.width * k)
            c.line(op.x0 * k, (PAGE_H - op.y0) * k, op.x1 * k, (PAGE_H - op.y1) * k)
        else:
            c.setFillColor(col(op.color))
            c.setFont(op.font, op.size * k)
            draw = {"left": c.drawString, "right": c.drawRightString, "center": c.drawCentredString}[op.align]
            draw(op.x * k, (PAGE_H - op.y) * k, op.text)
    c.showPage()
    c.save()


# ---------------------------------------------------------------- vendors, buyers, catalogues (all fictional)
@dataclass(frozen=True)
class Vendor:
    key: str
    name: str
    address: tuple
    state: str
    state_code: str
    currency: str
    template: str
    font: str
    number_pattern: str
    date_format: str
    tax_rate: float
    catalogue: str
    tax_id_label: str = "GSTIN"


VENDORS = [
    Vendor("saffron", "Saffron Loom Textiles Pvt Ltd", ("Plot 14, Sitapura Industrial Area", "Jaipur 302022"),
           "Rajasthan", "08", "INR", "gst_classic", "sans", "SLT/{fy}/{n:04d}", "%d/%m/%Y", 0.05, "textiles"),
    Vendor("kestrel", "Kestrel Agro Supplies", ("22 Grain Market Road", "Patiala 147001"), "Punjab", "03", "INR",
           "gst_classic", "serif", "KAS-{n:05d}", "%d-%m-%Y", 0.12, "agro"),
    Vendor("neelgiri", "Neelgiri Office Solutions LLP", ("Unit 7, Baner Business Park", "Pune 411045"),
           "Maharashtra", "27", "INR", "gst_modern", "sans", "NOS/{n:04d}/{yy}", "%d %b %Y", 0.18, "office"),
    Vendor("copperleaf", "Copperleaf Electricals", ("41 Naroda GIDC Phase 2", "Ahmedabad 382330"), "Gujarat",
           "24", "INR", "gst_modern", "serif", "CE{yy}{n:05d}", "%d-%b-%Y", 0.18, "electrical"),
    Vendor("bluefin", "Bluefin Freight Services Pvt Ltd", ("88 Outer Ring Road, Marathahalli", "Bengaluru 560037"),
           "Karnataka", "29", "INR", "gst_classic", "sans", "BFS/{fy}/{n:03d}", "%d.%m.%Y", 0.18, "freight"),
    Vendor("marigold", "Marigold Print House", ("B-19 Okhla Phase 1", "New Delhi 110020"), "Delhi", "07", "INR",
           "gst_modern", "sans", "MPH-{yy}-{n:04d}", "%d/%m/%Y", 0.12, "printing"),
    Vendor("harborview", "Harborview Analytics LLC", ("410 Alder Street, Suite 300", "Portland, OR 97204"),
           "Oregon", "OR", "USD", "us_simple", "sans", "HA-{n:05d}", "%B %d, %Y", 0.0, "consulting", "EIN"),
    Vendor("pinecrest", "Pinecrest Supply Co.", ("1820 Front Street", "Columbus, OH 43215"), "Ohio", "OH", "USD",
           "us_compact", "serif", "PSC{n:06d}", "%m/%d/%Y", 0.0725, "supplies", "Tax ID"),
    Vendor("northgate", "Northgate Design Studio", ("77 Larimer Square", "Denver, CO 80202"), "Colorado", "CO",
           "USD", "us_simple", "sans", "NDS-{yy}{n:03d}", "%b %d, %Y", 0.029, "design", "EIN"),
    Vendor("redwood", "Redwood Lane Software Inc.", ("500 Congress Avenue, Floor 9", "Austin, TX 78701"), "Texas",
           "TX", "USD", "us_compact", "sans", "RLS-{n:05d}", "%Y-%m-%d", 0.0825, "software", "Tax ID"),
    Vendor("lakeshore", "Lakeshore Office Goods", ("12 Lake Park Drive", "Madison, WI 53703"), "Wisconsin", "WI",
           "USD", "us_simple", "serif", "LOG/{n:05d}", "%m/%d/%Y", 0.055, "supplies", "EIN"),
]
VENDOR_BY_KEY = {v.key: v for v in VENDORS}

BUYERS = {
    "INR": [("Ananta Retail Pvt Ltd", ("5 Civil Lines", "Jaipur 302006"), "Rajasthan", "08"),
            ("Vrindavan Hospitality", ("Mall Road", "Patiala 147001"), "Punjab", "03"),
            ("Sahyadri Clinics LLP", ("FC Road, Shivajinagar", "Pune 411004"), "Maharashtra", "27"),
            ("Tapti Packaging Works", ("Ring Road", "Surat 395002"), "Gujarat", "24"),
            ("Kaveri Learning Centre", ("Jayanagar 4th Block", "Bengaluru 560011"), "Karnataka", "29")],
    "USD": [("Brightwater Dental Group", ("90 Pearl Street", "Hartford, CT 06103")),
            ("Summit Ridge Outfitters", ("2300 Mountain Road", "Boise, ID 83702")),
            ("Clearfield Learning Lab", ("16 College Avenue", "Ithaca, NY 14850")),
            ("Maple Hollow Bakery", ("8 Market Street", "Burlington, VT 05401"))],
}

# description, HSN/SAC, unit price range, quantity range, decimal quantities?
CATALOGUES = {
    "textiles": [("Cotton fabric roll, 40 count", "5208", (850, 2400), (2, 40), False),
                 ("Printed cotton bedsheet set", "6302", (450, 1300), (5, 60), False),
                 ("Handloom cushion cover", "6304", (180, 420), (10, 120), False),
                 ("Dyed yarn, per kg", "5205", (310, 520), (5, 80), True),
                 ("Block-printed table runner", "6302", (260, 640), (6, 50), False)],
    "agro": [("Hybrid wheat seed, 40 kg bag", "1001", (1800, 3200), (2, 30), False),
             ("Drip irrigation kit, 1 acre", "8424", (9500, 22000), (1, 4), False),
             ("Organic compost, per quintal", "3101", (650, 1200), (2, 25), True),
             ("Knapsack sprayer, 16 L", "8424", (1400, 2600), (1, 12), False),
             ("Mulching film roll", "3920", (2100, 3900), (1, 10), False)],
    "office": [("A4 copier paper, box of 5 reams", "4802", (1150, 1600), (2, 30), False),
               ("Ergonomic mesh chair", "9401", (5200, 9800), (1, 12), False),
               ("Laser toner cartridge", "8443", (2400, 5600), (1, 10), False),
               ("Whiteboard 4 x 3 ft", "9610", (1900, 3400), (1, 6), False),
               ("Annual printer maintenance", "998719", (6000, 15000), (1, 2), False)],
    "electrical": [("LED panel light 18 W", "9405", (420, 980), (10, 80), False),
                   ("Copper wire 2.5 sq mm, 90 m", "8544", (2800, 4300), (2, 20), False),
                   ("MCB distribution board, 8 way", "8537", (1600, 3200), (1, 10), False),
                   ("Ceiling fan, 1200 mm", "8414", (1900, 3600), (2, 24), False),
                   ("Installation labour, per hour", "998732", (350, 600), (4, 40), True)],
    "freight": [("FTL transport Bengaluru-Chennai", "996511", (18000, 32000), (1, 4), False),
                ("Part load, per kg", "996511", (9, 18), (200, 2400), False),
                ("Warehouse storage, per pallet-day", "996729", (45, 90), (20, 400), False),
                ("Loading and unloading charges", "996719", (1200, 3500), (1, 6), False),
                ("Transit insurance", "997136", (800, 2600), (1, 3), False)],
    "printing": [("Brochure A4, 4 page, 300 gsm", "4911", (14, 38), (200, 3000), False),
                 ("Visiting cards, box of 100", "4911", (180, 420), (5, 60), False),
                 ("Flex banner, per sq ft", "4911", (16, 32), (40, 600), True),
                 ("Annual report printing", "4901", (180, 420), (50, 500), False),
                 ("Design and proofing charges", "998391", (2500, 9000), (1, 3), False)],
    "consulting": [("Data pipeline review, per hour", None, (120, 210), (4, 60), True),
                   ("Dashboard build, fixed fee", None, (2400, 7800), (1, 2), False),
                   ("Analytics workshop, half day", None, (900, 1900), (1, 3), False),
                   ("Monthly reporting retainer", None, (1500, 4200), (1, 3), False)],
    "supplies": [("Copy paper, case of 10 reams", None, (38, 62), (1, 30), False),
                 ("Ballpoint pens, box of 50", None, (9, 21), (2, 40), False),
                 ("Stackable storage bins", None, (12, 34), (2, 24), False),
                 ("Desk lamp, LED", None, (24, 68), (1, 15), False),
                 ("Shipping boxes, bundle of 25", None, (18, 44), (1, 20), False)],
    "design": [("Brand identity package", None, (1800, 5200), (1, 1), False),
               ("Illustration, per hour", None, (65, 140), (3, 40), True),
               ("Print-ready brochure layout", None, (450, 1300), (1, 3), False),
               ("Website mockups, 5 pages", None, (1200, 3600), (1, 2), False)],
    "software": [("Annual license, team plan", None, (1200, 9600), (1, 3), False),
                 ("Additional user seats", None, (96, 240), (5, 60), False),
                 ("Implementation support, hour", None, (140, 220), (2, 30), True),
                 ("Premium support, 12 months", None, (1500, 4800), (1, 1), False)],
}

ERROR_TYPES_INR = ("total_mismatch", "line_sum_mismatch", "due_before_issue", "gstin_checksum")
ERROR_TYPES_USD = ("total_mismatch", "line_sum_mismatch", "due_before_issue")


# ---------------------------------------------------------------- helpers
def make_gstin(rng: np.random.Generator, state_code: str, name: str, valid: bool = True) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    pan = ("".join(rng.choice(list(letters), 3)) + rng.choice(["C", "F"]) + name[0].upper()
           + f"{int(rng.integers(0, 10000)):04d}" + rng.choice(list(letters)))
    body = f"{state_code}{pan}{int(rng.integers(1, 10))}Z"
    check = gstin_checksum(body)
    if not valid:
        check = GSTIN_CHARS[(GSTIN_CHARS.index(check) + int(rng.integers(1, 35))) % 36]
    return body + check


def make_ein(rng: np.random.Generator) -> str:
    return f"{int(rng.integers(10, 99))}-{int(rng.integers(1000000, 9999999))}"


def money(x: float, indian: bool) -> str:
    neg = x < 0
    whole, frac = f"{abs(x):.2f}".split(".")
    if indian and len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    else:
        whole = f"{int(whole):,}"
    return ("-" if neg else "") + f"{whole}.{frac}"


def qty_text(q: float) -> str:
    return f"{q:g}"


def fiscal_year(d: date) -> str:
    start = d.year if d.month >= 4 else d.year - 1
    return f"{start % 100:02d}-{(start + 1) % 100:02d}"


# ---------------------------------------------------------------- the invoice content
@dataclass
class InvoiceSpec:
    id: str
    vendor: str
    buyer: str
    buyer_address: tuple
    buyer_tax_id: str | None
    intra_state: bool
    vendor_tax_id: str
    invoice_number: str
    issue_date: date
    due_date: date | None
    line_items: list = field(default_factory=list)       # (description, hsn, qty, unit_price, amount)
    subtotal: float = 0.0
    tax_lines: list = field(default_factory=list)        # (label, amount)
    tax: float = 0.0
    total: float = 0.0
    document_errors: list = field(default_factory=list)


def make_spec(rng: np.random.Generator, idx: int, vendor: Vendor, error: str | None) -> InvoiceSpec:
    start = date(2025, 4, 1)
    issue = start + timedelta(days=int(rng.integers(0, 580)))
    terms = int(rng.choice([7, 15, 30, 30, 45, 60]))
    due = issue + timedelta(days=terms) if rng.random() > 0.1 else None   # ~10% say "Due on receipt"
    n = int(rng.integers(12, 9999))
    number = vendor.number_pattern.format(n=n, fy=fiscal_year(issue), yy=f"{issue.year % 100:02d}")
    if vendor.currency == "INR":
        b = BUYERS["INR"][int(rng.integers(0, len(BUYERS["INR"])))]
        buyer, baddr, bstate, bcode = b
        buyer_tax = make_gstin(rng, bcode, buyer)
        intra = bcode == vendor.state_code
        vendor_tax = make_gstin(rng, vendor.state_code, vendor.name, valid=error != "gstin_checksum")
    else:
        buyer, baddr = BUYERS["USD"][int(rng.integers(0, len(BUYERS["USD"])))]
        buyer_tax, intra = None, False
        vendor_tax = make_ein(rng)
    cat = CATALOGUES[vendor.catalogue]
    k = int(rng.integers(1, min(7, len(cat)) + 1))
    items = []
    for j in rng.choice(len(cat), size=k, replace=False):
        desc, hsn, (plo, phi), (qlo, qhi), dec = cat[int(j)]
        price = round(float(rng.uniform(plo, phi)), 0 if phi > 500 and rng.random() < 0.6 else 2)
        qty = float(rng.integers(qlo, qhi + 1))
        if dec and rng.random() < 0.6:
            qty = qty + 0.5
        items.append((desc, hsn, qty, price, round(qty * price, 2)))
    subtotal = round(sum(i[4] for i in items), 2)
    spec = InvoiceSpec(f"inv_{idx:04d}", vendor.key, buyer, baddr, buyer_tax, intra, vendor_tax, number, issue, due,
                       items)
    rate = vendor.tax_rate
    if vendor.currency == "INR":
        pct = f"{rate * 50:g}"
        if intra:
            half = round(subtotal * rate / 2, 2)
            spec.tax_lines = [(f"CGST @ {pct}%", half), (f"SGST @ {pct}%", half)]
        else:
            spec.tax_lines = [(f"IGST @ {rate * 100:g}%", round(subtotal * rate, 2))]
    else:
        spec.tax_lines = [(f"Sales Tax ({rate * 100:g}%)", round(subtotal * rate, 2))]
    spec.subtotal = subtotal
    spec.tax = round(sum(t[1] for t in spec.tax_lines), 2)
    spec.total = round(subtotal + spec.tax, 2)

    # a printed mistake: ground truth is what is printed, plus the error label the validators should raise
    if error == "total_mismatch":
        delta = max(1.0, round(spec.total * float(rng.uniform(0.01, 0.06))))
        spec.total = round(spec.total + delta * rng.choice([-1, 1]), 2)
    elif error == "line_sum_mismatch":
        delta = max(1.0, round(subtotal * float(rng.uniform(0.02, 0.08))))
        spec.subtotal = round(subtotal + delta * rng.choice([-1, 1]), 2)
        spec.total = round(spec.subtotal + spec.tax, 2)       # the rest of the page is consistent with the typo
    elif error == "due_before_issue":
        spec.due_date = issue - timedelta(days=int(rng.integers(3, 40)))
    if error:
        spec.document_errors = [error]
    return spec


# ---------------------------------------------------------------- layouts
def _header_kv(ops, labels_values, x_label, x_value, y0, step, size, font, value_align="right"):
    for i, (lab, val) in enumerate(labels_values):
        y = y0 + i * step
        ops.append(Txt(x_label, y, lab, size, font))
        ops.append(Txt(x_value, y, val, size, font, align=value_align))


def _due_text(spec: InvoiceSpec, v: Vendor) -> str:
    return spec.due_date.strftime(v.date_format) if spec.due_date else "Due on receipt"


def layout_gst_classic(spec: InvoiceSpec, v: Vendor) -> list:
    f, fb = v.font, f"{v.font}-bold"
    ops = [Txt(PAGE_W / 2, 95, "TAX INVOICE", 34, fb, "center"),
           Txt(70, 175, v.name, 30, fb), Txt(70, 212, v.address[0], 19, f), Txt(70, 240, v.address[1], 19, f),
           Txt(70, 270, f"GSTIN: {spec.vendor_tax_id}", 19, f),
           Txt(70, 298, f"State: {v.state} ({v.state_code})", 19, f)]
    _header_kv(ops, [("Invoice No:", spec.invoice_number), ("Invoice Date:", spec.issue_date.strftime(v.date_format)),
                     ("Due Date:", _due_text(spec, v))], 740, 1170, 175, 36, 20, f)
    ops.append(Rule(70, 325, 1170, 325, 2))
    ops += [Txt(70, 368, "Bill To:", 20, fb), Txt(70, 400, spec.buyer, 20, f),
            Txt(70, 428, spec.buyer_address[0], 19, f),
            Txt(70, 456, spec.buyer_address[1], 19, f), Txt(70, 484, f"GSTIN: {spec.buyer_tax_id}", 19, f)]
    top = 520
    ops += [Fill(70, top, 1170, top + 44), Rule(70, top, 1170, top), Rule(70, top + 44, 1170, top + 44)]
    hy = top + 30
    ops += [Txt(80, hy, "#", 19, fb), Txt(120, hy, "Description", 19, fb), Txt(580, hy, "HSN/SAC", 19, fb),
            Txt(760, hy, "Qty", 19, fb, "right"), Txt(940, hy, "Rate", 19, fb, "right"),
            Txt(1160, hy, "Amount", 19, fb, "right")]
    y = top + 44
    for i, (desc, hsn, q, p, amt) in enumerate(spec.line_items, 1):
        by = y + 30
        ops += [Txt(80, by, str(i), 19, f), Txt(120, by, desc, 19, f), Txt(580, by, hsn or "", 19, f),
                Txt(760, by, qty_text(q), 19, f, "right"), Txt(940, by, money(p, True), 19, f, "right"),
                Txt(1160, by, money(amt, True), 19, f, "right"), Rule(70, y + 44, 1170, y + 44, 1, LIGHT)]
        y += 44
    ops.append(Rule(70, y, 1170, y, 2))
    y += 45
    rows = [("Sub Total", spec.subtotal)] + spec.tax_lines
    for lab, val in rows:
        ops += [Txt(740, y, lab, 20, f), Txt(1160, y, money(val, True), 20, f, "right")]
        y += 36
    ops += [Rule(740, y - 22, 1170, y - 22, 1), Txt(740, y + 8, "Total (INR)", 22, fb),
            Txt(1160, y + 8, money(spec.total, True), 22, fb, "right")]
    ops += [Txt(1160, 1540, f"For {v.name}", 18, f, "right"), Txt(1160, 1600, "Authorised Signatory", 18, f, "right")]
    return ops


def layout_gst_modern(spec: InvoiceSpec, v: Vendor) -> list:
    f, fb = v.font, f"{v.font}-bold"
    blue = (30, 60, 120)
    ops = [Txt(70, 130, v.name, 30, fb, color=blue), Txt(70, 168, f"{v.address[0]}, {v.address[1]}", 19, f),
           Txt(70, 198, f"{v.state} · GSTIN {spec.vendor_tax_id}", 19, f), Txt(1170, 130, "INVOICE", 44, fb, "right",
                                                                                 blue)]
    _header_kv(ops, [("Invoice #", spec.invoice_number), ("Invoice Date", spec.issue_date.strftime(v.date_format)),
                     ("Due Date", _due_text(spec, v))], 760, 1170, 250, 36, 20, f)
    ops += [Txt(70, 260, "Billed To", 20, fb, color=blue), Txt(70, 292, spec.buyer, 20, f),
            Txt(70, 320, f"{spec.buyer_address[0]}, {spec.buyer_address[1]}", 19, f),
            Txt(70, 348, f"Buyer GSTIN: {spec.buyer_tax_id}", 19, f)]
    top = 440
    ops += [Rule(70, top, 1170, top, 3, blue), Rule(70, top + 46, 1170, top + 46, 1, blue)]
    hy = top + 32
    ops += [Txt(70, hy, "Item", 20, fb), Txt(780, hy, "Qty", 20, fb, "right"),
            Txt(980, hy, "Unit Price", 20, fb, "right"), Txt(1170, hy, "Total", 20, fb, "right")]
    y = top + 46
    for i, (desc, _hsn, q, p, amt) in enumerate(spec.line_items):
        if i % 2:
            ops.append(Fill(70, y + 2, 1170, y + 44, (244, 245, 248)))
        by = y + 31
        ops += [Txt(70, by, desc, 19, f), Txt(780, by, qty_text(q), 19, f, "right"),
                Txt(980, by, money(p, False), 19, f, "right"), Txt(1170, by, money(amt, False), 19, f, "right")]
        y += 46
    ops.append(Rule(70, y + 4, 1170, y + 4, 1, blue))
    y += 50
    for lab, val in [("Taxable Value", spec.subtotal)] + [(t.replace(" @", ""), a) for t, a in spec.tax_lines]:
        ops += [Txt(760, y, lab, 20, f), Txt(1170, y, money(val, False), 20, f, "right")]
        y += 36
    ops += [Fill(750, y - 24, 1180, y + 20, (232, 237, 246)), Txt(760, y + 6, "Grand Total (INR)", 22, fb),
            Txt(1170, y + 6, "Rs. " + money(spec.total, False), 22, fb, "right")]
    ops.append(Txt(70, 1600, "Thank you for your business.", 18, f, color=GREY))
    return ops


def layout_us_simple(spec: InvoiceSpec, v: Vendor) -> list:
    f, fb = v.font, f"{v.font}-bold"
    ops = [Txt(70, 120, "INVOICE", 46, fb), Txt(70, 190, v.name, 26, fb), Txt(70, 224, v.address[0], 19, f),
           Txt(70, 252, v.address[1], 19, f), Txt(70, 282, f"{v.tax_id_label}: {spec.vendor_tax_id}", 19, f)]
    _header_kv(ops, [("Invoice Number", spec.invoice_number), ("Date", spec.issue_date.strftime(v.date_format)),
                     ("Due Date", _due_text(spec, v))], 720, 1170, 190, 36, 20, f)
    ops += [Txt(70, 380, "Bill To", 20, fb), Txt(70, 412, spec.buyer, 20, f),
            Txt(70, 440, spec.buyer_address[0], 19, f),
            Txt(70, 468, spec.buyer_address[1], 19, f)]
    top = 520
    ops += [Fill(70, top, 1170, top + 44, (205, 208, 214))]
    hy = top + 30
    ops += [Txt(85, hy, "Description", 19, fb), Txt(740, hy, "Qty", 19, fb, "right"),
            Txt(950, hy, "Unit Price", 19, fb, "right"), Txt(1155, hy, "Amount", 19, fb, "right")]
    y = top + 44
    for desc, _hsn, q, p, amt in spec.line_items:
        by = y + 31
        ops += [Txt(85, by, desc, 19, f), Txt(740, by, qty_text(q), 19, f, "right"),
                Txt(950, by, money(p, False), 19, f, "right"), Txt(1155, by, money(amt, False), 19, f, "right"),
                Rule(70, y + 44, 1170, y + 44, 1, LIGHT)]
        y += 44
    y += 50
    for lab, val in [("Subtotal", spec.subtotal)] + spec.tax_lines:
        ops += [Txt(740, y, lab, 20, f), Txt(1155, y, "$" + money(val, False), 20, f, "right")]
        y += 36
    ops += [Rule(740, y - 22, 1170, y - 22, 2, INK), Txt(740, y + 10, "Total Due (USD)", 22, fb),
            Txt(1155, y + 10, "$" + money(spec.total, False), 22, fb, "right")]
    ops.append(Txt(70, 1600, "Please include the invoice number with your payment.", 18, f, color=GREY))
    return ops


def layout_us_compact(spec: InvoiceSpec, v: Vendor) -> list:
    f, fb = v.font, f"{v.font}-bold"
    ops = [Txt(70, 110, v.name, 28, fb), Txt(70, 146, f"{v.address[0]} · {v.address[1]}", 18, f),
           Txt(70, 174, f"{v.tax_id_label} {spec.vendor_tax_id}", 18, f), Rule(70, 200, 1170, 200, 1, INK)]
    _header_kv(ops, [("Invoice no.", spec.invoice_number), ("Issued", spec.issue_date.strftime(v.date_format)),
                     ("Payment due", _due_text(spec, v))], 70, 290, 250, 34, 20, f, value_align="left")
    ops += [Txt(700, 250, "Bill to", 20, fb), Txt(700, 284, spec.buyer, 20, f),
            Txt(700, 312, spec.buyer_address[0], 18, f),
            Txt(700, 340, spec.buyer_address[1], 18, f)]
    top = 420
    ops += [Rule(70, top, 1170, top, 1, INK), Rule(70, top + 44, 1170, top + 44, 1, INK)]
    hy = top + 30
    ops += [Txt(140, hy, "Qty", 19, fb, "right"), Txt(180, hy, "Item", 19, fb), Txt(920, hy, "Price", 19, fb, "right"),
            Txt(1170, hy, "Line total", 19, fb, "right")]
    y = top + 44
    for desc, _hsn, q, p, amt in spec.line_items:
        by = y + 31
        ops += [Txt(140, by, qty_text(q), 19, f, "right"), Txt(180, by, desc, 19, f),
                Txt(920, by, money(p, False), 19, f, "right"), Txt(1170, by, money(amt, False), 19, f, "right")]
        y += 42
    ops.append(Rule(70, y + 8, 1170, y + 8, 1, INK))
    y += 52
    for lab, val in [("Subtotal", spec.subtotal)] + [(t.replace("Sales ", "").replace("(", "").replace(")", ""), a)
                                                    for t, a in spec.tax_lines]:
        ops += [Txt(800, y, lab, 20, f), Txt(1170, y, money(val, False), 20, f, "right")]
        y += 34
    ops += [Txt(800, y + 8, "Balance due", 22, fb), Txt(1170, y + 8, "USD " + money(spec.total, False), 22, fb,
                                                         "right")]
    return ops


LAYOUTS = {"gst_classic": layout_gst_classic, "gst_modern": layout_gst_modern, "us_simple": layout_us_simple,
           "us_compact": layout_us_compact}


def layout(spec: InvoiceSpec) -> list:
    v = VENDOR_BY_KEY[spec.vendor]
    ops = LAYOUTS[v.template](spec, v)
    ops.append(Rule(70, 1660, 1170, 1660, 1, LIGHT))
    ops.append(Txt(PAGE_W / 2, 1690, FOOTER, 15, "sans", "center", GREY))
    return ops


# ---------------------------------------------------------------- scan degradation
def degrade(img: Image.Image, spec: dict, rng: np.random.Generator) -> tuple[Image.Image, dict]:
    """Make a clean render look scanned. Returns the image and the parameters actually applied."""
    import cv2

    applied = {"rotation_deg": 0.0}
    a = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = a.shape
    if spec["rotation"]:
        angle = float(rng.uniform(0.5, 1.0) * spec["rotation"] * rng.choice([-1, 1]))
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)        # counter-clockwise for positive angles
        a = cv2.warpAffine(a, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
        applied["rotation_deg"] = round(angle, 3)
    if spec["downscale"] < 1:
        small = cv2.resize(a, (int(w * spec["downscale"]), int(h * spec["downscale"])), interpolation=cv2.INTER_AREA)
        a = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if spec["blur"]:
        a = cv2.GaussianBlur(a, (0, 0), spec["blur"])
    if spec["contrast"] < 1:
        a = 255 - (255 - a) * spec["contrast"]
        a = a * np.linspace(1.0, 0.94, w, dtype=np.float32)[None, :]   # uneven lighting across the page
    if spec["noise"]:
        a = a + rng.normal(0, spec["noise"], a.shape).astype(np.float32)
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    applied.update({k: spec[k] for k in ("noise", "blur", "jpeg", "contrast", "downscale")})
    return out, applied


def _jpeg_bytes(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


# ---------------------------------------------------------------- dataset
def plan_dataset(n: int, params: dict, seed: int) -> list[dict]:
    """Which vendor, source kind, degradation level and printed mistake each invoice gets."""
    rng = np.random.default_rng(seed)
    n_text = round(n * params["share_pdf_text"])
    n_scan = round(n * params["share_pdf_scan"])
    kinds = ["pdf_text"] * n_text + ["pdf_scan"] * n_scan + ["image"] * (n - n_text - n_scan)
    levels = []
    img_i = scan_i = 0
    for k in kinds:
        if k == "pdf_text":
            levels.append(0)
        elif k == "pdf_scan":
            levels.append(1 + scan_i % 3)
            scan_i += 1
        else:
            levels.append(img_i % 4)
            img_i += 1
    order = rng.permutation(n)
    n_err = round(n * params["share_document_errors"])
    err_idx = set(rng.choice(n, size=n_err, replace=False).tolist())
    plan = []
    for i in range(n):
        j = int(order[i])
        v = VENDORS[i % len(VENDORS)]
        err = None
        if i in err_idx:
            types = ERROR_TYPES_INR if v.currency == "INR" else ERROR_TYPES_USD
            err = types[len([x for x in err_idx if x < i]) % len(types)]
        plan.append({"idx": i, "vendor": v.key, "kind": kinds[j], "level": levels[j], "error": err})
    return plan


def spec_to_truth(spec: InvoiceSpec, vendor: Vendor) -> dict:
    return {
        "vendor_name": vendor.name, "vendor_tax_id": spec.vendor_tax_id, "invoice_number": spec.invoice_number,
        "issue_date": spec.issue_date.isoformat(),
        "due_date": spec.due_date.isoformat() if spec.due_date else None, "currency": vendor.currency,
        "line_items": [{"description": d, "quantity": q, "unit_price": p, "amount": a}
                       for d, _h, q, p, a in spec.line_items],
        "subtotal": spec.subtotal, "tax": spec.tax, "total": spec.total,
    }


def generate(out_dir: Path, gt_dir: Path, params: dict, n: int | None = None, seed: int | None = None) -> list[dict]:
    n = n or params["n_invoices"]
    seed = params["seed"] if seed is None else seed
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + 1)
    records = []
    for p in plan_dataset(n, params, seed):
        v = VENDOR_BY_KEY[p["vendor"]]
        spec = make_spec(rng, p["idx"], v, p["error"])
        ops = layout(spec)
        kind, level = p["kind"], p["level"]
        applied: dict = {}
        if kind == "pdf_text":
            fname = f"{spec.id}.pdf"
            render_pdf(ops, out_dir / fname)
        else:
            img = render_image(ops)
            if level > 0:
                img, applied = degrade(img, params["degradation"][level], rng)
            q = params["degradation"][level]["jpeg"]
            if kind == "pdf_scan":
                fname = f"{spec.id}.pdf"
                page = Image.open(io.BytesIO(_jpeg_bytes(img, q or 90)))
                page.save(out_dir / fname, "PDF", resolution=DPI)
            elif q:
                fname = f"{spec.id}.jpg"
                (out_dir / fname).write_bytes(_jpeg_bytes(img, q))
            else:
                fname = f"{spec.id}.png"
                img.save(out_dir / fname)
        rec = {"id": spec.id, "file": fname, "synthetic": True, "vendor": v.key, "template": v.template,
               "source_kind": kind, "degradation_level": level, "degradation": applied,
               "document_errors": spec.document_errors, "truth": spec_to_truth(spec, v),
               "printed": {"issue_date": spec.issue_date.strftime(v.date_format),
                           "due_date": _due_text(spec, v), "date_format": v.date_format}}
        (gt_dir / f"{spec.id}.json").write_text(json.dumps(rec, indent=2))
        records.append(rec)
    (gt_dir / "_manifest.json").write_text(json.dumps({"synthetic": True, "seed": seed, "n": n, "params": params,
                                                      "vendors": [asdict(v) for v in VENDORS]}, indent=2,
                                                     default=str))
    return records
