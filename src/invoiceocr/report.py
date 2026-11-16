"""README images, rendered from outputs/metrics.json, outputs/results.csv and one re-OCR'd sample page."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from .config import Paths  # noqa: E402
from .schema import SCALAR_FIELDS  # noqa: E402

INK, MUTED, GRID = "#0e1a2b", "#5b6676", "#e3e7ee"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
GOOD, WARN, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlesize": 11.5, "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})
FIELD_LABELS = {"vendor_name": "vendor name", "vendor_tax_id": "vendor tax id (GSTIN/EIN)",
                "invoice_number": "invoice number", "issue_date": "issue date", "due_date": "due date",
                "currency": "currency", "subtotal": "subtotal", "tax": "tax", "total": "total"}


def _save(fig, out: Path, name: str) -> Path:
    p = out / name
    fig.savefig(p, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return p


def _suptitle(fig, text: str, sub: str | None = None):
    fig.suptitle(text, x=0.01, ha="left", fontsize=13, weight="bold", color=INK, y=1.02 if sub else 0.99)
    if sub:
        fig.text(0.01, 0.955, sub, ha="left", fontsize=9, color=MUTED)


# ---------------------------------------------------------------- architecture
def architecture(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    def box(x, y, w, h, title, sub, color=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white",
                                    ec=color, lw=2))
        ax.text(x + w / 2, y + h * 0.73, title, ha="center", va="center", fontsize=10.5, weight="bold", color=color)
        ax.text(x + w / 2, y + h * 0.33, sub, ha="center", va="center", fontsize=8, color=MUTED, linespacing=1.4)

    def arrow(x1, y1, x2, y2, color=MUTED, style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13, lw=1.5, color=color))

    ax.text(0.15, 5.3, "invoice-ocr-extraction: document → words with boxes → fields → validated JSON → approve or "
            "review", fontsize=13, weight="bold", color=INK)
    box(0.2, 3.0, 1.9, 1.6, "Document", "PNG / JPEG scan\nscanned PDF\ndigital PDF", INK)
    box(2.6, 3.65, 2.3, 1.25, "Text layer", "pdfplumber words\n(no OCR, conf 100)", VIOLET)
    box(2.6, 2.05, 2.3, 1.4, "OCR", "OpenCV: deskew, denoise\n(NL-means) · Tesseract 5\nimage_to_data (box+conf)",
        BLUE)
    arrow(2.1, 4.1, 2.6, 4.25)
    arrow(2.1, 3.5, 2.6, 2.9)
    box(5.4, 2.6, 2.1, 1.75, "Layout", "words → lines →\nsegments (column\ngaps kept apart)", INK)
    arrow(4.9, 4.25, 5.4, 3.7)
    arrow(4.9, 2.75, 5.4, 3.2)
    box(8.0, 3.55, 2.3, 1.3, "Rule extractor", "label anchors · table\ncolumns from header\nGSTIN/EIN · dates", BLUE)
    box(8.0, 1.85, 2.3, 1.35, "LLM extractor", "OpenAI structured output\nschema from pydantic\nonly with an API key",
        MUTED)
    arrow(7.5, 3.75, 8.0, 4.2)
    arrow(9.15, 3.55, 9.15, 3.2, MUTED)
    ax.text(9.22, 3.375, "hybrid: weak fields", fontsize=7.2, color=MUTED, va="center")
    box(10.8, 2.6, 2.05, 1.75, "Validate", "pydantic Invoice:\nformat, checksum,\narithmetic, dates", GOOD)
    arrow(10.3, 4.2, 10.8, 3.75)
    arrow(10.3, 2.5, 10.8, 3.1, MUTED)
    box(10.8, 0.35, 2.05, 1.45, "Route", "auto_approved\nor needs_review\n+ reasons", SERIOUS)
    arrow(11.82, 2.6, 11.82, 1.8)
    box(5.4, 0.35, 4.9, 1.3, "Evaluate (SYNTHETIC ground truth)",
        "field accuracy / F1 · line items · by scan level\nSTP rate · errors caught · OCR time per page", INK)
    arrow(10.8, 1.05, 10.3, 1.0)
    box(0.2, 0.35, 4.7, 1.3, "Generator", "11 fictional vendors · 4 layouts · INR / USD\nscan degradation L0-L3 · "
        "planted mistakes", INK)
    arrow(1.15, 1.65, 1.15, 3.0)
    ax.text(0.2, 0.08, "Served by `python -m invoiceocr extract` and POST /extract (Starlette). The LLM path is never "
            "called by the demo or CI.", fontsize=7.8, color=MUTED)
    return _save(fig, out, "architecture.png")


# ---------------------------------------------------------------- charts from metrics
def field_accuracy(out: Path, m: dict) -> Path:
    fields = list(SCALAR_FIELDS)
    acc = [m["fields"][f]["accuracy"] for f in fields]
    labels = [FIELD_LABELS[f] for f in fields] + ["line items (F1)", "whole table exact"]
    vals = acc + [m["line_items"]["f1"], m["line_items"]["tables_exact"]]
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(10, 4.9))
    colors = [BLUE if i < len(fields) else AQUA for i in order]
    ax.barh([labels[i] for i in order], [vals[i] for i in order], color=colors, height=0.62)
    for y, i in enumerate(order):
        ax.text(vals[i] + 0.006, y, f"{vals[i]:.1%}", va="center", fontsize=8.8, color=INK)
    ax.set_xlim(0, 1.09)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"share of {m['n_invoices']} SYNTHETIC invoices with the field exactly right (rule extractor)")
    worst = labels[int(order[0])]
    ax.set_title(f"Field accuracy across all scan conditions · weakest: {worst}", fontsize=12)
    return _save(fig, out, "field_accuracy.png")


def accuracy_by_condition(out: Path, m: dict) -> Path:
    bc = m["by_condition"]
    conds = list(bc)
    labels = [f"{bc[c]['label']}\nn={bc[c]['n']}" for c in conds]
    panels = [("field_accuracy", "Field accuracy (9 fields)", BLUE),
              ("line_item_f1", "Line-item F1", AQUA),
              ("stp_rate", "Approved without review (STP)", VIOLET)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)
    x = np.arange(len(conds))
    for ax, (key, title, color) in zip(axes, panels):
        vals = [bc[c][key] for c in conds]
        ax.bar(x, vals, color=color, width=0.62)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.015, f"{v:.0%}", ha="center", fontsize=8.6, color=INK)
        ax.set_xticks(x, labels, fontsize=7.6)
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
        ax.set_title(title)
    wrong = sum(bc[c]["auto_with_error"] for c in conds)
    axes[2].text(0.98, 0.97, f"auto-approved with a wrong field: {wrong}", transform=axes[2].transAxes, ha="right",
                 va="top", fontsize=8.4, color=INK)
    _suptitle(fig, "Accuracy falls with scan quality; the router lowers approvals instead of passing errors through",
              "Real Tesseract 5 runs on SYNTHETIC invoices. L1-L3 add rotation, blur, noise, low contrast and JPEG "
              "artifacts (params.yaml). Approvals below 100% on clean input are the planted mistakes, held for review.")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out, "accuracy_by_condition.png")


def review_routing(out: Path, m: dict) -> Path:
    r = m["routing"]
    s = r["status_by_correctness"]
    parts = [("auto-approved, all fields right", s["auto_correct"], GOOD),
             ("auto-approved, a field wrong", s["auto_wrong"], CRITICAL),
             ("review: printed mistake caught", s["review_planted_error"], SERIOUS),
             ("review: extraction error caught", s["review_extraction_error"], WARN),
             ("review: correct but unconfident", s["review_but_correct"], "#a7b0bd")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    left = 0
    n = m["n_invoices"]
    for label, v, color in parts:
        if v:
            a1.barh([0], [v], left=left, color=color, height=0.5, edgecolor="white", linewidth=2)
            if v / n > 0.06:
                a1.text(left + v / 2, 0, str(v), ha="center", va="center", fontsize=10, color=INK, weight="bold")
        left += v
    a1.set_xlim(0, n)
    a1.set_yticks([])
    a1.set_xlabel("invoices")
    handles = [Rectangle((0, 0), 1, 1, color=c) for _, _, c in parts]
    a1.legend(handles, [f"{lab} ({v})" for lab, v, _ in parts], loc="upper center", bbox_to_anchor=(0.5, -0.25),
              ncol=2, frameon=False, fontsize=8.6)
    a1.set_title(f"Where {n} invoices ended up · STP {r['stp_rate']:.0%}")
    a1.spines["left"].set_visible(False)
    reasons = r["reasons"]
    names = list(reasons)[::-1]
    a2.barh(names, [reasons[k] for k in names], color=BLUE, height=0.6)
    for i, k in enumerate(names):
        a2.text(reasons[k] + 0.4, i, str(reasons[k]), va="center", fontsize=8.6, color=INK)
    a2.set_xlabel("invoices with this reason (one invoice can have several)")
    a2.grid(axis="x", color=GRID)
    a2.set_axisbelow(True)
    a2.set_title("Why invoices were sent to review")
    n_err = r["n_with_extraction_error"]
    _suptitle(fig, f"Review routing: {n_err - r['n_auto_with_error']} of {n_err} invoices with a misread field were "
              f"held back; {m['document_errors']['caught']}/{m['document_errors']['n']} printed mistakes caught")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out, "review_routing.png")


def degradation_strip(out: Path, paths: Paths) -> Path:
    """The same header region at each scan level, from the generated files."""
    rows = list(csv.DictReader((paths.outputs / "results.csv").open()))
    picks = []
    for c in ("L0", "L1", "L2", "L3"):
        r = next((r for r in rows if r["condition"] == c and r["source_kind"] == "image"
                  and r["template"] == "gst_classic"), None)
        if r:
            picks.append((c, r))
    fig, axes = plt.subplots(1, len(picks), figsize=(14, 3.0))
    for ax, (c, r) in zip(np.atleast_1d(axes), picks):
        img = Image.open(paths.invoices / r["file"]).convert("L")
        ax.imshow(np.asarray(img.crop((40, 120, 640, 520))), cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(GRID)
        ax.set_title(f"{c} · fields right {sum(int(r[f'ok_{f}']) for f in SCALAR_FIELDS)}/9 · {r['status']}",
                     fontsize=9.5)
    _suptitle(fig, "Scan degradation levels (crops of SYNTHETIC invoices, as fed to the OCR stage)")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, out, "degradation_levels.png")


# ---------------------------------------------------------------- sample overlay
def overlay_image(image: np.ndarray, page, result, decision, out_path: Path, title: str | None = None) -> Path:
    """Page with every OCR word box (coloured by confidence) and the boxes the extracted fields came from."""
    h, w = image.shape[:2]
    crop_h = min(h, int(max([b[3] for boxes in result.evidence.values() for b in boxes] + [h * 0.5]) + 60))
    fig = plt.figure(figsize=(14, 8.2))
    ax = fig.add_axes((0.0, 0.0, 0.58, 0.93))
    top = int(max(0, min([b[1] for boxes in result.evidence.values() for b in boxes] + [h]) - 140))
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    for wd in page.words:
        if wd.y0 > crop_h:
            continue
        c = GOOD if wd.conf >= 90 else WARN if wd.conf >= 70 else CRITICAL
        ax.add_patch(Rectangle((wd.x0, wd.y0), wd.x1 - wd.x0, wd.y1 - wd.y0, fill=False, ec=c, lw=0.6, alpha=0.9))
    for field, boxes in result.evidence.items():
        for (x0, y0, x1, y1) in boxes:
            ax.add_patch(Rectangle((x0 - 4, y0 - 4), x1 - x0 + 8, y1 - y0 + 8, fill=False, ec=BLUE, lw=1.8))
        x0, y0, _, y1 = boxes[0]
        label = "line items" if field == "line_items" else field
        if x0 > 0.6 * w:                       # right-hand values: label to the left, clear of the box above
            ax.text(x0 - 10, (y0 + y1) / 2, label, fontsize=6.5, color="white", va="center", ha="right",
                    bbox={"boxstyle": "square,pad=0.15", "fc": BLUE, "ec": "none"})
        else:
            ax.text(x0 - 4, y0 - 7, label, fontsize=6.5, color="white", va="bottom",
                    bbox={"boxstyle": "square,pad=0.15", "fc": BLUE, "ec": "none"})
    ax.set_xlim(0, w)
    ax.set_ylim(crop_h, top)
    ax.axis("off")
    side = fig.add_axes((0.6, 0.0, 0.4, 0.93))
    side.axis("off")
    d = decision.invoice
    lines = [("vendor_name", d.vendor_name), ("vendor_tax_id", d.vendor_tax_id),
             ("invoice_number", d.invoice_number), ("issue_date", d.issue_date), ("due_date", d.due_date),
             ("currency", d.currency), ("subtotal", d.subtotal), ("tax", d.tax), ("total", d.total)]
    y = 0.97
    side.text(0, y, "Extracted (rule extractor)", fontsize=11, weight="bold", color=INK, transform=side.transAxes)
    y -= 0.055
    for k, v in lines:
        c = result.confidence.get(k, 0)
        side.text(0, y, k, fontsize=9, color=MUTED, transform=side.transAxes)
        side.text(0.36, y, f"{v if v is not None else '-'}", fontsize=9, color=INK, transform=side.transAxes)
        side.text(0.98, y, f"{c:.2f}", fontsize=8.5, color=MUTED, ha="right", transform=side.transAxes)
        y -= 0.042
    side.text(0.98, 0.97, "confidence", fontsize=8, color=MUTED, ha="right", transform=side.transAxes)
    y -= 0.02
    side.text(0, y, f"line items ({len(d.line_items)})", fontsize=9.5, weight="bold", color=INK,
              transform=side.transAxes)
    y -= 0.042
    for li in d.line_items[:7]:
        desc = (li.description or "-")[:30]
        side.text(0, y, f"{desc}", fontsize=8, color=INK, transform=side.transAxes)
        side.text(0.98, y, f"{li.quantity:g} × {li.unit_price:,.2f} = {li.amount:,.2f}"
                  if None not in (li.quantity, li.unit_price, li.amount) else "incomplete", fontsize=8, color=MUTED,
                  ha="right", transform=side.transAxes)
        y -= 0.036
    y -= 0.03
    ok = decision.status == "auto_approved"
    side.text(0, y, decision.status, fontsize=12, weight="bold", color=GOOD if ok else CRITICAL,
              transform=side.transAxes)
    y -= 0.045
    msgs = ["arithmetic ties out, GSTIN checksum valid, all unchecked fields confident"] if ok else \
        [f"{r.code}: {r.message}"[:70] for r in decision.reasons[:5]]
    for msg in msgs:
        side.text(0, y, msg, fontsize=8, color=MUTED, transform=side.transAxes)
        y -= 0.035
    y -= 0.02
    for c, lab in ((GOOD, "OCR word conf ≥ 90"), (WARN, "70-89"), (CRITICAL, "< 70"), (BLUE, "field evidence")):
        side.add_patch(Rectangle((0, y - 0.005), 0.03, 0.022, fill=False, ec=c, lw=1.6, transform=side.transAxes))
        side.text(0.045, y, lab, fontsize=8, color=MUTED, transform=side.transAxes)
        y -= 0.033
    fig.suptitle(title or "OCR boxes and extracted fields", x=0.01, ha="left", fontsize=13, weight="bold", color=INK)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out_path


def sample_overlay(out: Path, paths: Paths, params: dict) -> Path:
    from .extract.rules import RuleExtractor
    from .ocr import ocr_document
    from .routing import decide

    rows = list(csv.DictReader((paths.outputs / "results.csv").open()))
    pick = next((r for r in rows if r["condition"] == "L2" and r["template"] == "gst_classic"
                 and r["status"] == "auto_approved"), None) or next(r for r in rows if r["condition"] == "L2")
    p = params["ocr"]
    page, image = ocr_document(paths.invoices / pick["file"], {k: p[k] for k in ("psm", "deskew", "denoise",
                                                                                 "binarize")}, keep_image=True)
    res = RuleExtractor().extract(page)
    dec = decide(res, params["routing"]["min_field_confidence"])
    title = (f"SYNTHETIC invoice {pick['id']} at scan level L2 (deskewed {page.skew_deg:+.1f}°, "
             f"{len(page.words)} words, mean conf {page.mean_conf:.0f})")
    return overlay_image(image, page, res, dec, out / "sample_overlay.png", title)


def render_all(paths: Paths, params: dict) -> list[Path]:
    out = paths.images
    out.mkdir(parents=True, exist_ok=True)
    m = json.loads((paths.outputs / "metrics.json").read_text())
    return [architecture(out), field_accuracy(out, m), accuracy_by_condition(out, m), review_routing(out, m),
            degradation_strip(out, paths), sample_overlay(out, paths, params)]
