"""Run the pipeline on the SYNTHETIC set and score it against ground truth.

Writes one prediction file per invoice, ``outputs/results.csv`` (one row per invoice) and ``outputs/metrics.json``.
Scores: accuracy / precision / recall / F1 per field, line-item matching, results by scan condition, how often the
router approves automatically (straight-through processing), how many extraction mistakes it stops, whether it
catches the mistakes printed on purpose, and OCR time per page.
"""
from __future__ import annotations

import csv
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from .config import Paths
from .extract.hybrid import weak_fields
from .extract.rules import RuleExtractor
from .normalize import norm_text
from .ocr import OcrPage, ocr_document, tesseract_version
from .routing import decide
from .schema import SCALAR_FIELDS

log = logging.getLogger(__name__)
CONDITIONS = ["pdf_text", "L0", "L1", "L2", "L3"]
CONDITION_LABELS = {"pdf_text": "PDF text layer", "L0": "clean scan (L0)", "L1": "light (L1)", "L2": "medium (L2)",
                    "L3": "heavy (L3)"}


def condition(rec: dict) -> str:
    return "pdf_text" if rec["source_kind"] == "pdf_text" else f"L{rec['degradation_level']}"


def field_correct(field: str, truth, pred) -> bool:
    if truth is None or pred is None:
        return truth is None and pred is None
    if field == "vendor_name":
        return norm_text(truth) == norm_text(pred)
    if field in ("vendor_tax_id", "invoice_number"):
        return str(truth).replace(" ", "").upper() == str(pred).replace(" ", "").upper()
    if field in ("subtotal", "tax", "total"):
        return abs(float(truth) - float(pred)) < 0.005
    return truth == pred


def item_matches(t: dict, p: dict) -> bool:
    nums = all(p.get(k) is not None and abs(t[k] - p[k]) < 0.005 for k in ("quantity", "unit_price", "amount"))
    desc = SequenceMatcher(None, norm_text(t["description"]), norm_text(p.get("description"))).ratio() >= 0.8
    return nums and desc


def match_items(truth: list[dict], pred: list[dict]) -> int:
    used, hits = set(), 0
    for t in truth:
        for j, p in enumerate(pred):
            if j not in used and item_matches(t, p):
                used.add(j)
                hits += 1
                break
    return hits


def _ocr_job(args):
    path, settings, cache = args
    page, _ = ocr_document(Path(path), settings, Path(cache))
    return page.to_json()


def ocr_all(files: list[Path], settings: dict, cache_dir: Path, workers: int) -> list[OcrPage]:
    jobs = [(str(f), settings, str(cache_dir)) for f in files]
    if workers <= 1:
        return [OcrPage.from_json(_ocr_job(j)) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [OcrPage.from_json(d) for d in pool.map(_ocr_job, jobs, chunksize=2)]


def evaluate(paths: Paths, params: dict) -> dict:
    t0 = time.perf_counter()
    records = [json.loads(p.read_text()) for p in sorted(paths.ground_truth.glob("inv_*.json"))]
    if not records:
        raise FileNotFoundError(f"no ground truth in {paths.ground_truth}; run `python -m invoiceocr generate`")
    ocr_p = params["ocr"]
    settings = {k: ocr_p[k] for k in ("psm", "deskew", "denoise", "binarize")}
    log.info("OCR on %d documents with %d workers (cache: %s)", len(records), ocr_p["workers"], paths.ocr_cache)
    pages = ocr_all([paths.invoices / r["file"] for r in records], settings, paths.ocr_cache, ocr_p["workers"])
    t_ocr = time.perf_counter() - t0

    min_conf = params["routing"]["min_field_confidence"]
    llm_thr = params["routing"]["llm_fallback_confidence"]
    extractor = RuleExtractor()
    paths.predictions.mkdir(parents=True, exist_ok=True)
    rows = []
    results = []
    for rec, page in zip(records, pages):
        res = extractor.extract(page)
        results.append(res)
        dec = decide(res, min_conf)
        truth, pred = rec["truth"], res.draft.model_dump()
        correct = {f: field_correct(f, truth[f], pred[f]) for f in SCALAR_FIELDS}
        hits = match_items(truth["line_items"], pred["line_items"])
        table_exact = hits == len(truth["line_items"]) == len(pred["line_items"])
        weak = weak_fields(res, llm_thr)
        reasons = [r.code for r in dec.reasons]
        planted = rec["document_errors"]
        rows.append({
            "id": rec["id"], "file": rec["file"], "vendor": rec["vendor"], "template": rec["template"],
            "source_kind": rec["source_kind"], "condition": condition(rec), "ocr_source": page.source,
            "ocr_seconds": page.seconds, "ocr_mean_conf": round(page.mean_conf, 1), "skew_corrected": page.skew_deg,
            "rotation_applied": rec["degradation"].get("rotation_deg", 0.0),
            **{f"ok_{f}": int(v) for f, v in correct.items()},
            **{f"has_{f}": int(pred[f] is not None) for f in SCALAR_FIELDS},
            "items_truth": len(truth["line_items"]), "items_pred": len(pred["line_items"]), "items_matched": hits,
            "table_exact": int(table_exact), "all_correct": int(all(correct.values()) and table_exact),
            "status": dec.status, "reasons": ";".join(sorted(set(reasons))), "planted_error": ";".join(planted),
            "planted_caught": int(bool(planted) and dec.status == "needs_review"
                                  and all(p in reasons for p in planted)),
            "weak_fields": ";".join(weak),
        })
        (paths.predictions / f"{rec['id']}.json").write_text(json.dumps({
            "id": rec["id"], "synthetic": True, "decision": dec.model_dump(), "evidence": res.evidence,
            "notes": res.notes, "ocr": {"source": page.source, "seconds": page.seconds, "skew_deg": page.skew_deg,
                                        "mean_conf": round(page.mean_conf, 1), "words": len(page.words)}},
            indent=2, default=str))

    paths.outputs.mkdir(parents=True, exist_ok=True)
    with (paths.outputs / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    metrics = summarise(rows, records)
    # the review threshold is a business knob: what other settings would have done on the same extractions
    sweep = []
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        st = [decide(res, thr).status for res in results]
        auto = [r for r, s_ in zip(rows, st) if s_ == "auto_approved"]
        sweep.append({"threshold": thr, "stp_rate": round(len(auto) / len(rows), 4),
                      "auto_with_error": sum(1 for r in auto if not r["all_correct"])})
    metrics["routing"]["threshold_sweep"] = sweep
    metrics["ocr"]["settings"] = settings
    metrics["ocr"]["tesseract_version"] = tesseract_version()
    metrics["timing"] = {"ocr_stage_s": round(t_ocr, 1), "total_s": round(time.perf_counter() - t0, 1),
                         "workers": ocr_p["workers"]}
    (paths.outputs / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def _prf(tp: int, n_pred: int, n_truth: int) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_truth if n_truth else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def summarise(rows: list[dict], records: list[dict]) -> dict:
    n = len(rows)
    by_id = {r["id"]: r for r in records}
    fields = {}
    for f in SCALAR_FIELDS:
        acc = np.mean([r[f"ok_{f}"] for r in rows])
        preds = sum(r[f"has_{f}"] for r in rows)
        tp = sum(1 for r in rows if r[f"ok_{f}"] and r[f"has_{f}"])
        n_truth = sum(1 for r in rows if by_id[r["id"]]["truth"][f] is not None)
        fields[f] = {"accuracy": round(float(acc), 4), **_prf(tp, preds, n_truth), "n_truth": n_truth}
    it, ip, im = (sum(r[k] for r in rows) for k in ("items_truth", "items_pred", "items_matched"))
    items = {**_prf(im, ip, it), "n_truth": it, "n_pred": ip, "tables_exact": round(np.mean([r["table_exact"]
                                                                                         for r in rows]), 4)}
    auto = [r for r in rows if r["status"] == "auto_approved"]
    with_err = [r for r in rows if not r["all_correct"]]
    clean = [r for r in rows if r["all_correct"] and not r["planted_error"]]
    reasons: dict[str, int] = {}
    for r in rows:
        for c in filter(None, r["reasons"].split(";")):
            reasons[c] = reasons.get(c, 0) + 1
    planted = [r for r in rows if r["planted_error"]]
    by_type: dict[str, list[int]] = {}
    for r in planted:
        t = by_type.setdefault(r["planted_error"], [0, 0])
        t[0] += 1
        t[1] += r["planted_caught"]
    routing = {
        "stp_rate": round(len(auto) / n, 4), "n_auto": len(auto), "n_review": n - len(auto),
        "auto_precision": round(np.mean([r["all_correct"] for r in auto]), 4) if auto else None,
        "n_auto_with_error": sum(1 for r in auto if not r["all_correct"]),
        "n_with_extraction_error": len(with_err),
        "error_catch_rate": round(np.mean([r["status"] == "needs_review" for r in with_err]), 4) if with_err else None,
        "n_correct_and_clean": len(clean),
        "false_review_rate": round(np.mean([r["status"] == "needs_review" for r in clean]), 4) if clean else None,
        "reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "status_by_correctness": {
            "auto_correct": sum(1 for r in auto if r["all_correct"]),
            "auto_wrong": sum(1 for r in auto if not r["all_correct"]),
            "review_planted_error": sum(1 for r in rows if r["status"] == "needs_review" and r["planted_error"]),
            "review_extraction_error": sum(1 for r in rows if r["status"] == "needs_review" and not r["planted_error"]
                                           and not r["all_correct"]),
            "review_but_correct": sum(1 for r in rows if r["status"] == "needs_review" and not r["planted_error"]
                                      and r["all_correct"]),
        },
    }
    doc_errors = {"n": len(planted), "caught": sum(r["planted_caught"] for r in planted),
                  "rate": round(np.mean([r["planted_caught"] for r in planted]), 4) if planted else None,
                  "by_type": {k: {"n": v[0], "caught": v[1]} for k, v in sorted(by_type.items())}}
    by_cond = {}
    for c in CONDITIONS:
        rs = [r for r in rows if r["condition"] == c]
        if not rs:
            continue
        it_c = sum(r["items_truth"] for r in rs)
        ip_c = sum(r["items_pred"] for r in rs)
        im_c = sum(r["items_matched"] for r in rs)
        by_cond[c] = {
            "label": CONDITION_LABELS[c], "n": len(rs),
            "field_accuracy": round(float(np.mean([[r[f"ok_{f}"] for f in SCALAR_FIELDS] for r in rs])), 4),
            "line_item_f1": _prf(im_c, ip_c, it_c)["f1"],
            "all_correct": round(float(np.mean([r["all_correct"] for r in rs])), 4),
            "stp_rate": round(float(np.mean([r["status"] == "auto_approved" for r in rs])), 4),
            "auto_with_error": sum(1 for r in rs if r["status"] == "auto_approved" and not r["all_correct"]),
            "mean_ocr_conf": round(float(np.mean([r["ocr_mean_conf"] for r in rs])), 1),
            "mean_seconds": round(float(np.mean([r["ocr_seconds"] for r in rs])), 3),
            "per_field": {f: round(float(np.mean([r[f"ok_{f}"] for r in rs])), 4) for f in SCALAR_FIELDS},
        }
    tess = [r["ocr_seconds"] for r in rows if r["ocr_source"] == "tesseract"]
    text = [r["ocr_seconds"] for r in rows if r["ocr_source"] == "pdf_text"]
    ocr = {"pages_tesseract": len(tess), "pages_pdf_text": len(text),
           "tesseract_mean_s": round(float(np.mean(tess)), 3) if tess else None,
           "tesseract_median_s": round(float(np.median(tess)), 3) if tess else None,
           "tesseract_p95_s": round(float(np.percentile(tess, 95)), 3) if tess else None,
           "pdf_text_mean_s": round(float(np.mean(text)), 3) if text else None}
    skew = [(r["rotation_applied"], r["skew_corrected"]) for r in rows if r["ocr_source"] == "tesseract"]
    if skew:
        ocr["deskew_mean_abs_error_deg"] = round(float(np.mean([abs(a + b) for a, b in skew])), 3)
    needs_llm = [r for r in rows if r["weak_fields"]]
    weak_counts: dict[str, int] = {}
    for r in needs_llm:
        for f in r["weak_fields"].split(";"):
            weak_counts[f] = weak_counts.get(f, 0) + 1
    hybrid = {"invoices_needing_llm": len(needs_llm), "share_needing_llm": round(len(needs_llm) / n, 4),
              "mean_weak_fields_when_needed": round(float(np.mean([len(r["weak_fields"].split(";"))
                                                                   for r in needs_llm])), 2) if needs_llm else 0.0,
              "weak_field_counts": dict(sorted(weak_counts.items(), key=lambda kv: -kv[1]))}
    sources: dict[str, int] = {}
    for r in rows:
        sources[r["source_kind"]] = sources.get(r["source_kind"], 0) + 1
    return {"synthetic": True, "n_invoices": n, "sources": sources, "fields": fields, "line_items": items,
            "all_fields_correct": round(float(np.mean([r["all_correct"] for r in rows])), 4),
            "routing": routing, "document_errors": doc_errors, "by_condition": by_cond, "ocr": ocr, "hybrid": hybrid}
