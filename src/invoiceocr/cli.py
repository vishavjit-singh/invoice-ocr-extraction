"""``python -m invoiceocr generate | extract | eval | report | summary | serve``."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import config


def cmd_generate(args, params, paths):
    import shutil

    from .generate import generate

    for d in (paths.invoices, paths.ground_truth):
        if d.exists():
            shutil.rmtree(d)
    recs = generate(paths.invoices, paths.ground_truth, params["generate"], n=args.n, seed=args.seed)
    kinds: dict[str, int] = {}
    for r in recs:
        kinds[r["source_kind"]] = kinds.get(r["source_kind"], 0) + 1
    planted = sum(1 for r in recs if r["document_errors"])
    print(f"generated {len(recs)} SYNTHETIC invoices in {paths.invoices} ({kinds}); {planted} printed with a "
          f"deliberate mistake")


def cmd_extract(args, params, paths):
    from .extract import get_extractor
    from .ocr import ocr_document
    from .routing import decide

    p = params["ocr"]
    page, image = ocr_document(Path(args.file), {k: p[k] for k in ("psm", "deskew", "denoise", "binarize")},
                               keep_image=bool(args.overlay))
    result = get_extractor(args.extractor, params).extract(page)
    decision = decide(result, params["routing"]["min_field_confidence"])
    out = decision.model_dump(mode="json")
    out["ocr"] = {"source": page.source, "seconds": page.seconds, "skew_deg": page.skew_deg,
                  "mean_conf": round(page.mean_conf, 1), "words": len(page.words)}
    if result.notes:
        out["notes"] = result.notes
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.overlay:
        from .report import overlay_image

        overlay_image(image, page, result, decision, Path(args.overlay))
        print(f"wrote {args.overlay}", file=sys.stderr)


def cmd_eval(args, params, paths):
    from .evaluate import evaluate

    m = evaluate(paths, params)
    r = m["routing"]
    print(f"{m['n_invoices']} SYNTHETIC invoices · all fields correct {m['all_fields_correct']:.1%} · "
          f"STP {r['stp_rate']:.1%} · auto-approved with an error: {r['n_auto_with_error']} · "
          f"extraction errors sent to review {r['error_catch_rate']:.1%} · planted mistakes caught "
          f"{m['document_errors']['caught']}/{m['document_errors']['n']}")
    print(f"wrote {paths.outputs / 'metrics.json'} and {paths.outputs / 'results.csv'}")


def cmd_report(args, params, paths):
    from .report import render_all

    for p in render_all(paths, params):
        print(f"wrote {p}")


def cmd_summary(args, params, paths):
    m = json.loads((paths.outputs / "metrics.json").read_text())
    print(json.dumps(m, indent=2))


def cmd_serve(args, params, paths):
    import uvicorn

    uvicorn.run("invoiceocr.api:app", host=args.host, port=args.port, log_level="info")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="invoiceocr", description="Invoice OCR and structured extraction")
    ap.add_argument("--params", default=None, help="params.yaml to use")
    ap.add_argument("--root", default=str(config.ROOT), help="project root for data/ and outputs/")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="render the SYNTHETIC invoice set with ground truth")
    g.add_argument("--n", type=int, default=None)
    g.add_argument("--seed", type=int, default=None)
    e = sub.add_parser("extract", help="extract one invoice (image or PDF) and print the decision as JSON")
    e.add_argument("file")
    e.add_argument("--extractor", default="rules", choices=["rules", "hybrid", "llm"])
    e.add_argument("--overlay", default=None, help="also save the page with OCR boxes and extracted fields")
    sub.add_parser("eval", help="OCR + extract + validate every generated invoice and score it")
    sub.add_parser("report", help="render the charts in docs/images from outputs/")
    sub.add_parser("summary", help="print outputs/metrics.json")
    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    params = config.load_params(args.params)
    paths = config.Paths(Path(args.root))
    {"generate": cmd_generate, "extract": cmd_extract, "eval": cmd_eval, "report": cmd_report,
     "summary": cmd_summary, "serve": cmd_serve}[args.cmd](args, params, paths)
