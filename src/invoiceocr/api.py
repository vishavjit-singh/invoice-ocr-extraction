"""HTTP API (Starlette + pydantic).

FastAPI could not be installed in the environment this was built in, so the service uses Starlette, the ASGI
toolkit FastAPI is built on, with the pydantic models from ``schema.py`` for the response contract.

    GET  /health                    → status and Tesseract version
    GET  /schema                    → JSON schema of a validated Invoice
    POST /extract?extractor=rules   → multipart upload (field "file") or a raw image/PDF body
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from . import config
from .extract import NAMES, LLMError, get_extractor
from .ocr import ocr_document, tesseract_version
from .routing import decide
from .schema import Invoice

MAX_BYTES = 10 * 1024 * 1024
TYPES = {"image/png": ".png", "image/jpeg": ".jpg", "image/tiff": ".tif", "image/webp": ".webp",
         "application/pdf": ".pdf"}
MAGIC = [(b"%PDF", ".pdf"), (b"\x89PNG", ".png"), (b"\xff\xd8\xff", ".jpg"), (b"II*\x00", ".tif"),
         (b"MM\x00*", ".tif"), (b"RIFF", ".webp")]


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


def sniff(data: bytes) -> str | None:
    for magic, suffix in MAGIC:
        if data.startswith(magic):
            return suffix
    return None


def process_bytes(data: bytes, suffix: str, extractor: str, params: dict) -> dict:
    p = params["ocr"]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(data)
        page, _ = ocr_document(path, {k: p[k] for k in ("psm", "deskew", "denoise", "binarize")})
    result = get_extractor(extractor, params).extract(page)
    decision = decide(result, params["routing"]["min_field_confidence"])
    out = decision.model_dump(mode="json")
    out["ocr"] = {"source": page.source, "seconds": page.seconds, "skew_deg": page.skew_deg,
                  "mean_conf": round(page.mean_conf, 1), "words": len(page.words)}
    out["notes"] = result.notes
    return out


async def health(request: Request):
    return JSONResponse({"status": "ok", "tesseract": tesseract_version(),
                         "llm_available": bool(os.environ.get("OPENAI_API_KEY"))})


async def schema(request: Request):
    return JSONResponse(Invoice.model_json_schema())


async def extract(request: Request):
    extractor = request.query_params.get("extractor", "rules")
    if extractor not in NAMES:
        return _error(400, "bad_extractor", f"extractor must be one of {list(NAMES)}")
    if extractor == "llm" and not os.environ.get("OPENAI_API_KEY"):
        return _error(400, "llm_unavailable", "the llm extractor needs OPENAI_API_KEY on the server")
    ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if ctype == "multipart/form-data":
        form = await request.form(max_files=1)
        upload = form.get("file")
        if upload is None or isinstance(upload, str):
            return _error(400, "missing_file", "send the document in a form field called 'file'")
        data = await upload.read(MAX_BYTES + 1)
    else:
        data = await request.body()
    if not data:
        return _error(400, "empty_body", "no document received")
    if len(data) > MAX_BYTES:
        return _error(413, "too_large", f"documents up to {MAX_BYTES // (1024 * 1024)} MB")
    suffix = sniff(data)
    if suffix is None:
        return _error(415, "unsupported_type", f"send a PDF or an image ({', '.join(sorted(TYPES))})")
    try:
        out = await run_in_threadpool(process_bytes, data, suffix, extractor, request.app.state.params)
    except LLMError as e:
        return _error(502, "llm_error", str(e))
    except Exception as e:                       # unreadable image, corrupt PDF, ...
        return _error(422, "unreadable_document", f"{type(e).__name__}: {e}")
    return JSONResponse(out)


def create_app(params: dict | None = None) -> Starlette:
    app = Starlette(routes=[Route("/health", health), Route("/schema", schema),
                            Route("/extract", extract, methods=["POST"])])
    app.state.params = params or config.load_params()
    return app


app = create_app()
