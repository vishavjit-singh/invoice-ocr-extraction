"""Document → words with boxes and confidences.

PDFs with a text layer are read with pdfplumber and never OCR'd. Images and scanned PDFs go through OpenCV
preprocessing (grayscale, deskew by projection profile, denoise, optional adaptive threshold) and Tesseract
``image_to_data``, which returns every word with its box and a 0-100 confidence. Coordinates are always pixels at
150 dpi so the extractors see the same geometry whatever the source. Results are cached by file hash and settings.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PX_PER_PT = 150 / 72
cv2.setNumThreads(1)                         # parallelism comes from OCR worker processes, not threads
PIPELINE_VERSION = "2"                       # part of the cache key: bump when preprocessing changes
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float                   # 0-100; 100 for a PDF text layer

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def h(self) -> float:
        return self.y1 - self.y0


@dataclass
class OcrPage:
    words: list[Word]
    width: int
    height: int
    source: str                   # "pdf_text" or "tesseract"
    seconds: float
    skew_deg: float = 0.0
    settings: dict = field(default_factory=dict)

    @property
    def mean_conf(self) -> float:
        return float(np.mean([w.conf for w in self.words])) if self.words else 0.0

    def to_json(self) -> dict:
        d = asdict(self)
        d["words"] = [[w.text, round(w.x0, 1), round(w.y0, 1), round(w.x1, 1), round(w.y1, 1), round(w.conf, 1)]
                      for w in self.words]
        return d

    @classmethod
    def from_json(cls, d: dict) -> "OcrPage":
        d = dict(d)
        d["words"] = [Word(*w) for w in d["words"]]
        return cls(**d)


# ---------------------------------------------------------------- preprocessing
def estimate_skew(gray: np.ndarray, max_angle: float = 5.0) -> float:
    """Rotation (degrees, counter-clockwise) that makes text lines horizontal.

    Projection-profile search: rotate a downscaled binary image and keep the angle whose row sums are most
    peaked (text lines and table rules line up with pixel rows). Coarse 0.5° grid, then 0.1° around the best.
    """
    scale = 900 / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape
    centre = (w / 2, h / 2)

    def score(angle: float) -> float:
        m = cv2.getRotationMatrix2D(centre, angle, 1.0)
        rotated = cv2.warpAffine(binary, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        rows = rotated.sum(axis=1, dtype=np.float64)
        return float(np.sum(np.diff(rows) ** 2))

    coarse = np.arange(-max_angle, max_angle + 1e-9, 0.5)
    best = max(coarse, key=score)
    fine = np.arange(best - 0.5, best + 0.5 + 1e-9, 0.1)
    return round(float(max(fine, key=score)), 2)


def rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.05:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)


def estimate_noise(gray: np.ndarray) -> float:
    """Std of the high-frequency residual on paper pixels only (text edges would swamp a whole-page estimate)."""
    g = gray.astype(np.float32)
    smooth = cv2.GaussianBlur(g, (5, 5), 0)
    paper = smooth > np.percentile(smooth, 60) - 10
    return float(np.std((g - smooth)[paper])) if paper.any() else 0.0


def preprocess(img: Image.Image, deskew: bool = True, denoise: bool | str = "median", binarize: bool = False
               ) -> tuple[np.ndarray, float]:
    """Grayscale → deskew → denoise (noisy pages only: median or non-local means) → optional adaptive threshold.

    Returns the image and the correction angle.
    """
    gray = np.asarray(img.convert("L"))
    angle = estimate_skew(gray) if deskew else 0.0
    gray = rotate(gray, angle)
    if denoise:
        if estimate_noise(gray) > 3.0:      # only touch images that are actually noisy
            if denoise == "nlmeans":
                gray = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=5, searchWindowSize=11)
            else:
                gray = cv2.medianBlur(gray, 3)
    if binarize:
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    return gray, angle


# ---------------------------------------------------------------- engines
def tesseract_words(gray: np.ndarray, psm: int = 6) -> list[Word]:
    import pytesseract

    os.environ.setdefault("OMP_THREAD_LIMIT", "1")       # one core per page; parallelism comes from workers
    data = pytesseract.image_to_data(gray, lang="eng", config=f"--oem 1 --psm {psm}",
                                     output_type=pytesseract.Output.DICT)
    words = []
    for text, left, top, w, h, conf in zip(data["text"], data["left"], data["top"], data["width"], data["height"],
                                           data["conf"]):
        text = text.strip()
        if text and float(conf) >= 0:
            words.append(Word(text, left, top, left + w, top + h, float(conf)))
    return words


def pdf_text_words(path: Path) -> tuple[list[Word], int, int] | None:
    """Words from the first page's text layer, or None if the page has no text (a scanned PDF)."""
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        page = pdf.pages[0]
        raw = page.extract_words(keep_blank_chars=False, use_text_flow=False)
        if len(raw) < 5:
            return None
        k = PX_PER_PT
        words = [Word(w["text"], w["x0"] * k, w["top"] * k, w["x1"] * k, w["bottom"] * k, 100.0) for w in raw]
        return words, round(float(page.width) * k), round(float(page.height) * k)


def pdf_page_image(path: Path, dpi: int = 150) -> Image.Image:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        return pdf[0].render(scale=dpi / 72).to_pil().convert("RGB")
    finally:
        pdf.close()


def load_image(path_or_bytes: Path | bytes) -> Image.Image:
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    img = Image.open(src)
    img.load()
    return img


# ---------------------------------------------------------------- one entry point
def _cache_key(data: bytes, settings: dict) -> str:
    h = hashlib.sha256(data)
    h.update(json.dumps(settings, sort_keys=True).encode())
    return h.hexdigest()[:32]


def tesseract_version() -> str:
    try:
        import pytesseract

        return str(pytesseract.get_tesseract_version())
    except Exception:                        # tesseract missing: the caller will fail later with a clear error
        return "unknown"


def ocr_document(path: Path, settings: dict | None = None, cache_dir: Path | None = None,
                 keep_image: bool = False) -> tuple[OcrPage, np.ndarray | None]:
    """OCR one file. Returns the page and (if ``keep_image``) the preprocessed image the boxes refer to."""
    settings = {"psm": 6, "deskew": True, "denoise": "median", "binarize": False, **(settings or {})}
    settings = {k: settings[k] for k in ("psm", "deskew", "denoise", "binarize")}
    path = Path(path)
    data = path.read_bytes()
    key = _cache_key(data, {**settings, "tesseract": tesseract_version(), "pipeline": PIPELINE_VERSION})
    cached = cache_dir / f"{key}.json" if cache_dir else None
    if cached and cached.exists() and not keep_image:
        return OcrPage.from_json(json.loads(cached.read_text())), None

    t0 = time.perf_counter()
    image = None
    page: OcrPage
    text_layer = pdf_text_words(path) if path.suffix.lower() == ".pdf" else None
    if text_layer:
        words, w, h = text_layer
        page = OcrPage(words, w, h, "pdf_text", 0.0, 0.0, settings)
    else:
        img = pdf_page_image(path) if path.suffix.lower() == ".pdf" else load_image(path)
        gray, angle = preprocess(img, settings["deskew"], settings["denoise"], settings["binarize"])
        words = tesseract_words(gray, settings["psm"])
        page = OcrPage(words, gray.shape[1], gray.shape[0], "tesseract", 0.0, angle, settings)
        image = gray if keep_image else None
    page.seconds = round(time.perf_counter() - t0, 3)
    if cached:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(page.to_json()))
    if keep_image and image is None and page.source == "pdf_text":
        image = np.asarray(pdf_page_image(path).convert("L"))
    return page, image
