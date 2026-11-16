"""Rebuild reading order from word boxes: words → lines → segments.

A *line* is words whose vertical extents overlap. A *segment* is a run of words on a line with no large horizontal
gap, so "Invoice No:" and the value printed 300 px to its right become two segments of the same line, and the
left and right columns of a header block stay apart. The same code runs on Tesseract words and PDF text-layer words.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ocr import Word


@dataclass
class Segment:
    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def x0(self) -> float:
        return self.words[0].x0

    @property
    def x1(self) -> float:
        return self.words[-1].x1

    @property
    def y0(self) -> float:
        return min(w.y0 for w in self.words)

    @property
    def y1(self) -> float:
        return max(w.y1 for w in self.words)

    @property
    def height(self) -> float:
        return float(np.median([w.h for w in self.words]))

    @property
    def conf(self) -> float:
        return min(w.conf for w in self.words) / 100


@dataclass
class Line:
    words: list[Word]
    segments: list[Segment]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def y0(self) -> float:
        return min(w.y0 for w in self.words)

    @property
    def y1(self) -> float:
        return max(w.y1 for w in self.words)

    @property
    def cy(self) -> float:
        return float(np.median([w.cy for w in self.words]))


def build_lines(words: list[Word], gap_factor: float = 1.6) -> list[Line]:
    words = [w for w in words if w.text.strip() and w.h > 0]
    if not words:
        return []
    med_h = float(np.median([w.h for w in words]))
    rows: list[list[Word]] = []
    for w in sorted(words, key=lambda w: (w.cy, w.x0)):
        best, best_overlap = None, 0.0
        for row in rows[-4:]:                       # only recent rows can overlap vertically
            r0 = np.median([x.y0 for x in row])
            r1 = np.median([x.y1 for x in row])
            overlap = (min(r1, w.y1) - max(r0, w.y0)) / max(1.0, min(r1 - r0, w.h))
            if overlap > best_overlap:
                best, best_overlap = row, overlap
        if best is not None and best_overlap >= 0.45:
            best.append(w)
        else:
            rows.append([w])
    lines = []
    for row in rows:
        row.sort(key=lambda w: w.x0)
        segs, cur = [], [row[0]]
        for prev, w in zip(row, row[1:]):
            if w.x0 - prev.x1 > gap_factor * max(med_h, min(prev.h, w.h)):
                segs.append(Segment(cur))
                cur = []
            cur.append(w)
        segs.append(Segment(cur))
        lines.append(Line(row, segs))
    lines.sort(key=lambda ln: ln.cy)
    return lines


def layout_text(lines: list[Line]) -> str:
    """Plain text that keeps segments apart (`  |  `), used as LLM input and for debugging."""
    return "\n".join("  |  ".join(s.text for s in ln.segments) for ln in lines)
