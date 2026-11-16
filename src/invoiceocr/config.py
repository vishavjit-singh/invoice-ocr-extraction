"""Paths and parameters."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def invoices(self) -> Path:
        return self.data / "invoices"

    @property
    def ground_truth(self) -> Path:
        return self.data / "ground_truth"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def predictions(self) -> Path:
        return self.outputs / "predictions"

    @property
    def ocr_cache(self) -> Path:
        return self.root / ".cache" / "ocr"

    @property
    def images(self) -> Path:
        return self.root / "docs" / "images"


def load_params(path: Path | str | None = None) -> dict:
    p = Path(path) if path else ROOT / "params.yaml"
    params = yaml.safe_load(p.read_text())
    params["generate"]["degradation"] = {int(k): v for k, v in params["generate"]["degradation"].items()}
    return params
