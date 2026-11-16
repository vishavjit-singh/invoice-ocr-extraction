PY ?= python3
export PYTHONPATH := src

.PHONY: install demo generate eval report summary test lint serve docker clean clean-all

install:
	$(PY) -m pip install -r requirements-dev.txt

## Regenerate the SYNTHETIC invoices, OCR + extract + validate + score them, redraw docs/images
demo: clean generate eval report

generate:
	$(PY) -m invoiceocr generate

eval:
	$(PY) -m invoiceocr eval

report:
	$(PY) -m invoiceocr report

summary:
	$(PY) -m invoiceocr summary

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

serve:
	$(PY) -m invoiceocr serve

docker:
	docker compose up --build

## The OCR cache (.cache/ocr, keyed by file hash + settings + Tesseract version) survives `clean`
clean:
	rm -rf data outputs/predictions outputs/results.csv

clean-all: clean
	rm -rf .cache outputs
