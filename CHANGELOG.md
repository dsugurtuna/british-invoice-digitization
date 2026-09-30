# Changelog

## 3.0.0 (2026-09-30)

A correctness and honesty pass. CI had failed on every run since the first commit.

### Breaking

- Package renamed from `src` to `invoice_digitizer`; distribution renamed to
  `british-invoice-digitization`.
- Environment variable prefix changed from `ROYALAUDIT_` to `INVOICE_DIGITIZER_`. Settings
  are read from environment variables and an optional YAML file named by
  `INVOICE_DIGITIZER_CONFIG_FILE`.
- The API is built by `create_app()`; start it with `invoice-digitizer` or
  `uvicorn --factory invoice_digitizer.api.main:create_app`.
- Threshold updates and model reload need the `X-Admin-Key` header, and are off without a key.
- Missing weights stop start-up instead of silently falling back to COCO weights.
- Removed: the Streamlit dashboard (it did not parse), `utils/visualization.py` (it used
  attributes that did not exist), the no-op `extract_text` option, CSV/XML/webhook request
  options that were never implemented, and `requirements.txt`.

### Fixed

- Per-request thresholds leaked into later requests and raced between concurrent ones.
- The model could load twice under concurrent first use, and reload left a gap.
- File inputs reached the model in BGR order instead of RGB.
- Error results always raised a validation error (image width 0 against a minimum of 1).
- YAML configuration was mostly ignored, and nested settings read unprefixed environment
  variables.
- `torch.hub` loaded YOLOv5 from the default branch while everything said v7.0; now pinned.
- The Docker image did not install the application and requested a package Debian no longer
  ships; Prometheus scraped a port nothing listened on.

### Documentation

- Removed claims that nothing in the repository supports: a named client company,
  "98.5% mAP" and other accuracy figures, latency, cost and throughput tables, model size,
  "production-ready", "enterprise" and "thread-safe singleton" wording, OCR, PDF support,
  Kubernetes, TensorRT, Redis and PostgreSQL in the architecture.
- Removed a Roboflow export URL with an embedded key from the training notebook.
- Added `docs/WHY.md` and a table of what is and is not verified.

## 2.0.0 and earlier

Published as "RoyalAudit Digitizer". CI did not pass on these versions.
