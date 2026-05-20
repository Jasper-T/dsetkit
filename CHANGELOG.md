# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog
and this project adheres to Semantic Versioning.

---

## [0.1.0] - 2026-05-09

### Added

- Initial public release of `dsetkit`, a deep learning dataset infrastructure toolkit.
- Unified annotation schema:
  - `Annotation`, `AnnotationItem`, `BBox`
  - Common detection-oriented structure with extensibility fields.
- Multi-format annotation adapter system:
  - Supported formats: `coco`, `labelme`, `voc`, `yolo`
  - Unified APIs: `load(...)`, `dump(...)`, `convert(...)`.
- Batch conversion support in `annotations.convert`:
  - `batch_convert(...)` for large-scale conversion tasks
  - Optional multiprocessing with `ProcessPoolExecutor`
  - Error strategies: `errors="raise"` and `errors="skip"`.
- Dataset construction and indexing:
  - `Dataset` and `DatasetSample`
  - Auto image/label matching by filename stem
  - Label priority fallback when multiple suffixes are present.
- Detection evaluation foundation:
  - `Evaluator` base class with customizable `_load_predictions(...)`
  - IoU-based matching, AP/mAP-style calculation, Precision/Recall/F1 outputs
  - Per-class and overall metrics reporting.
- Utility modules:
  - Image metadata readers for PNG/JPEG/WebP/BMP
  - Natural-order image path traversal and stem index helpers.
- Optional visualization module:
  - `Plotter` for drawing detection annotations on images
  - OpenCV integration via optional dependency.
- Example/demo assets and scripts in `examples/` and `tests/`.

### Changed

- N/A (initial release)

### Fixed

- N/A (initial release)

### Breaking

- N/A (initial release)

---

## [0.1.1] - 202**-**-**
