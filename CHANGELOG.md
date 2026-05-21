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

## [0.1.1] - 2026-05-21

### Changed

- Renamed annotation IO helpers in `dsetkit.annotations.io` for clearer semantics:
  - `get_output_suffix(...)` → `get_label_file_suffix(...)`
  - `get_output_path(...)` → `get_label_path(...)`
  - `default_out_dir(...)` → `default_label_dir(...)`
  - `auto_out_path(...)` → `auto_label_path(...)`
- Introduced `get_label_dir_name(fmt)` helper to centralize format-to-directory-name lookup.
- `auto_label_path(...)` now appends the format-specific subdirectory (e.g. `labels/`, `annotations/`) under the user-provided `label_dir`, making behavior consistent with `default_label_dir(...)`.

### Fixed

- `dump_voc(...)` now coerces `ann.image_path` to `str` before writing it to the `<filename>` XML element, preventing failures when `image_path` is a `pathlib.Path`.

### Breaking

- The renamed functions in `dsetkit.annotations.io` are not aliased to their old names. Callers using `get_output_suffix`, `get_output_path`, `default_out_dir`, or `auto_out_path` must migrate to the new names.
- `auto_label_path(image_path, fmt, label_dir=...)` no longer treats `label_dir` as the final output directory; the format subdirectory is appended automatically. Pass the dataset root (the parent of the labels folder) instead of the labels folder itself.
