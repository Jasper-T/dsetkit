# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog
and this project adheres to Semantic Versioning.

---

## [0.2.0] - 2026-05-28

### Added

- Added `dsetkit.tools` single-file helper module with:
  - `convert_dataset(...)` / `convert_dirs(...)`
  - `plot_dataset(...)` / `plot_dirs(...)`
- Added `plot(...)` helper in `dsetkit.visualize.plot` to draw from `(image, anno_schema)` or `(image_path, label_path, fmt)`.
- Added `ensure_dir(path)` utility in `dsetkit.utils.file`.
- Added `tqdm` as a runtime dependency for progress bars in `dsetkit.tools`.

### Changed

- Refactored `Dataset` workflow:
  - Constructor now uses `source_format` (replacing `input_format`)
  - Added explicit `dataset.build()` lifecycle before iteration/indexing
  - Label matching now uses suffix filtering by `source_format` instead of mixed-format priority fallback
- Updated annotation adapters (`labelme`, `voc`, `yolo`, `coco`) to use explicit `width` / `height` passthrough and store image path metadata in `Annotation.extra`.
- Updated annotation I/O path helpers:
  - `get_label_path(...)` -> `new_label_path(...)`
  - `default_label_dir(...)` -> `new_label_dir(...)`
  - `auto_label_path(image_path, fmt, label_dir=...)` -> `auto_label_path(old_path, fmt, new_dir=...)`
- Kept OpenCV as an optional visualization dependency; `dsetkit.tools` now imports plotting support only when plotting helpers are used.

### Fixed

- Updated `Evaluator` to load annotations through `Dataset.source_format` after the 0.2.0 dataset API rename.

### Breaking

- `Dataset(input_format=...)` is removed; migrate to `Dataset(source_format=...)`.
- `Dataset` no longer builds indices at init time; callers must invoke `dataset.build()` before `len(dataset)`, iteration, or indexing.
- Built-in default format registry no longer includes `coco` in `FORMAT_SUFFIXES`; standard load/dump/convert workflows are now documented around `labelme` / `voc` / `yolo`.
- `Annotation.image_path` field has been removed; use `Annotation.extra` (e.g. `extra["image_path"]`) for source image metadata when needed.
- Renamed helpers in `dsetkit.annotations.io` are not aliased to old names (`get_label_path`, `default_label_dir`).

---

## [0.1.1] - 2026-05-21

### Changed

- Renamed annotation IO helpers in `dsetkit.annotations.io` for clearer semantics:
  - `get_output_suffix(...)` -> `get_label_file_suffix(...)`
  - `get_output_path(...)` -> `get_label_path(...)`
  - `default_out_dir(...)` -> `default_label_dir(...)`
  - `auto_out_path(...)` -> `auto_label_path(...)`
- Introduced `get_label_dir_name(fmt)` helper to centralize format-to-directory-name lookup.
- `auto_label_path(...)` now appends the format-specific subdirectory (e.g. `labels/`, `annotations/`) under the user-provided `label_dir`, making behavior consistent with `default_label_dir(...)`.

### Fixed

- `dump_voc(...)` now coerces `ann.image_path` to `str` before writing it to the `<filename>` XML element, preventing failures when `image_path` is a `pathlib.Path`.

### Breaking

- The renamed functions in `dsetkit.annotations.io` are not aliased to their old names. Callers using `get_output_suffix`, `get_output_path`, `default_out_dir`, or `auto_out_path` must migrate to the new names.
- `auto_label_path(image_path, fmt, label_dir=...)` no longer treats `label_dir` as the final output directory; the format subdirectory is appended automatically. Pass the dataset root (the parent of the labels folder) instead of the labels folder itself.

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
