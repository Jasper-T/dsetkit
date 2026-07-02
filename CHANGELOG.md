# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog
and this project adheres to Semantic Versioning.

---

## [Unreleased]

---

## [0.4.1] - 2026-07-02

### Changed

- Updated evaluator metrics to always report both the single-threshold AP column and the threshold-range AP column.
- `Evaluator.evaluate(iou=0.5)` now evaluates `0.50...0.95` with a fixed `0.05` IoU step and reports `mAP50` plus `mAP50-95`.
- `Evaluator.evaluate(iou=0.25)` now evaluates `0.25...0.95` with a fixed `0.05` IoU step and reports `mAP25` plus `mAP25-95`.
- Updated the printed evaluator table to include both AP columns.

### Breaking

- Removed the generic `metrics["mAP"]` compatibility alias from evaluator metric dictionaries; use the explicit keys such as `mAP50` and `mAP50-95` instead.

---

## [0.4.0] - 2026-06-30

### Added

- Added dataset-level `Dataset.stats()` for image, background image, and instance counts.
- Added `Dataset.ground_truth(sample)` to expose dataset annotations as evaluation-ready targets.
- Added common evaluation data structures in `dsetkit.detection`:
  - `AnnotationTarget`
  - `PredictionResult`
  - `EvaluationSample`
- Added tests for dataset statistics and Ultralytics-style evaluator metrics.

### Changed

- Refactored `Evaluator` around explicit prediction inputs and dataset-provided ground truth:
  - `Evaluator(dataset).evaluate(predictions=..., iou=...)` is now the primary API.
  - `_load_predictions(...)` remains available only as an optional extension point for custom prediction sources.
- Refactored `dsetkit.metrics` to operate on `EvaluationSample` objects while keeping numeric matching/AP computation in NumPy.
- Updated evaluator metrics to follow Ultralytics-style detection matching and AP computation.
- `Evaluator.evaluate(iou=...)` now uses dsetkit's own evaluation IoU semantics:
  - `iou=0.5` computes `mAP50`.
  - A sequence such as `np.linspace(0.5, 0.95, 10)` computes `mAP50-95`.
- Updated README files for the new evaluator API, dataset statistics, and evaluation data model.

### Breaking

- Removed the previous subclass-first evaluator flow as the documented primary API.
- Removed evaluator-side annotation loading/conversion responsibilities; ground truth now belongs to `Dataset`.
- Removed old detection structure names; use `AnnotationTarget`, `PredictionResult`, and `EvaluationSample`.

---

## [0.3.1] - 2026-06-26

### Changed

- Updated `split_tvt(...)` defaults in `dsetkit.split` to use a two-way `train/val` split ratio of `(0.8, 0.2)`.
- `split_tvt(...)` now accepts a plain txt filename and automatically resolves it relative to `dataset_root`.

### Added

- `split_tvt(...)` can now bootstrap from `dataset_root/images` when the source txt file does not exist.
- When bootstrapping from `dataset_root/images`, `split_tvt(...)` now saves the discovered image list to the requested txt file before writing split outputs.
- `split_tvt(...)` now returns the generated split mapping for downstream reuse.

---

## [0.3.0] - 2026-06-01

### Added

- Added `dsetkit.augment` module with flip and rotate helpers:
  - Schema-level: `flip_annotation`, `flip_annotation_horizontal`, `flip_annotation_vertical`, `rotate_annotation`
  - File-level: `flip_image`, `flip_label`, `rotate_image`, `rotate_label`, `rotate_sample`
- Added `dsetkit.split` module:
  - `split_paths(...)` �?shuffle image paths into train/val/test buckets
  - `save_split_txts(...)` �?write split lists to txt files
  - `split_tvt(...)` �?split paths from an existing txt file
- Extended `dsetkit.tools` with dataset-scale helpers:
  - `flip_dataset(...)` / `flip_dirs(...)`
  - `rotate_dataset(...)` / `rotate_dirs(...)`
  - `export_dataset(...)` / `export_dirs(...)` �?export image path lists to txt
  - `split_dataset(...)` / `split_dirs(...)` �?generate train/val/test txt splits
- Added `Annotation.require_size()` to validate and return `(width, height)`.
- Added `load_txt(...)`, `save_txt(...)`, and `rm_empty_dirs(...)` in `dsetkit.utils.file`.
- Added `resolve_image_wh(...)` in `dsetkit.utils.image` (shared by format adapters).
- Added tests: `tests/test_flip.py`, `tests/test_rotate.py`, `tests/test_split.py`.

### Changed

- `opencv-python` is now a core runtime dependency (required by augment and visualization workflows).
- Moved `resolve_image_wh(...)` from `dsetkit.annotations.formats.common` to `dsetkit.utils.image`.
- YOLO dumper now uses `ann.require_size()` instead of inline width/height checks.
- Removed `[visualize]` optional dependency group; OpenCV is installed via core dependencies.

### Removed

- `dsetkit.annotations.convert` module and `batch_convert(...)` API.
- `dsetkit.annotations.formats.common` module.

### Breaking

- `pip install -e ".[visualize]"` is removed; use `pip install -e .` (OpenCV is included in core dependencies).

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



