# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog
and this project adheres to Semantic Versioning.

---

## [Unreleased]

---

## [0.6.1] - 2026-10-22

### Added

- Add `Dataset.from_profile(profile)` to resume a scanner from an existing or JSON-restored single-source `DatasetProfile` without reading files, rescanning directories or reparsing labels. The scanner initially shares the supplied immutable result; subsequent checks replace only its own current Profile and preserve existing errors.
- Document resumed annotation/image checks and explicit rebuilding. Collection Profiles and non-Profile inputs are rejected; collection restoration continues to use `DatasetCollection(restored_collection_profile)`.

---

## [0.6.0] - 2026-10-22

### Changed

- Separate single-source DatasetProfile from DatasetCollectionProfile. Collection construction accepts Profiles, keeps sample-level partition membership and reports overall/partition statistics.
- Add DatasetCollection.split(rates, seed=22): pool current effective members, permanently exclude every stem conflict before allocation, and return a new snapshot sharing source facts. Exclusions never revive after repartitioning or restoration.
- Keep default scans annotation-only. Checks belong to Dataset; processing tools read missing image dimensions directly on demand and do not retain Dataset instances.
- Dataset/Collection val() uses normalized xyxy for YOLO GT and pixel xyxy for VOC/LabelMe; it performs no image or label I/O, independently of cached dimensions.
- Persist both Profile types with schema version 2. Collection serialization stores source facts once, sample membership and additional permanent exclusions; restoration derives statistics offline without legacy-schema compatibility.

- Add `Dataset.val(predictions)` and `DatasetCollection.val(predictions)` as cached-GT adapters to the pure Evaluator. IDs are stems qualified by split/source when needed; Profile has no val() method.

- Classify all duplicate-stem images within a directory or partition as image errors during Profile construction; preserve valid same-name images across different partitions. Raw inventories retain all supported images.
- `Evaluator` now accepts explicit prediction and GT mappings keyed by opaque nonblank string `sample_id`. GT defines the population; missing predictions are empty and unknown prediction IDs are errors. Explicit empty GT samples retain false positives.

- `DatasetProfile` now owns source descriptions, raw/processable/error samples, cached annotations, image sizes, check results and derived statistics. Dataset checks replace its current Profile and return the new result; Profiles are immutable and provide no checks.
- Conversion, plotting and export now consume `DatasetProfile` or `DatasetCollectionProfile`. Obtain it from `open_dataset(...).profile()` or `Dataset(...).build().profile()`. Evaluator is independent of Profile; callers may extract cached GT explicitly.
- Serialize complete results with `profile.export()` and restore with the matching `DatasetProfile.from_json(text)` or `DatasetCollectionProfile.from_json(text)` without reading source files or repeating content checks. `as_dict()` returns statistics only.
- Separate scanning (`dataset.py`), analysis results (`profile.py`), JSON structure (`_profile_io.py`) and detection validation (`annotations/validation.py`).

### Fixed

- Detect source image stem conflicts before reading labels or inspecting images; later checks skip path-conflicting samples and JSON restoration preserves their exclusions.
- Prefer actual image dimensions over annotation size metadata for validation and pixel annotations. Batch processing and single-file reads with an image path use image dimensions without changing source Profile caches.
- Require string LabelMe `imagePath` values and reject non-finite or overflowing numbers in restored Profile JSON. Source and sample paths in saved Profiles must be absolute.
- Sort and deduplicate validated IoU thresholds in `metrics` before AP calculation; emit accurate keys for single high thresholds.
- Include Markdown documentation and metric reference JSON fixtures in source distributions.
- Correct class mappings and output-directory examples in all README translations.

- Reject identical source/target annotation formats before conversion writes, and reject text merge outputs that refer to an input file.
- Share final class mapping validation across single-file and dataset conversions involving YOLO, including reordered named classes and explicit empty names.
- Record malformed annotation data per sample while propagating adapter and validation programming errors.
- Honor explicit empty visualization names, validate nonempty mappings, and raise when OpenCV cannot save an image.
- Match predictions in descending confidence order; verify matching and AP against Ultralytics 8.4.183 (`1da023f708beb274db82c2c04b034c1859ab3fba`).
- Preserve class array dtypes, reject invalid fractional IDs, and evaluate numeric class IDs without requiring names. Add `per_class_by_id` without changing named `per_class` keys.

### Removed

- Remove DatasetProfile.combine(), Profile/Collection check methods, collection flags on source Profiles and Dataset-instance input to Collection. No compatibility wrappers are provided.

- Removed Evaluator Profile input, prediction file loading hooks, path interpretation and terminal printing. `EvaluationSample.sample_id: str` replaces `image_id`; no compatibility wrappers are provided.

- Removed `DatasetView`, `export_scan/from_scan`, and scanner processing forwarding methods. Legacy scan dictionaries require consumer-side migration; no compatibility restoration layer is provided.

- Removed image augmentation: `dsetkit.augment` (including schema- and file-level flip/rotation helpers) and `dsetkit.tools.flip_dataset`, `flip_dirs`, `rotate_dataset`, and `rotate_dirs`. These APIs have no compatibility wrappers.

---

## [0.5.0] - 2026-08-14

### Added

- Added `Dataset.profile()` with positive/background/invalid image counts, orphan-label detection, per-class instance counts, and structured errors.
- Added lightweight image validation based on headers, trailers, and container lengths without full pixel decoding.
- Added `compare_class_schemas(..., mode="strict")` for exact class ID, name, and ordering compatibility checks.
- Added side-effect-free `discover_dataset(...)` for conventional LabelMe, VOC, and YOLO directory layouts.

### Changed

- Annotation format registration now carries suffix, output directory, and discovery-directory metadata.
- Dataset label suffix matching is now case-insensitive.
- Dataset discovery now reserves `candidates` for ambiguous layouts and leaves top-level annotation fields unset until one source is selected.

### Removed

- Removed `Dataset.stats()` and `DatasetStats`; use `Dataset.profile()` as the unified dataset statistics API.
- Removed the redundant `DatasetDiscovery.annotation_status` field; inspect the discovered paths, candidates, and errors directly.
- Removed the legacy `FORMAT_SUFFIXES` and `FORMAT_DIRS` snapshots; format metadata now has a single registry-backed source of truth.

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
- Initial example assets and demo scripts.

### Changed

- N/A (initial release)

### Fixed

- N/A (initial release)

### Breaking

- N/A (initial release)
