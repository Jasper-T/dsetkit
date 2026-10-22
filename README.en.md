# dsetkit

Language: [中文](README.md) | **English** | [日本語](README.ja.md)

**Deep learning dataset infrastructure toolkit** — a Python toolkit for object detection and annotation pipelines, with a unified annotation schema, multi-format conversion, dataset indexing, splitting, evaluation metrics, and visualization.

[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](pyproject.toml)

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Component responsibilities](#component-responsibilities)
- [Quick Start](#quick-start)
- [Supported Annotation Formats](#supported-annotation-formats)
- [Unified Data Model](#unified-data-model)
- [Project Structure](#project-structure)
- [Development and Build](#development-and-build)
- [License](#license)

---

## Features

| Module | Capability |
| ------ | ------ |
| **Annotation I/O** | Read/write **LabelMe / VOC / YOLO** with a unified `Annotation` schema |
| **Format Conversion** | Single-file `convert`; dataset-scale `convert_dataset` / `convert_dirs` |
| **Dataset** | Auto-pair images and labels by filename stem, with suffix filtering driven by `source_format` |
| **Dataset Profiling** | Count positive, background, and invalid images, orphan labels, and per-class instances |
| **Dataset Discovery** | Discover conventional image and LabelMe / VOC / YOLO annotation directories without side effects |
| **Schema Checks** | Strictly compare class IDs, names, and ordering without implicit remapping |
| **Split and Export** | Export image path lists to txt; randomly split into train/val/test by ratio |
| **Evaluation** | Ultralytics-style detection metrics over existing predictions: P / R / mAP@iou / F1 |
| **Image Utilities** | Read JPEG/PNG/WebP/BMP metadata without full decoding; natural sorting traversal |
| **Visualization** | `Plotter` draws detection boxes on images (OpenCV included in base install) |

---

## Installation

**Base install** (annotations, dataset, visualization, metrics):

```bash
pip install -e .
```

Requires **Python >= 3.11**. Core dependencies: `numpy>=2.0`, `natsort`, `opencv-python`, `tqdm`, `loguru`.

---

## Component responsibilities

Dataset has four core responsibilities: explicit-source scanning, annotation checking,
optional image checking and DatasetProfile generation. It does not discover arbitrary
layouts, partition data, generate training batches or train models. Dataset.val() is
only a cached-GT adapter to the independently callable Evaluator.

Components can be called independently with their documented inputs: annotation I/O,
image utilities, strict schema checks, path partitions, Plotter and Evaluator do not
require a Dataset. Collection and batch tools consume Profiles, not live scanners.
Profiles are shared result contracts; independence does not mean zero dependencies.
See the [complete class and function catalog](DATA_COLLECTIONS.md#component-responsibilities-and-independent-use).

PyTorch Dataset describes sample access and DataLoader coordinates loading/batching;
dsetkit prepares and analyzes reusable source/member results. A caller-owned adapter
connects a Profile to the training pipeline; no built-in PyTorch adapter is included.
See [integration boundaries](DATA_COLLECTIONS.md#relationship-to-pytorch-dataset-and-dataloader)
and the [official PyTorch tutorial](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html).

## Quick Start

### 1. Load and convert annotations

All supported formats are converted into the unified `Annotation` schema internally (`BBox` uses absolute `xyxy` coordinates).

```python
from dsetkit.annotations.io import load, dump, convert

# LabelMe JSON -> YOLO txt (class names are required for YOLO export)
convert(
    label_path="path/to/label.json",
    image_path="path/to/image.jpg",
    source_format="labelme",
    target_format="yolo",
    names=["smoke", "fire"],   # final category-to-ID order for YOLO export
    out_dir="output",
)
```

**Two-step flow** (helpful for debugging intermediate results):

```python
from pathlib import Path

ann = load(
    label_path="path/to/label.json",
    image_path="path/to/image.jpg",
    fmt="labelme",
    names=["smoke", "fire"],
)

Path("output").mkdir(parents=True, exist_ok=True)
dump(ann, "output/label.txt", fmt="yolo")
```

> **Note**: For **YOLO** read/write, pass `names` to `load` when possible (class id -> class name). For YOLO `dump`, `Annotation` must include valid `width`/`height`, and each object needs `category_id` or a resolvable class via `ann.names`.

### 2. Build dataset index

`open_dataset()` completes scanning; call `.profile()` for the complete analysis result. It scans the image directory and finds matching label files by stem, then filters by the suffix of `source_format` (`labelme -> .json`, `voc -> .xml`, `yolo -> .txt`).

```python
from dsetkit import open_dataset

profile = open_dataset(
    task_type="detection", image_dir="images/val",
    label_dir="labels",
    names=["person", "car"],
    source_format="voc",
).profile()

print(len(profile))
print(profile.as_dict())

for sample in profile:
    print(sample.image_path, sample.label_path)
    target = profile.ground_truth(sample)
```

`Dataset.profile()` returns a single-source `DatasetProfile`; `open_dataset().profile()` returns a `DatasetCollectionProfile` with source references, sample-level membership, cumulative exclusions and partition statistics. Images are unchecked by default; an explicit image check validates containers and fully decodes pixels. Structured diagnostics are available in `profile.errors`.


Use `open_dataset(task_type="detection", root="dataset", names=["object"], source_format="yolo")` for automatic layout discovery. Root and explicit directory modes are mutually exclusive; explicit label directories require an annotation format. Both return completed scans and enumerate only direct child files. Omitting `label_dir` in explicit mode does not search nearby directories. `Dataset` scans a single source; `DatasetCollection` assembles source scans. Conversion, plotting and export accept `DatasetProfile` or `DatasetCollectionProfile`; evaluation uses dataset.val() or the pure Evaluator with explicit predictions and GT.

#### 2.1 Discover a dataset layout

```python
from dsetkit import discover_parts, select_annotation_source, open_dataset

native, parts = discover_parts("path/to/dataset")
for part in parts:
    source = select_annotation_source(part.discovery, requested="yolo")
    print(part.split, part.discovery.image_dir, source.label_dir if source else None)

profile = open_dataset("path/to/dataset", task_type="detection", names=["object"], source_format="yolo").profile()
print(profile)
```

`discover_parts()` is the single layout-discovery entry point for flat, split-first and split-last layouts. Only direct child files are scanned. All annotation sources, including single sources, live in `part.discovery.candidates`. `select_annotation_source()` validates format selection and ambiguity; `open_dataset().profile()` returns complete analysis results that preserve split ownership. See [DATA_COLLECTIONS.md](DATA_COLLECTIONS.md) for the API and breaking changes.

#### 2.2 Compare class schemas strictly

```python
from dsetkit import compare_class_schemas

result = compare_class_schemas(
    ["cat", "dog"],
    ["cat", "person"],
    mode="strict",
)
print(result.compatible)   # False
print(result.differences)  # class_id=1: dog != person
```

Strict mode requires identical class counts, names, and ordering. It never sorts classes, ignores case, or creates an implicit mapping.

### 3. Evaluate object detection

`Evaluator` is a pure numerical evaluator of existing detections and explicit GT. It has no Profile dependency and performs no file access, inference or terminal printing. Box matching and metric calculation follow the Ultralytics reference:

```python
from dsetkit.evaluator import Evaluator

predictions = {
    "sample_001": [
        {"bbox": [10, 20, 80, 120], "class_id": 0, "conf": 0.91},
    ],
}
ground_truths = {
    "sample_001": [{"bbox": [12, 18, 82, 118], "class_id": 0}],
    "background_001": [],
}

metrics = Evaluator(names=["person"]).evaluate(
    predictions=predictions,
    ground_truths=ground_truths,
    conf_threshold=0.001,
    iou=0.5,
)
```

`iou` is dsetkit's starting evaluation IoU threshold: `iou=0.5` evaluates `0.50...0.95` with a fixed `0.05` step and reports both `mAP50` and `mAP50-95`; `iou=0.25` reports `mAP25` and `mAP25-95`.
Both mappings use unique nonblank string `sample_id` keys, with no path or suffix requirement and no path interpretation. GT defines the entire sample population: missing predictions are empty; prediction IDs absent from GT are errors. Keep background samples as explicit empty GT entries so their false positives participate in evaluation.
Values can be record lists as above or `PredictionResult` / `AnnotationTarget` objects. Prediction and GT boxes must share the same xyxy coordinates and class IDs. Pixel and normalized coordinates are both supported without image dimensions. Records also accept `cls`, or `label` with a `names` mapping.
Before AP calculation, `metrics` validates finite IoU thresholds in `(0, 1]`, sorts them and removes duplicates. Sequences use the normalized thresholds; a single threshold below 0.95 expands by 0.05 without exceeding 0.95. A single resulting threshold produces only its point metric, such as `mAP95` or `mAP100`.
Numeric `class_id`/`cls` evaluation does not require names; string labels require a mapping.
`per_class_by_id` includes every GT class; `per_class` keeps the existing name keys for named classes.
Matching uses descending confidence. P/R/F1 use the confidence threshold maximizing mean F1; mAP averages classes present in GT.
The calculation is checked against Ultralytics 8.4.183 (commit `1da023f708beb274db82c2c04b034c1859ab3fba`).
Results are returned as a dictionary; callers own display and printing. The `images` field counts all GT sample IDs, including empty GT samples.

### 3.1 Evaluate using dataset annotations as GT

`Dataset(...).build().val(predictions)` and `open_dataset(...).val(predictions)` use the current Profile's processable samples and cached annotations as GT, returning the same metrics dictionary. `open_dataset()` currently always returns `DatasetCollection`; both dataset types expose `val()`, while Profile does not.

```python
from dsetkit import open_dataset

dataset = open_dataset(
    task_type="detection", image_dir="images", label_dir="labels",
    source_format="yolo", names=["person"],
)
metrics = dataset.val({
    "sample_001": [
        {"bbox": [0.1, 0.2, 0.8, 0.9], "class_id": 0, "conf": 0.91},
    ],
})
```

Single-source IDs are suffix-free stems such as `sample_001`. Collection IDs use `profile.output_path(sample).with_suffix("").as_posix()`: `train/sample_001` for a unique split, `dataset_1/sample_001` for multiple unsplit sources, or `dataset_1/train/sample_001` for repeated splits. One unsplit source still uses bare stems.

Predictions use collection class IDs and each sample's GT coordinates: normalized xyxy for YOLO, pixel xyxy for VOC/LabelMe. Image checks and cached dimensions never change this convention. `val()` reads neither images nor label files and never rescans or checks files. Invalid samples are excluded, and the same missing/unknown prediction rules apply. Dataset types adapt GT; Evaluator remains purely numerical. Callers own display and printing.

### 4. Visualize annotations

```python
import cv2
from dsetkit.annotations.io import load
from dsetkit.visualize.plot import Plotter

img = cv2.imread("image.jpg")
ann = load(label_path="label.xml", image_path="image.jpg", fmt="voc")

plotter = Plotter(img)
for item in ann.items:
    plotter.detection_from_schema(item)
plotter.save("vis.jpg")
```

You can also draw one box directly with a custom caption:

```python
plotter = Plotter(img)
rendered = plotter.detection(
    bbox=[50, 40, 180, 160],
    class_id=1,
    text="person: 0.98",
)
cv2.imwrite("vis.jpg", rendered)
```

`Plotter.detection(...)` now accepts optional `class_id`, renders custom `text` when provided, and returns the current image array.

### 5. Dataset batch operations

Use `dsetkit.tools` for conversion/export, and `partition_paths()` with explicit persistence for partitioning:

```python
from pathlib import Path

from dsetkit import open_dataset, partition_paths
from dsetkit.tools import convert_dirs, export_dirs

# LabelMe -> YOLO (skips samples without labels)
convert_dirs(
    task_type="detection", image_dir="images/train",
    label_dir="labels/labelme",
    source_format="labelme",
    target_format="yolo",
    names=["cat", "dog"],
    out_dir="output",
)

# Export all image paths to a txt file
export_dirs(task_type="detection", image_dir="images/train", txt_path="all.txt")

# Split into train/val/test txt files (example ratios: 0.75 / 0.15 / 0.1)
profile = open_dataset(task_type="detection", image_dir="images/train", names=[]).profile()
partitions = partition_paths(profile.image_paths, rates=(0.75, 0.15, 0.1), seed=22)
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)
for name, paths in partitions.items():
    (output_dir / f"{name}.txt").write_text(
        "".join(f"{path}\n" for path in paths),
        encoding="utf-8",
    )
```

---

## Supported Annotation Formats

| Format | Identifier | File Suffix | Notes |
| ------ | -------- | ---------- | ------ |
| LabelMe | `labelme` | `.json` | Common interactive labeling export |
| Pascal VOC | `voc` | `.xml` | XML detection annotation |
| YOLO | `yolo` | `.txt` | Normalized `class cx cy w h` |

Formats are extensible through the registry (`register_format`). See `src/dsetkit/annotations/registry.py` and adapters under `formats/*.py`.

---

## Unified Data Model

```text
Annotation
├── width, height          # require_size() validates dimensions before export
├── names: list[str]
├── extra: dict[str, Any]
└── items: list[AnnotationItem]
    ├── category, category_id
    ├── bbox: BBox(x1, y1, x2, y2)   # absolute pixels, top-left to bottom-right
    └── extra / segmentation / keypoints (reserved)

EvaluationSample
├── sample_id: str
├── prediction: PredictionResult
│   ├── boxes: ndarray(N, 4)         # xyxy, float32
│   ├── cls: ndarray(N,)             # int class ids
│   └── conf: ndarray(N,)            # confidence scores
└── target: AnnotationTarget
    ├── boxes: ndarray(M, 4)         # xyxy, float32
    └── cls: ndarray(M,)             # int class ids
```

---

## Project Structure

```text
dsetkit/
├── src/dsetkit/
│   ├── annotations/       # schema, io, format adapters
│   ├── dataset.py         # Single-source scanning and checks
│   ├── collection.py      # Discovery orchestration and repartitioning
│   ├── collection_profile.py # Membership, permanent exclusions and partition statistics
│   ├── profile.py         # Results, statistics and processing access
│   ├── _profile_io.py     # JSON codec and structural validation
│   ├── partitions.py      # train/val/test splitting
│   ├── tools.py           # batch convert/plot/export helpers
│   ├── detection.py       # AnnotationTarget / PredictionResult / EvaluationSample
│   ├── evaluator.py       # pure prediction + GT pairing and evaluation
│   ├── metrics.py         # IoU, AP, P/R/F1
│   ├── utils/             # image metadata, file indexing
│   └── visualize/         # Plotter
└── tests/                 # SDK regression tests and metric reference data
```

---

## Development and Build

```bash
# editable install
pip install -e .

# build wheel
uv build --wheel
```


For release history, see [CHANGELOG.md](CHANGELOG.md).

---

## License

[MIT](pyproject.toml) · Author: Jasper Tao




## Scan options and processing membership

`Dataset`, `open_dataset()` and dataset-building `*_dirs` tools require
`task_type="detection"`; other tasks are rejected. `Dataset` and `open_dataset()`
define independent options: `check_annotations=True`, `check_images=False`.
Image path stem conflicts are excluded first, before parsing or checking their labels
or inspecting their images. Remaining selected annotations are parsed once.
The annotation check validates boxes; image checking additionally audits containers,
fully decodes images and checks boxes against actual image dimensions. Missing or
mismatched annotation size metadata alone is not an error when image dimensions are available.
Explicit image_info or checked image dimensions override annotation size in returned
pixel annotations without changing cached source annotations. Without image dimensions,
existing annotation size is used; missing size alone does not invalidate a box.
Previously recorded errors remain permanent. Batch conversion and plotting use checked
image dimensions or read the image header on demand, even when annotation size is present.
Single-file load/convert also prefer image dimensions when image_path is provided;
explicit width/height can supply known actual dimensions without another header read.
Disabling a check assumes processable content; it does not certify that content.
Unparseable annotations and unsupported detection structures remain errors.

`names` is optional. VOC/LabelMe infer names in first-occurrence order from
processable samples. YOLO preserves actual IDs; missing names
appear in `profile.unnamed_class_ids` and `warnings` without rejecting samples.
`class_ids`, nullable `class_names` and `class_instances` are parallel sequences.
Formats are normalized to lowercase. `convert()`, `convert_dataset()` and `convert_dirs()` reject identical source and target formats.
Conversions involving YOLO require a complete final class mapping: pass `names` explicitly for a single file; dataset conversion defaults to `profile.names`.
An explicit `names=[]` never falls back. VOC/LabelMe to YOLO assigns output IDs by category name in the final list.
Mixed YOLO/named-format collections require a shared explicit mapping.
`source_format` accepts real formats or None, never `"unannotated"`; a part without
an annotation directory has format None.

Dataset.check_images() checks known images when the caller chooses; Dataset.check_annotations() checks cached labels. Both update the scanner's current Profile and leave old results unchanged. Profile and Collection have no check methods.
Both Profile types provide export()/from_json(); restore collections with DatasetCollectionProfile.from_json(text), then DatasetCollection(restored) to continue splitting. JSON schema version 2 has no legacy reader. Restoration reads no source files; source caches are stored once, and collection additions preserve the full exclusion history.
`profile.as_dict()` returns statistics only. Conversion, plotting and export accept a Profile directly. Callers can extract GT for the independent Evaluator or use dataset.val().
Iteration, counts and processing reuse this membership and cached annotations;
pixel-based operations may still read the target image. Errors never become backgrounds.
`raw_samples` retains paths only; `invalid_samples` and `profile.errors` expose
exclusions. `total_images`/`total_label_files` count raw entries.
Sources are assumed immutable; an explicit build rescans them. Structural inspection
rejects symbolic links, special files and line breaks. Profile construction marks every
image with a duplicate stem in the same directory or partition as `invalid_image`,
regardless of `check_images`. Different partitions may share stems. Sample discovery remains fixed-depth. `inventory_dataset()` remains a task-independent raw inventory.

The former `export_scan/from_scan`, `DatasetView` and conversion/plotting/export forwarding methods are removed; dataset.val() provides evaluation separately. Saved legacy scan dictionaries require an explicit migration by the consumer.

LabelMe `imagePath` must be a string; missing or incorrectly typed values are annotation errors. Profile JSON requires absolute source and sample paths, and rejects non-finite numbers including exponent overflow.

## Collection snapshots and lazy image access

DatasetCollection accepts source Profiles, not Dataset instances. Collection construction retains native partitions; multiple unassigned sources share one scope. `collection.split((0.8, 0.2, 0), seed=22)` pools current valid members, permanently excludes every duplicate-stem candidate before allocation, and returns a new Collection. Different native partitions may share stems until pooling. Existing exclusions never return after a seed/ratio change or JSON restoration. File-invalid samples never participate in duplicate checks.

`profile.members` maps samples to train/val/test or None. `profile.excluded` contains cumulative sample errors; `profile.partition_stats` contains effective per-partition statistics, including empty native partitions. Overall raw totals and orphan-label diagnostics belong to the whole collection. `partition_paths()` remains a lower-level path utility and does not maintain collection exclusions.

Dataset instances can be released after scanning. Image sizes are optional and are not fetched during default scanning. Conversion and plotting directly read image headers only when missing dimensions are needed, and plotting reads pixels to render. Neither needs a Dataset instance or changes the Profile. `profile.annotation(sample, image_info=...)` accepts explicit dimensions for pixel-coordinate conversion; `ground_truth()` never needs images. File-error and duplicate-error exclusions are permanent in the current collection lineage.
