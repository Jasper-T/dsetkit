# dsetkit

Language: [中文](README.md) | **English** | [日本語](README.ja.md)

**Deep learning dataset infrastructure toolkit** — a Python toolkit for object detection and annotation pipelines, with a unified annotation schema, multi-format conversion, dataset indexing, evaluation metrics, and visualization.

[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](pyproject.toml)

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Annotation Formats](#supported-annotation-formats)
- [Unified Data Model](#unified-data-model)
- [Project Structure](#project-structure)
- [Development and Build](#development-and-build)
- [License](#license)

---

## Features

| Module | Capability |
|------|------|
| **Annotation I/O** | Read/write **COCO / LabelMe / VOC / YOLO** with a unified `Annotation` schema |
| **Format Conversion** | Single-file `convert` and large-scale `batch_convert` (multiprocessing supported) |
| **Dataset** | Auto-pair images and labels by filename stem, with priority selection when multiple formats coexist |
| **Evaluation** | IoU-based TP/FP matching, mAP@0.5, Precision / Recall / F1 (per-class and overall) |
| **Image Utilities** | Read JPEG/PNG/WebP/BMP metadata without full decoding; natural sorting traversal |
| **Visualization** | `Plotter` draws detection boxes on images (optional OpenCV dependency) |

---

## Installation

**Base install** (annotations, dataset, metrics):

```bash
pip install -e .
```

**With visualization** (requires OpenCV):

```bash
pip install -e ".[visualize]"
```

Requires **Python >= 3.11**. Core dependencies: `numpy`, `natsort`.

---

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
    names=["smoke", "fire"],   # names are typically required in YOLO-related load/dump
    out_dir="output/labels",
)
```

**Two-step flow** (helpful for debugging intermediate results):

```python
ann = load(
    label_path="path/to/label.json",
    image_path="path/to/image.jpg",
    fmt="labelme",
)

dump(ann, "output/label.txt", fmt="yolo")
```

> **Note**: For **YOLO** read/write, pass `names` to `load` when possible (class id -> class name). For YOLO `dump`, `Annotation` must include valid `width`/`height`, and each object needs `category_id` or a resolvable class via `ann.names`.

### 2. Batch conversion

Designed for very large datasets (100k+ samples). Outputs are written as `out_dir / {stem}{suffix}`.

```python
from dsetkit.annotations.convert import batch_convert
from dsetkit import Dataset

dataset = Dataset(
    image_dir="images/train",
    label_dir="labels/labelme",
    names=["cat", "dog"],
    input_format="labelme",
)

# convert only labeled samples
pairs = [s for s in dataset if s.label_path is not None]

paths = batch_convert(
    pairs,
    source_fmt="labelme",
    target_format="yolo",
    out_dir="labels/yolo",
    names=["cat", "dog"],
    max_workers=8,          # None for single-process; on Windows use __main__ guard for multiprocessing
    errors="raise",         # or "skip" to return (success_paths, failure_list)
)
```

### 3. Build dataset index

`Dataset` scans the image directory and finds matching label files by stem. If multiple suffixes exist for the same stem (for example `.txt` and `.json`), priority is: **yolo > coco/labelme > voc**.

```python
from dsetkit import Dataset

dataset = Dataset(
    image_dir="images/val",
    label_dir="labels",
    names=["person", "car"],
    input_format="voc",
)

print(len(dataset))
for sample in dataset:
    print(sample.image_path, sample.label_path)
```

### 4. Evaluate object detection

Subclass `Evaluator` and implement `_load_predictions` to compute metrics on a fixed annotation format:

```python
from pathlib import Path
from dsetkit import Dataset
from dsetkit.evaluator import Evaluator

class MyEvaluator(Evaluator):
    def _load_predictions(self, image_path: Path):
        # return list[dict], each item: bbox [x1,y1,x2,y2], label str, conf float
        ...

dataset = Dataset(
    image_dir="images/val",
    label_dir="labels/yolo",
    names=["person"],
    input_format="yolo",
)

metrics = MyEvaluator(dataset).evaluate(
    conf_threshold=0.5,
    iou_threshold=0.5,
    print_metrics=True,
)
# metrics includes mAP, precision, recall, f1, and per_class details
```

Terminal output follows a YOLO-style validation table (`Class / Instances / P / R / mAP50 / F1`).

### 5. Visualize annotations

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

See [`examples/`](examples/) and [`tests/demo.py`](tests/demo.py) for runnable examples.

---

## Supported Annotation Formats

| Format | Identifier | File Suffix | Notes |
|------|--------|----------|------|
| COCO | `coco` | `.json` | JSON for detection/instance tasks |
| LabelMe | `labelme` | `.json` | Common interactive labeling export |
| Pascal VOC | `voc` | `.xml` | XML detection annotation |
| YOLO | `yolo` | `.txt` | Normalized `class cx cy w h` |

Formats are extensible through the registry (`register_format`). See `src/dsetkit/annotations/registry.py` and adapters under `formats/*.py`.

---

## Unified Data Model

```text
Annotation
├── image_path, width, height
├── names: list[str]
└── items: list[AnnotationItem]
    ├── category, category_id
    ├── bbox: BBox(x1, y1, x2, y2)   # absolute pixels, top-left to bottom-right
    └── extra / segmentation / keypoints (reserved)
```

---

## Project Structure

```text
dsetkit/
├── src/dsetkit/
│   ├── annotations/       # schema, io, convert, format adapters
│   ├── dataset.py         # Dataset / DatasetSample
│   ├── evaluator.py       # detection evaluation base class
│   ├── metrics.py         # IoU, AP, P/R/F1
│   ├── utils/             # image metadata, file indexing
│   └── visualize/         # Plotter (optional opencv)
├── examples/              # conversion examples
└── tests/                 # demo and tests
```

---

## Development and Build

```bash
# editable install
pip install -e .

# build wheel
python -m build --wheel
```

Clean `__pycache__`: `scripts/clean_pycache.sh`

For release history, see [CHANGELOG.md](CHANGELOG.md).

---

## License

[MIT](pyproject.toml) · Author: Jasper Tao
