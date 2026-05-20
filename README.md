# dsetkit

Language: **中文** | [English](README.en.md) | [日本語](README.ja.md)

**Deep learning dataset infrastructure toolkit** — 面向目标检测与标注流水线的 Python 工具库：统一标注 schema、多格式互转、数据集索引、检测指标评估与可视化。

[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](pyproject.toml)

## 目录

- [特性](#特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [支持的标注格式](#支持的标注格式)
- [统一数据模型](#统一数据模型)
- [项目结构](#项目结构)
- [开发与打包](#开发与打包)
- [许可证](#许可证)

---

## 特性

| 模块 | 能力 |
|------|------|
| **标注 I/O** | 在统一 `Annotation` schema 下读写 **COCO / LabelMe / VOC / YOLO** |
| **格式转换** | 单文件 `convert`、大规模 `batch_convert`（支持多进程） |
| **数据集** | 按文件名 stem 自动配对图像与标签，多格式共存时按优先级选取 |
| **评估** | 基于 IoU 的 TP/FP 匹配、mAP@0.5、Precision / Recall / F1（类级与整体） |
| **图像工具** | 无需完整解码即可读取 JPEG/PNG/WebP/BMP 尺寸；自然排序遍历 |
| **可视化** | `Plotter` 在图像上绘制检测框（可选依赖 OpenCV） |

---

## 安装

**基础安装**（标注、数据集、指标）：

```bash
pip install -e .
```

**含可视化**（需要 OpenCV）：

```bash
pip install -e ".[visualize]"
```

要求 **Python ≥ 3.11**。核心依赖：`numpy`、`natsort`。

---

## 快速开始

### 1. 加载与转换标注

所有格式经内部适配器转为统一的 `Annotation`（绝对坐标 `xyxy` 的 `BBox`）。

```python
from dsetkit.annotations.io import load, dump, convert

# LabelMe JSON → YOLO txt（导出 YOLO 时需提供类别名）
convert(
    label_path="path/to/label.json",
    image_path="path/to/image.jpg",
    source_format="labelme",
    target_format="yolo",
    names=["smoke", "fire"],   # 与 YOLO 相关的 load/dump 通常需要 names
    out_dir="output/labels",
)
```

**两步式**（便于调试中间结果）：

```python
ann = load(
    label_path="path/to/label.json",
    image_path="path/to/image.jpg",
    fmt="labelme",
)

dump(ann, "output/label.txt", fmt="yolo")
```

> **注意**：读写 **YOLO** 格式时，`load` 建议传入 `names`（类别 id → 名称）；`dump` 到 YOLO 要求 `Annotation` 已具备 `width`/`height`，且每个目标有 `category_id` 或可通过 `ann.names` 解析类别。

### 2. 批量转换

适合十万级样本；输出路径统一为 `out_dir / {stem}{suffix}`。

```python
from dsetkit.annotations.convert import batch_convert
from dsetkit import Dataset

dataset = Dataset(
    image_dir="images/train",
    label_dir="labels/labelme",
    names=["cat", "dog"],
    input_format="labelme",
)

# 仅转换有标签的样本
pairs = [s for s in dataset if s.label_path is not None]

paths = batch_convert(
    pairs,
    source_fmt="labelme",
    target_format="yolo",
    out_dir="labels/yolo",
    names=["cat", "dog"],
    max_workers=8,          # None 为单进程；Windows 下多进程需在 __main__ 中调用
    errors="raise",         # 或 "skip" 返回 (成功路径, 失败列表)
)
```

### 3. 构建数据集索引

`Dataset` 扫描图像目录，按 stem 在标签目录中查找对应标注文件。若同一 stem 存在多种后缀（如 `.txt` 与 `.json`），按优先级选择：**yolo > coco/labelme > voc**。

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

### 4. 目标检测评估

继承 `Evaluator` 并实现 `_load_predictions`，即可对固定标注格式计算指标：

```python
from pathlib import Path
from dsetkit import Dataset
from dsetkit.evaluator import Evaluator

class MyEvaluator(Evaluator):
    def _load_predictions(self, image_path: Path):
        # 返回 list[dict]，每项: bbox [x1,y1,x2,y2], label str, conf float
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
# metrics 含 mAP、precision、recall、f1 及 per_class 明细
```

终端输出风格类似 YOLO 验证表（`Class / Instances / P / R / mAP50 / F1`）。

### 5. 可视化标注

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

示例数据与脚本见 [`examples/`](examples/) 与 [`tests/demo.py`](tests/demo.py)。

---

## 支持的标注格式

| 格式 | 标识符 | 文件后缀 | 说明 |
|------|--------|----------|------|
| COCO | `coco` | `.json` | 实例分割/检测 JSON |
| LabelMe | `labelme` | `.json` | 常见交互式标注导出 |
| Pascal VOC | `voc` | `.xml` | XML 检测标注 |
| YOLO | `yolo` | `.txt` | 归一化 `class cx cy w h` |

格式通过注册表扩展（`register_format`），详见 `src/dsetkit/annotations/registry.py` 与各 `formats/*.py` 适配器。

---

## 统一数据模型

```text
Annotation
├── image_path, width, height
├── names: list[str]
└── items: list[AnnotationItem]
    ├── category, category_id
    ├── bbox: BBox(x1, y1, x2, y2)   # 绝对像素，左上-右下
    └── extra / segmentation / keypoints（预留）
```

---

## 项目结构

```text
dsetkit/
├── src/dsetkit/
│   ├── annotations/       # schema、io、convert、各格式适配器
│   ├── dataset.py         # Dataset / DatasetSample
│   ├── evaluator.py       # 检测评估基类
│   ├── metrics.py         # IoU、AP、P/R/F1
│   ├── utils/             # 图像元信息、文件索引
│   └── visualize/         # Plotter（可选 opencv）
├── examples/              # 转换示例
└── tests/                 # demo 与测试
```

---

## 开发与打包

```bash
# 可编辑安装
pip install -e .

# 构建 wheel（可先备份旧 build 目录）
python -m build --wheel
```

清理 `__pycache__`：`scripts/clean_pycache.sh`

变更记录见 [CHANGELOG.md](CHANGELOG.md)。

---

## 许可证

[MIT](pyproject.toml) · Author: Jasper Tao
