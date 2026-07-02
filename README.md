# dsetkit

Language: **中文** | [English](README.en.md) | [日本語](README.ja.md)

**Deep learning dataset infrastructure toolkit** — 面向目标检测与标注流水线的 Python 工具库：统一标注 schema、多格式互转、数据集索引、划分与增强、检测指标评估与可视化。

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
| ------ | ------ |
| **标注 I/O** | 在统一 `Annotation` schema 下读写 **LabelMe / VOC / YOLO** |
| **格式转换** | 单文件 `convert`；数据集级 `convert_dataset` / `convert_dirs` |
| **数据集** | 按文件名 stem 自动配对图像与标签，并基于 `source_format` 过滤标签后缀 |
| **划分与导出** | 将图像路径列表导出为 txt，并按比例随机划分 train/val/test |
| **数据增强** | 水平/垂直翻转，90°/180°/270° 顺时针旋转（同步更新检测框） |
| **评估** | 对已有预测结果计算 Ultralytics 风格检测指标：P / R / mAP@iou / F1 |
| **图像工具** | 无需完整解码即可读取 JPEG/PNG/WebP/BMP 尺寸；自然排序遍历 |
| **可视化** | `Plotter` 在图像上绘制检测框（依赖 OpenCV，已包含在基础安装中） |

---

## 安装

**于础安装**（标注、数据集、增强、可视化、指标）：

```bash
pip install -e .
```

要求 **Python ≥ 3.11**。核心依赖：`numpy>=2.0`、`natsort`、`opencv-python`、`tqdm`。

---

## 快速开始

### 1. 加载与转换标注

所有格式经内部适配器转为统一的 `Annotation`（绝已坐标 `xyxy` 的 `BBox`）。

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

### 2. 构建数据集索引

`Dataset` 扫描图像目录，按 stem 在标签目录中查找已应标注文件，并按 `source_format` 已应后缀进行匹配（`labelme -> .json`, `voc -> .xml`, `yolo -> .txt`）。

```python
from dsetkit import Dataset

dataset = Dataset(
    image_dir="images/val",
    label_dir="labels",
    names=["person", "car"],
    source_format="voc",
)
dataset.build()

print(len(dataset))
print(dataset.stats().as_dict())  # images / backgrounds / instances

for sample in dataset:
    print(sample.image_path, sample.label_path)
    target = dataset.ground_truth(sample)
```

### 3. 目标检测评估

将已有预测结果传给 `Evaluator`；ground truth 由 `Dataset` 读取：

```python
from dsetkit import Dataset
from dsetkit.evaluator import Evaluator

dataset = Dataset(
    image_dir="images/val",
    label_dir="labels/yolo",
    names=["person"],
    source_format="yolo",
)
dataset.build()

predictions = {
    "image_001.jpg": [
        {"bbox": [10, 20, 80, 120], "label": "person", "conf": 0.91},
    ],
}

metrics = Evaluator(dataset).evaluate(
    predictions=predictions,
    conf_threshold=0.001,
    iou=0.5,
    print_metrics=True,
)
# metrics 含 precision、recall、f1、mAP50、mAP50-95 及 per_class 明细
```

`iou` 是 dsetkit 自己的评估 IoU 起始阈值：`iou=0.5` 会按固定 `0.05` 步长评估 `0.50...0.95`，同时输出 `mAP50` 和 `mAP50-95`；`iou=0.25` 则输出 `mAP25` 和 `mAP25-95`。
终端输出风格类似 YOLO 验证表（`Class / Images / Instances / P / R / mAPxx / mAPxx-95 / F1`）。

### 4. 可视化标注

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

也可以直接绘制单个检测框，并传入自定义显示文本：

```python
plotter = Plotter(img)
rendered = plotter.detection(
    bbox=[50, 40, 180, 160],
    class_id=1,
    text="person: 0.98",
)
cv2.imwrite("vis.jpg", rendered)
```

`Plotter.detection(...)` 现在支持可选 `class_id`、自定义 `text`，并直接返回当前绘制后的图像数组。

示例数据与脚本见 [`examples/`](examples/) 与 [`tests/demo.py`](tests/demo.py)。

### 5. 数据集批量操作

基于 `Dataset` 的便捷入口（`dsetkit.tools`），适合大规模转换、导出与划分：

```python
from dsetkit import Dataset
from dsetkit.tools import convert_dirs, export_dirs, split_dirs

# LabelMe → YOLO（跳过无标签样本）
convert_dirs(
    image_dir="images/train",
    label_dir="labels/labelme",
    source_format="labelme",
    target_format="yolo",
    names=["cat", "dog"],
    out_dir="labels/yolo",
)

# 导出全部图像路径
export_dirs(image_dir="images/train", txt_path="all.txt")

# 按比例划分 train/val/test txt（默认 0.75 / 0.15 / 0.1）
split_dirs(image_dir="images/train", out_dir="splits", seed=42)
```

### 6. 数据增强（翻转 / 旋转）

已图像与标注同步增强，输出写入 `out_dir/images/` 与已应格式标签目录：

```python
from dsetkit.tools import flip_dirs, rotate_dirs

# 水平翻转（direction=1）；垂直翻转为 direction=0
flip_dirs(
    image_dir="images/train",
    label_dir="labels/voc",
    source_format="voc",
    out_dir="augment/flipped",
    direction=1,
)

# 顺时针旋转 90°（亦支持 180、270）
rotate_dirs(
    image_dir="images/train",
    label_dir="labels/yolo",
    source_format="yolo",
    names=["person"],
    out_dir="augment/rot90",
    angle=90,
)
```

单样本或 schema 级 API 见 `dsetkit.augment`（如 `flip_annotation`、`rotate_label`）。

---

## 支持的标注格式

| 格式 | 标识符 | 文件后缀 | 说明 |
| ------ | -------- | ---------- | ------ |
| LabelMe | `labelme` | `.json` | 常见交互式标注导出 |
| Pascal VOC | `voc` | `.xml` | XML 检测标注 |
| YOLO | `yolo` | `.txt` | 归一化 `class cx cy w h` |

格式通过注册表扩展（`register_format`），详见 `src/dsetkit/annotations/registry.py` 与各 `formats/*.py` 适配器。

---

## 统一数据模型

```text
Annotation
├── width, height          # require_size() 可在导出/增强前校验尺寸
├── names: list[str]
├── extra: dict[str, Any]
└── items: list[AnnotationItem]
    ├── category, category_id
    ├── bbox: BBox(x1, y1, x2, y2)   # 绝已像素，左上-右下
    └── extra / segmentation / keypoints（预留）
```

---

## 项目结构

```text
dsetkit/
├── src/dsetkit/
│   ├── annotations/       # schema、io、各格式适配器
│   ├── augment/           # 翻转 / 旋转（图像 + 标注）
│   ├── dataset.py         # Dataset / DatasetSample / DatasetStats
│   ├── split.py           # train/val/test 划分
│   ├── tools.py           # 批量转换/可视化/增强/划分便捷入口
│   ├── evaluator.py       # 检测评估于类
│   ├── metrics.py         # NumPy IoU、AP、P/R/F1
│   ├── utils/             # 图像元信息、文件索引
│   └── visualize/         # Plotter
├── examples/              # 转换示例
└── tests/                 # demo 与测试
```

---

## 开发与打包

```bash
# 可编辑安装
pip install -e .

# 构建 wheel
uv build --wheel
```


变更记录见 [CHANGELOG.md](CHANGELOG.md)。

---

## 许可证

[MIT](pyproject.toml) · Author: Jasper Tao






