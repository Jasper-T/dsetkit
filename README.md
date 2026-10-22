# dsetkit

Language: **中文** | [English](README.en.md) | [日本語](README.ja.md)

**Deep learning dataset infrastructure toolkit** — 面向目标检测与标注流水线的 Python 工具库：统一标注 schema、多格式互转、数据集索引、划分、检测指标评估与可视化。

[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](pyproject.toml)

## 目录

- [特性](#特性)
- [安装](#安装)
- [组件职责与独立使用](#组件职责与独立使用)
- [与 PyTorch Dataset 和 DataLoader 的关系](#与-pytorch-dataset-和-dataloader-的关系)
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
| **数据集画像** | 区分正样本、背景图、无效图片与孤立标签，并统计各类别实例数 |
| **数据集发现** | 按标准目录约定无副作用地发现图片目录与 LabelMe / VOC / YOLO 标注源 |
| **Schema 检查** | 严格比较类别 ID、名称和顺序，不进行隐式重排 |
| **划分与导出** | 将图像路径列表导出为 txt，并按比例随机划分 train/val/test |
| **评估** | 对已有预测结果计算 Ultralytics 风格检测指标：P / R / mAP@iou / F1 |
| **图像工具** | 无需完整解码即可读取 JPEG/PNG/WebP/BMP 尺寸；自然排序遍历 |
| **可视化** | `Plotter` 在图像上绘制检测框（依赖 OpenCV，已包含在基础安装中） |

---

统一布局、视图与划分的推荐用法及接口变更见 [DATA_COLLECTIONS.md](DATA_COLLECTIONS.md)。

## 安装

**基础安装**（标注、数据集、可视化、指标）：

```bash
pip install -e .
```

要求 **Python ≥ 3.11**。核心依赖：`numpy>=2.0`、`natsort`、`opencv-python`、`tqdm`、`loguru`。

---

## 组件职责与独立使用

**dsetkit 按能力提供可单独调用、可以组合的组件。** 调用方可以只读一个标注、检查一张图片、转换一个文件、绘制一组框或计算一次指标，无需先创建 Dataset。需要来源扫描结果时再使用 Dataset，需要多来源组织和分区管理时再使用 DatasetCollection。

“解耦”指职责和输入输出契约明确，并不表示模块之间没有依赖。例如 Dataset 复用标注解析和检查函数，Collection 和批量处理工具消费 Profile，Evaluator 复用 metrics；这些依赖不要求扫描实例长期存活。Profile 是共享的结果契约，不是调用所有能力的总入口。

### 核心对象

| 类／入口 | 职责与输出 | 能力边界 | 适用场景与生命周期 |
| --- | --- | --- | --- |
| `Dataset` | 扫描明确的图片／标签来源、检查标注、可选检查图片，生成当前 `DatasetProfile` | 不发现未知布局、不管理 train/val/test、不划分、不训练、不组批；默认只检查标注，不读取图片 | 已知目录的单来源检查与分析；取得 Profile 后可释放；如需稍后在同一实例检查图片，可以保留 |
| `DatasetProfile` | 保存单来源清单、缓存标注、检查结果、可选图片尺寸及来源统计；提供结果访问、GT 适配、完整导出与恢复 | 不扫描、不检查文件、不分区、不提供 val；像素标注转换缺尺寸时由调用方提供 `image_info` | 扫描结果复用、持久化、下游处理；不依赖 Dataset 继续存活 |
| `DatasetCollection` | 消费来源 Profiles，组织成员、保留原生分区、维护永久排除并重新 split；`open()` 编排发现及临时扫描 | 汇总和重排不读取源图片或标签，不重新判定文件有效性，不移动文件 | 多来源分析、分区管理；仅保留集合结果，split 返回新 Collection |
| `DatasetCollectionProfile` | 保存来源引用、样本级分区成员、累计排除、集合类别映射及整体／分区统计 | 不复制缓存标注、不持有扫描器、不主动读取文件、不自行 split | 可独立恢复的集合快照；供重排、转换、绘图、导出和评估读取 |
| `Evaluator` | 接收显式 predictions 与 ground_truths，校验并配对 sample_id，委托 metrics 返回指标 | 无 Dataset／Profile、路径、图片、文件或模型依赖，不打印 | 内存中的预测评估；可以完全跳过扫描、发现及集合流程 |
| `Plotter` | 在传入图片数组的副本上绘制框、文字和标签，返回结果或显式保存／显示 | 不扫描、不划分、不计算指标；输入框与图片应使用相同像素坐标 | 已有图片数组的单图调试或渲染；无需 Profile 或 Dataset |

Dataset 的核心职责就是**扫描＋标注检查＋图片检查＋Profile 生成**。现有 `Dataset.val()` 和 `DatasetCollection.val()` 只是 GT 到纯 Evaluator 的便捷适配，不增加第二套评估算法，也不把转换、绘图、训练或加载批次纳入 Dataset 的职责。需要完全独立的评估时直接使用 Evaluator。

### 数据与结果类型

这些类用于描述输入或结果，本身不是另一个处理阶段。

| 类 | 描述的事实／能力 | 边界与使用位置 |
| --- | --- | --- |
| `BBox` | 一个 xyxy 框 | 普通标注 I/O 使用像素坐标；数值本身不会读取图片或推断坐标系 |
| `AnnotationItem` | 一个目标的类别、框和附加字段 | 检测流程目前只支持检测框；预留 segmentation/keypoints 等字段不代表已实现对应任务 |
| `Annotation` | 一张图片的标注、可选尺寸和名称表；`require_size()`／`to_array()` 访问已有数据 | 可单独构造、读写和检查；它是可变工作对象，Profile 返回的是副本 |
| `DatasetSample` | 图片路径及可选标签路径 | 是元数据记录，不是已加载图片或训练 tensor |
| `DatasetProfileError` | 错误路径、类别与原因 | 诊断记录；Collection 的 `excluded` 额外建立错误与样本身份的永久关联 |
| `DatasetDiscoveryCandidate` | 标注格式、候选标签目录及文件数 | 布局发现候选，不代表标注内容有效 |
| `DatasetDiscovery` | 图片目录及全部标注候选 | `select_annotation_source()` 再选择具体格式；单候选也保留在 candidates 中 |
| `DatasetPart` | 发现结果与原生分区标签 | 用于创建编排，不给单来源 Dataset 增加分区语义 |
| `FormatSpec` | 格式名称、扩展名、输出目录及发现别名 | 格式注册元数据；读写行为仍由对应 loader/dumper 提供 |
| `ClassSchemaDifference`／`ClassSchemaComparison` | 类别 ID、名称、顺序的差异和比较结论 | `compare_class_schemas()` 严格比较，不自动重排或改写映射 |
| `AnnotationTarget` | 数组形式的 GT 框和类别；支持空 GT | 不带来源或图片信息；数组转换和长度／类别 ID 检查不等于完整标注文件检查 |
| `PredictionResult` | 数组形式的预测框、类别和置信度；支持空预测 | 不运行模型、不加载预测文件 |
| `EvaluationSample` | 唯一非空字符串 ID 与一对预测／GT | ID 是普通字符串；该类型不验证文件路径，也不在多条记录间自动保证唯一性 |
| `ClassMetrics`／`DetectionMetrics` | 按类别和整体的指标结果；提供 `as_dict()` | 由数值计算生成，不读取数据、不打印 |
| `ImageInfo` | 图片宽高及可选通道、模式、格式信息 | 元信息记录，不是图片像素缓存 |
| `ImageValidationResult` | 图片检查是否通过、元信息及错误原因 | 单图检查结果，不会自动更新某个 Dataset／Collection |
| `InventoryDifference` | 两份文件摘要中的新增、删除和变化键 | 对比结果，不负责监控、重新扫描或更新集合 |

`AnnotationParseError`、`UnsupportedDatasetLayoutError`、`AnnotationSourceError`、`PartitionError` 分别表示标注解析、布局、标注源选择与路径分区错误。它们用于调用方区分失败原因，不是额外的业务组件。以下划线开头的模块、构造函数和 `_ProfileStats`／`_ProfileAccess` 是内部实现，不作为独立调用入口。

### 功能组件与独立调用入口

| 组件／入口 | 输入与能力 | 独立调用场景及边界 |
| --- | --- | --- |
| `annotations.io.load/dump/convert` | 指定格式的单标注读取、写出与转换；路径辅助函数生成目标标签目录／文件名 | 只转换一个文件，不需要 Dataset 或 Profile；写出与转换会写目标文件，所需图片尺寸可按需读取 |
| `annotations.io` 路径辅助函数 | `get_label_dir_name/get_label_file_suffix/new_label_path` 查询或生成路径；`new_label_dir/auto_label_path` 准备输出目录 | 只处理格式／输出路径，不扫描样本；后两者会创建目录 |
| `annotations.formats` | `load_labelme/dump_labelme`、`load_voc/dump_voc`、`load_yolo/dump_yolo` 实现具体格式读写 | 可直接指定文件调用；通常用统一 load/dump 入口，所需尺寸与类别映射仍由调用方提供 |
| `annotations.registry` | `register_format`、`get_loader/get_dumper/get_format_spec`、`list_formats/list_format_specs` | 扩展或查询格式适配器；注册改变进程内格式表，不创建扫描器 |
| `annotations.validation.validate_annotation` | 对内存 Annotation 检查检测框；已提供的图片尺寸优先于标注 size | 已有标注对象的独立校验，不查找文件或读取图片 |
| `compare_class_schemas` | 严格比较两份有序名称表 | 数据导入前检查类别定义；不是格式转换的名称重映射 |
| `discover_parts/select_annotation_source` | 识别受支持的固定布局并选择标注目录 | 仅查看布局与候选，不解析标注或检查图片内容；不递归猜测任意目录结构 |
| `open_dataset`／`DatasetCollection.open` | 发现或接受明确来源，临时扫描后组装 Collection | 创建便捷入口，始终返回 Collection；可以跳过它们，直接调用 Dataset 或底层函数 |
| `partition_paths/unique_paths` | 对明确路径排序、去重与按比例随机分配 | 无需文件存在或 Profile；这是路径操作，不继承 Collection 的排除历史 |
| `validate_partition_members/combine_partitions` | 检查同分区重复路径、跨分区重叠，或按原分区合并路径 | 不做按 stem 的永久排除，不自动重分；`normalize=True` 的路径解析有别于纯词法比较 |
| `save_partition_txts` | 把已明确的路径成员写入分区 TXT | 显式输出，不移动图片、不重新划分 |
| `DatasetCollection.split` | 汇集当前有效池，先永久排除同 stem 冲突，再分配新成员 | 需要集合结果，复用来源事实；改变种子、比例或恢复快照均不让错误样本复活 |
| `tools.convert_dataset/plot_dataset/export_dataset` | 以任一种 Profile 的有效成员批量转换、绘图或导出清单 | 需要结果契约，**不需要活着的 Dataset**；按实际需求直接读取尺寸或绘图像素，不重新解析源标注 |
| `tools.convert_dirs/plot_dirs/export_dirs` | 用明确目录临时扫描，再执行对应批量操作 | 一次调用完成流程；属于组合入口，不是另一套转换或绘图算法 |
| `visualize.plot.plot` | 已有图片数组／图片路径与 Annotation／标签文件的单图绘制 | 无需 Profile；只有显式提供文件路径时才读取对应文件，可直接使用内存对象 |
| `detection.prediction_from_records/target_from_records/class_id_from_record` | 将记录和名称映射转为数值预测／GT，或解析单条类别 ID | 无文件或 Profile 依赖；只适配输入，不负责模型推理或框匹配 |
| `metrics.detection_metrics` | `EvaluationSample` 列表到 `DetectionMetrics` | 可跳过 Evaluator 的映射适配；调用方负责样本配对和一致的类别／坐标 |
| `metrics.box_iou/match_predictions/compute_ap/ap_per_class` | 框 IoU、匹配与 AP 等数学计算 | 可直接计算数组；匹配函数要求预测已按置信度降序，不能跳过底层输入前提 |
| `metrics.normalize_iou_thresholds/expand_iou_thresholds/ap_keys_from_iou/smooth` | 阈值规范化、展开、指标键与曲线平滑 | 数值辅助；不读取数据、不判断划分、不做框 NMS；阈值去重不等于删除重复预测框 |
| `inventory_dataset/content_digest/file_hashes` | 生成文件清单或按内容读取文件并计算 SHA256 | 盘点与变化检测，不判断训练样本有效性；摘要计算会读取内容，不能误称为仅查看目录 |
| `compare_file_hashes/fingerprint_identity` | 对比已有摘要、对调用方定义的 identity 计算摘要 | 不自动监控或重扫；identity 的含义和路径策略由调用方决定 |
| `utils.image.read_image_info/resolve_image_wh` | 获取宽高，或优先使用已显式提供的宽高 | 尺寸有需要时读取图片头，不提前为全体样本准备 |
| `utils.image.validate_image` | 独立检查一张图片，`decode=True` 时增加完整解码 | 不需要 Dataset；调用它不会改变既有 Profile 或永久排除池 |
| `utils.image.iter_image_paths/get_image_paths/register_reader/image_to_base64` | 枚举直接图片文件、注册元信息读取器或编码图片内容 | 与集合组织解耦；枚举不等于有效性检查，Base64 编码会读取文件内容 |
| `utils.file.load_txt/save_txt/merge_txt_files/replace_text_in_txt` | 显式读写、合并或替换 TXT | 不生成或检查数据集成员；替换可原地写回，合并输出不能指向输入文件 |
| `utils.file.find_parent_dirs_with_subfolder/build_stem_index/validate_source_tree` | 目录搜索、按给定格式优先级建立 stem 索引、检查目录结构 | 不是 discovery 的任意布局识别；格式优先索引不等于 Collection 的重名全员排除 |
| `utils.file.ensure_dir/flatten_dir/rm_empty_dirs` | 创建目录、移动文件以扁平化目录、删除空目录 | 仅显式调用时执行；均不是 Dataset 扫描的隐式步骤 |

扫描和集合组织对源数据只读；这个承诺不应被扩大为“整个工具库没有写操作”。格式输出、清单保存、图片保存及显式文件整理均有各自的写入边界。

### 可以跳过扫描流程的例子

```python
from pathlib import Path
from dsetkit.annotations import Annotation, AnnotationItem, BBox
from dsetkit.annotations.validation import validate_annotation
from dsetkit.evaluator import Evaluator
from dsetkit import partition_paths

# 只检查内存标注，不读任何文件。
annotation = Annotation(width=100, height=80,
                        items=[AnnotationItem("object", 0, BBox(10, 10, 30, 40))])
validate_annotation(annotation, source_format="voc")

# 只评估内存数据，sample_id 不需要对应文件。
metrics = Evaluator().evaluate(
    {"sample": [{"bbox": [.1, .2, .5, .6], "class_id": 0, "conf": .9}]},
    {"sample": [{"bbox": [.1, .2, .5, .6], "class_id": 0}], "background": []},
)

# 只分配给定路径，不扫描目录，不维护集合排除记录。
groups = partition_paths([Path(f"/data/image_{i}.jpg") for i in range(10)],
                         rates=(.8, .2, 0), seed=22)
```

## 与 PyTorch Dataset 和 DataLoader 的关系

这三者处于不同职责层次，名称相近不代表可以直接互换。

| 维度 | dsetkit `Dataset`／`DatasetCollection`／Profile | PyTorch `Dataset`／`IterableDataset` | PyTorch `DataLoader` |
| --- | --- | --- | --- |
| 核心问题 | 来源里有什么、哪些样本有效、如何分区及保存分析结果 | 如何取得一个训练样本或样本流 | 如何按策略取得并交付一批样本 |
| 主要输出 | 来源／集合 Profile、成员路径、缓存标注、诊断和统计 | 调用方定义的单样本内容 | 经加载和整理的样本或 batch |
| 扫描与检查 | 有显式扫描、标注检查和可选图片检查 | 可以由自定义实现安排，不是统一的内容检查协议 | 不提供 dsetkit 的标注格式检查或永久排除模型 |
| 划分与采样 | 持久的 train/val/test 成员关系，重排继承累计排除 | 定义样本访问方式 | 通过 sampler/shuffle 等配置加载顺序，不等于重新划分集合 |
| 图片与 tensor | 默认不读图片，不自动生成训练 tensor | 可实现图片读取、变换和 tensor 转换 | 调用数据源并整理返回值；提供组批、worker、预取、内存固定等机制 |
| 生命周期 | Dataset 扫描后可释放，Profile 可离线保存和复用 | 参与训练时的样本访问 | 参与训练／推理时的数据迭代 |

PyTorch 的 map-style Dataset 使用 `__getitem__`／`__len__` 提供样本访问；IterableDataset 提供样本流。DataLoader 负责加载编排，两者也按职责分离。参见官方 [Dataset 类型说明](https://docs.pytorch.org/docs/stable/data.html#dataset-types) 和 [Dataset／DataLoader 教程](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)。

适合的组合关系是：

```text
源图片与标注
    → dsetkit 扫描／检查
    → 来源 Profiles → 可选 Collection 组织／重排
    → 调用方的 PyTorch Dataset 适配层（读取有效成员、按需解码、变换和 tensor 转换）
    → PyTorch DataLoader（采样顺序、组批、并行加载）
    → 训练或推理
```

这是推荐的集成方式，当前库没有内置 PyTorch 适配器。dsetkit.Dataset 不是 torch.utils.data.Dataset 的子类，也不提供训练样本索引协议。Profile 虽然能索引，但返回的是 DatasetSample 路径记录，不能据此认为已经具备图像加载、检测目标整理或默认组批能力。

适配层需明确选择集合分区、统一模型所需类别和坐标约定，并将图像变换同步应用到框。当前 Profile 的 GT：YOLO 为归一化 xyxy，VOC／LabelMe 为像素 xyxy；每个预测／GT 对必须一致。检测目标数量可变时，应配置适合目标结构的 `collate_fn`，不将其责任放进 dsetkit 扫描层。DataLoader 的加载机制参见官方 [DataLoader 文档](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)。

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
    out_dir="output",
)
```

**两步式**（便于调试中间结果）：

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

> **注意**：读写 **YOLO** 格式时，`load` 建议传入 `names`（类别 id → 名称）；`dump` 到 YOLO 要求 `Annotation` 已具备 `width`/`height`，且每个目标有 `category_id` 或可通过 `ann.names` 解析类别。

### 2. 构建数据集索引

`open_dataset()` 是统一扫描入口，无需再次调用 `build()`；通过 `.profile()` 取得完整分析结果。显式模式接受任意图片、标签目录名称，按 stem 关联文件并按 `source_format` 过滤后缀（`labelme -> .json`, `voc -> .xml`, `yolo -> .txt`）。

```python
from dsetkit import open_dataset

profile = open_dataset(
    task_type="detection", image_dir="images/val",
    label_dir="labels",
    names=["person", "car"],
    source_format="voc",
).profile()

print(len(profile))

print(profile.positive_images)    # 有至少一个有效实例的图片
print(profile.background_images)  # 无标签或标签中没有实例的图片
print(profile.invalid_images)     # 图片损坏或标注无法读取
print(profile.orphan_labels)      # 没有对应图片的标签文件
print(profile.class_instances)    # 按类别 ID 顺序排列的实例数

for sample in profile:
    print(sample.image_path, sample.label_path)
    target = profile.ground_truth(sample)
```

`Dataset.profile()` 返回单来源 `DatasetProfile`；`open_dataset().profile()` 返回含来源引用、分区成员、累计排除和分区统计的 `DatasetCollectionProfile`。默认不检查图片；显式启用图片检查时检查容器并完整解码。错误明细位于 `profile.errors`。


也可使用 `open_dataset(task_type="detection", root="dataset", names=["object"], source_format="yolo")` 自动发现固定布局。`root` 与 `image_dir` / `label_dir` 互斥；显式标签目录必须同时指定实际标注格式。显式模式省略 `label_dir` 时只处理图片，不搜索默认目录。两种模式都只读取扫描目录的直接子文件。`Dataset` 负责单来源扫描，`DatasetCollection` 组装扫描结果；转换、绘图和导出接收两种 Profile；评估可使用数据集的 `val()`，或向纯 Evaluator 显式提供预测与 GT。

### 2.1 发现数据集布局

```python
from dsetkit import discover_parts, select_annotation_source, open_dataset

native, parts = discover_parts("path/to/dataset")
for part in parts:
    source = select_annotation_source(part.discovery, requested="yolo")
    print(part.split, part.discovery.image_dir, source.label_dir if source else None)

profile = open_dataset("path/to/dataset", task_type="detection", names=["object"], source_format="yolo").profile()
print(profile)
```

`discover_parts()` 是统一的目录发现入口，仅识别平面、分区在外、分区在内三种固定布局；每个目录只读取直接子文件。每个分区的标注源统一位于 `part.discovery.candidates`，单一来源也不例外；`select_annotation_source()` 校验指定格式与歧义，`open_dataset().profile()` 提供保留分区归属的完整分析结果。标签目录按格式注册表发现，YOLO/VOC/LabelMe 分别匹配 `.txt`/`.xml`/`.json`。

### 2.2 严格比较类别 schema

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

strict 模式要求类别数量、名称和顺序完全一致；不会自动排序、忽略大小写或建立类别 mapping。

### 3. 目标检测评估

`Evaluator` 是纯数学评估器，直接比较已有检测结果与 GT；不依赖 Profile，不读取图片或标注文件，不执行模型推理或终端打印。检测框匹配和指标计算参考 Ultralytics：

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

`iou` 是 dsetkit 自己的评估 IoU 起始阈值：`iou=0.5` 会按固定 `0.05` 步长评估 `0.50...0.95`，同时输出 `mAP50` 和 `mAP50-95`；`iou=0.25` 则输出 `mAP25` 和 `mAP25-95`。
预测与 GT 字典以非空字符串 `sample_id` 为键；标识不要求路径或后缀，也不做路径解析。GT 决定全部评估样本，缺失预测按空预测计算；预测出现 GT 没有的标识时报错。背景样本必须显式保留为空 GT，误检会参与评估。
每个值可以是上面的记录列表，也可以是 `PredictionResult`／`AnnotationTarget`。预测与 GT 均使用相同类别 ID 和相同坐标系的 `xyxy`；可以是像素坐标或归一化坐标，无需图片尺寸。记录也可用 `cls` 或带 `names` 映射的 `label` 表示类别。
`metrics` 在 AP 计算前校验 IoU 阈值为 `(0, 1]` 内的有限数，并升序排序、去重；序列使用规范化后的阈值计算，单阈值低于 0.95 时按 0.05 步长展开至不超过 0.95。仅有一个实际阈值时只输出相应的单点指标，例如 `mAP95` 或 `mAP100`。
数值 `class_id`／`cls` 的评估不要求 names；字符串 label 需要名称映射。
`per_class_by_id` 包含全部 GT 类别的明细，原 `per_class` 保留已命名类别的名称键。
匹配按置信度降序进行；P／R／F1 取平均 F1 最佳的置信度阈值，mAP 对 GT 中出现的类别求平均。
计算口径对照 Ultralytics 8.4.183（提交 `1da023f708beb274db82c2c04b034c1859ab3fba`）。
指标以字典返回，显示和打印由调用方负责。返回字段 `images` 表示 GT 字典中的样本数，包含空 GT 样本。

### 3.1 用数据集标注作 GT

`Dataset(...).build().val(predictions)` 和 `open_dataset(...).val(predictions)` 使用当前 Profile 的可处理样本和缓存标注作 GT，返回同样的指标字典。`open_dataset()` 当前始终返回 `DatasetCollection`；两个数据集类型都提供 `val()`，Profile 不提供。

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

单来源的标识为不带后缀的 stem，例如 `sample_001`。集合的标识为 `profile.output_path(sample).with_suffix("").as_posix()`：唯一分区使用 `train/sample_001`，多个未分区来源使用 `dataset_1/sample_001`，重复分区使用 `dataset_1/train/sample_001`。只有一个未分区来源时仍直接使用 stem。

预测必须使用集合的类别 ID，并与对应样本 GT 使用相同坐标：YOLO 为归一化 `xyxy`，VOC／LabelMe 为像素 `xyxy`。是否执行图片检查不会改变坐标约定。`val()` 不读取图片或标签文件，不重新扫描或检查。无效样本不进入评估，缺失预测和未知标识遵循 Evaluator 的同样规则。Dataset／DatasetCollection 负责适配 GT，Evaluator 保持纯计算；结果的展示和打印由调用方负责。

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

### 5. 数据集批量操作

转换与导出使用 `dsetkit.tools`。集合重新划分使用 `new_collection = collection.split((0.8, 0.2, 0), seed=22)`：先汇集当前有效成员并永久排除候选池中的全部同 stem 冲突，再分配新分区，旧结果保持不变。`profile.members` 保存样本级分区，`profile.excluded` 保存累计错误，`profile.partition_stats` 提供分区统计。无原生划分的多个来源共用未分区作用域；原生不同分区可以同名。

底层 `partition_paths()` 只分配路径，不维护集合排除历史；以下示例显式保存路径划分：

```python
from pathlib import Path

from dsetkit import open_dataset, partition_paths
from dsetkit.tools import convert_dirs, export_dirs

# LabelMe → YOLO（跳过无标签样本）
convert_dirs(
    task_type="detection", image_dir="images/train",
    label_dir="labels/labelme",
    source_format="labelme",
    target_format="yolo",
    names=["cat", "dog"],
    out_dir="output",
)

# 导出全部图像路径
export_dirs(task_type="detection", image_dir="images/train", txt_path="all.txt")

# 按比例划分 train/val/test txt（示例比例 0.75 / 0.15 / 0.1）
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
├── width, height          # require_size() 可在导出前校验尺寸
├── names: list[str]
├── extra: dict[str, Any]
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
│   ├── annotations/       # schema、io、各格式适配器
│   ├── dataset.py         # 单来源扫描与检查
│   ├── collection.py      # 发现编排与集合重排
│   ├── collection_profile.py # 成员、累计排除与分区统计
│   ├── profile.py         # 完整分析结果、统计与处理取数
│   ├── _profile_io.py     # Profile JSON 编解码与结构检查
│   ├── partitions.py      # train/val/test 划分
│   ├── tools.py           # 批量转换/可视化/导出便捷入口
│   ├── evaluator.py       # 无文件依赖的预测与 GT 配对评估
│   ├── metrics.py         # NumPy IoU、AP、P/R/F1
│   ├── utils/             # 图像元信息、文件索引
│   └── visualize/         # Plotter
└── tests/                 # SDK 回归测试与指标对照数据
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





## 样本校验与处理范围

`Dataset`、`open_dataset()` 和 `*_dirs` 入口要求 `task_type="detection"`，目前只开放目标检测。
`open_dataset()` 与 `Dataset` 的两个独立检查开关为 `check_annotations=True`、`check_images=False`。
构建时先检查图片路径重名，冲突样本直接排除，不解析或检查其标签，也不检查图片；之后只解析剩余样本的所选标注。默认检查检测框，不读取图片。无法解析的标注和非检测结构仍记为错误。
启用图片检查时检查图片容器并完整解码，使用实际图片尺寸检查框的边界；标注 size 缺失或与图片不一致本身不构成错误。
显式 `image_info` 或已检查的图片尺寸优先于标注 size，并用于返回的像素标注，不修改缓存中的原始标注。没有图片尺寸时使用已有标注 size，缺失 size 本身不影响有效框；已产生的错误仍保留。
批量转换和绘图需要尺寸时，优先使用图片检查缓存，否则按需读取图片头，即使标注已有 size 也以图片为准。
单文件 `load/convert` 提供 `image_path` 时同样优先使用图片尺寸；已有实际宽高可显式传入 `width/height`，避免重复读取。
关闭相应检查表示按正常样本假定使用，不代表内容已经验证。

```python
from dsetkit import Dataset, DatasetProfile

scanner = Dataset(image_dir="dataset/images", label_dir="dataset/labels",
                  source_format="yolo", task_type="detection").build()
original = scanner.profile()  # 默认只检查标注，不读图片
profile = scanner.check_images()  # 调用方选择合适时机检查图片
state = profile.export()
restored = DatasetProfile.from_json(state)  # 不读取源文件或重复检查
```

Profile 是不可变结果，没有检查或 val 方法。检查入口仅在 Dataset 上，更新它的当前 Profile；旧结果及已建立的 Collection 不会跟随改变。DatasetCollection 接收来源 Profiles，建立后不保留 Dataset 实例。转换和绘图按实际需要直接读取图片头或像素，不依赖 Dataset 的生命周期，不提前读取尺寸。`profile.annotation(sample, image_info=...)` 可使用显式图片元信息进行像素坐标转换；`ground_truth()` 无需图片。

两种 Profile 均提供 `export()`／对应类型的 `from_json()`。集合使用 `DatasetCollectionProfile.from_json(text)` 恢复，再通过 `DatasetCollection(restored)` 继续重排。JSON 结构版本为 2，不兼容旧结构；来源事实只保存一份，集合额外保存成员与新增排除，完整恢复累计错误池。`as_dict()` 仅为统计摘要。

`Dataset.from_profile(source_profile)` 可从已有或 JSON 恢复的单来源 `DatasetProfile` 恢复扫描器；创建时不读取文件、扫描目录或重新解析标签，`profile()` 初始返回传入的同一结果。之后可调用 `check_annotations()` 检查缓存标注，或调用 `check_images()` 读取并检查已知图片；检查只替换该扫描器的当前结果，旧 Profile、其他扫描器及已有错误保持不变。显式调用 `build()` 才会按保存的来源和检查选项重新扫描。该入口不接受集合 Profile。

旧 `export_scan/from_scan`、`DatasetView`、`DatasetProfile.combine()` 及 Profile／Collection 检查接口已删除，不提供旧兼容层。

`names` 可省略。VOC／LabelMe 从可处理样本中按名称首次出现的顺序生成有序类别表。
YOLO 保留实际 ID，未命名类别由 `profile.unnamed_class_ids` 和 `warnings` 提示，
不补出未出现的 ID，也不会因为名称缺失排除样本。`class_ids`、`class_names`、`class_instances`
按位置对应；`class_instances` 不能用原始 ID 直接索引。
格式输入统一为小写；`convert()`、`convert_dataset()` 和 `convert_dirs()` 禁止源格式与目标格式相同。
涉及 YOLO 的转换需要完整的最终名称映射；单文件转换显式传入 `names`，数据集转换默认使用 `profile.names`。
显式传入 `names=[]` 不会回退；VOC／LabelMe 转 YOLO 按名称在最终列表中的位置生成输出 ID。
混合 YOLO 和命名格式的集合需要共享的显式类别映射，以免把不明 ID 误认为另一来源的名称。

`source_format` 仅接受真实标注格式或 None（自动发现）；不存在 `unannotated` 格式。
未发现标注目录的部分，其格式为 None。无标注文件的图片和合法空标注是背景，
无法解析或检查失败的样本作为错误排除。迭代、索引、统计、转换和导出使用本次扫描的可处理成员。
`raw_samples` 仅保留图片／标注路径；`invalid_samples` 和 `profile.errors` 记录排除结果。
`total_images`／`total_label_files` 记录原始数量，`images`／`label_files` 记录处理成员及其标注数量。
普通处理复用已解析标注和 profile；需要像素或尺寸的操作仍可读取目标图片。

物理来源在扫描后按不可变数据使用，显式 `build()` 才重新扫描。
结构检查拒绝整个数据集树中的符号链接、特殊文件与路径换行；
同目录或同分区的重复 stem 全部记为 `invalid_image` 并永久排除，不受图片检查开关影响。不同原生分区之间允许同名；重新 split 合成一个候选池时，先永久排除冲突再分配。排除池只增不减，改变种子、比例或 JSON 恢复均不恢复错误样本。已因文件错误排除的样本不再参与重名检查。
样本发现只收录规定目录的直接子文件。`inventory_dataset()` 保留任务无关的原始文件清单与摘要能力。

LabelMe 的 `imagePath` 必须是字符串，缺失或类型错误记为标注错误。Profile JSON 中的来源与样本路径必须为绝对路径；非有限数字及指数溢出数值会被拒绝。
