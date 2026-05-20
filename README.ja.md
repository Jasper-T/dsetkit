# dsetkit

Language: [中文](README.md) | [English](README.en.md) | **日本語**

**Deep learning dataset infrastructure toolkit** — 物体検出とアノテーション処理のための Python ツールキット。統一アノテーション schema、複数形式変換、データセット索引、評価指標、可視化を提供します。

[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](pyproject.toml)

## 目次

- [特徴](#特徴)
- [インストール](#インストール)
- [クイックスタート](#クイックスタート)
- [対応アノテーション形式](#対応アノテーション形式)
- [統一データモデル](#統一データモデル)
- [プロジェクト構成](#プロジェクト構成)
- [開発とビルド](#開発とビルド)
- [ライセンス](#ライセンス)

---

## 特徴

| モジュール | 機能 |
|------|------|
| **Annotation I/O** | 統一 `Annotation` schema で **COCO / LabelMe / VOC / YOLO** を読み書き |
| **形式変換** | 単体 `convert` と大規模 `batch_convert`（マルチプロセス対応） |
| **データセット** | ファイル名 stem で画像とラベルを自動対応付け。複数形式が共存する場合は優先度で選択 |
| **評価** | IoU ベースの TP/FP マッチング、mAP@0.5、Precision / Recall / F1（クラス別・全体） |
| **画像ユーティリティ** | フルデコードなしで JPEG/PNG/WebP/BMP の情報を取得、自然順ソートで走査 |
| **可視化** | `Plotter` で画像上に検出ボックスを描画（OpenCV は任意依存） |

---

## インストール

**基本インストール**（アノテーション、データセット、評価指標）:

```bash
pip install -e .
```

**可視化込み**（OpenCV が必要）:

```bash
pip install -e ".[visualize]"
```

**Python >= 3.11** が必要です。主要依存: `numpy`, `natsort`。

---

## クイックスタート

### 1. アノテーションの読み込みと変換

対応形式は内部で統一 `Annotation` schema に変換されます（`BBox` は絶対座標 `xyxy`）。

```python
from dsetkit.annotations.io import load, dump, convert

# LabelMe JSON -> YOLO txt（YOLO 出力時はクラス名が必要）
convert(
    label_path="path/to/label.json",
    image_path="path/to/image.jpg",
    source_format="labelme",
    target_format="yolo",
    names=["smoke", "fire"],   # YOLO 関連の load/dump では names が必要になることが多い
    out_dir="output/labels",
)
```

**2ステップ方式**（中間結果を確認しやすい）:

```python
ann = load(
    label_path="path/to/label.json",
    image_path="path/to/image.jpg",
    fmt="labelme",
)

dump(ann, "output/label.txt", fmt="yolo")
```

> **注意**: **YOLO** の読み書きでは、可能なら `load` に `names`（class id -> class name）を渡してください。YOLO への `dump` には `Annotation` の `width`/`height` が必要で、各オブジェクトは `category_id` もしくは `ann.names` から解決可能である必要があります。

### 2. バッチ変換

大規模データ（10万件以上）向け。出力は `out_dir / {stem}{suffix}` に保存されます。

```python
from dsetkit.annotations.convert import batch_convert
from dsetkit import Dataset

dataset = Dataset(
    image_dir="images/train",
    label_dir="labels/labelme",
    names=["cat", "dog"],
    input_format="labelme",
)

# ラベル付きサンプルのみ変換
pairs = [s for s in dataset if s.label_path is not None]

paths = batch_convert(
    pairs,
    source_fmt="labelme",
    target_format="yolo",
    out_dir="labels/yolo",
    names=["cat", "dog"],
    max_workers=8,          # None で単一プロセス。Windows の並列実行は __main__ ガードが必要
    errors="raise",         # "skip" の場合は (成功パス, 失敗一覧) を返す
)
```

### 3. データセット索引の構築

`Dataset` は画像ディレクトリを走査し、stem で対応するラベルを検索します。同一 stem に複数拡張子（例: `.txt` と `.json`）がある場合、優先度は **yolo > coco/labelme > voc** です。

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

### 4. 物体検出評価

`Evaluator` を継承して `_load_predictions` を実装すると、固定アノテーション形式で指標を計算できます。

```python
from pathlib import Path
from dsetkit import Dataset
from dsetkit.evaluator import Evaluator

class MyEvaluator(Evaluator):
    def _load_predictions(self, image_path: Path):
        # list[dict] を返す: bbox [x1,y1,x2,y2], label str, conf float
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
# metrics には mAP, precision, recall, f1, per_class が含まれる
```

端末出力は YOLO 形式の検証テーブル（`Class / Instances / P / R / mAP50 / F1`）に近い形式です。

### 5. アノテーション可視化

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

実行例は [`examples/`](examples/) と [`tests/demo.py`](tests/demo.py) を参照してください。

---

## 対応アノテーション形式

| 形式 | 識別子 | 拡張子 | 説明 |
|------|--------|----------|------|
| COCO | `coco` | `.json` | 検出/インスタンス向け JSON |
| LabelMe | `labelme` | `.json` | 一般的な対話型アノテーション出力 |
| Pascal VOC | `voc` | `.xml` | XML 形式の検出アノテーション |
| YOLO | `yolo` | `.txt` | 正規化 `class cx cy w h` |

形式拡張はレジストリ（`register_format`）で行います。`src/dsetkit/annotations/registry.py` と `formats/*.py` を参照してください。

---

## 統一データモデル

```text
Annotation
├── image_path, width, height
├── names: list[str]
└── items: list[AnnotationItem]
    ├── category, category_id
    ├── bbox: BBox(x1, y1, x2, y2)   # 絶対ピクセル座標（左上-右下）
    └── extra / segmentation / keypoints（拡張用）
```

---

## プロジェクト構成

```text
dsetkit/
├── src/dsetkit/
│   ├── annotations/       # schema, io, convert, 各形式アダプタ
│   ├── dataset.py         # Dataset / DatasetSample
│   ├── evaluator.py       # 検出評価の基底クラス
│   ├── metrics.py         # IoU, AP, P/R/F1
│   ├── utils/             # 画像メタ情報、ファイル索引
│   └── visualize/         # Plotter（opencv は任意）
├── examples/              # 変換サンプル
└── tests/                 # demo とテスト
```

---

## 開発とビルド

```bash
# editable install
pip install -e .

# wheel build
python -m build --wheel
```

`__pycache__` の削除: `scripts/clean_pycache.sh`

変更履歴: [CHANGELOG.md](CHANGELOG.md)

---

## ライセンス

[MIT](pyproject.toml) · Author: Jasper Tao
