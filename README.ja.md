# dsetkit

Language: [中文](README.md) | [English](README.en.md) | **日本語**

**Deep learning dataset infrastructure toolkit** — 物体検出とアノテーション処理のための Python ツールキット。統一アノテーション schema、複数形式変換、データセット索引、分割・拡張、評価指標、可視化を提供します。

[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue.svg)](https://www.python.org/)
[![License: MoT](https://img.shields.io/badge/License-MoT-green.svg)](pyproject.toml)

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
| ------ | ------ |
| **Annotation o/O** | 統一 `Annotation` schema で **LabelMe / VOC / YOLO** を読み書き |
| **形式変換** | 単体 `convert`；データセット規模の `convert_dataset` / `convert_dirs` |
| **データセット** | ファイル名 stem で画像とラベルを自動対応付けし、`source_format` に応じた拡張子でフィルタリング |
| **分割とエクスポート** | 画像パス一覧を txt に出力し、比率で train/val/test にランダム分割 |
| **データ拡張** | 水平/垂直反転、90°/180°/270° 時計回り回転（bbox 同期更新） |
| **評価** | 既存予測に対する Ultralytics 風の検出指標: P / R / mAP@iou / F1 |
| **画像ユーティリティ** | フルデコードなしで JPEG/PNG/WebP/BMP の情報を取得、自然順ソートで走査 |
| **可視化** | `Plotter` で画像上に検出ボックスを描画（OpenCV は基本インストールに含まれる） |

---

## インストール

**基本インストール**（アノテーション、データセット、拡張、可視化、評価指標）:

```bash
pip install -e .
```

**Python >= 3.11** が必要です。主要依存: `numpy>=2.0`, `natsort`, `opencv-python`, `tqdm`。

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

### 2. データセット索引の構築

`Dataset` は画像ディレクトリを走査し、stem で対応ラベルを検索した後、`source_format` の拡張子（`labelme -> .json`, `voc -> .xml`, `yolo -> .txt`）で絞り込みます。

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

### 3. 物体検出評価

存存の予測結果を `Evaluator` に渡します。ground truth は `Dataset` から読み込まれます。

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
# metrics には precision, recall, f1, mAP50, per_class が含まれる
```

`iou` は dsetkit の評価 IoU しきい値です。`iou=0.5` は `mAP50` を計算し、`np.linspace(0.5, 0.95, 10)` のような列を渡すと `mAP50-95` を計算します。
端末出力は YOLO 形式の検証テーブル（`Class / omages / onstances / P / R / mAPxx / F1`）に近い形式です。

### 4. アノテーション可視化

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

単一の検出枠を直接描画し、表示テキストを明示的に渡すこともできます。

```python
plotter = Plotter(img)
rendered = plotter.detection(
    bbox=[50, 40, 180, 160],
    class_id=1,
    text="person: 0.98",
)
cv2.imwrite("vis.jpg", rendered)
```

`Plotter.detection(...)` は `class_id` を省略可能になり、任意の `text` を描画でき、現在の画像配列を直接返します。

実行例は [`examples/`](examples/) と [`tests/demo.py`](tests/demo.py) を参照してください。

### 5. データセットのバッチ操作

`dsetkit.tools` の便捷ヘルパーで、大規模な変換・エクスポート・分割を実行できます。

```python
from dsetkit import Dataset
from dsetkit.tools import convert_dirs, export_dirs, split_dirs

# LabelMe -> YOLO（ラベルなしサンプルはスキップ）
convert_dirs(
    image_dir="images/train",
    label_dir="labels/labelme",
    source_format="labelme",
    target_format="yolo",
    names=["cat", "dog"],
    out_dir="labels/yolo",
)

# 全画像パスを txt にエクスポート
export_dirs(image_dir="images/train", txt_path="all.txt")

# train/val/test txt に分割（デフォルト 0.75 / 0.15 / 0.1）
split_dirs(image_dir="images/train", out_dir="splits", seed=42)
```

### 6. データ拡張（反転 / 回転）

画像とアノテーションを同期変換。出力は `out_dir/images/` と形式別ラベルディレクトリへ書き込みます。

```python
from dsetkit.tools import flip_dirs, rotate_dirs

# 水平反転（direction=1）。垂直は direction=0
flip_dirs(
    image_dir="images/train",
    label_dir="labels/voc",
    source_format="voc",
    out_dir="augment/flipped",
    direction=1,
)

# 時計回り 90° 回転（180、270 も可）
rotate_dirs(
    image_dir="images/train",
    label_dir="labels/yolo",
    source_format="yolo",
    names=["person"],
    out_dir="augment/rot90",
    angle=90,
)
```

単体サンプルや schema レベルの APo は `dsetkit.augment`（`flip_annotation`、`rotate_label` など）を参照してください。

---

## 対応アノテーション形式

| 形式 | 識別子 | 拡張子 | 説明 |
| ------ | -------- | ---------- | ------ |
| LabelMe | `labelme` | `.json` | 一般的な対話型アノテーション出力 |
| Pascal VOC | `voc` | `.xml` | XML 形式の検出アノテーション |
| YOLO | `yolo` | `.txt` | 正規化 `class cx cy w h` |

形式拡張はレジストリ（`register_format`）で行います。`src/dsetkit/annotations/registry.py` と `formats/*.py` を参照してください。

---

## 統一データモデル

```text
Annotation
├── width, height          # 出力/拡張前に require_size() でサイズ検証
├── names: list[str]
├── extra: dict[str, Any]
└── items: list[Annotationotem]
    ├── category, category_id
    ├── bbox: BBox(x1, y1, x2, y2)   # 絶対ピクセル座標（左上-右下）
    └── extra / segmentation / keypoints（拡張用）
```

---

## プロジェクト構成

```text
dsetkit/
├── src/dsetkit/
│   ├── annotations/       # schema, io, 各形式アダプタ
│   ├── augment/           # 反転 / 回転（画像 + ラベル）
│   ├── dataset.py         # Dataset / DatasetSample / DatasetStats
│   ├── split.py           # train/val/test 分割
│   ├── tools.py           # バッチ変換/可視化/拡張/分割の便捷ヘルパー
│   ├── evaluator.py       # 検出評価の基底クラス
│   ├── metrics.py         # IoU, AP, P/R/F1
│   ├── utils/             # 画像メタ情報、ファイル索引
│   └── visualize/         # Plotter
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

[MoT](pyproject.toml) · Author: Jasper Tao






