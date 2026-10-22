# dsetkit

Language: [中文](README.md) | [English](README.en.md) | **日本語**

**Deep learning dataset infrastructure toolkit** — 物体検出とアノテーション処理のための Python ツールキット。統一アノテーション schema、複数形式変換、データセット索引、分割、評価指標、可視化を提供します。

[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](pyproject.toml)

## 目次

- [特徴](#特徴)
- [インストール](#インストール)
- [コンポーネントの責務](#コンポーネントの責務)
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
| **データセットプロファイル** | 正例・背景・無効画像、孤立ラベル、クラス別インスタンス数を集計 |
| **データセット検出** | 標準的な画像および LabelMe / VOC / YOLO ラベルディレクトリを副作用なしで検出 |
| **Schema チェック** | クラス ID・名称・順序を暗黙の並べ替えなしで厳密比較 |
| **分割とエクスポート** | 画像パス一覧を txt に出力し、比率で train/val/test にランダム分割 |
| **評価** | 既存予測に対する Ultralytics 風の検出指標: P / R / mAP@iou / F1 |
| **画像ユーティリティ** | フルデコードなしで JPEG/PNG/WebP/BMP の情報を取得、自然順ソートで走査 |
| **可視化** | `Plotter` で画像上に検出ボックスを描画（OpenCV は基本インストールに含まれる） |

---

## インストール

**基本インストール**（アノテーション、データセット、可視化、評価指標）:

```bash
pip install -e .
```

**Python >= 3.11** が必要です。主要依存: `numpy>=2.0`, `natsort`, `opencv-python`, `tqdm`。

---

## コンポーネントの責務

Dataset の中心は、明示ソースの走査・注釈検査・任意の画像検査・DatasetProfile の生成です。
レイアウト検出、分割、学習バッチ生成、モデル学習は担当しません。既存の val() は
独立した Evaluator にキャッシュ済み GT を渡す便宜的な入口です。

注釈 I/O、画像ユーティリティ、クラス定義比較、パス分配、Plotter、Evaluator は
必要な入力だけで単独利用できます。Collection とバッチ処理は Profile を受け取り、
走査済み Dataset の存続を要求しません。Profile は結果の契約です。疎結合は依存が
一切ないことではなく、責務と入力・出力が明確であることを意味します。
[全クラス・機能の責務一覧](DATA_COLLECTIONS.md#component-responsibilities-and-independent-use)を参照してください。

PyTorch Dataset は学習時のサンプル取得、DataLoader はロードとバッチ編成を担当します。
dsetkit の結果を、利用側の PyTorch Dataset アダプタ経由で接続します。組み込みの
PyTorch アダプタはありません。[連携の境界](DATA_COLLECTIONS.md#relationship-to-pytorch-dataset-and-dataloader)と
[PyTorch 公式チュートリアル](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)を参照してください。

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
    out_dir="output",
)
```

**2ステップ方式**（中間結果を確認しやすい）:

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

> **注意**: **YOLO** の読み書きでは、可能なら `load` に `names`（class id -> class name）を渡してください。YOLO への `dump` には `Annotation` の `width`/`height` が必要で、各オブジェクトは `category_id` もしくは `ann.names` から解決可能である必要があります。

### 2. データセット索引の構築

`open_dataset()` は走査を完了し、`.profile()` で完全な分析結果を返します。画像ディレクトリを走査し、stem で対応ラベルを検索した後、`source_format` の拡張子（`labelme -> .json`, `voc -> .xml`, `yolo -> .txt`）で絞り込みます。

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

`Dataset.profile()` は単一ソースの `DatasetProfile`、`open_dataset().profile()` はソース参照、サンプル所属、累積除外、分割統計を持つ `DatasetCollectionProfile` を返します。既定では画像を検査しません。明示的な画像検査ではコンテナと全画素を検証し、診断内容を `profile.errors` に返します。


`open_dataset(task_type="detection", root="dataset", names=["object"], source_format="yolo")` で自動検出も利用できます。root と明示ディレクトリは同時指定できません。明示ラベルには形式が必要で、label_dir を省略した場合は他の場所を検索しません。両モードは構築済みビューを返し、直下のファイルだけを走査します。`Dataset` は単一ソースを走査し、`DatasetCollection` は走査結果を組み立てます。変換・可視化・エクスポートは `DatasetProfile` または `DatasetCollectionProfile` を受け取り、評価はデータセットの `val()` または予測と GT を明示する Evaluator を使います。

#### 2.1 データセットレイアウトの検出

```python
from dsetkit import discover_parts, select_annotation_source, open_dataset

native, parts = discover_parts("path/to/dataset")
for part in parts:
    source = select_annotation_source(part.discovery, requested="yolo")
    print(part.split, part.discovery.image_dir, source.label_dir if source else None)

profile = open_dataset("path/to/dataset", task_type="detection", names=["object"], source_format="yolo").profile()
print(profile)
```

`discover_parts()` は固定レイアウトを認識し、直下のファイルだけを走査します。単一の注釈ソースも `part.discovery.candidates` に格納されます。`select_annotation_source()` は形式と曖昧さを検証し、`open_dataset().profile()` は分区を保持する完全な分析結果を返します。API と破壊的変更は [DATA_COLLECTIONS.md](DATA_COLLECTIONS.md) を参照してください。

#### 2.2 クラス schema の厳密比較

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

strict モードではクラス数、名称、順序が完全に一致する必要があります。自動ソート、大文字小文字の無視、暗黙の mapping は行いません。

### 3. 物体検出評価

`Evaluator` は既存の検出結果と明示的な GT を比較する純粋な数値評価クラスです。Profile に依存せず、ファイルの読み込み、推論、端末への出力は行いません。ボックスのマッチングと指標計算は Ultralytics を参考にしています。

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

`iou` は dsetkit の評価 IoU の開始しきい値です。`iou=0.5` は `0.50...0.95` を固定 `0.05` ステップで評価し、`mAP50` と `mAP50-95` の両方を出力します。`iou=0.25` は `mAP25` と `mAP25-95` を出力します。
両マッピングのキーは空白のみでない文字列 `sample_id` です。パスや拡張子の要件はなく、パスとして解釈しません。GT が評価対象全体を定義し、予測の欠落は空の予測、GT に存在しない予測 ID はエラーとします。背景サンプルは空の GT として明示し、誤検出も評価に含めます。
値は上記のレコードリスト、または `PredictionResult`／`AnnotationTarget` です。予測と GT は同じクラス ID と同じ `xyxy` 座標系を使います。ピクセル座標と正規化座標の両方を扱え、画像寸法は不要です。レコードは `cls`、または `names` マッピングがある場合の `label` も受け付けます。
`metrics` は AP 計算前に IoU しきい値が `(0, 1]` の有限数であることを検査し、昇順に整列して重複を除去します。配列は正規化後のしきい値を使い、0.95 未満の単一しきい値は 0.05 刻みで 0.95 以下まで展開します。実際のしきい値が一つの場合は `mAP95` や `mAP100` の単点指標のみ出力します。
数値の `class_id`／`cls` による評価には names は不要です。文字列ラベルには名称マッピングが必要です。
`per_class_by_id` は GT の全クラスを含み、従来の `per_class` は名称キーを保持します。
マッチングは信頼度の降順で行い、P／R／F1 は平均 F1 が最大となるしきい値を使用します。mAP は GT に出現するクラスの平均です。
計算は Ultralytics 8.4.183（コミット `1da023f708beb274db82c2c04b034c1859ab3fba`）と照合しています。
結果は辞書で返し、表示や印刷は呼び出し側が担当します。`images` は空 GT を含む GT サンプルの総数です。

### 3.1 データセットの注釈を GT として評価する

`Dataset(...).build().val(predictions)` と `open_dataset(...).val(predictions)` は、現在の Profile の処理可能サンプルとキャッシュ済み注釈を GT として使い、同じ指標辞書を返します。`open_dataset()` は現在常に `DatasetCollection` を返します。両データセット型は `val()` を提供し、Profile は提供しません。

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

単一ソースの ID は拡張子を除いた stem（例：`sample_001`）です。集合では `profile.output_path(sample).with_suffix("").as_posix()` を使います。一意な分割は `train/sample_001`、複数の未分割ソースは `dataset_1/sample_001`、同じ分割の複数ソースは `dataset_1/train/sample_001` です。未分割の単一ソースでは stem のみです。

予測にはコレクションのクラス ID と各 GT と同じ座標を使います。YOLO は正規化 xyxy、VOC／LabelMe はピクセル xyxy です。画像検査や寸法キャッシュで座標系は変わりません。`val()` は画像もラベルファイルも読み込まず、再走査やファイル検査もしません。無効サンプルを除外し、予測の欠落や未知 ID は Evaluator と同じ規則に従います。データセット型が GT を適合し、Evaluator は純粋な数値計算を担当します。表示や印刷は呼び出し側が担当します。

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

### 5. データセットのバッチ操作

変換・エクスポートは `dsetkit.tools`、分割は `partition_paths()` と明示的な保存を使用します。

```python
from pathlib import Path

from dsetkit import open_dataset, partition_paths
from dsetkit.tools import convert_dirs, export_dirs

# LabelMe -> YOLO（ラベルなしサンプルはスキップ）
convert_dirs(
    task_type="detection", image_dir="images/train",
    label_dir="labels/labelme",
    source_format="labelme",
    target_format="yolo",
    names=["cat", "dog"],
    out_dir="output",
)

# 全画像パスを txt にエクスポート
export_dirs(task_type="detection", image_dir="images/train", txt_path="all.txt")

# train/val/test txt に分割（例: 0.75 / 0.15 / 0.1）
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
├── width, height          # 出力前に require_size() でサイズ検証
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
│   ├── dataset.py         # 単一ソースの走査と検査
│   ├── collection.py      # 検出の編成と再分割
│   ├── collection_profile.py # 所属、永久除外と分割統計
│   ├── profile.py         # 完全な分析結果と処理用データ
│   ├── _profile_io.py     # JSON と構造検査
│   ├── partitions.py      # train/val/test 分割
│   ├── tools.py           # バッチ変換/可視化/エクスポートの便捷ヘルパー
│   ├── evaluator.py       # ファイル非依存の予測と GT の評価
│   ├── metrics.py         # IoU, AP, P/R/F1
│   ├── utils/             # 画像メタ情報、ファイル索引
│   └── visualize/         # Plotter
└── tests/                 # SDK 回帰テストと指標比較データ
```

---

## 開発とビルド

```bash
# editable install
pip install -e .

# wheel をビルド
uv build --wheel
```


変更履歴: [CHANGELOG.md](CHANGELOG.md)

---

## ライセンス

[MIT](pyproject.toml) · Author: Jasper Tao




## 走査オプションと処理対象

構築には `task_type="detection"` が必要です。`Dataset` と `open_dataset()` の既定値は
`check_annotations=True`、`check_images=False` です。最初に画像パスの stem 重複を除外し、
重複サンプルの注釈解析・検査と画像検査を省略します。残りの注釈は一度だけ解析します。
画像検査を有効にするとコンテナ検査と完全デコードを行い、実際の画像寸法で検出枠を検証します。
画像寸法がある場合、注釈 size の欠落や不一致だけではエラーになりません。
明示的な image_info または検査済み画像寸法を注釈 size より優先し、返すピクセル注釈に適用します。
元の注釈キャッシュは変更しません。画像寸法がない場合は既存の注釈 size を使い、size の欠落だけで枠を無効にはしません。
既存のエラーは保持します。一括変換・可視化では画像検査キャッシュを使うか、必要時に画像ヘッダーを読み、注釈 size より優先します。
単一ファイルの load/convert でも image_path があれば画像寸法を優先します。実寸が既知なら width/height を明示し、再読み込みを省略できます。
検査を無効にした内容は正常と仮定します。解析できない注釈は背景に変換しません。

`names` は省略可能です。VOC／LabelMe は処理可能なサンプルから出現順に名称を収集し、
YOLO は実際の ID を保持します。未指定名称は `unnamed_class_ids` と `warnings` に表示され、
サンプルの除外理由にはなりません。`class_ids`、`class_names`、`class_instances` は位置で対応します。
形式名は小文字に統一します。`convert()`、`convert_dataset()`、`convert_dirs()` は同形式への変換を拒否します。
YOLO が関わる変換には完全な最終名称マッピングが必要です。単一ファイルでは `names` を明示し、データセットでは省略時に `profile.names` を使います。
明示的な `names=[]` はフォールバックしません。VOC／LabelMe から YOLO への変換では、最終名称リストの位置で出力 ID を決定します。
`source_format` は実際の形式または None のみで、`unannotated` は受け付けません。
注釈ディレクトリのない部分の形式は None です。

Dataset.check_images() は呼び出し側が選んだ時点で画像を検査し、Dataset.check_annotations() はキャッシュ済み注釈を検査します。現在の Profile を置き換え、以前の結果は変更しません。Profile と Collection に検査メソッドはありません。
両 Profile 型が export()/from_json() を提供します。集合は DatasetCollectionProfile.from_json(text) で復元し、DatasetCollection(restored) で再分割を続けられます。JSON 構造バージョンは 2 で旧形式の互換層はありません。ソースのキャッシュと集合の追加除外から完全な累積除外をオフラインで復元します。
`profile.as_dict()` は統計のみを返します。変換・可視化・エクスポートは Profile を直接受け取ります。評価では呼び出し側が GT を取り出して独立した Evaluator に渡すか、データセットの `val()` を使います。
元データは走査後不変と仮定し、明示的な build で再走査します。
シンボリックリンク、特殊ファイル、改行を含むパスは拒否します。
Profile 構築時、同じディレクトリまたは同じ分割内の重複 stem を持つ画像はすべて `invalid_image` として除外します。
この検査は `check_images` に関係なく行い、異なる分割間の同名画像は許可します。
サンプル検出は指定位置の直下のみです。`raw_samples` はパスのみ、
`invalid_samples` と `profile.errors` は除外結果を保持します。

旧 `export_scan/from_scan`、`DatasetView`、Dataset の変換・可視化・エクスポート転送メソッドは削除しました。評価には個別に `val()` を提供します。旧 scan データの移行は利用側で明示的に行います。

LabelMe の `imagePath` は文字列が必須です。欠落や型の誤りは注釈エラーとして記録します。Profile JSON のソースとサンプルのパスは絶対パスを要求し、指数オーバーフローを含む非有限数を拒否します。

## コレクションの結果と必要時の画像読み込み

DatasetCollection は Dataset インスタンスではなくソース Profiles を受け取ります。元の分割を保持し、未分割ソースは一つの未割当領域にまとめます。collection.split((0.8, 0.2, 0), seed=22) は現在の有効メンバーを一つに集め、重複 stem の全サンプルを永久除外してから分配し、新しい Collection を返します。異なる元分割の同名画像は集約前には許可します。種、比率、JSON 復元で除外を解除しません。ファイルエラーで除外された画像は重複検査に参加しません。

profile.members はサンプルの train/val/test または None への所属、profile.excluded は累積エラー、profile.partition_stats は有効メンバーの分割統計を保持します。空の元分割も統計を残します。partition_paths() は低レベルのパス分配で、集合の除外履歴を管理しません。

Dataset は走査後に解放できます。画像寸法は任意のキャッシュで、既定の走査では先読みしません。変換や可視化は必要な寸法を画像ヘッダーから直接読み、可視化は描画用の画素を読みます。Dataset の存続は不要で、Profile は変更しません。profile.annotation(sample, image_info=...) に寸法を渡してピクセル座標へ変換でき、ground_truth() は画像を必要としません。
