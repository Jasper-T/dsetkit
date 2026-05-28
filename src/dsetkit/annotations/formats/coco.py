import json

from ..schema import Annotation, AnnotationItem, BBox
from ..registry import register_format


def load_coco(
    path: str,
    image_path: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = {
        i["id"]: i
        for i in data.get("images", [])
    }

    categories = {
        c["id"]: c["name"]
        for c in data.get("categories", [])
    }

    anns_by_img = {}

    for a in data.get("annotations", []):
        anns_by_img.setdefault(
            a["image_id"],
            [],
        ).append(a)

    results = []

    for img_id, img in images.items():
        items = []

        for a in anns_by_img.get(img_id, []):
            bbox = None

            if "bbox" in a:
                x, y, w, h = a["bbox"]

                bbox = BBox(
                    x1=x,
                    y1=y,
                    x2=x + w,
                    y2=y + h,
                )

            cat_id = a.get("category_id")
    
            items.append(
                AnnotationItem(
                    category=categories.get(
                        cat_id,
                        str(cat_id),
                    ),
                    category_id=cat_id,
                    bbox=bbox,
                    segmentation=a.get("segmentation"),
                    extra={
                        "area": a.get("area"),
                        "iscrowd": a.get("iscrowd", 0),
                    },
                )
            )

        ann = Annotation(
            width=width if width is not None else img.get("width"),
            height=height if height is not None else img.get("height"),
            items=items,
            names=list(categories.values()),
            extra={"image_path": image_path or img["file_name"]},
        )

        results.append(ann)

    # current API keeps single-image behavior
    if len(results) != 1:
        raise ValueError(
            "Only single-image COCO files supported"
        )

    return results[0]


def dump_coco(
    anns: list[Annotation] | Annotation,
    out_path: str,
):
    if isinstance(anns, Annotation):
        anns = [anns]

    images = []
    annotations = []
    categories = []

    category_to_id = {}

    # global ontology
    for ann in anns:
        for name in ann.names:
            if name not in category_to_id:
                category_to_id[name] = (
                    len(category_to_id) + 1
                )

        for item in ann.items:
            if item.category not in category_to_id:
                category_to_id[item.category] = (
                    len(category_to_id) + 1
                )

    categories = [
        {
            "id": cid,
            "name": name,
        }
        for name, cid in category_to_id.items()
    ]

    ann_id = 1

    for img_id, ann in enumerate(anns, start=1):
        image_path = ann.extra.get("image_path", "")
        images.append(
            {
                "id": img_id,
                "file_name": image_path,
                "width": ann.width,
                "height": ann.height,
            }
        )

        for item in ann.items:
            bbox = None
            area = 0

            if item.bbox is not None:
                x1 = item.bbox.x1
                y1 = item.bbox.y1
                x2 = item.bbox.x2
                y2 = item.bbox.y2

                bbox = [
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1,
                ]

                area = (
                    (x2 - x1)
                    * (y2 - y1)
                )

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_to_id[
                        item.category
                    ],
                    "bbox": bbox,
                    "segmentation": item.segmentation,
                    "area": item.extra.get(
                        "area",
                        area,
                    ),
                    "iscrowd": item.extra.get(
                        "iscrowd",
                        0,
                    ),
                }
            )

            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            coco,
            f,
            indent=2,
            ensure_ascii=False,
        )


register_format(
    "coco",
    loader=load_coco,
    dumper=dump_coco,
)