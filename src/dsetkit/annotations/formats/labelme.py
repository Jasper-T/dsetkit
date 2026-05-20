import json
from pathlib import Path

from ..schema import Annotation, AnnotationItem, BBox
from ..registry import register_format
from .common import resolve_image_path, resolve_image_wh


def _parse_int(v):
    try:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return int(float(v))
    except (ValueError, TypeError):
        return None


def load_labelme(
    path: str,
    image_path: str | None = None,
    names: list[str] = [],
    **kwargs,
) -> Annotation:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_path = resolve_image_path(path, image_path, data.get("imagePath"))
    width = _parse_int(data.get("imageWidth"))
    height = _parse_int(data.get("imageHeight"))
    
    if image_path:
        width, height = resolve_image_wh(image_path, width, height)
    
    ann = Annotation(
        image_path=image_path,
        width=width,
        height=height,
        names=names,
    )

    for shape in data.get("shapes", []):
        points = shape.get("points") or []
        if not isinstance(points, list) or len(points) < 2:
            continue

        xs, ys = [], []
        for p in points:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            try:
                xs.append(float(p[0]))
                ys.append(float(p[1]))
            except (TypeError, ValueError):
                continue
        
        category = shape.get("label", "")
        category_id = names.index(category) if category in names else None
        item = AnnotationItem(
            category=category,
            category_id=category_id,
                bbox=BBox(
                    x1=min(xs),
                    y1=min(ys),
                    x2=max(xs),
                    y2=max(ys),
                ),
                extra={
                    "shape_type": shape.get("shape_type", "polygon"),
                },
            )
        ann.items.append(item)

    return ann


def dump_labelme(
    ann: Annotation,
    out_path: str,
):
    data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [],
        "imagePath": Path(ann.image_path).name if ann.image_path else "",
        "imageData": None,
        "imageHeight": ann.height,
        "imageWidth": ann.width,
    }

    for item in ann.items:
        if item.bbox is None:
            continue

        shape_type = item.extra.get("shape_type", "rectangle")

        # detection-only: always export rectangle points
        points = [
            [item.bbox.x1, item.bbox.y1],
            [item.bbox.x2, item.bbox.y2],
        ]

        data["shapes"].append(
            {
                "label": item.category,
                "points": points,
                "group_id": item.extra.get("group_id"),
                "shape_type": "rectangle",
                "flags": item.extra.get("flags", {}),
            }
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )

register_format(
    "labelme",
    loader=load_labelme,
    dumper=dump_labelme,
)