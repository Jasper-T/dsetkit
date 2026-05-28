import json
from pathlib import Path

from ..schema import Annotation, AnnotationItem, BBox
from ..registry import register_format
from .common import resolve_image_wh


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
    width: int | None = None,
    height: int | None = None,
    names: list[str] | None = None,
    **kwargs,
) -> Annotation:
    names = names or []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_name = data.get("imagePath")
    extra = {"imagePath": image_name}
    json_width = _parse_int(data.get("imageWidth"))
    json_height = _parse_int(data.get("imageHeight"))
    width = width if width is not None else json_width
    height = height if height is not None else json_height
    
    if image_path:
        extra["image_path"] = image_path
        width, height = resolve_image_wh(image_path, width, height)
    
    ann = Annotation(
        width=width,
        height=height,
        names=names,
        extra=extra,
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
    image_name = ann.extra.get("imagePath", "")
    image_path = ann.extra.get("image_path", image_name)
    data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [],
        "imagePath": Path(image_path).name if image_path else "",
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