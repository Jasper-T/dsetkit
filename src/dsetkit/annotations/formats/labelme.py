import json
import math
from pathlib import Path

from ...utils.image import resolve_image_wh
from ..registry import register_format
from ..schema import Annotation, AnnotationItem, AnnotationParseError, BBox


def _parse_int(v):
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    number = _parse_float(v)
    if not number.is_integer():
        raise AnnotationParseError(f"LabelMe dimensions must be integers: {v!r}")
    return int(number)


def _parse_float(value):
    if not isinstance(value, (str, int, float)):
        raise AnnotationParseError(f"Invalid LabelMe number: {value!r}")
    try:
        number = float(value)
    except (ValueError, OverflowError) as error:
        raise AnnotationParseError(f"Invalid LabelMe number: {value!r}") from error
    if not math.isfinite(number):
        raise AnnotationParseError(f"Non-finite LabelMe number: {value!r}")
    return number


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
    if not isinstance(data, dict):
        raise AnnotationParseError("LabelMe annotation must be an object")

    image_name = data.get("imagePath")
    if not isinstance(image_name, str):
        raise AnnotationParseError("LabelMe imagePath must be a string")
    extra = {"imagePath": image_name}
    json_width = _parse_int(data.get("imageWidth"))
    json_height = _parse_int(data.get("imageHeight"))
    if image_path:
        extra["image_path"] = image_path
        width, height = resolve_image_wh(image_path, width, height)
    else:
        width = width if width is not None else json_width
        height = height if height is not None else json_height
    
    ann = Annotation(
        width=width,
        height=height,
        names=names,
        extra=extra,
    )

    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        raise AnnotationParseError("LabelMe shapes must be a list")
    for shape in shapes:
        if not isinstance(shape, dict) or shape.get("shape_type") != "rectangle":
            raise AnnotationParseError("Only LabelMe rectangle shapes are supported for detection")
        points = shape.get("points")
        if not isinstance(points, list) or len(points) != 2:
            raise AnnotationParseError("A LabelMe rectangle requires exactly two points")
        if any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in points):
            raise AnnotationParseError("Invalid LabelMe rectangle point")
        xs = [_parse_float(p[0]) for p in points]
        ys = [_parse_float(p[1]) for p in points]
        category = shape.get("label")
        if not isinstance(category, str) or not category.strip():
            raise AnnotationParseError("Missing LabelMe detection label")
        if names and category not in names:
            raise AnnotationParseError(f"Unknown LabelMe category: {category}")
        item = AnnotationItem(
            category=category,
            category_id=names.index(category) if category in names else None,
            bbox=BBox(min(xs), min(ys), max(xs), max(ys)),
            extra={"shape_type": "rectangle"},
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
    suffix=".json",
    label_dir_name="labelme",
    discovery_dir_names=("Jsons", "labelme", "Labelme"),
)
