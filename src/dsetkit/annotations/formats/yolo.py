from ..schema import Annotation, AnnotationItem, BBox
from ..registry import register_format
from ...utils.image  import resolve_image_wh


def load_yolo(
    path: str,
    image_path: str | None = None,
    width: int | None = None,
    height: int | None = None,
    names: list[str] | None = None,
):
    if image_path:
        width, height = resolve_image_wh(image_path, width, height)
    elif width is None or height is None:
        raise ValueError("width and height are required when image_path is not provided")

    ann = Annotation(
        width=width,
        height=height,
        names=names or [],
        extra={"image_path": image_path} if image_path else {},
    )

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            cls, cx, cy, bw, bh = map(float, parts)

            cls_id = int(cls)

            if names and cls_id < len(names):
                category = names[cls_id]
            else:
                category = str(cls_id)

            x1 = (cx - bw / 2) * width
            y1 = (cy - bh / 2) * height
            x2 = (cx + bw / 2) * width
            y2 = (cy + bh / 2) * height

            item = AnnotationItem(
                category=category,
                category_id=cls_id,
                bbox=BBox(x1, y1, x2, y2),
            )
            ann.items.append(item)

    return ann


def dump_yolo(
    ann: Annotation,
    out_path: str,
):
    width, height = ann.require_size()

    lines = []

    for item in ann.items:
        if item.bbox is None:
            continue

        if item.category_id is not None:
            cls_id = item.category_id
        elif ann.names:
            try:
                cls_id = ann.names.index(item.category)
            except ValueError as e:
                raise ValueError(f"Category not found in names: {item.category}") from e
        else:
            raise ValueError(f"Missing category_id for: {item.category}")

        cx = ((item.bbox.x1 + item.bbox.x2) / 2) / width
        cy = ((item.bbox.y1 + item.bbox.y2) / 2) / height
        bw = (item.bbox.x2 - item.bbox.x1) / width
        bh = (item.bbox.y2 - item.bbox.y1) / height

        lines.append(f"{cls_id} {cx} {cy} {bw} {bh}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


register_format(
    "yolo",
    loader=load_yolo,
    dumper=dump_yolo,
)