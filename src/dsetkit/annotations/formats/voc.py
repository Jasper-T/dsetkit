import xml.etree.ElementTree as ET

from ..schema import Annotation, AnnotationItem, BBox
from ..registry import register_format
from .common import resolve_image_path, resolve_image_wh


def _parse_xml_int(parent, tag: str) -> int | None:
    if parent is None:
        return None
    text = parent.findtext(tag)
    if not text or not text.strip():
        return None
    return int(text)


def _get_float(b, tag: str, default=0.0) -> float:
    if b is None:
        return default
    text = b.findtext(tag)
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def load_voc(
    path: str,
    image_path: str | None = None,
    names: list[str] = [],
    **kwargs,
) -> Annotation:
    tree = ET.parse(path)
    root = tree.getroot()

    filename = root.findtext("filename")
    image_path = resolve_image_path(path, image_path, filename)
    
    size = root.find("size")
    width = _parse_xml_int(size, "width")
    height = _parse_xml_int(size, "height")

    if image_path:
        width, height = resolve_image_wh(image_path, width, height)
    
    ann = Annotation(
        image_path=image_path,
        width=width,
        height=height,
        names=names,
    )

    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name is None:
            continue

        b = obj.find("bndbox")
        if b is None:
            continue
        
        category_id = names.index(name) if name in names else None
        item = AnnotationItem(
            category=name,
            category_id=category_id,
            bbox=BBox(
                x1=_get_float(b, "xmin"),
                y1=_get_float(b, "ymin"),
                x2=_get_float(b, "xmax"),
                y2=_get_float(b, "ymax"),
            ),
            extra={
                "shape_type": "rectangle",
            },
        )
        ann.items.append(item)

    return ann


def dump_voc(
    ann: Annotation,
    out_path: str,
):
    root = ET.Element("annotation")

    ET.SubElement(
        root,
        "filename",
    ).text = str(ann.image_path)

    size = ET.SubElement(root, "size")

    ET.SubElement(
        size,
        "width",
    ).text = str(ann.width or 0)

    ET.SubElement(
        size,
        "height",
    ).text = str(ann.height or 0)

    ET.SubElement(
        size,
        "depth",
    ).text = "3"

    for item in ann.items:
        if item.bbox is None:
            continue

        obj = ET.SubElement(root, "object")

        ET.SubElement(
            obj,
            "name",
        ).text = item.category

        bndbox = ET.SubElement(
            obj,
            "bndbox",
        )

        ET.SubElement(
            bndbox,
            "xmin",
        ).text = str(int(item.bbox.x1))

        ET.SubElement(
            bndbox,
            "ymin",
        ).text = str(int(item.bbox.y1))

        ET.SubElement(
            bndbox,
            "xmax",
        ).text = str(int(item.bbox.x2))

        ET.SubElement(
            bndbox,
            "ymax",
        ).text = str(int(item.bbox.y2))

    ET.indent(root, space="  ")

    tree = ET.ElementTree(root)

    tree.write(
        out_path,
        encoding="utf-8",
        xml_declaration=True,
    )


register_format(
    "voc",
    loader=load_voc,
    dumper=dump_voc,
)