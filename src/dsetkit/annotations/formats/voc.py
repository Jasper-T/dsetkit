import xml.etree.ElementTree as ET

from ...utils.image import resolve_image_wh
from ..registry import register_format
from ..schema import Annotation, AnnotationItem, AnnotationParseError, BBox


def _parse_xml_int(parent, tag: str) -> int | None:
    if parent is None:
        return None
    text = parent.findtext(tag)
    if not text or not text.strip():
        return None
    try:
        return int(text)
    except ValueError as error:
        raise AnnotationParseError(f"Invalid VOC dimension {tag}: {text!r}") from error


def _get_float(parent, tag: str) -> float:
    text = parent.findtext(tag)
    if text is None or not text.strip():
        raise AnnotationParseError(f"Missing VOC coordinate: {tag}")
    try:
        return float(text)
    except ValueError as error:
        raise AnnotationParseError(f"Invalid VOC coordinate {tag}: {text!r}") from error


def load_voc(
    path: str,
    image_path: str | None = None,
    width: int | None = None,
    height: int | None = None,
    names: list[str] | None = None,
    **kwargs,
) -> Annotation:
    names = names or []
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag != "annotation":
        raise AnnotationParseError("Expected a VOC annotation root")

    file_name = root.findtext("filename")
    extra = {
        "filename": file_name
    }

    size = root.find("size")
    xml_width = _parse_xml_int(size, "width")
    xml_height = _parse_xml_int(size, "height")
    if image_path:
        extra["image_path"] = image_path
        width, height = resolve_image_wh(image_path, width, height)
    else:
        width = width if width is not None else xml_width
        height = height if height is not None else xml_height
    
    ann = Annotation(
        width=width,
        height=height,
        names=names,
        extra=extra,
    )

    for obj in root.findall("object"):
        name = obj.findtext("name")
        if not name or not name.strip():
            raise AnnotationParseError("Missing VOC object name")

        b = obj.find("bndbox")
        if b is None:
            raise AnnotationParseError("Missing VOC detection bounding box")
        
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
    file_name = ann.extra.get("filename", "")
    image_path = ann.extra.get("image_path", file_name)
    
    ET.SubElement(
        root,
        "filename",
    ).text = str(image_path)

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
        ).text = str(item.bbox.x1)

        ET.SubElement(
            bndbox,
            "ymin",
        ).text = str(item.bbox.y1)

        ET.SubElement(
            bndbox,
            "xmax",
        ).text = str(item.bbox.x2)

        ET.SubElement(
            bndbox,
            "ymax",
        ).text = str(item.bbox.y2)

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
    suffix=".xml",
    label_dir_name="xmls",
    discovery_dir_names=("Annotations", "xmls", "Xmls"),
)
