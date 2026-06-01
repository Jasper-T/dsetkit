from pathlib import Path
from typing import Literal

import cv2

from ..annotations.io import dump, get_label_file_suffix, load
from ..annotations.schema import Annotation, BBox
from ..utils.file import ensure_dir

# Same as cv2.flip flipCode: 0 = vertical, 1 = horizontal
FlipDirection = Literal[0, 1]


def _validate_direction(direction: int) -> FlipDirection:
    if direction not in (0, 1):
        raise ValueError("direction must be 0 (vertical) or 1 (horizontal)")
    return direction  # type: ignore[return-value]


def _flip_suffix(direction: FlipDirection) -> str:
    return "_fliph" if direction == 1 else "_flipv"


def flip_annotation_horizontal(ann: Annotation) -> Annotation:
    """Flip bounding boxes for a horizontal image flip (x axis mirror)."""
    width, _ = ann.require_size()
    for item in ann.items:
        if item.bbox is None:
            continue
        bbox = item.bbox
        item.bbox = BBox(width - bbox.x2, bbox.y1, width - bbox.x1, bbox.y2)
    return ann


def flip_annotation_vertical(ann: Annotation) -> Annotation:
    """Flip bounding boxes for a vertical image flip (y axis mirror)."""
    _, height = ann.require_size()
    for item in ann.items:
        if item.bbox is None:
            continue
        bbox = item.bbox
        item.bbox = BBox(bbox.x1, height - bbox.y2, bbox.x2, height - bbox.y1)
    return ann


def flip_annotation(ann: Annotation, direction: FlipDirection = 1) -> Annotation:
    """Flip bounding boxes in schema coordinates for the given direction (0/1, cv2.flip)."""
    direction = _validate_direction(direction)
    if direction == 1:
        return flip_annotation_horizontal(ann)
    return flip_annotation_vertical(ann)


def flip_image(
    image_path: Path,
    out_img_dir: Path,
    direction: FlipDirection = 1,
) -> Path:
    """Flip one image file and write to out_img_dir."""
    direction = _validate_direction(direction)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    flipped = cv2.flip(image, direction)
    stem = image_path.stem
    out_img_path = out_img_dir / f"{stem}{_flip_suffix(direction)}{image_path.suffix}"
    ensure_dir(out_img_dir)
    cv2.imwrite(str(out_img_path), flipped)
    return out_img_path


def flip_label(
    label_path: Path,
    image_path: Path,
    out_label_dir: Path,
    source_format: str,
    direction: FlipDirection = 1,
    names: list[str] | None = None,
    target_format: str | None = None,
) -> Path:
    """Flip one label file and write to out_label_dir."""
    direction = _validate_direction(direction)
    output_format = target_format or source_format

    load_kwargs: dict = {
        "label_path": str(label_path),
        "image_path": str(image_path),
        "fmt": source_format,
    }
    if names is not None:
        load_kwargs["names"] = names
    ann = load(**load_kwargs)
    ann = flip_annotation(ann, direction=direction)
    stem = image_path.stem
    label_suffix = get_label_file_suffix(output_format)
    out_label_path = out_label_dir / f"{stem}{_flip_suffix(direction)}{label_suffix}"
    ensure_dir(out_label_dir)
    dump(ann, str(out_label_path), fmt=output_format)
    return out_label_path


def flip_sample(
    image_path: Path,
    label_path: Path,
    out_img_dir: Path,
    out_label_dir: Path,
    source_format: str,
    direction: FlipDirection = 1,
    names: list[str] | None = None,
    target_format: str | None = None,
) -> None:
    flipped_image_path = flip_image(image_path, out_img_dir, direction=direction)
    flipped_label_path = flip_label(label_path, image_path, out_label_dir, source_format, direction=direction, names=names, target_format=target_format)
    return flipped_image_path, flipped_label_path

