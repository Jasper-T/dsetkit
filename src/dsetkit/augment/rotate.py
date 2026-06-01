from pathlib import Path
from typing import Literal

import cv2

from ..annotations.io import dump, get_label_file_suffix, load
from ..annotations.schema import Annotation, BBox
from ..utils.file import ensure_dir

# Clockwise rotation in degrees (orthogonal only; axis-aligned bbox preserved)
RotateAngle = Literal[90, 180, 270]

_CV2_ROTATE = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _validate_angle(angle: int) -> RotateAngle:
    if angle not in (90, 180, 270):
        raise ValueError("angle must be 90, 180, or 270 (degrees clockwise)")
    return angle  # type: ignore[return-value]


def _rotate_suffix(angle: RotateAngle) -> str:
    return f"_rot{angle}"


def _transform_point(x: float, y: float, width: int, height: int, angle: RotateAngle) -> tuple[float, float]:
    """Map a point from source image coords to rotated image coords (clockwise)."""
    if angle == 90:
        return y, width - x
    if angle == 180:
        return width - x, height - y
    return height - y, x


def _rotate_bbox(bbox: BBox, width: int, height: int, angle: RotateAngle) -> BBox:
    corners = (
        (bbox.x1, bbox.y1),
        (bbox.x2, bbox.y1),
        (bbox.x1, bbox.y2),
        (bbox.x2, bbox.y2),
    )
    xs: list[float] = []
    ys: list[float] = []
    for x, y in corners:
        nx, ny = _transform_point(x, y, width, height, angle)
        xs.append(nx)
        ys.append(ny)
    return BBox(min(xs), min(ys), max(xs), max(ys))


def _rotated_size(width: int, height: int, angle: RotateAngle) -> tuple[int, int]:
    if angle in (90, 270):
        return height, width
    return width, height


def rotate_annotation(ann: Annotation, angle: RotateAngle = 90) -> Annotation:
    """Rotate bounding boxes and update annotation size for clockwise orthogonal rotation."""
    angle = _validate_angle(angle)
    width, height = ann.require_size()
    new_width, new_height = _rotated_size(width, height, angle)
    for item in ann.items:
        if item.bbox is None:
            continue
        item.bbox = _rotate_bbox(item.bbox, width, height, angle)
    ann.width = new_width
    ann.height = new_height
    return ann


def rotate_image(
    image_path: Path,
    out_img_dir: Path,
    angle: RotateAngle = 90,
) -> Path:
    """Rotate one image file clockwise and write to out_img_dir."""
    angle = _validate_angle(angle)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    rotated = cv2.rotate(image, _CV2_ROTATE[angle])
    stem = image_path.stem
    out_img_path = out_img_dir / f"{stem}{_rotate_suffix(angle)}{image_path.suffix}"
    ensure_dir(out_img_dir)
    cv2.imwrite(str(out_img_path), rotated)
    return out_img_path


def rotate_label(
    label_path: Path,
    image_path: Path,
    out_label_dir: Path,
    source_format: str,
    angle: RotateAngle = 90,
    names: list[str] | None = None,
    target_format: str | None = None,
) -> Path:
    """Rotate one label file and write to out_label_dir."""
    angle = _validate_angle(angle)
    output_format = target_format or source_format

    load_kwargs: dict = {
        "label_path": str(label_path),
        "image_path": str(image_path),
        "fmt": source_format,
    }
    if names is not None:
        load_kwargs["names"] = names
    ann = load(**load_kwargs)
    ann = rotate_annotation(ann, angle=angle)
    stem = image_path.stem
    label_suffix = get_label_file_suffix(output_format)
    out_label_path = out_label_dir / f"{stem}{_rotate_suffix(angle)}{label_suffix}"
    ensure_dir(out_label_dir)
    dump(ann, str(out_label_path), fmt=output_format)
    return out_label_path


def rotate_sample(
    image_path: Path,
    label_path: Path,
    out_img_dir: Path,
    out_label_dir: Path,
    source_format: str,
    angle: RotateAngle = 90,
    names: list[str] | None = None,
    target_format: str | None = None,
) -> tuple[Path, Path]:
    rotated_image_path = rotate_image(image_path, out_img_dir, angle=angle)
    rotated_label_path = rotate_label(
        label_path,
        image_path,
        out_label_dir,
        source_format,
        angle=angle,
        names=names,
        target_format=target_format,
    )
    return rotated_image_path, rotated_label_path
