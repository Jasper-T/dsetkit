"""Detection geometry checks, independent of file discovery and serialization."""

import numpy as np

from .schema import Annotation, AnnotationParseError


def validate_annotation(
    annotation: Annotation | None,
    *,
    source_format: str | None,
    image_size: tuple[int, int] | None = None,
) -> None:
    if annotation is None:
        return
    actual = image_size
    if source_format == "yolo":
        width, height = 1, 1
    else:
        width, height = actual or (annotation.width, annotation.height)
    if (width is not None and width <= 0) or (height is not None and height <= 0):
        raise AnnotationParseError("Annotation dimensions must be positive")
    for item in annotation.items:
        if item.bbox is None or item.segmentation is not None:
            raise AnnotationParseError("Only detection bounding boxes are supported")
        coords = [item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2]
        x1, y1, x2, y2 = coords
        tolerance = max(width or 1, height or 1) * 1e-9
        if (
            not np.isfinite(coords).all()
            or x2 <= x1
            or y2 <= y1
            or min(x1, y1) < -tolerance
            or (width is not None and x2 > width + tolerance)
            or (height is not None and y2 > height + tolerance)
        ):
            raise AnnotationParseError(f"Invalid detection bounding box: {coords}")
