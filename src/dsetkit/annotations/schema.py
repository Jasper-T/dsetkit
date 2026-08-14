from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class ClassSchemaDifference:
    class_id: int
    left_name: str | None
    right_name: str | None
    reason: str

    def as_dict(self) -> dict[str, int | str | None]:
        return {
            "class_id": self.class_id,
            "left_name": self.left_name,
            "right_name": self.right_name,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ClassSchemaComparison:
    compatible: bool
    differences: tuple[ClassSchemaDifference, ...]

    def as_dict(self) -> dict:
        return {
            "compatible": self.compatible,
            "differences": [difference.as_dict() for difference in self.differences],
        }


def compare_class_schemas(
    left_names: Sequence[str],
    right_names: Sequence[str],
    mode: str = "strict",
) -> ClassSchemaComparison:
    """Compare class IDs, names, and ordering without implicit remapping."""
    if mode != "strict":
        raise ValueError(f"Unsupported class schema comparison mode: {mode}")

    differences: list[ClassSchemaDifference] = []
    size = max(len(left_names), len(right_names))
    for class_id in range(size):
        left_name = left_names[class_id] if class_id < len(left_names) else None
        right_name = right_names[class_id] if class_id < len(right_names) else None
        if left_name == right_name:
            continue
        if left_name is None:
            reason = "missing_on_left"
        elif right_name is None:
            reason = "missing_on_right"
        else:
            reason = "name_mismatch"
        differences.append(
            ClassSchemaDifference(
                class_id=class_id,
                left_name=left_name,
                right_name=right_name,
                reason=reason,
            )
        )

    return ClassSchemaComparison(
        compatible=not differences,
        differences=tuple(differences),
    )


@dataclass
class BBox:
    # xyxy (absolute)
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class AnnotationItem:
    category: str
    category_id: Optional[int] = None

    bbox: Optional[BBox] = None

    # future-proof
    segmentation: Optional[Any] = None
    keypoints: Optional[Any] = None
    track_id: Optional[int] = None

    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Annotation:
    width: Optional[int] = None
    height: Optional[int] = None

    items: List[AnnotationItem] = field(default_factory=list)

    # class_id -> class_name
    names: List[str] = field(default_factory=list)

    extra: Dict[str, Any] = field(default_factory=dict)

    def require_size(self) -> tuple[int, int]:
        if self.width is None or self.height is None:
            raise ValueError("Annotation width/height required")
        return self.width, self.height

    def to_array(self):
        rows = []
        for item in self.items:
            if item.bbox is None:
                continue
            rows.append([
                item.bbox.x1,
                item.bbox.y1,
                item.bbox.x2,
                item.bbox.y2,
                item.category_id,
            ])
        return np.asarray(rows, dtype=np.float32)
