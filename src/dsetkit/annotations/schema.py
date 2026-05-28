import numpy as np

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


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