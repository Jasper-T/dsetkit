from copy import deepcopy
from pathlib import Path
from typing import Sequence, Tuple, Union

import cv2
import numpy as np

from ..annotations.io import load
from ..annotations._mapping import validate_class_ids, validate_names
from ..annotations.schema import Annotation, AnnotationItem


class Plotter:
    def __init__(
        self,
        image: np.ndarray,
        line_width: int = None,
        font_scale: float = None,
        font_thickness: int = None,
        padding: int = None,
        text_color=(255, 255, 255),
        font=cv2.FONT_HERSHEY_SIMPLEX,
    ):
        self.image = image.copy()

        h, w = image.shape[:2]
        self.height, self.width = h, w
        
        scale = max(h, w) / 1000

        self.line_width = (
            line_width
            if line_width is not None
            else max(int(scale * 2), 1)
        )

        self.font_scale = (
            font_scale
            if font_scale is not None
            else max(scale * 0.5, 0.5)
        )

        self.font_thickness = (
            font_thickness
            if font_thickness is not None
            else max(int(scale * 2), 1)
        )

        self.padding = (
            padding
            if padding is not None
            else max(int(scale * 3), 2)
        )

        self.text_color = text_color

        self.font = font

    @staticmethod
    def random_color(class_id: int) -> Tuple[int, int, int]:
        """
        stable color from class_id
        """

        rng = np.random.default_rng(class_id)

        return tuple(
            map(int, rng.integers(0, 255, size=3))
        )

    def box(
        self,
        bbox: Sequence[int],
        color: Tuple[int, int, int],
    ):
        x1, y1, x2, y2 = map(int, bbox)

        cv2.rectangle(
            self.image,
            (x1, y1),
            (x2, y2),
            color,
            self.line_width,
        )

        return self

    def rectangle_fill(
        self,
        pt1,
        pt2,
        color,
    ):
        cv2.rectangle(
            self.image,
            pt1,
            pt2,
            color,
            -1,
        )

        return self

    def text(
        self,
        text: str,
        position,
        color=None,
    ):
        if color is None:
            color = self.text_color

        cv2.putText(
            self.image,
            text,
            position,
            self.font,
            self.font_scale,
            color,
            self.font_thickness,
            cv2.LINE_AA,
        )

        return self

    def label(
        self,
        point: Sequence[int],
        text: str,
        color,
    ):
        """
        label background attached to bbox top
        """

        x1, y1 = map(int, point)

        (tw, th), baseline = cv2.getTextSize(
            text,
            self.font,
            self.font_scale,
            self.font_thickness,
        )

        bg_x1 = x1
        bg_y2 = y1

        bg_x2 = bg_x1 + tw + self.padding * 2
        bg_y1 = bg_y2 - th - baseline - self.padding * 2

        # top overflow
        if bg_y1 < 0:
            bg_y1 = y1
            bg_y2 = bg_y1 + th + baseline + self.padding * 2

        cv2.rectangle(
            self.image,
            (bg_x1, bg_y1),
            (bg_x2, bg_y2),
            color,
            -1,
        )

        text_x = bg_x1 + self.padding
        text_y = bg_y2 - baseline - self.padding

        self.text(
            text,
            (text_x, text_y),
        )

        return self


    def detection(
        self,
        bbox: Sequence[int],
        class_id: int = None,
        text: str = None, 
    ):  
        if class_id is None:
            color = self.random_color(0)
        else:
            color = self.random_color(int(class_id))

        self.box(
            bbox,
            color,
        )

        if text is None and class_id is not None:
            text = str(class_id)
        if text is not None:
            point = (bbox[0], bbox[1])
            self.label(
                point=point,
                text=text,
                color=color,
            )

        return self.get()
    
    
    def detection_from_schema(self, item: AnnotationItem):
        if item.bbox is None:
            return self.get()
        bbox = [item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2]
        name = item.category or ""
        score = item.extra.get("score")
        class_id = item.category_id or 0
        text = name if score is None else f"{name}: {score:.2f}"
        
        return self.detection(
            bbox=bbox,
            class_id=class_id,
            text=text,
        )

    def show(
        self,
        win_name: str = "image",
    ):
        cv2.imshow(win_name, self.image)

        cv2.waitKey(0)

        cv2.destroyAllWindows()

    def save(
        self,
        save_path: Union[str, Path],
    ):
        save_path = Path(save_path).resolve()

        saved = cv2.imwrite(
            str(save_path),
            self.image,
        )
        if not saved:
            raise OSError(f"Failed to save image: {save_path}")

    def get(self):
        return self.image


def plot(
    image: np.ndarray | None = None, 
    image_path: str | Path | None = None, 
    label_path: str | Path | None = None,
    anno_schema: Annotation | None = None,
    fmt: str  = 'yolo', 
    names: list[str] = None,
    save_path: str | Path | None = None
):

    if image is None:
        if image_path is None:
            raise ValueError("image_path is required when image is not provided")
        image = cv2.imread(
            str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        if image is None:
            raise ValueError(f"Cannot decode image: {image_path}")
    
    if anno_schema is None:
        if label_path is None:
            raise ValueError("label_path is required when anno_schema is not provided")
        
        anno_schema = load(
            label_path=label_path, 
            image_path=image_path, 
            fmt=fmt, 
            width=image.shape[1],
            height=image.shape[0],
        )
    final_names = names if names is not None else anno_schema.names
    validate_names(final_names)
    anno_schema = deepcopy(anno_schema)
    native_categories = list(dict.fromkeys(item.category for item in anno_schema.items))
    for item in anno_schema.items:
        if item.category_id is None:
            if final_names and item.category not in final_names:
                raise ValueError(f"Visualization requires a name mapping for category: {item.category}")
            item.category_id = (final_names or native_categories).index(item.category)
    if final_names:
        validate_class_ids(
            [item.category_id for item in anno_schema.items], final_names, context="Visualization"
        )
    for item in anno_schema.items:
        item.category = final_names[int(item.category_id)] if final_names else str(item.category_id)

    plotter = Plotter(image)
    for item in anno_schema.items:
        plotter.detection_from_schema(item)

    if save_path is not None:
        plotter.save(save_path)

    return plotter.get()
