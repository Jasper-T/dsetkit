from pathlib import Path
import cv2
import numpy as np

from typing import Sequence, Tuple, Optional, Union 

from ..annotations.schema import AnnotationItem


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
        bbox: Sequence[int],
        text: str,
        color,
    ):
        """
        label background attached to bbox top
        """

        x1, y1, _, _ = map(int, bbox)

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
        class_id: int,
        name: str = "",
        score: Optional[float] = None,
    ):
        color = self.random_color(class_id)

        self.box(
            bbox,
            color,
        )

        if name != "":
            text = name

            if score is not None:
                text += f": {score:.2f}"

            self.label(
                bbox,
                text,
                color,
            )

        return self
    
    def detection_from_schema(self, item: AnnotationItem):
        return self.detection(
            bbox=[item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2],
            class_id=item.category_id or 0,
            name=item.category or "",
            score=item.extra.get("score"),
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

        cv2.imwrite(
            save_path,
            self.image,
        )

    def get(self):
        return self.image