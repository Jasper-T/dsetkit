from pathlib import Path
from ...utils.image import read_image_info


def resolve_image_wh(image_path, width, height):
    if width is not None and height is not None:
        return width, height
    
    info = read_image_info(image_path)
    return info.width, info.height