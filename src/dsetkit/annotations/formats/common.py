from pathlib import Path
from ...utils.image import read_image_info


def resolve_image_path(base_path, image_path, filename=None):
    if image_path:
        p = Path(image_path)
        if p.is_file():
            return str(p)

        # relative to base
        p2 = Path(base_path).parent / image_path
        if p2.is_file():
            return str(p2)

    if filename:
        p = Path(base_path).parent / filename
        if p.is_file():
            return str(p)
    
    return None # None indicates no image path found


def resolve_image_wh(image_path, width, height):
    if width is not None and height is not None:
        return width, height
    
    info = read_image_info(image_path)
    return info.width, info.height