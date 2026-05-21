from pathlib import Path

from .registry import get_dumper, get_loader
from .formats import FORMAT_SUFFIXES
from .schema import Annotation

# trigger adapter registration
from . import formats  # noqa: F401


FORMAT_DIRS = {
    "coco": "coco",
    "labelme": "labelme",
    "voc": "xmls",
    "yolo": "labels",
}

def get_label_dir_name(fmt: str) -> str:
    name = FORMAT_DIRS.get(fmt)
    if name is None:
        raise ValueError(f"Unknown format: {fmt}")
    return name


def get_label_file_suffix(fmt: str) -> str:
    suffix = FORMAT_SUFFIXES.get(fmt)
    if suffix is None:
        raise ValueError(f"Unknown format: {fmt}")
    return suffix


def get_label_path(image_path: str, label_dir: str, fmt: str) -> str:
    suffix = get_label_file_suffix(fmt)
    
    label_file = f"{Path(image_path).stem}{suffix}"
    
    label_path = Path(label_dir) / label_file
    return str(label_path)


def default_label_dir(image_path: str, fmt: str) -> str:
    name = get_label_dir_name(fmt)
    label_dir = Path(image_path).parent.parent / name
    ensure_dir(label_dir)
    return str(label_dir)


def auto_label_path(image_path: str, fmt: str, label_dir: str | None = None) -> str:
    if label_dir is None:
        label_dir = default_label_dir(image_path, fmt)
    else:
        name = get_label_dir_name(fmt)
        label_dir = Path(label_dir) / name
        ensure_dir(label_dir)

    label_path = get_label_path(image_path, label_dir, fmt)
    return label_path


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load(
    *,
    label_path: str,
    image_path: str,
    fmt: str,
    **kwargs,
) -> Annotation:
    """
    Load annotation file into unified Annotation schema.
    fmt is the target format.
    supported formats are:
    - coco
    - labelme
    - voc
    - yolo
    if the target format is not supported, an error will be raised.
    """

    if not fmt:
        raise ValueError("fmt must be explicitly specified")

    loader = get_loader(fmt)

    load_kwargs = dict(kwargs)

    load_kwargs.setdefault("image_path", image_path)

    return loader(label_path, **load_kwargs)


def dump(
    ann,
    path: str,
    fmt: str,
    **kwargs,
):
    """
    Dump Annotation schema to target format.
    target format is one of the supported formats.
    supported formats are:
    - coco
    - labelme
    - voc
    - yolo
    if the target format is not supported, an error will be raised.
    """
    dumper = get_dumper(fmt)

    return dumper(ann, path, **kwargs)



def convert(
    *,
    label_path: str,
    image_path: str,
    target_format: str,
    source_format: str,
    out_dir: str | None = None,
    **kwargs,
) -> str:
    """
    Convert annotation format.
    fmt is the source format; target_format is the output format.
    supported formats are:
    - coco
    - labelme
    - voc
    - yolo
    if a format is not supported, an error will be raised.
    """

    ann = load(
        label_path=label_path,
        image_path=image_path,
        fmt=source_format,
        **kwargs,
    )

    out_path = auto_label_path(image_path, target_format, out_dir)

    dump(ann, out_path, fmt=target_format)

    return out_path

