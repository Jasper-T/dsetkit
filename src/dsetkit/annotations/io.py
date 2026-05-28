from pathlib import Path

from .registry import get_dumper, get_loader
from .formats import FORMAT_SUFFIXES
from .schema import Annotation
from ..utils.file import ensure_dir

# trigger adapter registration
from . import formats  # noqa: F401


FORMAT_DIRS = {
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


def new_label_path(old_path: str, label_dir: str, fmt: str) -> str:
    suffix = get_label_file_suffix(fmt)
    
    new_file_name = f"{Path(old_path).stem}{suffix}"
    
    new_path = Path(label_dir) / new_file_name
    return str(new_path)


def new_label_dir(old_path: str, fmt: str, new_dir: str | None = None) -> str:
    name = get_label_dir_name(fmt)
    if new_dir is None:
        new_dir = Path(old_path).parent.parent
    else:
        new_dir = Path(new_dir)
    
    new_dir = new_dir / name
    ensure_dir(new_dir)
    return str(new_dir)


def auto_label_path(old_path: str, fmt: str, new_dir: str | None = None) -> str:
    new_dir = new_label_dir(old_path, fmt, new_dir)
    return new_label_path(old_path, new_dir, fmt)


def load(
    *,
    label_path: str,
    image_path: str | None = None,
    width: int | None = None,
    height: int | None = None,
    fmt: str,
    **kwargs,
) -> Annotation:
    """
    Load annotation file into unified Annotation schema.
    fmt is the target format.
    supported formats are:
    - labelme
    - voc
    - yolo
    if the target format is not supported, an error will be raised.
    """

    if not fmt:
        raise ValueError("fmt must be explicitly specified")

    loader = get_loader(fmt)

    load_kwargs = dict(kwargs)
    if image_path is not None:
        load_kwargs.setdefault("image_path", image_path)
    if width is not None:
        load_kwargs.setdefault("width", width)
    if height is not None:
        load_kwargs.setdefault("height", height)

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
    image_path: str | None = None,
    target_format: str,
    source_format: str,
    out_dir: str | None = None,
    **kwargs,
) -> str:
    """
    Convert annotation format.
    fmt is the source format; target_format is the output format.
    supported formats are:
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

    out_path = auto_label_path(
        old_path = label_path, 
        fmt = target_format, 
        new_dir = out_dir
    )

    dump(ann, out_path, fmt=target_format)

    return out_path

