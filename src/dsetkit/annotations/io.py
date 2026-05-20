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

def get_output_suffix(fmt: str) -> str:
    suffix = FORMAT_SUFFIXES.get(fmt)
    if suffix is None:
        raise ValueError(f"Unknown format: {fmt}")
    return suffix


def get_output_path(image_path: str, out_dir: str, fmt: str) -> str:
    suffix = get_output_suffix(fmt)
    label_file = f"{Path(image_path).stem}{suffix}"
    out_path = Path(out_dir) / label_file
    return str(out_path)


def default_out_dir(image_path: str, fmt: str) -> str:
    name = FORMAT_DIRS.get(fmt)
    if name is None:
        raise ValueError(f"Unknown format: {fmt}")

    out_dir = Path(image_path).parent.parent / name
    ensure_dir(out_dir)
    return str(out_dir)


def auto_out_path(image_path: str, fmt: str, out_dir: str | None = None) -> str:
    if out_dir is None:
        out_dir = default_out_dir(image_path, fmt)
    else:
        out_dir = Path(out_dir)
        ensure_dir(out_dir)

    out_path = get_output_path(image_path, out_dir, fmt)
    return out_path


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

    out_path = auto_out_path(image_path, target_format, out_dir)

    dump(ann, out_path, fmt=target_format)

    return out_path

