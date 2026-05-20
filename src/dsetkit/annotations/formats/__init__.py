# Register all annotation adapters with the global registry.
from . import coco  # noqa: F401
from . import labelme  # noqa: F401
from . import voc  # noqa: F401
from . import yolo  # noqa: F401


FORMAT_SUFFIXES = {
    "coco": ".json",
    "labelme": ".json",
    "voc": ".xml",
    "yolo": ".txt",
}