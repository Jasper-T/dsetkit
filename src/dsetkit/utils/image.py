import struct
import base64

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final

from natsort import natsorted


JPEG_SOF_MARKERS: Final[set[int]] = { 
    0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF
}


@dataclass(slots=True)
class ImageInfo:
    width: int
    height: int

    channels: int | None = None

    mode: str | None = None
    format: str | None = None

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def shape(self) -> tuple[int, ...]:
        if self.channels is None:
            return (self.height, self.width)

        return (
            self.height,
            self.width,
            self.channels,
        )


_READERS: Final[
    dict[str, Callable[[Path], ImageInfo]]
] = {}


def register_reader(*suffixes: str):

    def decorator(
        func: Callable[[Path], ImageInfo],
    ):

        for suffix in suffixes:
            _READERS[suffix] = func

        return func

    return decorator


def read_image_info(
    path: str | Path,
) -> ImageInfo:

    path = Path(path)

    reader = _READERS.get(
        path.suffix.lower(),
    )

    if reader is None:
        raise ValueError(
            f"Unsupported image type: {path}"
        )

    return reader(path)


def iter_image_paths(
    directory: str | Path,
) -> Iterator[Path]:

    directory = Path(directory)

    if not directory.is_dir():
        raise ValueError(
            f"Invalid directory: {directory}"
        )

    for path in directory.iterdir():

        if not path.is_file():
            continue

        if path.suffix.lower() not in _READERS:
            continue

        yield path.resolve()


def get_image_paths(
    directory: str | Path,
    sort: bool = True,
) -> list[Path]:

    paths = iter_image_paths(directory)

    if sort:
        return natsorted(paths)

    return list(paths)


@register_reader(".png")
def _read_png_info(
    path: Path,
) -> ImageInfo:

    with path.open("rb") as f:
        header = f.read(29)

    if len(header) < 29:
        raise ValueError(
            f"Incomplete PNG header: {path}"
        )

    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(
            f"Invalid PNG image: {path}"
        )

    width, height = struct.unpack(
        ">II",
        header[16:24],
    )

    color_type = header[25]

    mode_map = {
        0: ("L", 1),
        2: ("RGB", 3),
        3: ("P", 1),
        4: ("LA", 2),
        6: ("RGBA", 4),
    }

    mode, channels = mode_map.get(
        color_type,
        (None, None),
    )

    return ImageInfo(
        width=width,
        height=height,
        channels=channels,
        mode=mode,
        format="PNG",
    )

@register_reader(".jpg", ".jpeg")
def _read_jpeg_info(
    path: Path,
) -> ImageInfo:

    NO_LENGTH_MARKERS = {
        0x01,
        0xD0,
        0xD1,
        0xD2,
        0xD3,
        0xD4,
        0xD5,
        0xD6,
        0xD7,
        0xD8,
        0xD9,
    }

    with path.open("rb") as f:

        if f.read(2) != b"\xff\xd8":
            raise ValueError(
                f"Invalid JPEG image: {path}"
            )

        while True:

            byte = f.read(1)

            if not byte:
                break

            if byte != b"\xff":
                continue

            marker = f.read(1)

            while marker == b"\xff":
                marker = f.read(1)

            if not marker:
                break

            marker_int = marker[0]

            if marker_int in NO_LENGTH_MARKERS:
                continue

            size_bytes = f.read(2)

            if len(size_bytes) != 2:
                break

            segment_size = struct.unpack(
                ">H",
                size_bytes,
            )[0]

            if segment_size < 2:
                break

            if marker_int in JPEG_SOF_MARKERS:

                segment = f.read(
                    segment_size - 2
                )

                if len(segment) < 6:
                    break

                height, width = struct.unpack(
                    ">HH",
                    segment[1:5],
                )

                channels = segment[5]

                mode_map = {
                    1: "L",
                    3: "RGB",
                    4: "CMYK",
                }

                return ImageInfo(
                    width=width,
                    height=height,
                    channels=channels,
                    mode=mode_map.get(
                        channels,
                    ),
                    format="JPEG",
                )

            f.seek(segment_size - 2, 1)

    raise ValueError(
        f"Cannot read JPEG image info: {path}"
    )

@register_reader(".webp")
def _read_webp_info(
    path: Path,
) -> ImageInfo:

    with path.open("rb") as f:
        header = f.read(30)

    if len(header) < 30:
        raise ValueError(
            f"Incomplete WebP header: {path}"
        )

    if header[:4] != b"RIFF":
        raise ValueError(
            f"Invalid WebP image: {path}"
        )

    if header[8:12] != b"WEBP":
        raise ValueError(
            f"Invalid WebP image: {path}"
        )

    chunk = header[12:16]

    if chunk != b"VP8X":
        raise ValueError(
            f"Unsupported WebP format: {path}"
        )

    width = (
        int.from_bytes(
            header[24:27],
            "little",
        ) + 1
    )

    height = (
        int.from_bytes(
            header[27:30],
            "little",
        ) + 1
    )

    return ImageInfo(
        width=width,
        height=height,
        channels=None,
        mode=None,
        format="WEBP",
    )

@register_reader(".bmp")
def _read_bmp_info(
    path: Path,
) -> ImageInfo:

    with path.open("rb") as f:
        header = f.read(54)

    if len(header) < 54:
        raise ValueError(
            f"Incomplete BMP header: {path}"
        )

    if header[:2] != b"BM":
        raise ValueError(
            f"Invalid BMP image: {path}"
        )

    width = struct.unpack(
        "<i",
        header[18:22],
    )[0]

    height = struct.unpack(
        "<i",
        header[22:26],
    )[0]

    bits_per_pixel = struct.unpack(
        "<H",
        header[28:30],
    )[0]

    channels_map = {
        1: 1,
        4: 1,
        8: 1,
        16: 3,
        24: 3,
        32: 4,
    }

    mode_map = {
        1: "P",
        4: "P",
        8: "P",
        16: "RGB",
        24: "RGB",
        32: "RGBA",
    }

    return ImageInfo(
        width=abs(width),
        height=abs(height),
        channels=channels_map.get(
            bits_per_pixel,
        ),
        mode=mode_map.get(
            bits_per_pixel,
        ),
        format="BMP",
    )


IMAGE_SUFFIXES: Final[
    tuple[str, ...]
] = tuple(_READERS)

IMAGE_SUFFIX_SET: Final[
    set[str]
] = set(_READERS)


def image_to_base64(image_path):

    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
