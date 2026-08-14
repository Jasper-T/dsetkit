import base64
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dsetkit.utils.image import read_image_info, validate_image

VALID_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class TestImageValidation(unittest.TestCase):
    def test_validates_png_without_full_decode(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image.png"
            path.write_bytes(VALID_PNG)

            result = validate_image(path)

            self.assertTrue(result.valid)
            self.assertEqual(result.info.size, (1, 1))

    def test_rejects_png_missing_end_chunk(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image.png"
            path.write_bytes(VALID_PNG[:-12])

            result = validate_image(path)

            self.assertFalse(result.valid)
            self.assertEqual(result.error_kind, "invalid_image")

    def test_reads_all_common_webp_header_variants(self) -> None:
        variants = {
            "VP8X": _webp_vp8x(3, 5),
            "VP8": _webp_vp8(3, 5),
            "VP8L": _webp_vp8l(3, 5),
        }
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for name, content in variants.items():
                with self.subTest(variant=name):
                    path = root / f"{name}.webp"
                    path.write_bytes(content)
                    info = read_image_info(path)
                    self.assertEqual(info.size, (3, 5))
                    self.assertTrue(validate_image(path).valid)


def _riff_webp(payload: bytes) -> bytes:
    content = b"WEBP" + payload
    return b"RIFF" + len(content).to_bytes(4, "little") + content


def _webp_vp8x(width: int, height: int) -> bytes:
    payload = (
        b"\x00\x00\x00\x00"
        + (width - 1).to_bytes(3, "little")
        + (height - 1).to_bytes(3, "little")
    )
    return _riff_webp(b"VP8X" + len(payload).to_bytes(4, "little") + payload)


def _webp_vp8(width: int, height: int) -> bytes:
    payload = (
        b"\x00\x00\x00"
        + b"\x9d\x01\x2a"
        + width.to_bytes(2, "little")
        + height.to_bytes(2, "little")
    )
    return _riff_webp(b"VP8 " + len(payload).to_bytes(4, "little") + payload)


def _webp_vp8l(width: int, height: int) -> bytes:
    bits = (width - 1) | ((height - 1) << 14)
    payload = b"\x2f" + bits.to_bytes(4, "little") + b"\x00" * 5
    return _riff_webp(b"VP8L" + len(payload).to_bytes(4, "little") + payload)


if __name__ == "__main__":
    unittest.main()
