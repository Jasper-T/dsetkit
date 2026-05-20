from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal, overload

from .formats import FORMAT_SUFFIXES
from .registry import get_dumper, get_loader

# ensure format adapters are registered before get_loader / get_dumper
from . import formats  # noqa: F401

from .io import dump, load

if TYPE_CHECKING:
    from ..dataset import DatasetSample


@dataclass(frozen=True)
class BatchConvertFailure:
    """One failed item when ``batch_convert(..., errors='skip')``."""

    label_path: str
    message: str


class BatchConvertItemError(RuntimeError):
    """Raised when ``batch_convert(..., errors='raise')`` fails on one pair."""

    def __init__(self, label_path: str, message: str):
        self.label_path = label_path
        super().__init__(
            f"batch_convert failed for label_path={label_path!r}: {message}"
        )


def _ensure_out_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _out_path_for_label(label_path: str, out_dir: str, target_format: str) -> str:
    suffix = FORMAT_SUFFIXES[target_format]
    if suffix is None:
        raise ValueError(f"Unknown target format: {target_format}")
    return str(Path(out_dir) / f"{Path(label_path).stem}{suffix}")


def _convert_one_impl(
    label_path: str,
    image_path: str,
    source_fmt: str,
    target_format: str,
    out_path: str,
    load_kwargs: dict,
) -> str:
    ann = load(
        label_path=label_path,
        image_path=image_path,
        fmt=source_fmt,
        **load_kwargs,
    )
    dump(ann, out_path, fmt=target_format)
    return out_path


def _batch_convert_worker_strict(
    args: tuple[str, str, str, str, str, dict],
) -> str:
    label_path, image_path, source_fmt, target_format, out_path, load_kwargs = args
    try:
        return _convert_one_impl(
            label_path,
            image_path,
            source_fmt,
            target_format,
            out_path,
            load_kwargs,
        )
    except Exception as e:
        raise BatchConvertItemError(
            label_path,
            f"{type(e).__name__}: {e}",
        ) from e


def _batch_convert_worker_skip(
    args: tuple[str, str, str, str, str, dict],
) -> tuple[str | None, str, str | None]:
    label_path, image_path, source_fmt, target_format, out_path, load_kwargs = args
    try:
        out = _convert_one_impl(
            label_path,
            image_path,
            source_fmt,
            target_format,
            out_path,
            load_kwargs,
        )
        return (out, label_path, None)
    except Exception as e:
        return (None, label_path, f"{type(e).__name__}: {e}")


def _coerce_sample_pair(item: tuple[str, str] | DatasetSample) -> tuple[str, str]:
    if isinstance(item, tuple):
        if len(item) != 2:
            raise ValueError(
                f"Expected (label_path, image_path) with length 2, got {len(item)}"
            )
        return str(item[0]), str(item[1])
    from ..dataset import DatasetSample

    if isinstance(item, DatasetSample):
        if not item.label_path:
            raise ValueError(
                "DatasetSample.label_path must not be None for batch_convert"
            )
        return item.label_path, item.image_path
    raise TypeError(
        f"Expected tuple[str, str] or DatasetSample, got {type(item).__name__}"
    )


@overload
def batch_convert(
    pairs: Iterable[tuple[str, str] | DatasetSample],
    *,
    source_fmt: str,
    target_format: str,
    out_dir: str,
    max_workers: int | None = None,
    errors: Literal["raise"] = "raise",
    **kwargs,
) -> list[str]: ...


@overload
def batch_convert(
    pairs: Iterable[tuple[str, str] | DatasetSample],
    *,
    source_fmt: str,
    target_format: str,
    out_dir: str,
    max_workers: int | None = None,
    errors: Literal["skip"],
    **kwargs,
) -> tuple[list[str], list[BatchConvertFailure]]: ...


def batch_convert(
    pairs: Iterable[tuple[str, str] | DatasetSample],
    *,
    source_fmt: str,
    target_format: str,
    out_dir: str,
    max_workers: int | None = None,
    errors: Literal["raise", "skip"] = "raise",
    **kwargs,
) -> list[str] | tuple[list[str], list[BatchConvertFailure]]:
    """
    Convert many label/image pairs from ``source_fmt`` to ``target_format`` in one
    output directory.

    This is intended for large jobs (e.g. 100k+ files): the output directory is
    created once (``mkdir(parents=True, exist_ok=True)``), and optional process
    workers run :func:`~dsetkit.annotations.io.load` →
    :func:`~dsetkit.annotations.io.dump` per pair without per-item directory
    creation.

    Parameters
    ----------
    pairs :
        Iterable of ``(label_path, image_path)`` or ``DatasetSample`` instances.
        ``label_path`` must always be set (samples without labels should be
        filtered by the caller).
    source_fmt, target_format :
        Explicit formats (e.g. ``\"yolo\"``, ``\"voc\"``). Required so formats are
        not auto-detected once per file.
    out_dir :
        All outputs are written as ``out_dir / (stem(label_path) + suffix)`` where
        ``suffix`` comes from ``FORMAT_SUFFIXES[target_format]``.
    max_workers :
        ``None`` (default): run sequentially in the current process.
        Integer ``>= 1``: use ``ProcessPoolExecutor`` with that many workers.
        Extra ``**kwargs`` are forwarded to :func:`~dsetkit.annotations.io.load`
        only; they must be picklable when ``max_workers`` is not ``None``.
    errors :
        * ``\"raise\"`` (default): stop on the first failure and raise
          :class:`BatchConvertItemError` with the failing ``label_path`` in the
          message and the original exception as ``__cause__``.
        * ``\"skip\"``: continue; return ``(written_paths, failures)`` where
          ``failures`` is a list of :class:`BatchConvertFailure` (one per failed
          ``label_path``, order follows ``pairs``).

    Returns
    -------
    list[str] or tuple[list[str], list[BatchConvertFailure]]
        With ``errors='raise'``: paths in the same order as ``pairs``. With
        ``errors='skip'``: ``(ok_paths, failures)`` where ``ok_paths`` preserves the
        relative order of successful pairs, and ``failures`` lists
        :class:`BatchConvertFailure` in the order failed pairs were encountered.

    Notes
    -----
    On Windows, multiprocessing requires the usual ``if __name__ == \"__main__\"``
    guard in scripts that call this with ``max_workers`` set.
    """
    if not out_dir:
        raise ValueError("out_dir must be a non-empty string")

    suffix = FORMAT_SUFFIXES.get(target_format)
    if suffix is None:
        raise ValueError(f"Unknown target format: {target_format!r}")

    get_loader(source_fmt)
    get_dumper(target_format)

    _ensure_out_dir(out_dir)

    load_kwargs = dict(kwargs)
    jobs: list[tuple[str, str, str, str, str, dict]] = []

    for item in pairs:
        label_path, image_path = _coerce_sample_pair(item)
        out_path = _out_path_for_label(label_path, out_dir, target_format)
        jobs.append(
            (label_path, image_path, source_fmt, target_format, out_path, load_kwargs)
        )

    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be None or >= 1")

    if errors not in ("raise", "skip"):
        raise ValueError(f"errors must be 'raise' or 'skip', got {errors!r}")

    if max_workers is None:
        return _batch_convert_sequential(jobs, errors=errors)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        if errors == "raise":
            return list(executor.map(_batch_convert_worker_strict, jobs))

        results = list(executor.map(_batch_convert_worker_skip, jobs))

    written: list[str] = []
    failures: list[BatchConvertFailure] = []
    for out_path, label_path, err in results:
        if err is not None:
            failures.append(BatchConvertFailure(label_path=label_path, message=err))
        elif out_path is not None:
            written.append(out_path)
    return written, failures


def _batch_convert_sequential(
    jobs: list[tuple[str, str, str, str, str, dict]],
    *,
    errors: Literal["raise", "skip"],
) -> list[str] | tuple[list[str], list[BatchConvertFailure]]:
    written: list[str] = []
    failures: list[BatchConvertFailure] = []

    for job in jobs:
        label_path, image_path, source_fmt, target_format, out_path, load_kwargs = job
        if errors == "raise":
            try:
                written.append(
                    _convert_one_impl(
                        label_path,
                        image_path,
                        source_fmt,
                        target_format,
                        out_path,
                        load_kwargs,
                    )
                )
            except Exception as e:
                raise BatchConvertItemError(
                    label_path,
                    f"{type(e).__name__}: {e}",
                ) from e
        else:
            try:
                written.append(
                    _convert_one_impl(
                        label_path,
                        image_path,
                        source_fmt,
                        target_format,
                        out_path,
                        load_kwargs,
                    )
                )
            except Exception as e:
                failures.append(
                    BatchConvertFailure(
                        label_path=label_path,
                        message=f"{type(e).__name__}: {e}",
                    )
                )

    if errors == "skip":
        return written, failures
    return written
