from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from .annotations.io import convert as convert_annotation
from .dataset import Dataset
from .utils.file import ensure_dir


def convert_dataset(
    dataset: Dataset,
    target_format: str,
    names: list[str] | None = None,
    out_dir: str | Path | None = None,
) -> None:
    names = names or dataset.names

    for sample in tqdm(dataset, desc="Converting"):
        if sample.label_path is None:
            continue

        convert_annotation(
            label_path=sample.label_path,
            image_path=sample.image_path,
            target_format=target_format,
            source_format=dataset.source_format,
            names=names,
            out_dir=out_dir,
        )


def convert_dirs(
    image_dir: str | Path,
    label_dir: str | Path,
    source_format: str,
    target_format: str,
    names: list[str],
    out_dir: str | Path | None = None,
) -> None:
    dataset = Dataset(
        image_dir=image_dir,
        label_dir=label_dir,
        names=names,
        source_format=source_format,
    )
    dataset.build()
    convert_dataset(dataset, target_format=target_format, names=names, out_dir=out_dir)


def plot_dataset(
    dataset: Dataset,
    names: list[str] | None = None,
    out_dir: str | Path | None = None,
) -> None:
    from .visualize.plot import plot

    names = names or dataset.names

    if out_dir is None:
        out_dir = Path(dataset.image_dir).parent / "annotations"

    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    for sample in tqdm(dataset, desc="Plotting"):
        if sample.label_path is None:
            continue

        plot(
            label_path=sample.label_path,
            image_path=sample.image_path,
            names=names,
            fmt=dataset.source_format,
            save_path=out_dir / sample.image_path.name,
        )


def plot_dirs(
    image_dir: str | Path,
    label_dir: str | Path,
    source_format: str,
    names: list[str],
    out_dir: str | Path | None = None,
) -> None:
    dataset = Dataset(
        image_dir=image_dir,
        label_dir=label_dir,
        names=names,
        source_format=source_format,
    )
    dataset.build()
    plot_dataset(dataset, names=names, out_dir=out_dir)


__all__ = [
    "convert_dataset",
    "convert_dirs",
    "plot_dataset",
    "plot_dirs",
]
