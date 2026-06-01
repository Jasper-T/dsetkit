from pathlib import Path
from typing import Sequence

from tqdm import tqdm

from .annotations.io import (
    FORMAT_DIRS,
    convert as convert_annotation,
)
from .augment.flip import FlipDirection, _validate_direction, flip_image, flip_label
from .augment.rotate import RotateAngle, _validate_angle, rotate_image, rotate_label
from .dataset import Dataset
from .split import save_split_txts, split_paths
from .utils.file import ensure_dir, save_txt


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


def flip_dataset(
    dataset: Dataset,
    out_dir: str | Path,
    direction: FlipDirection = 1,
    names: list[str] | None = None,
    target_format: str | None = None,
) -> None:
    direction = _validate_direction(direction)
    direction_name = "horizontal" if direction == 1 else "vertical"
    source_format = dataset.source_format

    out_img_dir = Path(out_dir) / "images"
    out_label_dir = (
        Path(out_dir) / FORMAT_DIRS[source_format]
        if dataset.label_dir is not None
        else None
    )
    ensure_dir(out_img_dir)
    if out_label_dir is not None:
        ensure_dir(out_label_dir)

    for sample in tqdm(dataset, desc=f"Flipping ({direction_name})"):
        flip_image(sample.image_path, out_img_dir, direction=direction)
        if (
            sample.label_path is not None
            and sample.label_path.is_file()
            and out_label_dir is not None
        ):
            flip_label(
                sample.label_path,
                sample.image_path,
                out_label_dir,
                source_format=source_format,
                direction=direction,
                names=names,
                target_format=target_format,
            )


def flip_dirs(
    image_dir: str | Path,
    label_dir: str | Path | None,
    source_format: str,
    out_dir: str | Path,
    direction: FlipDirection = 1,
    names: list[str] | None = None,
    target_format: str | None = None,
) -> None:
    dataset = Dataset(
        image_dir=image_dir,
        label_dir=label_dir,
        names=names or [],
        source_format=source_format,
    )
    dataset.build()
    flip_dataset(
        dataset,
        out_dir=out_dir,
        direction=direction,
        names=names,
        target_format=target_format,
    )


def rotate_dataset(
    dataset: Dataset,
    out_dir: str | Path,
    angle: RotateAngle = 90,
    names: list[str] | None = None,
    target_format: str | None = None,
) -> None:
    angle = _validate_angle(angle)
    source_format = dataset.source_format

    out_img_dir = Path(out_dir) / "images"
    out_label_dir = (
        Path(out_dir) / FORMAT_DIRS[source_format]
        if dataset.label_dir is not None
        else None
    )
    ensure_dir(out_img_dir)
    if out_label_dir is not None:
        ensure_dir(out_label_dir)

    for sample in tqdm(dataset, desc=f"Rotating ({angle}° CW)"):
        rotate_image(sample.image_path, out_img_dir, angle=angle)
        if (
            sample.label_path is not None
            and sample.label_path.is_file()
            and out_label_dir is not None
        ):
            rotate_label(
                sample.label_path,
                sample.image_path,
                out_label_dir,
                source_format=source_format,
                angle=angle,
                names=names,
                target_format=target_format,
            )


def rotate_dirs(
    image_dir: str | Path,
    label_dir: str | Path | None,
    source_format: str,
    out_dir: str | Path,
    angle: RotateAngle = 90,
    names: list[str] | None = None,
    target_format: str | None = None,
) -> None:
    dataset = Dataset(
        image_dir=image_dir,
        label_dir=label_dir,
        names=names or [],
        source_format=source_format,
    )
    dataset.build()
    rotate_dataset(
        dataset,
        out_dir=out_dir,
        angle=angle,
        names=names,
        target_format=target_format,
    )


def export_dataset(dataset: Dataset, txt_path: str | Path) -> None:
    save_txt([sample.image_path for sample in dataset], txt_path)


def export_dirs(image_dir: str | Path, txt_path: str | Path) -> None:
    dataset = Dataset(names=[], image_dir=image_dir)
    dataset.build()
    export_dataset(dataset, txt_path)


def split_dataset(
    dataset: Dataset,
    out_dir: str | Path,
    rates: Sequence[float] = (0.75, 0.15, 0.1),
    seed: int = 42,
    add_time: bool = False,
) -> dict[str, Path]:
    splits = split_paths(
        [sample.image_path for sample in dataset],
        rates=rates,
        seed=seed,
    )
    return save_split_txts(splits, out_dir, add_time=add_time)


def split_dirs(
    image_dir: str | Path,
    out_dir: str | Path,
    rates: Sequence[float] = (0.75, 0.15, 0.1),
    seed: int = 42,
    add_time: bool = False,
) -> dict[str, Path]:
    dataset = Dataset(names=[], image_dir=image_dir)
    dataset.build()
    return split_dataset(
        dataset,
        out_dir=out_dir,
        rates=rates,
        seed=seed,
        add_time=add_time,
    )
