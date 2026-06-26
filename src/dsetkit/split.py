import random
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .utils.file import load_txt, save_txt
from .utils.image import get_image_paths

def _validate_rates(rates: Sequence[float]) -> None:
    if len(rates) not in (1, 2, 3):
        raise ValueError("rates must contain train, train/val, or train/val/test ratios.")
    if any(rate <= 0 for rate in rates):
        raise ValueError("All split ratios must be positive.")
    if sum(rates) > 1.0:
        raise ValueError("Split ratios must sum to 1.0 or less.")
    if len(rates) == 1 and rates[0] >= 1.0:
        raise ValueError("A single train ratio must be less than 1.0 so val can use the remainder.")


def split_paths(
    image_paths: Sequence[Path],
    rates: Sequence[float] = (0.75, 0.15, 0.1),
    seed: int = 42,
) -> dict[str, list[Path]]:
    """Shuffle image paths and split them into train/val/test buckets."""
    _validate_rates(rates)

    if not image_paths:
        raise ValueError("No image paths provided.")

    shuffled = list(image_paths)
    random.Random(seed).shuffle(shuffled)

    train_end = int(len(shuffled) * rates[0])
    splits: dict[str, list[Path]] = {"train": shuffled[:train_end]}

    if len(rates) == 1:
        splits["val"] = shuffled[train_end:]
    else:
        val_end = train_end + int(len(shuffled) * rates[1])
        splits["val"] = shuffled[train_end:val_end]
        splits["test"] = shuffled[val_end:]

    return splits


def save_split_txts(
    splits: dict[str, list[Path]],
    out_dir: str | Path,
    add_time: bool = False,
) -> dict[str, Path]:
    """Write split path lists to train/val/test txt files."""
    out_dir = Path(out_dir)
    date_suffix = f"_{datetime.now().strftime('%Y%m%d')}" if add_time else ""

    output_paths: dict[str, Path] = {}
    for split_name, paths in splits.items():
        txt_path = out_dir / f"{split_name}{date_suffix}.txt"
        save_txt(paths, txt_path)
        output_paths[split_name] = txt_path

    return output_paths


def split_tvt(
    dataset_root: str | Path,
    txt_file_name: str,
    rates: Sequence[float] = (0.8, 0.2),
    seed: int = 42,
    add_time: bool = False,
) -> dict[str, list[Path]]:
    """Split image paths into train/val/test buckets and save the split txt files."""
    dataset_root = Path(dataset_root)
    txt_path = dataset_root / txt_file_name

    if txt_path.is_file():
        image_paths = [Path(path) for path in load_txt(txt_path)] 
    else:
        image_dir = dataset_root / "images"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Images dir not found: {image_dir}")

        image_paths = get_image_paths(image_dir)
        save_txt(image_paths, txt_path)
    
    if not image_paths:
        raise SystemExit("No images found. Please check the images directory.")

    splits = split_paths(image_paths, rates=rates, seed=seed)
    save_split_txts(splits, dataset_root, add_time=add_time)
    return splits

