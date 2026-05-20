from pathlib import Path
from typing import Iterable, Union, Mapping

from natsort import natsorted


def merge_txt_files(
    txt_paths: Iterable[Union[str, Path]], output_txt_path: Union[str, Path]
) -> None:
    """Merge multiple txt files into a new txt file in sequence."""
    output_path = Path(output_txt_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out_f:
        for txt_path in txt_paths:
            current_txt_path = Path(txt_path).resolve()
            if not current_txt_path.is_file():
                continue

            with current_txt_path.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)


def build_stem_index(
    directory: Path,
    format_suffixes: dict[str, str],
    format_priority: dict[str, int],
) -> dict[str, Path]:

    directory = Path(directory)
    valid_suffixes = set(format_suffixes.values())

    temp: dict[str, tuple[int, Path]] = {}

    for path in directory.iterdir():

        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if suffix not in valid_suffixes:
            continue

        fmt = next(
            (
                f for f, s in format_suffixes.items()
                if s == suffix
            ),
            None,
        )

        if fmt is None:
            continue

        priority = format_priority.get(fmt, 10_000)

        if (
            path.stem not in temp
            or priority < temp[path.stem][0]
        ):
            temp[path.stem] = (priority, path)

    return {k: v[1] for k, v in temp.items()}