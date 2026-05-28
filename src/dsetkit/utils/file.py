from pathlib import Path
from typing import Iterable, Union, Mapping

from natsort import natsorted


def find_parent_dirs_with_subfolder(
    root_dir: str | Path,
    target_folder_name: str,
    exclude_empty: bool = False
) -> list[Path]:
    """
    Recursively search under root_dir for directories containing target_folder_name,
    and return their parent directories.

    Args:
        root_dir (str): Root directory to search in.
        target_folder_name (str): Name of the target subdirectory to match.
        exclude_empty (bool): If True, exclude empty target folders.

    Returns:
        list[Path]: A list of unique parent directory Paths.
    """
    root = Path(root_dir)
    result = set()

    for current_dir in root.rglob("*"):
        if current_dir.is_dir() and current_dir.name == target_folder_name:

            if exclude_empty and not any(current_dir.iterdir()):
                continue

            result.add(current_dir.parent.resolve())

    return list(result)


def load_txt(txt_path: str | Path) -> list:
    txt_path = Path(txt_path).resolve()
    if not txt_path.is_file():
        raise FileNotFoundError(f"txt not exists: {txt_path}")
    with txt_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def save_txt(paths: list, txt_path: str | Path):
    txt_path = Path(txt_path).resolve()
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w", encoding="utf-8") as f:
        for path in paths:
            f.write(str(path) + "\n")


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


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


def replace_text_in_txt(
    txt_path: Union[str, Path],
    old_text: str,
    new_text: str,
    output_txt_path: Union[str, Path, None] = None,
) -> None:
    """替换 txt 中的指定字符/字符串，可原地替换或输出到新文件。"""
    src_path = Path(txt_path).resolve()
    if not src_path.is_file():
        return

    target_path = Path(output_txt_path).resolve() if output_txt_path else src_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    content = src_path.read_text(encoding="utf-8")
    replaced_content = content.replace(old_text, new_text)
    target_path.write_text(replaced_content, encoding="utf-8")


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