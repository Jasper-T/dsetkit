import os
import argparse


DEFAULT_IGNORE_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".idea",
    ".vscode",
    "dist",
    "build",
}


def print_tree(root: str, prefix: str = "", ignore_dirs: set[str] | None = None) -> None:
    if ignore_dirs is None:
        ignore_dirs = DEFAULT_IGNORE_DIRS

    try:
        items = sorted(os.listdir(root))
    except PermissionError:
        return
    except FileNotFoundError:
        raise FileNotFoundError(f"路径不存在: {root}")

    items = [i for i in items if i not in ignore_dirs]

    for index, item in enumerate(items):
        path = os.path.join(root, item)
        is_last = index == len(items) - 1

        connector = "└── " if is_last else "├── "
        print(prefix + connector + item)

        if os.path.isdir(path):
            extension = "    " if is_last else "│   "
            print_tree(path, prefix + extension, ignore_dirs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="打印目录树结构"
    )

    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="要打印的目录路径，默认当前目录"
    )

    parser.add_argument(
        "--ignore",
        nargs="*",
        default=list(DEFAULT_IGNORE_DIRS),
        help="需要忽略的目录名"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    project_path = os.path.abspath(args.path)
    ignore_dirs = set(args.ignore)

    print(project_path)
    print_tree(project_path, ignore_dirs=ignore_dirs)