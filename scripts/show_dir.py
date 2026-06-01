import os

# 需要忽略的目录
IGNORE_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".idea",
    ".vscode",
    "dist",
    "build"
}

def print_tree(root, prefix=""):
    try:
        items = sorted(os.listdir(root))
    except PermissionError:
        return

    items = [i for i in items if i not in IGNORE_DIRS]

    for index, item in enumerate(items):
        path = os.path.join(root, item)
        is_last = index == len(items) - 1

        connector = "└── " if is_last else "├── "
        print(prefix + connector + item)

        if os.path.isdir(path):
            extension = "    " if is_last else "│   "
            print_tree(path, prefix + extension)


if __name__ == "__main__":
    project_path = "."  # 当前目录
    print(project_path)
    print_tree(project_path)