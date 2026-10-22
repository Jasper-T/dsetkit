"""Class mapping checks shared by conversion and visualization."""

from .schema import Annotation


def validate_names(names: list[str]) -> None:
    if (
        not isinstance(names, list)
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Class names must be a list of unique nonblank strings")


def validate_class_ids(class_ids, names: list[str], *, context: str) -> None:
    missing = {
        class_id for class_id in class_ids
        if class_id is None or not 0 <= class_id < len(names) or class_id != int(class_id)
    }
    if missing:
        raise ValueError(f"{context} requires names for class IDs: {sorted(missing, key=str)}")


def validate_conversion_mapping(
    source_format: str | None,
    target_format: str,
    names: list[str],
    *,
    class_ids=(),
    categories=(),
) -> None:
    if "yolo" not in (source_format, target_format):
        return
    validate_names(names)
    if source_format == "yolo":
        validate_class_ids(class_ids, names, context=f"Conversion to {target_format}")
    elif target_format == "yolo":
        missing = sorted(set(categories) - set(names))
        if missing:
            raise ValueError(f"Conversion to yolo requires names for categories: {missing}")


def apply_conversion_mapping(
    annotation: Annotation, source_format: str, target_format: str, names: list[str]
) -> None:
    if source_format == "yolo":
        for item in annotation.items:
            item.category = names[int(item.category_id)]
    elif target_format == "yolo":
        for item in annotation.items:
            item.category_id = names.index(item.category)
    if "yolo" in (source_format, target_format):
        annotation.names = list(names)
