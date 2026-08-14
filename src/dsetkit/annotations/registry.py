from dataclasses import dataclass
from typing import Callable, Dict

LOADERS: Dict[str, Callable] = {}
DUMPERS: Dict[str, Callable] = {}
FORMAT_SPECS: Dict[str, "FormatSpec"] = {}


@dataclass(frozen=True, slots=True)
class FormatSpec:
    name: str
    suffix: str
    label_dir_name: str
    discovery_dir_names: tuple[str, ...]


def register_format(
    name: str,
    loader=None,
    dumper=None,
    *,
    suffix: str | None = None,
    label_dir_name: str | None = None,
    discovery_dir_names: tuple[str, ...] = (),
):
    name = name.lower()
    if loader:
        LOADERS[name] = loader
    if dumper:
        DUMPERS[name] = dumper

    metadata = (suffix, label_dir_name, discovery_dir_names)
    if any(value for value in metadata):
        if suffix is None or label_dir_name is None:
            raise ValueError(
                "suffix and label_dir_name are required when registering format metadata"
            )
        normalized_suffix = suffix.lower()
        if not normalized_suffix.startswith("."):
            normalized_suffix = f".{normalized_suffix}"
        directory_names = discovery_dir_names or (label_dir_name,)
        FORMAT_SPECS[name] = FormatSpec(
            name=name,
            suffix=normalized_suffix,
            label_dir_name=label_dir_name,
            discovery_dir_names=tuple(directory_names),
        )


def get_loader(name: str):
    if name not in LOADERS:
        raise ValueError(f"Unknown format: {name}")
    return LOADERS[name]


def get_dumper(name: str):
    if name not in DUMPERS:
        raise ValueError(f"Unknown format: {name}")
    return DUMPERS[name]


def get_format_spec(name: str) -> FormatSpec:
    name = name.lower()
    if name not in FORMAT_SPECS:
        raise ValueError(f"Unknown format metadata: {name}")
    return FORMAT_SPECS[name]


def list_formats():
    return list(LOADERS.keys())


def list_format_specs() -> list[FormatSpec]:
    return list(FORMAT_SPECS.values())
