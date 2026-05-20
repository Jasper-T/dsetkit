from typing import Callable, Dict

LOADERS: Dict[str, Callable] = {}
DUMPERS: Dict[str, Callable] = {}


def register_format(name: str, loader=None, dumper=None):
    if loader:
        LOADERS[name] = loader
    if dumper:
        DUMPERS[name] = dumper


def get_loader(name: str):
    if name not in LOADERS:
        raise ValueError(f"Unknown format: {name}")
    return LOADERS[name]


def get_dumper(name: str):
    if name not in DUMPERS:
        raise ValueError(f"Unknown format: {name}")
    return DUMPERS[name]


def list_formats():
    return list(LOADERS.keys())