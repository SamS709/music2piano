"""Helpers for numbered dataset roots such as data/data_0 and data/data_1."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Union

PathLike = Union[str, Path]


def dataset_roots(path: PathLike) -> List[Path]:
    """Return numbered dataset directories contained by ``path``.

    A directory that directly contains YAML metadata is also accepted for
    backwards compatibility.
    """
    root = Path(path)
    if any(root.glob("*.yaml")):
        return [root]
    prefix = root.name + "_"
    candidates = []
    for candidate in root.glob(f"{prefix}*"):
        if not candidate.is_dir():
            continue
        candidate_match = re.fullmatch(rf"{re.escape(prefix)}(\d+)", candidate.name)
        if candidate_match:
            index = int(candidate_match.group(1))
            candidates.append((index, candidate))
    return [candidate for _, candidate in sorted(candidates)]


def metadata_files(path: PathLike) -> List[Path]:
    """Return metadata files from all numbered roots under ``path``."""
    files = []
    for root in dataset_roots(path):
        files.extend(root.glob("*.yaml"))
    return sorted(files)


def next_dataset_root(path: PathLike) -> Path:
    """Return the next numbered child dataset directory."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        candidate = root / f"{root.name}_{index}"
        if not candidate.exists():
            return candidate
        index += 1


def mirrored_root(source_root: PathLike, output_root: PathLike) -> Path:
    """Map data/data_0 to data_clean/data_clean_0."""
    source = Path(source_root)
    output = Path(output_root)
    match = re.fullmatch(r".+_(\d+)", source.name)
    if match is None:
        return output
    return output / f"{output.name}_{match.group(1)}"
