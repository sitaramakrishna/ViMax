"""Archive and restore earlier versions of an item's files."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from hermes_story import items, project

VERSION_RE = re.compile(r"v([0-9]+)")


class VersionError(Exception):
    """A requested version does not exist."""


def _folder(pid: str, item: str) -> Path:
    return project.safe_child(project.path(pid), "versions", items.folder(items.parse(item)))


def list(pid: str, item: str) -> list[int]:
    """Return the archived version numbers of ``item``, oldest first."""
    base = _folder(pid, item)
    if not base.is_dir():
        return []
    found = []
    for child in base.iterdir():
        match = VERSION_RE.fullmatch(child.name)
        if child.is_dir() and match:
            found.append(int(match.group(1)))
    return sorted(found)


def archive(pid: str, item: str, files: list[Path]) -> int:
    """Copy the existing ``files`` into the next ``v<N>`` folder and return N."""
    base = _folder(pid, item)
    n = max(list(pid, item), default=0) + 1
    dest = project.safe_child(project.path(pid), "versions", base.name, f"v{n}")
    dest.mkdir(parents=True, mode=0o700)
    for src in files:
        src = Path(src)
        if src.is_file():
            shutil.copy2(src, dest / src.name)
    return n


def restore(pid: str, item: str, n: int, dest_dir: Path) -> list[Path]:
    """Copy the files of ``v<n>`` back into ``dest_dir`` and return their paths."""
    base = _folder(pid, item)
    src_dir = project.safe_child(project.path(pid), "versions", base.name, f"v{int(n)}")
    if not src_dir.is_dir():
        raise VersionError(f"no version {n} for {item!r}")
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    restored = []
    for src in sorted(src_dir.iterdir()):
        if src.is_file():
            target = dest / src.name
            shutil.copy2(src, target)
            restored.append(target)
    return restored
