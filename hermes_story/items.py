"""Item ids for hermes_story.

An item names one addressable piece of a film::

    story            the story text
    scene:<i>        one scene
    char:<name>      one character
    sheet:<name>     one character sheet
    shot:<i>.<j>     one shot of a scene
    prompt:<i>.<j>   the prompt for a shot
    frame:<i>.<j>    the first/last frame of a shot
    clip:<i>.<j>     the video clip of a shot

Names are free text (1-60 characters, no control characters). The on-disk
folder is derived from a hash so that any name is a safe single path part.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

SCENE_RE = re.compile(r"(scene):(\d+)")
SHOT_RE = re.compile(r"(shot|prompt|frame|clip):(\d+)\.(\d+)")
NAME_RE = re.compile(r"(char|sheet):(.{1,60})")

STAGE_OF = {
    "story": "script",
    "scene": "script",
    "char": "characters",
    "sheet": "sheets",
    "shot": "storyboard",
    "prompt": "prompts",
    "frame": "frames",
    "clip": "clips",
}


class ItemError(ValueError):
    """An item id is malformed."""


@dataclass(frozen=True)
class Item:
    kind: str
    scene: int | None = None
    shot: int | None = None
    name: str | None = None


def _reject_control(name: str) -> None:
    if any(unicodedata.category(c).startswith("C") for c in name):
        raise ItemError(f"control character in item name: {name!r}")


def parse(item: str) -> Item:
    """Parse an item id, raising :class:`ItemError` when it is malformed."""
    if not isinstance(item, str):
        raise ItemError(f"item must be a string: {item!r}")
    if item == "story":
        return Item("story")

    match = SCENE_RE.fullmatch(item)
    if match:
        return Item("scene", scene=int(match.group(2)))

    match = SHOT_RE.fullmatch(item)
    if match:
        return Item(match.group(1), scene=int(match.group(2)), shot=int(match.group(3)))

    match = NAME_RE.fullmatch(item)
    if match:
        name = match.group(2)
        _reject_control(name)
        return Item(match.group(1), name=name)

    raise ItemError(f"bad item id: {item!r}")


def fmt(item: Item) -> str:
    """Render an item back to its canonical string form."""
    if item.kind == "story":
        return "story"
    if item.kind == "scene":
        return f"scene:{item.scene}"
    if item.kind in ("char", "sheet"):
        return f"{item.kind}:{item.name}"
    if item.kind in ("shot", "prompt", "frame", "clip"):
        return f"{item.kind}:{item.scene}.{item.shot}"
    raise ItemError(f"unknown item kind: {item.kind!r}")


def folder(item: Item) -> str:
    """A safe single path part, unique per item id."""
    if not isinstance(item, Item):
        item = parse(item)
    digest = hashlib.sha1(fmt(item).encode("utf-8")).hexdigest()[:10]
    return f"{item.kind}-{digest}"
