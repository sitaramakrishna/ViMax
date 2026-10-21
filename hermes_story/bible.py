"""The per-project ``bible.json``: options, characters, current versions, revisions."""

from __future__ import annotations

import datetime

from hermes_story import items, project


def _default(pid: str) -> dict:
    return {
        "id": pid,
        "options": {},
        "characters": {},
        "current": {},
        "revisions": [],
    }


def _key(item: str) -> str:
    return items.fmt(items.parse(item))


def load(pid: str) -> dict:
    """Read ``bible.json``, filling in any missing top-level keys."""
    bible = project.read_json(project.path(pid) / "bible.json", None)
    if not isinstance(bible, dict):
        bible = _default(pid)
    for key, value in _default(pid).items():
        bible.setdefault(key, value)
    return bible


def save(pid: str, bible: dict) -> None:
    project.write_json(project.path(pid) / "bible.json", bible)


def set_current(pid: str, item: str, v: int) -> None:
    bible = load(pid)
    bible["current"][_key(item)] = int(v)
    save(pid, bible)


def add_revision(pid: str, item: str, feedback: str, from_v: int, to_v: int) -> None:
    """Record a retake, keeping ``feedback`` verbatim, and point at ``to_v``."""
    key = _key(item)
    bible = load(pid)
    bible["revisions"].append(
        {
            "item": key,
            "feedback": feedback,
            "from": int(from_v),
            "to": int(to_v),
            "at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
    )
    bible["current"][key] = int(to_v)
    save(pid, bible)


def set_cameo(pid: str, character: str, person: str | None) -> None:
    """Attach (or clear, with ``None``) the real person behind a character."""
    bible = load(pid)
    entry = bible["characters"].setdefault(character, {})
    entry["cameo"] = person
    save(pid, bible)
