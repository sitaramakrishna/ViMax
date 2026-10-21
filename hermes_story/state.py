"""The per-project ``state.json``: stage and item statuses, and gating."""

from __future__ import annotations

from hermes_story import items, project

STAGES = ["script", "characters", "sheets", "storyboard", "prompts", "frames", "clips"]
PAID = ["sheets", "frames", "clips"]
STATUSES = ["pending", "running", "review", "approved", "failed"]


class StateError(ValueError):
    """An unknown stage, status or malformed state."""


class NotReady(Exception):
    """An earlier stage has not been approved yet."""


def _default() -> dict:
    return {"stages": {}, "items": {}, "errors": []}


def load(pid: str) -> dict:
    """Read ``state.json``, filling in any missing top-level keys."""
    state = project.read_json(project.path(pid) / "state.json", None)
    if not isinstance(state, dict):
        state = _default()
    for key, value in _default().items():
        state.setdefault(key, value)
    return state


def save(pid: str, state: dict) -> None:
    project.write_json(project.path(pid) / "state.json", state)


def _check_stage(stage: str) -> None:
    if stage not in STAGES:
        raise StateError(f"unknown stage: {stage!r}")


def _check_status(status: str) -> None:
    if status not in STATUSES:
        raise StateError(f"unknown status: {status!r}")


def _record_error(state: dict, kind: str, name: str, error: str) -> None:
    state["errors"].append({"kind": kind, "name": name, "error": error})


def set_stage(pid: str, stage: str, status: str, error: str | None = None) -> None:
    _check_stage(stage)
    _check_status(status)
    state = load(pid)
    entry: dict = {"status": status}
    if error is not None:
        entry["error"] = error
        _record_error(state, "stage", stage, error)
    state["stages"][stage] = entry
    save(pid, state)


def set_item(pid: str, item: str, status: str, error: str | None = None) -> None:
    _check_status(status)
    key = items.fmt(items.parse(item))
    state = load(pid)
    entry: dict = {"status": status}
    if error is not None:
        entry["error"] = error
        _record_error(state, "item", key, error)
    state["items"][key] = entry
    save(pid, state)


def stage_status(pid: str, stage: str) -> str:
    """Status of ``stage``, defaulting to ``pending`` when never set."""
    entry = load(pid)["stages"].get(stage)
    if isinstance(entry, dict):
        return entry.get("status", "pending")
    if isinstance(entry, str):
        return entry
    return "pending"


def item_status(pid: str, item: str) -> str:
    key = items.fmt(items.parse(item))
    entry = load(pid)["items"].get(key)
    if isinstance(entry, dict):
        return entry.get("status", "pending")
    if isinstance(entry, str):
        return entry
    return "pending"


def require_ready(pid: str, stage: str) -> None:
    """Raise :class:`NotReady` unless every earlier stage is approved."""
    _check_stage(stage)
    index = STAGES.index(stage)
    earlier = STAGES[:index]
    if not earlier:
        return
    state = load(pid)
    not_approved = []
    for name in earlier:
        entry = state["stages"].get(name)
        if isinstance(entry, dict):
            status = entry.get("status", "pending")
        elif isinstance(entry, str):
            status = entry
        else:
            status = "pending"
        if status != "approved":
            not_approved.append(name)
    if not_approved:
        raise NotReady(f"{stage} needs approved: {', '.join(not_approved)}")
