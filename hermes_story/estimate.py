"""Cost estimates for hermes_story stages.

Every estimate is read from the files already in the project's ``work/``
folder, so it costs nothing and never calls a model.  Before the prompts exist
the frames and clips estimates fall back to the storyboard's shot count and
assume every shot is a ``medium`` variation (the most expensive case).
"""

from __future__ import annotations

from pathlib import Path

from hermes_story import items as item_ids, project, state

#: Variation types that need both a first and a last frame.
VARIATION_WITH_LAST = ("medium", "large")


def _work(pid: str) -> Path:
    return project.path(pid) / "work"


def _json(pid: str, *parts: str):
    return project.read_json(_work(pid).joinpath(*parts), None)


def characters(pid: str) -> list[dict]:
    """The character entries in ``characters.json`` (empty before the stage)."""
    raw = _json(pid, "characters.json")
    if not isinstance(raw, list):
        return []
    return [entry for entry in raw if isinstance(entry, dict)]


def scenes(pid: str) -> list[int]:
    """Scene indices, from ``script.json`` when present, else the folders."""
    script = _json(pid, "script.json")
    if isinstance(script, list):
        return list(range(len(script)))
    found = []
    for child in _work(pid).glob("scene_*"):
        suffix = child.name[len("scene_"):]
        if child.is_dir() and suffix.isdigit():
            found.append(int(suffix))
    return sorted(found)


def storyboard(pid: str, scene: int) -> list[dict] | None:
    """The scene's storyboard entries, or ``None`` when it does not exist."""
    raw = _json(pid, f"scene_{scene}", "storyboard.json")
    if not isinstance(raw, list):
        return None
    return [entry for entry in raw if isinstance(entry, dict)]


def shot_description(pid: str, scene: int, shot: int) -> dict | None:
    raw = _json(pid, f"scene_{scene}", "shots", str(shot), "shot_description.json")
    return raw if isinstance(raw, dict) else None


def cameras(pid: str, scene: int) -> list[dict]:
    raw = _json(pid, f"scene_{scene}", "camera_tree.json")
    if not isinstance(raw, list):
        return []
    return [entry for entry in raw if isinstance(entry, dict)]


def _shot_dir_indexes(pid: str, scene: int) -> list[int]:
    shots = _work(pid) / f"scene_{scene}" / "shots"
    if not shots.is_dir():
        return []
    return sorted(
        int(child.name)
        for child in shots.iterdir()
        if child.is_dir() and child.name.isdigit()
    )


def shot_indexes(pid: str, scene: int) -> list[int]:
    """Shot indices from the storyboard, else from the ``shots/`` folders."""
    briefs = storyboard(pid, scene)
    if briefs is not None:
        return [
            int(entry["idx"])
            for entry in briefs
            if isinstance(entry.get("idx"), int)
        ]
    return _shot_dir_indexes(pid, scene)


def has_parent(camera: dict) -> bool:
    """A camera that needs a transition video from its parent shot."""
    return (
        camera.get("parent_cam_idx") is not None
        or camera.get("parent_shot_idx") is not None
    )


# -- paid stages ---------------------------------------------------------


def sheets(pid: str, cfg) -> float:
    """One visible character needs three image calls (front, side, back)."""
    visible = [entry for entry in characters(pid) if entry.get("is_visible", True)]
    return len(visible) * 3 * cfg.price_image_call


def frames(pid: str, cfg) -> float:
    """Candidate images per shot, plus a frame and a transition video per child camera."""
    total = 0.0
    for scene in scenes(pid):
        for shot in shot_indexes(pid, scene):
            description = shot_description(pid, scene, shot)
            if description is None:
                # Before the prompts exist: assume a medium variation.
                candidates = cfg.num_image_candidates * 2
            elif description.get("variation_type") in VARIATION_WITH_LAST:
                candidates = cfg.num_image_candidates * 2
            else:
                candidates = cfg.num_image_candidates
            total += candidates * cfg.price_image_call
        for camera in cameras(pid, scene):
            if has_parent(camera):
                total += cfg.price_image_call
                total += cfg.video_duration_s * cfg.price_video_second
    return total


def clips(pid: str, cfg) -> float:
    """One video per shot, priced per second."""
    shots = sum(len(shot_indexes(pid, scene)) for scene in scenes(pid))
    return shots * cfg.video_duration_s * cfg.price_video_second


# -- text stages ---------------------------------------------------------


def _has_storyboard(pid: str, scene: int) -> bool:
    return storyboard(pid, scene) is not None


def _prompts_done(pid: str, scene: int) -> bool:
    shots = shot_indexes(pid, scene)
    return bool(shots) and all(
        shot_description(pid, scene, shot) is not None for shot in shots
    )


def text_calls(pid: str, stage: str) -> int:
    """How many chat calls ``stage`` still has to make, given the files on disk."""
    work = _work(pid)
    if stage == "script":
        calls = 0
        if not (work / "story.txt").is_file():
            calls += 1
        if not (work / "script.json").is_file():
            calls += 1
        return calls
    if stage == "characters":
        return 0 if (work / "characters.json").is_file() else 1
    if stage == "storyboard":
        return sum(1 for scene in scenes(pid) if not _has_storyboard(pid, scene))
    if stage == "prompts":
        return sum(1 for scene in scenes(pid) if not _prompts_done(pid, scene))
    return 0


def text(pid: str, stage: str, cfg) -> float:
    return text_calls(pid, stage) * cfg.price_chat_call


# -- public API ----------------------------------------------------------


def stage_estimate(pid: str, stage: str, cfg) -> float:
    """Estimated cost in USD of running ``stage`` from the current files."""
    if stage == "sheets":
        return sheets(pid, cfg)
    if stage == "frames":
        return frames(pid, cfg)
    if stage == "clips":
        return clips(pid, cfg)
    if stage in ("script", "characters", "storyboard", "prompts"):
        return text(pid, stage, cfg)
    raise ValueError(f"unknown stage: {stage!r}")


def film_estimate(pid: str, cfg) -> dict[str, float]:
    """Estimated cost of every stage, in stage order."""
    return {stage: stage_estimate(pid, stage, cfg) for stage in state.STAGES}


def item_estimate(pid: str, item: str, cfg) -> float:
    """Estimated cost of retaking one item."""
    kind = item_ids.parse(item).kind
    if kind == "sheet":
        return 3 * cfg.price_image_call
    if kind == "frame":
        return cfg.num_image_candidates * cfg.price_image_call
    if kind == "clip":
        return cfg.video_duration_s * cfg.price_video_second
    return cfg.price_chat_call
