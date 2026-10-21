"""The ViMax-backed stages: text, character sheets and first frames.

Each paid-for ViMax call is guarded so the ledger can attribute it to an item,
even though the text stages themselves only spend chat tokens.  ViMax's own
pipeline methods load their output file when it already exists, so re-running
an approved stage is free and simply rebuilds the :class:`StageResult` from
disk.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from agents.character_portraits_generator import (
    CharacterPortraitsGenerator,
    prompt_template_front,
)
from hermes_story import bible, guarded, project, state
from hermes_story import items as item_ids
from interfaces import CharacterInScene, ShotBriefDescription, ShotDescription
from utils.text import safe_path_component

DEFAULT_DURATION_S = 8

#: Appended to the front-portrait prompt when a real person is the model.
CAMEO_FRONT_RULE = "Keep the face and identity of the person in the reference photos."

#: Photo extensions accepted from a cameo folder.
CAMEO_PHOTO_EXTS = (".jpg", ".jpeg", ".png")

#: Storyboard-only rule: dialogue is added later, so keep mouths out of frame.
DIALOGUE_RULE = (
    "Dialogue shots: prefer framings where speaking mouths are not clearly "
    "visible (over-the-shoulder, wide shot, back view, silhouette, voice-over "
    "on a reaction shot). Voices are added later; video has no sound."
)


@dataclass
class StageResult:
    """What one stage produced: items, cost and any error messages."""

    stage: str
    items: list[dict] = field(default_factory=list)
    cost_usd: float = 0.0
    errors: list[str] = field(default_factory=list)


def user_requirement(bible_data: dict, for_stage: str) -> str:
    """The requirement text handed to ViMax, built from the film's options."""
    options = bible_data.get("options") or {}
    style = options.get("style") or "cinematic"
    shape = options.get("shape") or "16:9"
    length_s = int(options.get("length_s") or 0)
    duration_s = int(options.get("duration_s") or DEFAULT_DURATION_S)
    shots = length_s // duration_s if duration_s > 0 else 0
    text = (
        f"Style: {style}. Aspect ratio: {shape}. "
        f"Total length about {length_s} seconds, so about {shots} shots "
        f"of {duration_s} seconds across all scenes."
    )
    if for_stage == "storyboard":
        text += "\n" + DIALOGUE_RULE
    return text


# -- reading the working tree -------------------------------------------


def _work(engine) -> Path:
    return engine.project_dir / "work"


def _relative(engine, path: Path) -> str:
    return Path(path).relative_to(engine.project_dir).as_posix()


def _row(engine, item: str, text: str, paths: tuple[Path, ...] = ()) -> dict:
    return {
        "item": item,
        "version": 0,
        "kind": "text",
        "text": text,
        "paths": [_relative(engine, path) for path in paths],
    }


def _scene_indexes(engine) -> list[int]:
    """Scene indices, from ``script.json`` when present, else the folders."""
    script = project.read_json(_work(engine) / "script.json", None)
    if isinstance(script, list):
        return list(range(len(script)))
    found = []
    for child in _work(engine).glob("scene_*"):
        suffix = child.name[len("scene_"):]
        if child.is_dir() and suffix.isdigit():
            found.append(int(suffix))
    return sorted(found)


def _load_script(engine) -> list:
    script = project.read_json(_work(engine) / "script.json", None)
    if not isinstance(script, list):
        raise FileNotFoundError("no script.json in the project work dir")
    return script


def _load_characters(engine) -> list[CharacterInScene]:
    raw = project.read_json(_work(engine) / "characters.json", None)
    if not isinstance(raw, list):
        raise FileNotFoundError("no characters.json in the project work dir")
    return [CharacterInScene.model_validate(entry) for entry in raw]


def _script_rows(engine) -> list[dict]:
    work = _work(engine)
    rows = []
    story_path = work / "story.txt"
    if story_path.is_file():
        rows.append(
            _row(
                engine,
                "story",
                story_path.read_text(encoding="utf-8"),
                paths=(story_path,),
            )
        )
    script_path = work / "script.json"
    script = project.read_json(script_path, None)
    if isinstance(script, list):
        for i, text in enumerate(script):
            rows.append(_row(engine, f"scene:{i}", str(text), paths=(script_path,)))
    return rows


def _character_rows(engine) -> list[dict]:
    characters_path = _work(engine) / "characters.json"
    rows = []
    for character in _load_characters(engine):
        rows.append(
            _row(
                engine,
                f"char:{character.identifier_in_scene}",
                str(character),
                paths=(characters_path,),
            )
        )
    return rows


def _storyboard_path(engine, scene: int) -> Path:
    return _work(engine) / f"scene_{scene}" / "storyboard.json"


def _storyboard_rows(engine) -> list[dict]:
    rows = []
    for scene in _scene_indexes(engine):
        path = _storyboard_path(engine, scene)
        raw = project.read_json(path, None)
        if not isinstance(raw, list):
            continue
        for entry in raw:
            brief = ShotBriefDescription.model_validate(entry)
            text = brief.visual_desc
            if brief.audio_desc:
                text += "\n" + brief.audio_desc
            rows.append(_row(engine, f"shot:{scene}.{brief.idx}", text, paths=(path,)))
    return rows


def _prompt_rows(engine) -> list[dict]:
    rows = []
    for scene in _scene_indexes(engine):
        raw = project.read_json(_storyboard_path(engine, scene), None)
        if not isinstance(raw, list):
            continue
        for entry in raw:
            brief = ShotBriefDescription.model_validate(entry)
            path = (
                _work(engine)
                / f"scene_{scene}"
                / "shots"
                / str(brief.idx)
                / "shot_description.json"
            )
            data = project.read_json(path, None)
            if not isinstance(data, dict):
                continue
            shot = ShotDescription.model_validate(data)
            text = f"{shot.ff_desc}\n{shot.motion_desc}"
            rows.append(_row(engine, f"prompt:{scene}.{shot.idx}", text, paths=(path,)))
    return rows


def _registry_path(engine) -> Path:
    return _work(engine) / "character_portraits_registry.json"


def _load_registry(engine) -> dict:
    """ViMax's ``character_portraits_registry.json``, or ``{}`` when absent."""
    raw = project.read_json(_registry_path(engine), None)
    return raw if isinstance(raw, dict) else {}


def _sheet_rows(engine) -> list[dict]:
    rows = []
    for name, views in _load_registry(engine).items():
        if not isinstance(views, dict):
            continue
        paths = []
        for view in ("front", "side", "back"):
            entry = views.get(view)
            if isinstance(entry, dict) and entry.get("path"):
                paths.append(Path(entry["path"]))
        if not paths:
            continue
        row = _row(engine, f"sheet:{name}", "", paths=tuple(paths))
        row["kind"] = "album"
        rows.append(row)
    return rows


def _shot_indexes(engine, scene: int) -> list[int]:
    raw = project.read_json(_storyboard_path(engine, scene), None)
    if isinstance(raw, list):
        return [ShotBriefDescription.model_validate(entry).idx for entry in raw]
    shots_dir = _work(engine) / f"scene_{scene}" / "shots"
    if not shots_dir.is_dir():
        return []
    return sorted(
        int(child.name)
        for child in shots_dir.iterdir()
        if child.is_dir() and child.name.isdigit()
    )


def _frame_rows(engine, scene: int | None = None) -> list[dict]:
    scenes = [scene] if scene is not None else _scene_indexes(engine)
    rows = []
    for i in scenes:
        for j in _shot_indexes(engine, i):
            folder = _work(engine) / f"scene_{i}" / "shots" / str(j)
            first = folder / "first_frame.png"
            if not first.is_file():
                continue
            paths = [first]
            last = folder / "last_frame.png"
            if last.is_file():
                paths.append(last)
            row = _row(engine, f"frame:{i}.{j}", "", paths=tuple(paths))
            row["kind"] = "image"
            rows.append(row)
    return rows


# -- cameo photos --------------------------------------------------------


class CameoError(Exception):
    """A cameo person has no usable consent or photos."""


def cameo_photos(person: str) -> list[Path]:
    """Photos of ``person`` from ``$HERMES_STORY_CAMEO_DIR/<person>/``.

    The folder must be accompanied by a ``consent.json`` object; without one
    the cameo fails closed.  At most three photos are returned, sorted by name.
    """
    base = os.environ.get("HERMES_STORY_CAMEO_DIR", "").strip()
    if not base:
        raise CameoError("HERMES_STORY_CAMEO_DIR is not set")
    name = str(person or "").strip()
    if (
        not name
        or name in (".", "..")
        or "/" in name
        or "\\" in name
        or "\x00" in name
    ):
        raise CameoError(f"bad cameo person: {person!r}")
    folder = Path(base).expanduser() / name
    if not folder.is_dir():
        raise CameoError(f"no cameo folder for {name!r}: {folder}")
    consent = folder / "consent.json"
    if not consent.is_file():
        raise CameoError(f"no consent.json for cameo person {name!r}")
    try:
        payload = json.loads(consent.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CameoError(f"bad consent.json for cameo person {name!r}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CameoError(f"bad consent.json for cameo person {name!r}: not an object")
    photos = sorted(
        (
            child
            for child in folder.iterdir()
            if child.is_file() and child.suffix.lower() in CAMEO_PHOTO_EXTS
        ),
        key=lambda child: child.name,
    )
    if not photos:
        raise CameoError(f"no photos for cameo person {name!r}")
    return photos[:3]


# -- stages --------------------------------------------------------------


async def _script(engine) -> list[dict]:
    data = bible.load(engine.pid)
    options = data.get("options") or {}
    requirement = user_requirement(data, "script")
    idea = engine.idea()
    with guarded.item("story"):
        story = await idea.develop_story(
            idea=options["idea"], user_requirement=requirement, quiet=True
        )
    with guarded.item("scene:0"):
        await idea.write_script_based_on_story(
            story=story, user_requirement=requirement, quiet=True
        )
    return _script_rows(engine)


async def _characters(engine) -> list[dict]:
    story = (_work(engine) / "story.txt").read_text(encoding="utf-8")
    with guarded.item("char:characters"):
        await engine.idea().extract_characters(story=story, quiet=True)

    rows = _character_rows(engine)
    data = bible.load(engine.pid)
    known = data.setdefault("characters", {})
    for row in rows:
        name = item_ids.parse(row["item"]).name
        if name not in known:
            known[name] = {"cameo": None, "voice": None}
    bible.save(engine.pid, data)
    return rows


async def _storyboard(engine) -> list[dict]:
    requirement = user_requirement(bible.load(engine.pid), "storyboard")
    characters = _load_characters(engine)
    for scene, text in enumerate(_load_script(engine)):
        pipeline = engine.scene(scene)
        with guarded.item(f"shot:{scene}.0"):
            await pipeline.design_storyboard(
                script=text,
                characters=characters,
                user_requirement=requirement,
                quiet=True,
            )
    return _storyboard_rows(engine)


async def _prompts(engine) -> list[dict]:
    requirement = user_requirement(bible.load(engine.pid), "prompts")
    characters = _load_characters(engine)
    for scene, text in enumerate(_load_script(engine)):
        pipeline = engine.scene(scene)
        storyboard = await pipeline.design_storyboard(
            script=text,
            characters=characters,
            user_requirement=requirement,
            quiet=True,
        )
        with guarded.item(f"prompt:{scene}.0"):
            await pipeline.decompose_visual_descriptions(
                shot_brief_descriptions=storyboard,
                characters=characters,
                quiet=True,
            )
    return _prompt_rows(engine)


def _cameo_of(engine, name: str) -> str | None:
    entry = (bible.load(engine.pid).get("characters") or {}).get(name)
    if not isinstance(entry, dict):
        return None
    person = entry.get("cameo")
    return person if isinstance(person, str) and person.strip() else None


async def _front_portrait(
    engine,
    generator: CharacterPortraitsGenerator,
    character: CharacterInScene,
    style: str,
    path: Path,
) -> None:
    """Generate the front view, using consented photos for a cameo person."""
    person = _cameo_of(engine, character.identifier_in_scene)
    if person is None:
        output = await generator.generate_front_portrait(character, style)
    else:
        photos = cameo_photos(person)
        features = (
            "(static) "
            + (character.static_features or "")
            + "; (dynamic) "
            + (character.dynamic_features or "")
        )
        prompt = prompt_template_front.format(
            identifier=character.identifier_in_scene,
            features=features,
            style=style,
        )
        prompt = prompt.rstrip() + "\n" + CAMEO_FRONT_RULE
        output = await engine.image.generate_single_image(
            prompt=prompt, reference_image_paths=[str(photo) for photo in photos]
        )
    output.save(str(path))


async def _sheets(engine) -> list[dict]:
    data = bible.load(engine.pid)
    style = (data.get("options") or {}).get("style") or "cinematic"
    characters = _load_characters(engine)
    generator = CharacterPortraitsGenerator(image_generator=engine.image)
    registry = _load_registry(engine)
    root = _work(engine) / "character_portraits"
    root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for character in characters:
        if not character.is_visible:
            continue
        name = character.identifier_in_scene
        folder = root / f"{character.idx}_{safe_path_component(name)}"
        folder.mkdir(parents=True, exist_ok=True)
        front_path = folder / "front.png"
        side_path = folder / "side.png"
        back_path = folder / "back.png"
        errors: list[str] = []

        with guarded.item(f"sheet:{name}"):
            try:
                if not front_path.is_file():
                    await _front_portrait(engine, generator, character, style, front_path)
            except Exception as exc:
                errors.append(str(exc) or type(exc).__name__)

            if front_path.is_file():
                for path, view in ((side_path, "side"), (back_path, "back")):
                    if path.is_file():
                        continue
                    try:
                        if view == "side":
                            output = await generator.generate_side_portrait(
                                character, str(front_path)
                            )
                        else:
                            output = await generator.generate_back_portrait(
                                character, str(front_path)
                            )
                        output.save(str(path))
                    except Exception as exc:
                        errors.append(str(exc) or type(exc).__name__)

        views = {}
        for view, path in (
            ("front", front_path),
            ("side", side_path),
            ("back", back_path),
        ):
            if path.is_file():
                views[view] = {
                    "path": str(path),
                    "description": f"A {view} view portrait of {name}.",
                }
        if views:
            registry[name] = views
            project.write_json(_registry_path(engine), registry)

        row = _row(
            engine,
            f"sheet:{name}",
            "",
            paths=tuple(
                Path(views[view]["path"])
                for view in ("front", "side", "back")
                if view in views
            ),
        )
        row["kind"] = "album"
        if errors:
            row["_error"] = "; ".join(errors)
        rows.append(row)

    project.write_json(_registry_path(engine), registry)
    return rows


async def _frames(engine) -> list[dict]:
    characters = _load_characters(engine)
    registry_file = _registry_path(engine)
    if not registry_file.is_file():
        raise FileNotFoundError("no character_portraits_registry.json in the project work dir")
    registry = _load_registry(engine)
    requirement = user_requirement(bible.load(engine.pid), "frames")

    rows: list[dict] = []
    for scene, text in enumerate(_load_script(engine)):
        pipeline = engine.scene(scene)
        storyboard = await pipeline.design_storyboard(
            script=text,
            characters=characters,
            user_requirement=requirement,
            quiet=True,
        )
        shot_descriptions = await pipeline.decompose_visual_descriptions(
            shot_brief_descriptions=storyboard,
            characters=characters,
            quiet=True,
        )
        camera_tree = await pipeline.construct_camera_tree(
            shot_descriptions=shot_descriptions, quiet=True
        )
        priority_shot_idxs = [
            camera.parent_shot_idx
            for camera in camera_tree
            if camera.parent_shot_idx is not None
        ]

        async def run_camera(camera, _pipeline=pipeline, _scene=scene):
            first = camera.active_shot_idxs[0] if camera.active_shot_idxs else 0
            with guarded.item(f"frame:{_scene}.{first}"):
                await _pipeline.generate_frames_for_single_camera(
                    camera=camera,
                    shot_descriptions=shot_descriptions,
                    characters=characters,
                    character_portraits_registry=registry,
                    priority_shot_idxs=priority_shot_idxs,
                )

        # ViMax starts every camera at once; the ledger caps the spend.
        await asyncio.gather(*(run_camera(camera) for camera in camera_tree))
        rows.extend(_frame_rows(engine, scene))
    return rows


_STAGES = {
    "script": _script,
    "characters": _characters,
    "sheets": _sheets,
    "storyboard": _storyboard,
    "prompts": _prompts,
    "frames": _frames,
}

_READERS = {
    "script": _script_rows,
    "characters": _character_rows,
    "sheets": _sheet_rows,
    "storyboard": _storyboard_rows,
    "prompts": _prompt_rows,
    "frames": _frame_rows,
}


# -- bookkeeping ---------------------------------------------------------


def _stamp(pid: str, rows: list[dict], *, save: bool) -> None:
    """Fill in each row's version, recording 1 for items never seen before."""
    data = bible.load(pid)
    current = data.setdefault("current", {})
    changed = False
    for row in rows:
        key = item_ids.fmt(item_ids.parse(row["item"]))
        if key not in current:
            current[key] = 1
            changed = True
        row["version"] = int(current.get(key, 1))
    if save and changed:
        bible.save(pid, data)


def _finish(engine, stage: str, rows: list[dict]) -> list[str]:
    """Stamp rows, set their item statuses and collect per-item errors."""
    _stamp(engine.pid, rows, save=True)
    errors: list[str] = []
    for row in rows:
        error = row.pop("_error", None)
        if error:
            errors.append(error)
        state.set_item(engine.pid, row["item"], "failed" if error else "review", error)
    state.set_stage(engine.pid, stage, "review")
    return errors


async def run_stage(engine, stage: str) -> StageResult:
    """Run ``stage`` for ``engine``'s film and report the resulting items."""
    run = _STAGES.get(stage)
    if run is None:
        raise ValueError(f"unknown stage: {stage!r}")

    pid = engine.pid
    if state.stage_status(pid, stage) == "approved":
        rows = _READERS[stage](engine)
        _stamp(pid, rows, save=False)
        return StageResult(stage, rows, 0.0, [])

    state.require_ready(pid, stage)
    spent_before = engine.ledger.spent()
    state.set_stage(pid, stage, "running")
    try:
        rows = await run(engine)
    except Exception as exc:
        message = str(exc) or type(exc).__name__
        state.set_stage(pid, stage, "failed", message)
        return StageResult(stage, [], engine.ledger.spent() - spent_before, [message])

    errors = _finish(engine, stage, rows)
    return StageResult(stage, rows, engine.ledger.spent() - spent_before, errors)
