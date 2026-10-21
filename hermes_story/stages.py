"""The text stages: script, characters, storyboard and prompts.

Each paid-for ViMax call is guarded so the ledger can attribute it to an item,
even though the text stages themselves only spend chat tokens.  ViMax's own
pipeline methods load their output file when it already exists, so re-running
an approved stage is free and simply rebuilds the :class:`StageResult` from
disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from hermes_story import bible, guarded, project, state
from hermes_story import items as item_ids
from interfaces import CharacterInScene, ShotBriefDescription, ShotDescription

DEFAULT_DURATION_S = 8

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


_STAGES = {
    "script": _script,
    "characters": _characters,
    "storyboard": _storyboard,
    "prompts": _prompts,
}

_READERS = {
    "script": _script_rows,
    "characters": _character_rows,
    "storyboard": _storyboard_rows,
    "prompts": _prompt_rows,
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


def _finish(engine, stage: str, rows: list[dict]) -> None:
    _stamp(engine.pid, rows, save=True)
    for row in rows:
        state.set_item(engine.pid, row["item"], "review")
    state.set_stage(engine.pid, stage, "review")


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

    _finish(engine, stage, rows)
    return StageResult(stage, rows, engine.ledger.spent() - spent_before, [])
