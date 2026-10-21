"""Shared fixtures for the image-stage tests.

``eng_after_text`` runs every stage before ``sheets`` with canned replies so a
test can exercise one image stage without repeating the text setup.  The fake
inner generators are exposed as ``image_inner`` / ``video_inner`` so a test can
inspect or steer the calls the guarded generators make.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_story import config, engine, project, stages, state
from tests.hermes_story.fakes import FakeImageGen, FakeStageChat, FakeVideoGen, canned

PID = "20261021-0905-demo"

REPO = Path(__file__).resolve().parents[2]

OPTIONS = {
    "idea": "a robot learns to paint",
    "style": "warm watercolor",
    "shape": "16:9",
    "length_s": 16,
    "duration_s": 8,
    "cap_usd": 10,
}

#: A camera tree for scene 0 with one child camera, so the frames estimate has
#: to account for a transition video as well as the extra first frame.
CAMERA_TREE_WITH_PARENT = [
    {"idx": 0, "active_shot_idxs": [0], "parent_cam_idx": None, "parent_shot_idx": None},
    {"idx": 1, "active_shot_idxs": [0], "parent_cam_idx": 0, "parent_shot_idx": 0},
]


def _approve(pid: str, stage: str) -> None:
    state.set_stage(pid, stage, "approved")


def _run(engine_obj, stage: str) -> stages.StageResult:
    result = asyncio.run(stages.run_stage(engine_obj, stage))
    assert not result.errors, result.errors
    return result


@pytest.fixture
def story_root(tmp_path, monkeypatch):
    """An empty projects root shared by the CLI and the in-process fixtures."""
    monkeypatch.setenv("HERMES_STORY_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture
def eng_after_text(story_root):
    """Every text stage is approved; ``sheets`` is still pending."""
    project.create(PID, OPTIONS)

    chat = FakeStageChat(canned.TEXT_FLOW)
    image = FakeImageGen()
    sheet_image = FakeImageGen()
    video = FakeVideoGen()
    eng = engine.Engine(
        PID, config.load(), chat=chat, image=image, sheet_image=sheet_image, video=video
    )
    # Test-only hooks: the production engine exposes none of these.
    eng.chat_calls = lambda: chat.calls
    eng.image_inner = image
    eng.sheet_image_inner = sheet_image
    eng.video_inner = video

    for stage in ("script", "characters"):
        _run(eng, stage)
        _approve(PID, stage)

    # The text stages are gated on every earlier stage, including the paid
    # sheets stage they do not actually need.  Approve it only for the gate,
    # then put it back so the test's sheets run is not treated as a re-read.
    _approve(PID, "sheets")
    for stage in ("storyboard", "prompts"):
        _run(eng, stage)
        _approve(PID, stage)
    state.set_stage(PID, "sheets", "pending")
    return eng


@pytest.fixture
def eng_after_sheets(eng_after_text):
    """``eng_after_text`` plus generated and approved character sheets."""
    _run(eng_after_text, "sheets")
    _approve(eng_after_text.pid, "sheets")
    return eng_after_text


@pytest.fixture
def eng_after_frames(eng_after_sheets):
    """``eng_after_sheets`` plus generated and approved first frames."""
    _run(eng_after_sheets, "frames")
    _approve(eng_after_sheets.pid, "frames")
    return eng_after_sheets


@pytest.fixture
def eng_after_clips(eng_after_frames):
    """``eng_after_frames`` plus generated and approved video clips."""
    _run(eng_after_frames, "clips")
    _approve(eng_after_frames.pid, "clips")
    return eng_after_frames


@pytest.fixture
def new_film(story_root):
    """A project just created, with every stage still pending."""
    project.create(PID, OPTIONS)
    return PID


@pytest.fixture
def film_ready_for_sheets(eng_after_text):
    """The id of a film whose earlier stages are approved and sheets pending."""
    return eng_after_text.pid


@pytest.fixture
def clips_film(eng_after_clips):
    """The id of a film with every stage, clips included, already produced."""
    return eng_after_clips.pid


@pytest.fixture
def film_with_clip_v2(clips_film, cli):
    """A finished film whose ``clip:0.0`` was retaken once (so v1 is archived).

    The film is built in-process: the fake chat replays its canned replies from
    the start in every subprocess, so a multi-stage film cannot be built by
    separate CLI calls.
    """
    feedback = cli.write("feedback-v2.txt", "slower")
    out = cli(
        "retake", clips_film, "clip:0.0", "--feedback-file", feedback, "--approved-spend"
    )
    assert out["ok"], out["errors"]
    return clips_film

@pytest.fixture
def eng_for_estimate(eng_after_text):
    """The canned film with prompts present and one child camera in scene 0."""
    project.write_json(
        eng_after_text.project_dir / "work" / "scene_0" / "camera_tree.json",
        CAMERA_TREE_WITH_PARENT,
    )
    return eng_after_text


@pytest.fixture
def cli(story_root):
    """Run ``python -m hermes_story`` in a subprocess with the fakes enabled.

    ``cli(...)`` returns the parsed JSON line, ``cli.raw(...)`` the completed
    process and ``cli.write(name, text)`` a path for ``--idea-file`` and
    friends.  The fake providers only exist when ``PYTEST_CURRENT_TEST`` is set.
    """

    def raw(*argv, env=None):
        child = dict(os.environ)
        child.pop("OPENROUTER_API_KEY", None)
        child["HERMES_STORY_ROOT"] = str(story_root)
        child["HERMES_STORY_FAKE"] = "1"
        child.setdefault("PYTEST_CURRENT_TEST", "tests/hermes_story/test_cli.py")
        if env:
            child.update(env)
        return subprocess.run(
            [sys.executable, "-m", "hermes_story", *map(str, argv)],
            cwd=REPO,
            env=child,
            capture_output=True,
            text=True,
            timeout=600,
        )

    def call(*argv, env=None):
        return json.loads(raw(*argv, env=env).stdout)

    def write(name, text):
        path = story_root / "cli-files" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return str(path)

    call.raw = raw
    call.write = write
    return call


@pytest.fixture(autouse=True)
def _example_config_only(monkeypatch):
    """Tests never read the developer's local configs/hermes-story.yaml."""
    from hermes_story import config as _config
    monkeypatch.setenv("HERMES_STORY_CONFIG", str(_config.EXAMPLE_CONFIG))
