"""Shared fixtures for the image-stage tests.

``eng_after_text`` runs every stage before ``sheets`` with canned replies so a
test can exercise one image stage without repeating the text setup.  The fake
inner generators are exposed as ``image_inner`` / ``video_inner`` so a test can
inspect or steer the calls the guarded generators make.
"""

from __future__ import annotations

import asyncio

import pytest

from hermes_story import config, engine, project, stages, state
from tests.hermes_story.fakes import FakeImageGen, FakeStageChat, FakeVideoGen, canned

PID = "20261021-0905-demo"

OPTIONS = {
    "idea": "a robot learns to paint",
    "style": "warm watercolor",
    "shape": "16:9",
    "length_s": 16,
    "cap_usd": 10,
}


def _approve(pid: str, stage: str) -> None:
    state.set_stage(pid, stage, "approved")


def _run(engine_obj, stage: str) -> stages.StageResult:
    result = asyncio.run(stages.run_stage(engine_obj, stage))
    assert not result.errors, result.errors
    return result


@pytest.fixture
def eng_after_text(tmp_path, monkeypatch):
    """Every text stage is approved; ``sheets`` is still pending."""
    monkeypatch.setenv("HERMES_STORY_ROOT", str(tmp_path))
    project.create(PID, OPTIONS)

    chat = FakeStageChat(canned.TEXT_FLOW)
    image = FakeImageGen()
    video = FakeVideoGen()
    eng = engine.Engine(PID, config.load(), chat=chat, image=image, video=video)
    # Test-only hooks: the production engine exposes none of these.
    eng.chat_calls = lambda: chat.calls
    eng.image_inner = image
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
