import asyncio

import pytest

from hermes_story import bible, config, engine, project, stages, state
from tests.hermes_story.fakes import FakeChat, FakeImageGen, FakeVideoGen, canned


@pytest.fixture
def eng(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_STORY_ROOT", str(tmp_path))
    project.create(
        "20261021-0905-demo",
        {
            "idea": "a robot learns to paint",
            "style": "warm watercolor",
            "shape": "16:9",
            "length_s": 16,
            "cap_usd": 10,
        },
    )
    chat = FakeChat(canned.TEXT_FLOW)
    e = engine.Engine(
        "20261021-0905-demo",
        config.load(),
        chat=chat,
        image=FakeImageGen(),
        video=FakeVideoGen(),
    )
    # Test-only hook: the engine has no chat_calls in production.
    e.chat_calls = lambda: chat.calls
    return e


def test_script_stage_items_per_scene(eng):
    r = asyncio.run(stages.run_stage(eng, "script"))
    assert [i["item"] for i in r.items] == ["story", "scene:0", "scene:1"]
    assert state.stage_status(eng.pid, "script") == "review"
    assert r.cost_usd == 0 or r.cost_usd < 0.05
    assert all(i["version"] == 1 for i in r.items)
    assert bible.load(eng.pid)["current"]["scene:0"] == 1
    assert all(i["kind"] == "text" for i in r.items)


def test_rerun_approved_is_noop(eng):
    asyncio.run(stages.run_stage(eng, "script"))
    state.set_stage(eng.pid, "script", "approved")
    n = len(eng.chat_calls())
    r = asyncio.run(stages.run_stage(eng, "script"))
    assert len(eng.chat_calls()) == n and [i["item"] for i in r.items][0] == "story"


def test_storyboard_requirement_has_mouth_rule(eng):
    req = stages.user_requirement(bible.load(eng.pid), "storyboard")
    assert "mouths are not clearly visible" in req and "about 2 shots" in req


def test_storyboard_per_scene_items(eng):
    for s in ["script", "characters"]:
        asyncio.run(stages.run_stage(eng, s))
        state.set_stage(eng.pid, s, "approved")
    state.set_stage(eng.pid, "sheets", "approved")  # sheets are not needed for text storyboard
    r = asyncio.run(stages.run_stage(eng, "storyboard"))
    assert all(i["item"].startswith("shot:") for i in r.items)
    assert [i["item"] for i in r.items] == ["shot:0.0", "shot:1.0"]


def test_characters_stage_items_and_bible(eng):
    asyncio.run(stages.run_stage(eng, "script"))
    state.set_stage(eng.pid, "script", "approved")
    r = asyncio.run(stages.run_stage(eng, "characters"))
    assert [i["item"] for i in r.items] == ["char:Ana", "char:Bolt"]
    assert bible.load(eng.pid)["characters"]["Ana"] == {"cameo": None, "voice": None}
    assert state.item_status(eng.pid, "char:Bolt") == "review"


def test_prompts_stage_items(eng):
    for s in ["script", "characters"]:
        asyncio.run(stages.run_stage(eng, s))
        state.set_stage(eng.pid, s, "approved")
    state.set_stage(eng.pid, "sheets", "approved")
    asyncio.run(stages.run_stage(eng, "storyboard"))
    state.set_stage(eng.pid, "storyboard", "approved")

    r = asyncio.run(stages.run_stage(eng, "prompts"))
    assert [i["item"] for i in r.items] == ["prompt:0.0", "prompt:1.0"]
    assert "Static camera" in r.items[0]["text"]
    assert state.stage_status(eng.pid, "prompts") == "review"

    # A second run finds every shot description on disk and asks the model nothing.
    n = len(eng.chat_calls())
    again = asyncio.run(stages.run_stage(eng, "prompts"))
    assert [i["item"] for i in again.items] == ["prompt:0.0", "prompt:1.0"]
    assert len(eng.chat_calls()) == n


def test_earlier_stage_must_be_approved(eng):
    asyncio.run(stages.run_stage(eng, "script"))
    with pytest.raises(state.NotReady):
        asyncio.run(stages.run_stage(eng, "characters"))


def test_stage_failure_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_STORY_ROOT", str(tmp_path))
    project.create("20261021-0905-demo", {})
    chat = FakeChat(canned.TEXT_FLOW)
    e = engine.Engine(
        "20261021-0905-demo",
        config.load(),
        chat=chat,
        image=FakeImageGen(),
        video=FakeVideoGen(),
    )
    r = asyncio.run(stages.run_stage(e, "script"))
    assert r.items == [] and r.errors
    assert state.stage_status(e.pid, "script") == "failed"


def test_unknown_stage_is_rejected(eng):
    with pytest.raises(ValueError):
        asyncio.run(stages.run_stage(eng, "nonsense"))
