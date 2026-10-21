import subprocess
import sys
from pathlib import Path

from hermes_story import config, engine, guarded, ledger, project
from tests.hermes_story.fakes import FakeChat, FakeImageGen, FakeVideoGen


def test_engine_wraps_and_scopes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_STORY_ROOT", str(tmp_path))
    project.create("20261021-0905-demo", {"cap_usd": 5})
    e = engine.Engine(
        "20261021-0905-demo",
        config.load(),
        chat=FakeChat([]),
        image=FakeImageGen(),
        video=FakeVideoGen(),
    )
    assert isinstance(e.idea().image_generator, guarded.GuardedImageGenerator)
    s = e.scene(2)
    assert s.working_dir.endswith("work/scene_2")
    assert isinstance(s.video_generator, guarded.GuardedVideoGenerator)
    assert e.ledger.cap == 5


def test_engine_cap_falls_back_to_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_STORY_ROOT", str(tmp_path))
    project.create("20261021-0905-demo", {})
    e = engine.Engine(
        "20261021-0905-demo",
        config.load(),
        chat=FakeChat([]),
        image=FakeImageGen(),
        video=FakeVideoGen(),
    )
    assert e.ledger.cap == config.load().default_cap_usd


def test_engine_shares_ledger_with_generators(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_STORY_ROOT", str(tmp_path))
    project.create("20261021-0905-demo", {"cap_usd": 1})
    e = engine.Engine(
        "20261021-0905-demo",
        config.load(),
        chat=FakeChat([]),
        image=FakeImageGen(),
        video=FakeVideoGen(),
    )
    idea = e.idea()
    assert idea.working_dir == str(Path(tmp_path) / "20261021-0905-demo" / "work")
    assert isinstance(idea.video_generator, guarded.GuardedVideoGenerator)
    assert idea.image_generator.ledger is e.ledger
    assert isinstance(idea.image_generator.ledger, ledger.Ledger)


def test_engine_uses_injected_chat(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_STORY_ROOT", str(tmp_path))
    project.create("20261021-0905-demo", {})
    chat = FakeChat([])
    e = engine.Engine(
        "20261021-0905-demo",
        config.load(),
        chat=chat,
        image=FakeImageGen(),
        video=FakeVideoGen(),
    )
    assert e.chat is chat


def test_never_imports_agent_runtime():
    repo = Path(__file__).resolve().parents[2]
    code = (
        "import hermes_story.engine, sys; "
        "bad = [m for m in sys.modules if m == 'agent_runtime' or m.startswith('agent_runtime.')]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], cwd=repo, check=True)
