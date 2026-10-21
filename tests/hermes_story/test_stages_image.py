# tests/hermes_story/test_stages_image.py
import asyncio
import json

import pytest

from hermes_story import bible, ledger, stages, state


def test_sheets_three_views_per_character(eng_after_text):
    r = asyncio.run(stages.run_stage(eng_after_text, "sheets"))
    albums = [i for i in r.items if i["kind"] == "album"]
    assert len(albums) == 2 and all(len(a["paths"]) == 3 for a in albums)
    reg = json.loads((eng_after_text.project_dir / "work/character_portraits_registry.json").read_text())
    assert set(reg) == {"Ana", "Bolt"}


def test_cameo_photo_is_front_reference(eng_after_text, tmp_path, monkeypatch):
    cam = tmp_path / "cameo" / "owner"; cam.mkdir(parents=True)
    (cam / "consent.json").write_text('{"person":"owner"}'); (cam / "a.jpg").write_bytes(b"x")
    monkeypatch.setenv("HERMES_STORY_CAMEO_DIR", str(tmp_path / "cameo"))
    bible.set_cameo(eng_after_text.pid, "Ana", "owner")
    asyncio.run(stages.run_stage(eng_after_text, "sheets"))
    front_calls = [c for c in eng_after_text.image_inner.calls if str(cam / "a.jpg") in (c["refs"] or [])]
    assert len(front_calls) == 1


def test_cameo_without_consent_fails_closed(eng_after_text, tmp_path, monkeypatch):
    (tmp_path / "cameo" / "bob").mkdir(parents=True)
    monkeypatch.setenv("HERMES_STORY_CAMEO_DIR", str(tmp_path / "cameo"))
    bible.set_cameo(eng_after_text.pid, "Ana", "bob")
    r = asyncio.run(stages.run_stage(eng_after_text, "sheets"))
    assert any("consent" in e for e in r.errors)
    assert state.load(eng_after_text.pid)["items"]["sheet:Ana"]["status"] == "failed"


def test_side_failure_is_visible(eng_after_text):
    eng_after_text.image_inner.fail_on = lambda prompt, refs: "side" in prompt.lower()
    r = asyncio.run(stages.run_stage(eng_after_text, "sheets"))
    assert r.errors and state.load(eng_after_text.pid)["items"]["sheet:Ana"]["status"] == "failed"


def test_frames_stop_at_cap(eng_after_sheets):
    led = eng_after_sheets.ledger; led.cap = led.spent() + 0.05   # room for one image only
    r = asyncio.run(stages.run_stage(eng_after_sheets, "frames"))
    assert any("cap" in e for e in r.errors)
    assert state.stage_status(eng_after_sheets.pid, "frames") == "failed"
