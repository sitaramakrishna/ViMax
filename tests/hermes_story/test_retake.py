import asyncio

import pytest

from hermes_story import bible, items, retake, stages


def test_text_retake_versions_and_feedback(eng_after_text):
    fb = "make Ana older\n../../.env $(rm -rf /)"
    eng_after_text.chat.queue(
        '{"idx":0,"identifier_in_scene":"Ana","is_visible":true,'
        '"static_features":"older woman","dynamic_features":null}'
    )
    r = asyncio.run(retake.retake(eng_after_text, "char:Ana", fb))
    assert r.items[0]["version"] == 2
    b = bible.load(eng_after_text.pid)
    assert b["revisions"][-1]["feedback"] == fb
    last_msg = eng_after_text.chat.calls[-1]
    assert "OWNER FEEDBACK" in str(last_msg) and fb in str(last_msg)
    assert not (eng_after_text.project_dir.parent / ".env").exists()
    assert "older woman" in (
        eng_after_text.project_dir / "work/characters.json"
    ).read_text()


def test_clip_retake_only_that_clip(eng_after_clips):
    before = len(eng_after_clips.video_inner.calls)
    asyncio.run(retake.retake(eng_after_clips, "clip:0.0", "slower"))
    calls = eng_after_clips.video_inner.calls[before:]
    assert len(calls) == 1 and "Revision request: slower" in calls[0]["prompt"]


def test_cascade_for_sheet(eng_after_clips):
    assert retake.cascade(eng_after_clips.pid, "sheet:Ana") == [
        "frame:0.0",
        "clip:0.0",
    ]


def test_invalidate_then_rerun_regenerates(eng_after_clips):
    retake.invalidate(eng_after_clips.pid, ["clip:0.0"])
    before = len(eng_after_clips.video_inner.calls)
    asyncio.run(stages.run_stage(eng_after_clips, "clips"))
    assert len(eng_after_clips.video_inner.calls) == before + 1


def test_retake_counts_toward_cap(eng_after_clips):
    eng_after_clips.ledger.cap = eng_after_clips.ledger.spent() + 0.01
    r = asyncio.run(retake.retake(eng_after_clips, "clip:0.0", "again"))
    assert r.errors and "cap" in r.errors[0]
    assert (
        eng_after_clips.project_dir / "work/scene_0/shots/0/video.mp4"
    ).exists()  # restored v-current


def test_edit_replaces_scene_text_and_rejects_media(eng_after_text):
    r = retake.edit(eng_after_text.pid, "scene:1", "A new second scene.")
    assert r.items[0]["item"] == "scene:1"
    assert "A new second scene." in (
        eng_after_text.project_dir / "work/script.json"
    ).read_text()

    with pytest.raises(items.ItemError):
        retake.edit(eng_after_text.pid, "clip:0.0", "nope")


def test_sheet_retake_regenerates_three_views(eng_after_sheets):
    front = eng_after_sheets.project_dir / "work/character_portraits/0_Ana/front.png"
    before = len(eng_after_sheets.image_inner.calls)
    r = asyncio.run(retake.retake(eng_after_sheets, "sheet:Ana", "curly hair"))
    calls = eng_after_sheets.image_inner.calls[before:]
    assert r.items[0]["version"] == 2
    assert len(calls) == 3
    assert "Revision request: curly hair" in calls[0]["prompt"]
    assert str(front) in calls[0]["refs"]


def test_frame_retake_appends_revision(eng_after_frames):
    selector = (
        eng_after_frames.project_dir
        / "work/scene_0/shots/0/first_frame_selector_output.json"
    )
    r = asyncio.run(retake.retake(eng_after_frames, "frame:0.0", "closer"))
    assert r.items[0]["version"] == 2
    assert (
        eng_after_frames.project_dir / "work/scene_0/shots/0/first_frame.png"
    ).exists()
    assert "Revision request: closer" in selector.read_text()


def test_edit_keeps_old_version(eng_after_clips):
    from hermes_story import versions
    r = retake.edit(eng_after_clips.pid, "scene:0", "new scene text")
    assert r.items[0]["version"] == 2
    assert versions.list(eng_after_clips.pid, "scene:0") == [1]


def test_invalidate_refuses_partial_shared_file(eng_after_clips):
    import pytest
    from hermes_story.items import ItemError
    with pytest.raises(ItemError):
        retake.invalidate(eng_after_clips.pid, ["char:Ana"])      # Bolt shares characters.json
    assert (eng_after_clips.project_dir / "work/characters.json").exists()
    assert retake.invalidate(eng_after_clips.pid, ["char:Ana", "char:Bolt"]) == ["char:Ana", "char:Bolt"]
