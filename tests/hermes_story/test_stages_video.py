# tests/hermes_story/test_stages_video.py
import asyncio

from hermes_story import stages


def test_clips_sequential_and_ordered(eng_after_frames):
    r = asyncio.run(stages.run_stage(eng_after_frames, "clips"))
    order = [c["item"] for c in eng_after_frames.video_inner.calls]
    assert order == sorted(order, key=lambda s: tuple(map(int, s.split(":")[1].split("."))))
    assert eng_after_frames.video_inner.max_concurrent == 1


def test_clips_stop_cleanly_at_cap(eng_after_frames):
    eng_after_frames.ledger.cap = eng_after_frames.ledger.spent() + 0.30   # one 8 s clip ($0.24)
    r = asyncio.run(stages.run_stage(eng_after_frames, "clips"))
    assert len([i for i in r.items if i["kind"] == "video"]) == 1
    assert any(e.startswith("cap reached before clip:") for e in r.errors)


def test_missing_frame_skipped(eng_after_frames):
    (eng_after_frames.project_dir / "work/scene_1/shots/0/first_frame.png").unlink()
    r = asyncio.run(stages.run_stage(eng_after_frames, "clips"))
    assert "frame missing for clip:1.0" in r.errors


def test_existing_clips_need_no_cap_room(eng_after_frames):
    asyncio.run(stages.run_stage(eng_after_frames, "clips"))
    eng_after_frames.ledger.cap = eng_after_frames.ledger.spent()   # no room left
    before = len(eng_after_frames.video_inner.calls)
    r = asyncio.run(stages.run_stage(eng_after_frames, "clips"))
    assert not r.errors and len(eng_after_frames.video_inner.calls) == before
    assert len([i for i in r.items if i["kind"] == "video"]) == 2
