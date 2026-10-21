"""Cost estimates, read from the canned film's files."""

import pytest

from hermes_story import estimate, project, state


def test_canned_film_estimates(eng_for_estimate):
    cfg = eng_for_estimate.cfg
    pid = eng_for_estimate.pid
    assert estimate.sheets(pid, cfg) == pytest.approx(2 * 3 * cfg.price_image_call)
    assert estimate.clips(pid, cfg) == pytest.approx(
        2 * cfg.video_duration_s * cfg.price_video_second
    )
    assert estimate.frames(pid, cfg) == pytest.approx(
        (2 + 4) * cfg.price_image_call
        + cfg.price_image_call
        + cfg.video_duration_s * cfg.price_video_second
    )


def test_stage_estimate_matches_the_helpers(eng_for_estimate):
    cfg = eng_for_estimate.cfg
    pid = eng_for_estimate.pid
    assert estimate.stage_estimate(pid, "sheets", cfg) == estimate.sheets(pid, cfg)
    assert estimate.stage_estimate(pid, "frames", cfg) == estimate.frames(pid, cfg)
    assert estimate.stage_estimate(pid, "clips", cfg) == estimate.clips(pid, cfg)


def test_film_estimate_covers_every_stage_in_order(eng_for_estimate):
    values = estimate.film_estimate(eng_for_estimate.pid, eng_for_estimate.cfg)
    assert list(values) == state.STAGES


def test_before_prompts_assume_medium(eng_for_estimate):
    pid = eng_for_estimate.pid
    cfg = eng_for_estimate.cfg
    for path in (eng_for_estimate.project_dir / "work").glob(
        "scene_*/shots/*/shot_description.json"
    ):
        path.unlink()
    assert estimate.frames(pid, cfg) == pytest.approx(
        2 * cfg.num_image_candidates * 2 * cfg.price_image_call
        + cfg.price_image_call
        + cfg.video_duration_s * cfg.price_video_second
    )
    assert estimate.clips(pid, cfg) == pytest.approx(
        2 * cfg.video_duration_s * cfg.price_video_second
    )


def test_empty_project_estimates_are_zero(new_film):
    from hermes_story import config

    cfg = config.load()
    values = estimate.film_estimate(new_film, cfg)
    assert values["sheets"] == 0
    assert values["frames"] == 0
    assert values["clips"] == 0
    assert values["script"] == pytest.approx(2 * cfg.price_chat_call)


def test_produced_text_stages_need_no_calls(eng_for_estimate):
    cfg = eng_for_estimate.cfg
    pid = eng_for_estimate.pid
    assert estimate.text_calls(pid, "script") == 0
    assert estimate.text_calls(pid, "characters") == 0
    assert estimate.text_calls(pid, "storyboard") == 0


def test_item_estimate(eng_for_estimate):
    cfg = eng_for_estimate.cfg
    pid = eng_for_estimate.pid
    assert estimate.item_estimate(pid, "sheet:Ana", cfg) == pytest.approx(
        3 * cfg.price_image_call
    )
    assert estimate.item_estimate(pid, "frame:0.0", cfg) == pytest.approx(
        cfg.num_image_candidates * cfg.price_image_call
    )
    assert estimate.item_estimate(pid, "clip:0.0", cfg) == pytest.approx(
        cfg.video_duration_s * cfg.price_video_second
    )
    assert estimate.item_estimate(pid, "scene:0", cfg) == pytest.approx(
        cfg.price_chat_call
    )


def test_unknown_stage_is_rejected(eng_for_estimate):
    with pytest.raises(ValueError):
        estimate.stage_estimate(eng_for_estimate.pid, "nonsense", eng_for_estimate.cfg)
