import pytest

from hermes_story import bible, project, state


@pytest.fixture
def pid(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_STORY_ROOT", str(tmp_path))
    project.create("20261021-0905-demo", {"idea": "x"})
    return "20261021-0905-demo"


def test_revision_log_keeps_feedback_verbatim(pid):
    fb = "make it night\n../../.env $(rm -rf /)"
    bible.add_revision(pid, "frame:0.1", fb, 1, 2)
    b = bible.load(pid)
    assert b["revisions"][-1]["feedback"] == fb
    assert b["current"]["frame:0.1"] == 2


def test_stage_gating(pid):
    with pytest.raises(state.NotReady):
        state.require_ready(pid, "characters")
    state.set_stage(pid, "script", "approved")
    state.require_ready(pid, "characters")  # no raise
    assert state.stage_status(pid, "sheets") == "pending"
