import datetime
import json
import os

import pytest

from hermes_story import project


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_STORY_ROOT", str(tmp_path))
    return tmp_path


def test_new_id_format():
    pid = project.new_id("A Robot learns to Paint!", datetime.datetime(2026, 10, 21, 9, 5))
    assert pid == "20261021-0905-a-robot-learns-to-paint"


def test_new_id_non_ascii_falls_back():
    pid = project.new_id("ராமு கதை", datetime.datetime(2026, 10, 21, 9, 5))
    assert pid == "20261021-0905-story"


@pytest.mark.parametrize("bad", ["20261021-0905-demo\n", "../x", "20261021-0905-A", "x", "20261021-0905-a/b", ""])
def test_path_rejects_bad_ids(bad):
    with pytest.raises(project.ProjectError):
        project.path(bad)


def test_create_layout(root):
    p = project.create("20261021-0905-demo", {"idea": "x"})
    for name in ["bible.json", "state.json", "ledger.json", "messages.json"]:
        assert (p / name).is_file()
    assert (p / "work").is_dir() and (p / "versions").is_dir()


def test_safe_child_blocks_escape_and_symlink(root, tmp_path):
    p = project.create("20261021-0905-demo", {})
    with pytest.raises(project.ProjectError):
        project.safe_child(p, "..", "other")
    os.symlink("/etc", p / "work" / "link")
    with pytest.raises(project.ProjectError):
        project.safe_child(p, "work", "link", "hosts")


def test_write_json_atomic(root):
    p = project.create("20261021-0905-demo", {})
    project.write_json(p / "state.json", {"a": 1})
    assert json.loads((p / "state.json").read_text()) == {"a": 1}
    assert not [f for f in os.listdir(p) if f.endswith(".tmp")]


def test_lock_busy(root):
    project.create("20261021-0905-demo", {})
    with project.lock("20261021-0905-demo"):
        with pytest.raises(project.Busy):
            with project.lock("20261021-0905-demo"):
                pass


def test_project_dir_is_private(root):
    p = project.create("20261021-0905-demo", {})
    assert (p.stat().st_mode & 0o777) == 0o700
