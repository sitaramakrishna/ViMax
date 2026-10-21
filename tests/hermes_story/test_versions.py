from hermes_story import project, versions


def test_archive_and_restore(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_STORY_ROOT", str(tmp_path))
    p = project.create("20261021-0905-demo", {})
    f = p / "work" / "story.txt"
    f.write_text("v1 text")
    assert versions.archive("20261021-0905-demo", "story", [f]) == 1
    f.write_text("v2 text")
    assert versions.archive("20261021-0905-demo", "story", [f]) == 2
    assert versions.list("20261021-0905-demo", "story") == [1, 2]
    versions.restore("20261021-0905-demo", "story", 1, p / "work")
    assert f.read_text() == "v1 text"
