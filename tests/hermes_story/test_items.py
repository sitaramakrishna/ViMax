import pytest

from hermes_story import items


@pytest.mark.parametrize(
    "s",
    [
        "story",
        "scene:0",
        "char:Ana",
        "char:Dr. Ana/Bot",
        "char:ராமு",
        "sheet:Ana",
        "shot:1.3",
        "prompt:0.0",
        "frame:2.1",
        "clip:1.3",
    ],
)
def test_roundtrip(s):
    assert items.fmt(items.parse(s)) == s


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "clip:1",
        "clip:a.b",
        "char:",
        "char:" + "x" * 61,
        "char:a\nb",
        "foo:1",
        "scene:-1",
    ],
)
def test_rejects(bad):
    with pytest.raises(items.ItemError):
        items.parse(bad)


def test_folder_is_single_safe_part():
    f = items.folder(items.parse("char:../../etc"))
    assert "/" not in f and ".." not in f and f.startswith("char-")
