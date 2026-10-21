"""The CLI's JSON contract: one line per command, stable error tokens."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from hermes_story import bible, project, versions
from tests.hermes_story.fakes import fake_log_has_reference

REPO = Path(__file__).resolve().parents[2]


def test_new_and_status(cli):
    out = cli("new", "--idea-file", cli.write("idea.txt", "a robot paints"), "--length", "60")
    assert out["ok"] and out["stages"]["script"] == "pending"
    assert cli("status", out["id"])["id"] == out["id"]


def test_paid_needs_approval(cli, film_ready_for_sheets):
    out = cli("run", film_ready_for_sheets, "sheets")
    assert not out["ok"] and out["errors"][0].startswith("needs_approval")
    assert out["cost"]["this_call_usd"] == 0


def test_not_ready(cli, new_film):
    assert cli("run", new_film, "characters")["errors"][0].startswith("not_ready")


def test_busy(cli, film_ready_for_sheets):
    with project.lock(film_ready_for_sheets):
        assert cli("run", film_ready_for_sheets, "storyboard")["errors"][0].startswith("busy")


def test_key_never_printed(cli, new_film, monkeypatch):
    out = cli.raw(
        "run",
        new_film,
        "script",
        env={"OPENROUTER_API_KEY": "sk-secret-123", "HERMES_STORY_FAKE_FAIL": "sk-secret-123"},
    )
    assert "sk-secret-123" not in out.stdout + out.stderr


def test_one_json_line(cli, new_film):
    r = cli.raw("status", new_film)
    assert len(r.stdout.strip().splitlines()) == 1


def test_message_map(cli, new_film):
    cli("msg-record", new_film, "555", "scene:1", "2")
    assert cli("msg-lookup", new_film, "555")["items"][0] == {
        "item": "scene:1",
        "version": 2,
        "kind": None,
        "text": None,
        "paths": [],
    }


def test_cli_never_imports_agent_runtime():
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            "import hermes_story.cli, sys; assert not any("
            "m.split('.')[0] in ('agent_runtime','main_agent') for m in sys.modules)",
        ],
        cwd=REPO,
        capture_output=True,
    )
    assert r.returncode == 0, r.stderr


# -- extras --------------------------------------------------------------


def test_run_script_then_approve(cli, new_film):
    out = cli("run", new_film, "script")
    assert out["ok"] and out["stages"]["script"] == "review"
    assert [item["item"] for item in out["items"]] == ["story", "scene:0", "scene:1"]
    assert cli("approve", new_film, "script")["stages"]["script"] == "approved"


def test_approved_run_is_free_even_without_a_key(cli, film_ready_for_sheets):
    out = cli("run", film_ready_for_sheets, "script")
    assert out["ok"] and out["cost"]["this_call_usd"] == 0
    assert [item["item"] for item in out["items"]][0] == "story"


def test_approve_requires_review(cli, new_film):
    out = cli("approve", new_film, "script")
    assert not out["ok"] and out["errors"][0].startswith("not_ready")


def test_estimate_reports_the_frames_formula(cli, film_ready_for_sheets):
    out = cli("estimate", film_ready_for_sheets, "sheets")
    assert out["ok"] and out["cost"]["this_call_usd"] == pytest.approx(2 * 3 * 0.06)
    every = cli("estimate", film_ready_for_sheets)
    assert set(every["estimates"]) == {
        "script",
        "characters",
        "sheets",
        "storyboard",
        "prompts",
        "frames",
        "clips",
    }


def test_cap_command_updates_the_ledger(cli, new_film):
    out = cli("cap", new_film, "3.5")
    assert out["ok"] and out["cost"]["cap_usd"] == 3.5
    assert cli("status", new_film)["cost"]["cap_usd"] == 3.5
    assert project.read_json(project.path(new_film) / "ledger.json", {})["cap_usd"] == 3.5
    assert cli("cap", new_film, "-1")["errors"][0].startswith("bad_item")


def test_cameo_bind_and_clear(cli, film_ready_for_sheets):
    assert cli("cameo-bind", film_ready_for_sheets, "Ana", "owner")["ok"]
    assert bible.load(film_ready_for_sheets)["characters"]["Ana"]["cameo"] == "owner"
    assert cli("cameo-bind", film_ready_for_sheets, "Ana", "none")["ok"]
    assert bible.load(film_ready_for_sheets)["characters"]["Ana"]["cameo"] is None


def test_edit_cascade_invalidate(cli, clips_film):
    text = cli.write("scene.txt", "A new second scene.")
    out = cli("edit", clips_film, "scene:1", "--text-file", text)
    assert out["ok"] and out["items"][0]["version"] == 2

    cascade = cli("cascade", clips_film, "sheet:Ana")
    assert cascade["cascade"] == ["frame:0.0", "clip:0.0"]
    assert [item["item"] for item in cascade["items"]] == ["frame:0.0", "clip:0.0"]

    invalidated = cli("invalidate", clips_film, "clip:0.0")
    assert invalidated["invalidated"] == ["clip:0.0"]


def test_retake_of_a_paid_item_needs_approval(cli, clips_film):
    feedback = cli.write("feedback.txt", "slower")
    out = cli("retake", clips_film, "clip:0.0", "--feedback-file", feedback)
    assert not out["ok"] and out["errors"][0].startswith("needs_approval")
    assert out["estimates"]["clip:0.0"] == pytest.approx(8 * 0.03)


def test_retake_with_approval_uses_the_versioned_files(cli, clips_film):
    feedback = cli.write("feedback.txt", "slower")
    out = cli(
        "retake",
        clips_film,
        "clip:0.0",
        "--feedback-file",
        feedback,
        "--approved-spend",
    )
    assert out["ok"], out["errors"]
    assert out["items"][0]["version"] == 2


def test_bad_id_and_unknown_stage(cli):
    assert cli("status", "nope")["errors"][0].startswith("bad_id")
    assert cli("list")["ok"]


def test_list_reports_the_film(cli, new_film):
    out = cli("list")
    assert out["ok"] and any(entry["id"] == new_film for entry in out["projects"])


def test_usage_error_is_json(cli):
    r = cli.raw("run", "20261021-0905-demo")
    assert r.returncode == 1
    payload = json.loads(r.stdout)
    assert payload["errors"][0].startswith("bad_item")


def test_cap_rejects_nan_and_huge(cli, new_film):
    for bad in ("nan", "inf", "-1", "1000"):
        out = cli("cap", new_film, bad)
        assert not out["ok"] and out["errors"][0].startswith("bad_item")


def test_state_commands_respect_lock(cli, new_film):
    with project.lock(new_film):
        assert cli("cap", new_film, "5")["errors"][0].startswith("busy")
        assert cli("approve", new_film, "script")["errors"][0].startswith("busy")

# -- restore / items / reference photos ---------------------------------

def test_restore_clip_brings_back_v1(cli, film_with_clip_v2):
    out = cli("restore", film_with_clip_v2, "clip:0.0", "1")
    assert out["ok"] and out["items"][0]["item"] == "clip:0.0" and out["items"][0]["version"] == 3
    # v2 was archived before being replaced
    assert 2 in versions.list(film_with_clip_v2, "clip:0.0")

def test_restore_refuses_shared_file_items(cli, film_with_clip_v2):
    out = cli("restore", film_with_clip_v2, "char:Ana", "1")
    assert out["errors"][0].startswith("bad_item")

def test_restore_unknown_version(cli, film_with_clip_v2):
    assert cli("restore", film_with_clip_v2, "clip:0.0", "9")["errors"][0].startswith("bad_item")

def test_items_for_one_stage(cli, film_with_clip_v2):
    out = cli("items", film_with_clip_v2, "clips")
    assert [i["item"] for i in out["items"]] == ["clip:0.0", "clip:1.0"]
    assert out["items"][0]["paths"]

def test_items_all_stages(cli, film_with_clip_v2):
    out = cli("items", film_with_clip_v2)
    assert out["ok"] and {"story", "clip:1.0"} <= {i["item"] for i in out["items"]}

def test_ref_add_becomes_front_reference(cli, film_ready_for_sheets, tmp_path):
    img = tmp_path / "me.png"; Image.new("RGB", (8, 8)).save(img)
    assert cli("ref-add", film_ready_for_sheets, "Ana", str(img), "--kind", "self")["ok"]
    cli("run", film_ready_for_sheets, "sheets", "--approved-spend")
    assert fake_log_has_reference(film_ready_for_sheets, "refs/")

def test_ref_add_consented_needs_text(cli, film_ready_for_sheets, tmp_path):
    img = tmp_path / "x.png"; Image.new("RGB", (8, 8)).save(img)
    assert cli("ref-add", film_ready_for_sheets, "Ana", str(img), "--kind", "consented")["errors"][0].startswith("consent")

def test_ref_add_rejects_non_image(cli, film_ready_for_sheets, tmp_path):
    bad = tmp_path / "x.png"; bad.write_text("not an image")
    assert cli("ref-add", film_ready_for_sheets, "Ana", str(bad), "--kind", "not_person")["errors"][0].startswith("bad_item")

def test_ref_add_keeps_three_and_clear_empties(cli, film_ready_for_sheets, tmp_path):
    img = tmp_path / "me.png"; Image.new("RGB", (8, 8)).save(img)
    for _ in range(4):
        out = cli("ref-add", film_ready_for_sheets, "Ana", str(img), "--kind", "self")
        assert out["ok"], out["errors"]
    assert len(out["refs"]) == 3
    folder = project.path(film_ready_for_sheets) / "refs"
    files = [p for p in folder.rglob("ref-*")]
    assert len(files) == 3 and all((p.stat().st_mode & 0o777) == 0o600 for p in files)
    assert (project.path(film_ready_for_sheets) / "versions").is_dir()
    assert cli("ref-clear", film_ready_for_sheets, "Ana")["refs"] == []
    assert not [p for p in folder.rglob("ref-*")]

def test_ref_add_unknown_character(cli, film_ready_for_sheets, tmp_path):
    img = tmp_path / "me.png"; Image.new("RGB", (8, 8)).save(img)
    out = cli("ref-add", film_ready_for_sheets, "Nobody", str(img), "--kind", "self")
    assert out["errors"][0].startswith("bad_item")
