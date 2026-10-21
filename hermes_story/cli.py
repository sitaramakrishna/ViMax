"""The ``python -m hermes_story`` command line.

Every command prints exactly one JSON line on stdout; anything the ViMax
pipelines print goes to stderr.  The exit status is 0 when the JSON ``ok`` is
true and 1 otherwise.  Error strings always start with a stable token
(``busy``, ``not_ready``, ``needs_approval``, ``cap``, ``bad_item``,
``bad_id``, ``provider`` or ``consent``) so the caller can branch on them.

The OpenRouter key is only ever read from ``OPENROUTER_API_KEY``; it is
removed from every error message before anything is printed.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime
import json
import math
import os
import re
import sys
import types
from pathlib import Path

from hermes_story import bible, config, estimate, ledger, project, state
from hermes_story import items as item_ids
from hermes_story import retake, stages

#: Stable prefixes of the ``errors`` entries.
TOKENS = (
    "busy ",
    "not_ready ",
    "needs_approval ",
    "cap ",
    "bad_item ",
    "bad_id ",
    "provider ",
    "consent ",
)

#: Largest cap accepted for one film, in USD.
MAX_CAP_USD = 100.0

#: Item kinds whose retake spends money.
PAID_ITEM_KINDS = ("sheet", "frame", "clip")

_AUTH_RE = re.compile(r"(?i)\bauthorization\b\s*[:=]\s*(?:bearer\s+)?\S+")
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")


# -- redaction -----------------------------------------------------------


def redact(value) -> str:
    """Remove the API key value and any ``Authorization`` header text."""
    text = str(value)
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if len(key) >= 4:
        text = text.replace(key, "[redacted]")
    text = _AUTH_RE.sub("Authorization: [redacted]", text)
    text = _BEARER_RE.sub("Bearer [redacted]", text)
    return text


def _tag(message) -> str:
    """Give ``message`` one of the stable error prefixes."""
    text = redact(message)
    if text.startswith(TOKENS):
        return text
    low = text.lower()
    if "consent" in low:
        return "consent " + text
    if "cap $" in low or "cap reached" in low:
        return "cap " + text
    if "nothing to retake" in low or "cannot retake" in low or "nothing to edit" in low:
        return "bad_item " + text
    return "provider " + text


def _error_text(exc: BaseException) -> str:
    """Map an exception to ``"<token> <message>"``."""
    if isinstance(exc, project.Busy):
        code = "busy"
    elif isinstance(exc, state.NotReady):
        code = "not_ready"
    elif isinstance(exc, item_ids.ItemError):
        code = "bad_item"
    elif isinstance(exc, stages.CameoError):
        code = "consent"
    elif isinstance(exc, ledger.CapExceeded):
        code = "cap"
    elif isinstance(exc, project.ProjectError):
        code = "bad_id"
    elif isinstance(exc, (state.StateError, ValueError)):
        code = "bad_item"
    else:
        code = "provider"
    return f"{code} {redact(exc)}"


# -- JSON payload --------------------------------------------------------


def _money(value) -> float:
    try:
        return round(max(0.0, float(value)), 6)
    except (TypeError, ValueError):
        return 0.0


def _cap(pid: str, cfg) -> float:
    data = project.read_json(project.path(pid) / "ledger.json", None)
    if isinstance(data, dict):
        try:
            return float(data.get("cap_usd"))
        except (TypeError, ValueError):
            pass
    return float(getattr(cfg, "default_cap_usd", 0.0) or 0.0)


def _cost(pid, cfg, this_call: float = 0.0) -> dict:
    spent = 0.0
    cap = float(getattr(cfg, "default_cap_usd", 0.0) or 0.0)
    if isinstance(pid, str) and project.ID_RE.fullmatch(pid):
        try:
            cap = _cap(pid, cfg)
            data = project.read_json(project.path(pid) / "ledger.json", None)
            entries = data.get("entries") if isinstance(data, dict) else None
            if isinstance(entries, list):
                for entry in entries:
                    if (
                        isinstance(entry, dict)
                        and entry.get("status") in ledger.SPENT_STATUSES
                    ):
                        spent += _money(entry.get("usd"))
        except Exception:
            spent = 0.0
    return {
        "this_call_usd": _money(this_call),
        "spent_usd": _money(spent),
        "cap_usd": _money(cap),
    }


def _stages(pid) -> dict:
    result = {stage: "pending" for stage in state.STAGES}
    if not isinstance(pid, str) or not project.ID_RE.fullmatch(pid):
        return result
    try:
        data = state.load(pid)["stages"]
    except Exception:
        return result
    for stage in state.STAGES:
        entry = data.get(stage)
        if isinstance(entry, dict):
            result[stage] = entry.get("status", "pending")
        elif isinstance(entry, str):
            result[stage] = entry
    return result


def _item_statuses(pid: str) -> dict:
    try:
        entries = state.load(pid)["items"]
    except Exception:
        return {}
    result = {}
    for key, entry in entries.items():
        if isinstance(entry, dict):
            result[key] = entry.get("status", "pending")
        elif isinstance(entry, str):
            result[key] = entry
    return result


def _payload(
    cmd: str,
    *,
    pid=None,
    cfg=None,
    stage=None,
    items=None,
    errors=None,
    this_call: float = 0.0,
    estimates=None,
    extra=None,
) -> dict:
    errors = [_tag(error) for error in (errors or [])]
    payload = {
        "ok": not errors,
        "cmd": cmd,
        "id": pid,
        "stage": stage,
        "items": list(items or []),
        "cost": _cost(pid, cfg, this_call),
        "stages": _stages(pid),
        "errors": errors,
    }
    if estimates is not None:
        payload["estimates"] = {key: _money(value) for key, value in estimates.items()}
    if extra:
        payload.update(extra)
    return payload


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


# -- reading the working tree -------------------------------------------


def _reader_engine(pid: str):
    return types.SimpleNamespace(pid=pid, project_dir=project.path(pid))


def _read_stage(pid: str, stage: str) -> list[dict]:
    """The existing items of an approved stage, without calling a model."""
    reader = stages._READERS.get(stage)
    if reader is None:
        return []
    try:
        rows = reader(_reader_engine(pid))
    except Exception:
        return []
    stages._stamp(pid, rows, save=False)
    return rows


def _items_for(pid: str, stage: str | None = None) -> list[dict]:
    names = [stage] if stage else state.STAGES
    rows: list[dict] = []
    for name in names:
        rows.extend(_read_stage(pid, name))
    return rows


def _item_rows(pid: str, keys: list[str]) -> list[dict]:
    current = bible.load(pid).get("current") or {}
    rows = []
    for key in keys:
        try:
            version = int(current.get(key, 0))
        except (TypeError, ValueError):
            version = 0
        rows.append(
            {"item": key, "version": version, "kind": None, "text": None, "paths": []}
        )
    return rows


def _messages(pid: str) -> dict:
    raw = project.read_json(project.path(pid) / "messages.json", None)
    return raw if isinstance(raw, dict) else {}


# -- shared argument checks ---------------------------------------------


def _check_stage(stage: str) -> str:
    if stage not in state.STAGES:
        raise item_ids.ItemError(f"unknown stage: {stage}")
    return stage


def _require_project(pid: str) -> Path:
    path = project.path(pid)
    if not path.is_dir():
        raise project.ProjectError(f"no such project: {pid}")
    return path


def _check_cap(value) -> float:
    try:
        cap = float(value)
    except (TypeError, ValueError):
        raise item_ids.ItemError(f"cap must be a number: {value!r}")
    if not math.isfinite(cap) or not 0 <= cap <= MAX_CAP_USD:
        raise item_ids.ItemError(f"cap must be between 0 and {MAX_CAP_USD:g} USD")
    return cap


def _read_file(name: str, what: str) -> str:
    try:
        return Path(name).read_text(encoding="utf-8")
    except OSError as exc:
        raise item_ids.ItemError(f"cannot read {what}: {exc}")


# -- engine --------------------------------------------------------------


def _fake_providers():
    """Test-only providers, behind ``HERMES_STORY_FAKE=1`` and pytest."""
    if os.environ.get("HERMES_STORY_FAKE") != "1":
        return None
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    from tests.hermes_story import fakes

    failure = os.environ.get("HERMES_STORY_FAKE_FAIL")
    chat = (
        fakes.FakeFailChat(failure)
        if failure
        else fakes.FakeStageChat(fakes.canned.TEXT_FLOW)
    )
    return {"chat": chat, "image": fakes.FakeImageGen(), "video": fakes.FakeVideoGen()}


def _engine(pid: str, cfg):
    from hermes_story.engine import Engine

    return Engine(pid, cfg, **(_fake_providers() or {}))


# -- commands ------------------------------------------------------------


def cmd_new(args) -> dict:
    cfg = config.load()
    idea = _read_file(args.idea_file, "idea file").strip()
    if not idea:
        raise item_ids.ItemError("idea file is empty")
    cap = _check_cap(args.cap if args.cap is not None else cfg.default_cap_usd)

    now = datetime.datetime.now()
    pid = project.new_id(idea, now)
    suffix = 2
    while project.path(pid).exists():
        pid = project.new_id(f"{idea} {suffix}", now)
        suffix += 1

    options = {
        "idea": idea,
        "style": args.style or "cinematic",
        "shape": args.shape or "16:9",
        "length_s": int(args.length),
        "cap_usd": cap,
        "duration_s": cfg.video_duration_s,
    }
    project.create(pid, options)
    return _payload("new", pid=pid, cfg=cfg, extra={"options": options})


def cmd_status(args) -> dict:
    cfg = config.load()
    _require_project(args.id)
    return _payload(
        "status",
        pid=args.id,
        cfg=cfg,
        items=_items_for(args.id),
        extra={"item_statuses": _item_statuses(args.id)},
    )


def cmd_list(args) -> dict:
    cfg = config.load()
    projects = []
    root = project.root()
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if child.is_dir() and project.ID_RE.fullmatch(child.name):
                projects.append(
                    {
                        "id": child.name,
                        "stages": _stages(child.name),
                        "cost": _cost(child.name, cfg),
                    }
                )
    return _payload("list", cfg=cfg, extra={"projects": projects})


def cmd_estimate(args) -> dict:
    cfg = config.load()
    _require_project(args.id)
    if args.stage:
        stage = _check_stage(args.stage)
        value = estimate.stage_estimate(args.id, stage, cfg)
        return _payload(
            "estimate",
            pid=args.id,
            cfg=cfg,
            stage=stage,
            this_call=value,
            estimates={stage: value},
        )
    values = estimate.film_estimate(args.id, cfg)
    return _payload(
        "estimate",
        pid=args.id,
        cfg=cfg,
        this_call=sum(values.values()),
        estimates=values,
    )


def cmd_run(args) -> dict:
    cfg = config.load()
    _require_project(args.id)
    stage = _check_stage(args.stage)

    with project.lock(args.id):
        if state.stage_status(args.id, stage) == "approved":
            return _payload(
                "run",
                pid=args.id,
                cfg=cfg,
                stage=stage,
                items=_read_stage(args.id, stage),
            )

        state.require_ready(args.id, stage)

        if stage in state.PAID and not args.approved_spend:
            value = estimate.stage_estimate(args.id, stage, cfg)
            return _payload(
                "run",
                pid=args.id,
                cfg=cfg,
                stage=stage,
                errors=[
                    f"needs_approval {stage} estimated ${value:.2f}; "
                    "re-run with --approved-spend"
                ],
                estimates={stage: value},
            )

        eng = _engine(args.id, cfg)
        with contextlib.redirect_stdout(sys.stderr):
            result = asyncio.run(stages.run_stage(eng, stage))

    return _payload(
        "run",
        pid=args.id,
        cfg=cfg,
        stage=stage,
        items=result.items,
        errors=result.errors,
        this_call=result.cost_usd,
    )


def cmd_approve(args) -> dict:
    cfg = config.load()
    _require_project(args.id)
    stage = _check_stage(args.stage)
    with project.lock(args.id):
        if state.stage_status(args.id, stage) != "review":
            raise state.NotReady(f"{stage} is not in review")
        state.set_stage(args.id, stage, "approved")
    return _payload(
        "approve",
        pid=args.id,
        cfg=cfg,
        stage=stage,
        items=_read_stage(args.id, stage),
    )


def cmd_retake(args) -> dict:
    cfg = config.load()
    _require_project(args.id)
    parsed = item_ids.parse(args.item)
    stage = item_ids.STAGE_OF[parsed.kind]
    feedback = _read_file(args.feedback_file, "feedback file")

    with project.lock(args.id):
        if parsed.kind in PAID_ITEM_KINDS and not args.approved_spend:
            item_id = item_ids.fmt(parsed)
            value = estimate.item_estimate(args.id, args.item, cfg)
            return _payload(
                "retake",
                pid=args.id,
                cfg=cfg,
                stage=stage,
                errors=[
                    f"needs_approval {item_id} estimated ${value:.2f}; "
                    "re-run with --approved-spend"
                ],
                estimates={item_id: value},
            )

        eng = _engine(args.id, cfg)
        with contextlib.redirect_stdout(sys.stderr):
            result = asyncio.run(retake.retake(eng, args.item, feedback))

    return _payload(
        "retake",
        pid=args.id,
        cfg=cfg,
        stage=result.stage or stage,
        items=result.items,
        errors=result.errors,
        this_call=result.cost_usd,
    )


def cmd_edit(args) -> dict:
    cfg = config.load()
    _require_project(args.id)
    new_text = _read_file(args.text_file, "text file")
    with project.lock(args.id):
        result = retake.edit(args.id, args.item, new_text)
    return _payload(
        "edit",
        pid=args.id,
        cfg=cfg,
        stage=result.stage,
        items=result.items,
        errors=result.errors,
        this_call=result.cost_usd,
    )


def cmd_cascade(args) -> dict:
    cfg = config.load()
    _require_project(args.id)
    keys = retake.cascade(args.id, args.item)
    return _payload(
        "cascade",
        pid=args.id,
        cfg=cfg,
        items=_item_rows(args.id, keys),
        extra={"cascade": keys},
    )


def cmd_invalidate(args) -> dict:
    cfg = config.load()
    _require_project(args.id)
    with project.lock(args.id):
        touched = retake.invalidate(args.id, args.items)
    return _payload(
        "invalidate",
        pid=args.id,
        cfg=cfg,
        items=_item_rows(args.id, touched),
        extra={"invalidated": touched},
    )


def cmd_cameo_bind(args) -> dict:
    cfg = config.load()
    _require_project(args.id)
    parsed = item_ids.parse(f"char:{args.character}")
    person = args.person.strip()
    if person.lower() == "none":
        person = None
    with project.lock(args.id):
        bible.set_cameo(args.id, parsed.name, person)
    return _payload(
        "cameo-bind",
        pid=args.id,
        cfg=cfg,
        extra={"character": parsed.name, "cameo": person},
    )


def cmd_cap(args) -> dict:
    cfg = config.load()
    _require_project(args.id)
    cap = _check_cap(args.usd)
    path = project.path(args.id) / "ledger.json"
    with project.lock(args.id):
        data = project.read_json(path, None)
        entries = []
        if isinstance(data, dict) and isinstance(data.get("entries"), list):
            entries = data["entries"]
        project.write_json(path, {"cap_usd": cap, "entries": entries})
    return _payload("cap", pid=args.id, cfg=cfg, extra={"cap_usd": _money(cap)})


def cmd_msg_record(args) -> dict:
    cfg = config.load()
    _require_project(args.id)
    parsed = item_ids.parse(args.item)
    version = int(args.version)
    message_id = str(args.message_id)
    messages = _messages(args.id)
    messages[message_id] = {"item": item_ids.fmt(parsed), "version": version}
    project.write_json(project.path(args.id) / "messages.json", messages)
    return _payload(
        "msg-record",
        pid=args.id,
        cfg=cfg,
        extra={"message_id": message_id, "recorded": messages[message_id]},
    )


def cmd_msg_lookup(args) -> dict:
    cfg = config.load()
    _require_project(args.id)
    message_id = str(args.message_id)
    entry = _messages(args.id).get(message_id)
    rows = []
    if isinstance(entry, dict):
        rows.append(
            {
                "item": entry.get("item"),
                "version": entry.get("version"),
                "kind": None,
                "text": None,
                "paths": [],
            }
        )
    return _payload(
        "msg-lookup",
        pid=args.id,
        cfg=cfg,
        items=rows,
        extra={"message_id": message_id},
    )


HANDLERS = {
    "new": cmd_new,
    "status": cmd_status,
    "list": cmd_list,
    "estimate": cmd_estimate,
    "run": cmd_run,
    "approve": cmd_approve,
    "retake": cmd_retake,
    "edit": cmd_edit,
    "cascade": cmd_cascade,
    "invalidate": cmd_invalidate,
    "cameo-bind": cmd_cameo_bind,
    "cap": cmd_cap,
    "msg-record": cmd_msg_record,
    "msg-lookup": cmd_msg_lookup,
}


# -- argparse ------------------------------------------------------------


def _length(value: str) -> int:
    try:
        seconds = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"length must be whole seconds: {value!r}")
    if not 4 <= seconds <= 180:
        raise argparse.ArgumentTypeError("length must be 4 to 180 seconds")
    return seconds


class _UsageError(Exception):
    """Bad command line; reported as a JSON ``bad_item`` error."""


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise _UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="python -m hermes_story", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)

    new = sub.add_parser("new", help="create a film from an idea file")
    new.add_argument("--idea-file", required=True)
    new.add_argument("--style", default="cinematic")
    new.add_argument("--shape", choices=["16:9", "9:16"], default="16:9")
    new.add_argument("--length", type=_length, default=30)
    new.add_argument("--cap", type=float, default=None)

    status = sub.add_parser("status", help="show one film's stages and items")
    status.add_argument("id")

    sub.add_parser("list", help="list every film")

    est = sub.add_parser("estimate", help="estimate the cost of one or all stages")
    est.add_argument("id")
    est.add_argument("stage", nargs="?", default=None)

    run = sub.add_parser("run", help="run one stage")
    run.add_argument("id")
    run.add_argument("stage")
    run.add_argument("--approved-spend", action="store_true")

    approve = sub.add_parser("approve", help="approve a stage in review")
    approve.add_argument("id")
    approve.add_argument("stage")

    retake = sub.add_parser("retake", help="retake one item with feedback")
    retake.add_argument("id")
    retake.add_argument("item")
    retake.add_argument("--feedback-file", required=True)
    retake.add_argument("--approved-spend", action="store_true")

    edit = sub.add_parser("edit", help="replace a text item without a model call")
    edit.add_argument("id")
    edit.add_argument("item")
    edit.add_argument("--text-file", required=True)

    cascade = sub.add_parser("cascade", help="list the items that depend on an item")
    cascade.add_argument("id")
    cascade.add_argument("item")

    invalidate = sub.add_parser("invalidate", help="discard items so they regenerate")
    invalidate.add_argument("id")
    invalidate.add_argument("items", nargs="+")

    cameo = sub.add_parser("cameo-bind", help="attach a consenting person to a character")
    cameo.add_argument("id")
    cameo.add_argument("character")
    cameo.add_argument("person")

    cap = sub.add_parser("cap", help="set the film's spending cap")
    cap.add_argument("id")
    cap.add_argument("usd")

    record = sub.add_parser("msg-record", help="remember which item a message made")
    record.add_argument("id")
    record.add_argument("message_id")
    record.add_argument("item")
    record.add_argument("version")

    lookup = sub.add_parser("msg-lookup", help="look up the item a message made")
    lookup.add_argument("id")
    lookup.add_argument("message_id")

    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except _UsageError as exc:
        _emit(_payload("", errors=[f"bad_item {exc}"]))
        return 1

    handler = HANDLERS[args.command]
    try:
        payload = handler(args)
    except Exception as exc:
        payload = _payload(
            args.command,
            pid=getattr(args, "id", None),
            errors=[_error_text(exc)],
        )
    _emit(payload)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
