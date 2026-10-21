"""Project ids, directories, atomic JSON files and the per-film lock."""

import contextlib
import datetime
import fcntl
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path

ID_RE = re.compile(r"^[0-9]{8}-[0-9]{4}-[a-z0-9-]{1,40}$")
REPO = Path(__file__).resolve().parent.parent


class ProjectError(Exception):
    """Invalid project id, path or layout."""


class Busy(Exception):
    """Another process holds the project lock."""


def root() -> Path:
    return Path(os.environ.get("HERMES_STORY_ROOT") or REPO / "projects").resolve()


def new_id(slug_source: str, now: datetime.datetime) -> str:
    s = unicodedata.normalize("NFKD", slug_source).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")[:40].strip("-") or "story"
    return f"{now:%Y%m%d-%H%M}-{s}"


def path(pid: str) -> Path:
    if not isinstance(pid, str) or not ID_RE.fullmatch(pid):
        raise ProjectError(f"bad project id: {pid!r}")
    return root() / pid


def safe_child(base: Path, *parts: str) -> Path:
    base = Path(base).resolve()
    cur = base
    for part in parts:
        if part in ("", ".", "..") or "/" in part or "\x00" in part:
            raise ProjectError(f"bad path part: {part!r}")
        cur = cur / part
        if cur.is_symlink():
            raise ProjectError(f"symlink in path: {cur}")
    resolved = cur.resolve()
    if resolved != base and not resolved.is_relative_to(base):
        raise ProjectError(f"path escapes project: {cur}")
    return cur


def read_json(p: Path, default):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def write_json(p: Path, obj) -> None:
    p = Path(p)
    fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, p)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def create(pid: str, options: dict) -> Path:
    p = path(pid)
    root().mkdir(parents=True, exist_ok=True, mode=0o700)
    p.mkdir(mode=0o700, exist_ok=False)
    (p / "work").mkdir(mode=0o700)
    (p / "versions").mkdir(mode=0o700)
    write_json(
        p / "bible.json",
        {"id": pid, "options": options, "characters": {}, "current": {}, "revisions": []},
    )
    write_json(p / "state.json", {"stages": {}, "items": {}, "errors": []})
    write_json(p / "ledger.json", {"cap_usd": options.get("cap_usd"), "entries": []})
    write_json(p / "messages.json", {})
    return p


@contextlib.contextmanager
def lock(pid: str):
    f = open(path(pid) / ".lock", "a+")
    try:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Busy(pid)
        yield
    finally:
        f.close()
