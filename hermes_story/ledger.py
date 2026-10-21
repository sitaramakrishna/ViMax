"""The per-project ``ledger.json``: reservations, costs and the per-film cap.

Every paid call is reserved before it starts, then settled as ``committed``
(the call returned), ``uncertain`` (the request may have reached the provider)
or ``released`` (the call failed before anything was sent).  Only ``committed``
and ``uncertain`` entries count as spent; open reservations count against the
cap so that concurrent calls cannot overrun it.
"""

from __future__ import annotations

import asyncio
import datetime
import os
import threading
import uuid
from pathlib import Path

from hermes_story import project

#: Entries in these states have been paid for.
SPENT_STATUSES = ("committed", "uncertain")

# One mutex per ledger file, so two Ledger instances in one process cannot
# interleave a read-modify-write.
_MUTEXES: dict[Path, threading.RLock] = {}
_ASYNC_LOCKS: dict[Path, asyncio.Lock] = {}
_REGISTRY_GUARD = threading.Lock()


def _mutex(path: Path) -> threading.RLock:
    with _REGISTRY_GUARD:
        return _MUTEXES.setdefault(path, threading.RLock())


def _async_lock(path: Path) -> asyncio.Lock:
    with _REGISTRY_GUARD:
        return _ASYNC_LOCKS.setdefault(path, asyncio.Lock())


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _money(value) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


class LedgerError(Exception):
    """Malformed ledger data or an unknown entry."""


class CapExceeded(Exception):
    """A call would push the film over its spending cap."""

    def __init__(self, spent: float, est: float, cap: float) -> None:
        self.spent = float(spent)
        self.est = float(est)
        self.cap = float(cap)
        super().__init__(
            f"cap ${self.cap:.2f} exceeded: ${self.spent:.2f} spent + ${self.est:.2f} estimate"
        )


def _spent(entries: list) -> float:
    total = 0.0
    for entry in entries:
        if isinstance(entry, dict) and entry.get("status") in SPENT_STATUSES:
            total += _money(entry.get("usd"))
    return total


def _open(entries: list) -> float:
    total = 0.0
    for entry in entries:
        if isinstance(entry, dict) and entry.get("status") == "reserved":
            total += _money(entry.get("est"))
    return total


class Ledger:
    """Read/write access to ``ledger.json`` for one project."""

    def __init__(self, pid: str, cap_usd: float) -> None:
        self.pid = pid
        self.cap_usd = float(cap_usd)
        self.path = project.path(pid) / "ledger.json"
        self._mutex = _mutex(self.path)
        #: Serialises callers that want to reserve from async code.
        self.lock = _async_lock(self.path)
        self._settle_stale()

    @property
    def cap(self) -> float:
        return self.cap_usd

    @cap.setter
    def cap(self, value: float) -> None:
        self.cap_usd = float(value)

    def _settle_stale(self) -> None:
        """A reservation left by another (crashed) process may have been sent: charge it."""
        with self._mutex:
            entries = self._load()
            changed = False
            for entry in entries:
                if isinstance(entry, dict) and entry.get("status") == "reserved" and entry.get("proc") != os.getpid():
                    entry["status"] = "uncertain"
                    entry["usd"] = _money(entry.get("est"))
                    changed = True
            if changed:
                self._save(entries)

    # -- storage ---------------------------------------------------------

    def _load(self) -> list:
        data = project.read_json(self.path, None)
        if not isinstance(data, dict):
            return []
        entries = data.get("entries")
        return entries if isinstance(entries, list) else []

    def _save(self, entries: list) -> None:
        project.write_json(self.path, {"cap_usd": self.cap_usd, "entries": entries})

    # -- queries ---------------------------------------------------------

    def entries(self) -> list[dict]:
        """A copy of every ledger entry, oldest first."""
        with self._mutex:
            return [dict(entry) for entry in self._load() if isinstance(entry, dict)]

    def spent(self) -> float:
        """Total cost of committed and uncertain entries."""
        with self._mutex:
            return _spent(self._load())

    # -- transitions -----------------------------------------------------

    def reserve(self, kind: str, item: str, est_usd: float, *, model: str = "") -> str:
        """Reserve ``est_usd`` for ``item`` and return the new entry id."""
        est = _money(est_usd)
        with self._mutex:
            entries = self._load()
            spent = _spent(entries)
            if spent + _open(entries) + est > self.cap_usd + 1e-9:
                raise CapExceeded(spent, est, self.cap_usd)
            entry = {
                "id": uuid.uuid4().hex,
                "kind": kind,
                "item": item,
                "model": model,
                "est": est,
                "usd": None,
                "status": "reserved",
                "proc": os.getpid(),
                "at": _now(),
            }
            entries.append(entry)
            self._save(entries)
            return entry["id"]

    def commit(self, entry_id: str, usd: float) -> None:
        """Settle a reservation at its actual cost."""
        self._set(entry_id, status="committed", usd=_money(usd))

    def uncertain(self, entry_id: str) -> None:
        """The request may have reached the provider: charge the estimate."""
        with self._mutex:
            entries = self._load()
            entry = self._find(entries, entry_id)
            entry["usd"] = _money(entry.get("est"))
            entry["status"] = "uncertain"
            self._save(entries)

    def release(self, entry_id: str) -> None:
        """The call failed before submission: it costs nothing."""
        self._set(entry_id, status="released", usd=0.0)

    # -- internals -------------------------------------------------------

    @staticmethod
    def _find(entries: list, entry_id: str) -> dict:
        for entry in entries:
            if isinstance(entry, dict) and entry.get("id") == entry_id:
                return entry
        raise LedgerError(f"unknown ledger entry: {entry_id!r}")

    def _set(self, entry_id: str, *, status: str, usd: float) -> None:
        with self._mutex:
            entries = self._load()
            entry = self._find(entries, entry_id)
            entry["usd"] = usd
            entry["status"] = status
            self._save(entries)
