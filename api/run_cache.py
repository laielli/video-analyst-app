"""
Content-addressed cache + persistence for free-text run-docs (the cost lever + the permalink
substrate). A free-text question fires a BILLED Azure OpenAI codegen call on an unauthenticated
GET; this layer serves an identical repeat question for free and deterministically, and turns the
same content-address into a stable shareable permalink (`run_id`).

KEY-NORMALIZATION CONTRACT (the only deliberately-open knob — see the plan's DESIGN-ROOM):
    The cache is content-addressed on (normalized_query_text, clip_id).
    - clip_id enters the key RAW — it is already a controlled enum (canned.CLIPS), not user prose.
    - The free-text question is normalized by `normalize_query` with the CONSERVATIVE default:
        casefold + strip + collapse internal whitespace, and NOTHING ELSE.
      No punctuation stripping, no synonym/lemmatization. This is the lowest-false-merge-risk
      choice: two *meaningfully different* questions never collide (a wrong merge would serve a
      stale answer for a paid question — the worst failure for a cost cache). It still merges the
      free, obvious variants (case, leading/trailing/internal whitespace).
    - The key is a fixed sha256 hex digest of f"{clip_id}\\x00{normalize_query(text)}". The NUL
      separator keeps clip_id and query unambiguous. The key derivation is FIXED; only the
      normalizer is the knob. Changing the normalizer changes existing keys, so bump CACHE_FORMAT.

`run_id` IS the cache key (full sha256 hex, no truncation) — sharing the same question twice
yields the same link. The `^[0-9a-f]{64}$` route guard depends on this fixed length.

PERSISTENCE: one JSON file per entry under `root/<key>.json`, written atomically (temp + os.replace),
mirroring api/scripts/precompute.py's atomic-write discipline. Each entry is an envelope
`{"cache_format": CACHE_FORMAT, "doc": <run-doc>}`. `get` validates the version AND the doc against
run_doc.schema.json; any parse/version/schema failure is a MISS and the bad file is best-effort
deleted (self-healing). Bumping CACHE_FORMAT is the kill-switch that invalidates stale entries
without a TTL. Eviction is a soft cap (evict-oldest-by-mtime, FIFO — NOT LRU) on `put`.

The `RunStore` Protocol + constructor injection is the Azure-Blob seam: a future BlobRunStore
implements the same get/put. Concurrency posture is single-process, best-effort: atomic per-entry
writes + a tolerant eviction sweep, no file locks.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Protocol, runtime_checkable

import jsonschema

API_DIR = Path(__file__).resolve().parent
_RD_SCHEMA = json.loads((API_DIR / "schema" / "run_doc.schema.json").read_text())

# Bump to invalidate ALL stored entries immediately (the no-TTL kill-switch). Bump whenever the
# run-doc format, the codegen prompt, or the normalizer changes in a way that would make a stored
# entry stale or mis-keyed.
CACHE_FORMAT = 1

_WS_RE = re.compile(r"\s+")
_KEY_RE = re.compile(r"^[0-9a-f]{64}$")


def normalize_query(text: str) -> str:
    """Conservative normalizer (see module docstring): casefold + strip + collapse internal
    whitespace, nothing else. Case- and whitespace-insensitive; meaning-preserving."""
    t = text.strip().casefold()
    t = _WS_RE.sub(" ", t)
    return t


def cache_key(text: str, clip_id: str) -> str:
    """The content-address: sha256 hex of (clip_id RAW, NUL, normalized question). Also the run_id.
    Stable across processes, so a permalink survives restarts."""
    payload = f"{clip_id}\x00{normalize_query(text)}".encode()
    return hashlib.sha256(payload).hexdigest()


def is_valid_key(key: str) -> bool:
    """A well-formed key is exactly 64 lowercase hex chars (the sha256 digest shape). Anything
    else (path traversal, wrong length, uppercase) is rejected — defense in depth so a raw query
    param can never compose a file path."""
    return bool(_KEY_RE.match(key))


def _schema_ok(doc: object) -> bool:
    try:
        jsonschema.Draft202012Validator(_RD_SCHEMA).validate(doc)
        return True
    except jsonschema.ValidationError:
        return False
    except Exception:  # noqa: BLE001 — any schema oddity is treated as invalid -> miss
        return False


@runtime_checkable
class RunStore(Protocol):
    """The persistence seam. A future BlobRunStore implements these two methods unchanged."""

    def get(self, key: str) -> dict | None: ...

    def put(self, key: str, doc: dict) -> None: ...


class FileRunStore:
    """File-backed RunStore: one JSON envelope per entry at root/<key>.json.

    - get: read + parse + version-check + schema-validate the wrapped doc; any failure is a miss
      and the bad file is best-effort deleted (self-healing). Refuses malformed keys outright.
    - put: schema-validate before writing (never persist a doc we'd reject on read), write
      atomically, then enforce the soft entry cap by evicting oldest-by-mtime.
    """

    def __init__(self, root: Path | str, *, max_entries: int | None = 1000):
        self.root = Path(root)
        self.max_entries = max_entries

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> dict | None:
        # Defense in depth: never let a raw query param compose a path. (run=../../etc/passwd etc.)
        if not is_valid_key(key):
            return None
        path = self._path(key)
        try:
            raw = path.read_text()
        except (FileNotFoundError, OSError):
            return None
        try:
            envelope = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            self._best_effort_delete(path)  # corrupt JSON -> self-heal
            return None
        if not isinstance(envelope, dict) or envelope.get("cache_format") != CACHE_FORMAT:
            self._best_effort_delete(path)  # stale format (kill-switch) -> self-heal
            return None
        doc = envelope.get("doc")
        if not isinstance(doc, dict) or not _schema_ok(doc):
            self._best_effort_delete(path)  # schema-invalid -> self-heal
            return None
        return doc

    def put(self, key: str, doc: dict) -> None:
        if not is_valid_key(key):
            raise ValueError(f"invalid cache key: {key!r}")
        # Never persist a doc we'd reject on read (keeps the cache self-consistent).
        if not _schema_ok(doc):
            raise ValueError("refusing to cache a run-doc that fails run_doc.schema.json")
        envelope = {"cache_format": CACHE_FORMAT, "doc": doc}
        text = json.dumps(envelope, indent=2, ensure_ascii=False) + "\n"
        self._atomic_write(self._path(key), text)
        self._evict()

    # ---- internals -------------------------------------------------------------------------

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(text)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    @staticmethod
    def _best_effort_delete(path: Path) -> None:
        try:
            path.unlink()
        except OSError:
            pass

    def _evict(self) -> None:
        """Soft cap: keep at most max_entries .json files, evicting oldest-by-write-time first.
        FIFO by mtime, NOT LRU (reads don't touch mtime — deliberately, per the plan). Tolerant of
        races: a file vanishing mid-sweep is ignored (single-process, best-effort, no locks)."""
        if self.max_entries is None:
            return
        try:
            entries = list(self.root.glob("*.json"))
        except OSError:
            return
        if len(entries) <= self.max_entries:
            return
        # Sort oldest-first by mtime; tolerate files that vanish between glob and stat.
        def _mtime(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except OSError:
                return float("inf")  # missing -> sort last, effectively skipped

        entries.sort(key=_mtime)
        for path in entries[: len(entries) - self.max_entries]:
            self._best_effort_delete(path)
