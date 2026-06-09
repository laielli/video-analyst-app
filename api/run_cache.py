"""
Content-addressed cache for grounded free-text run-docs + the substrate for shareable
permalinks. A free-text question fires a BILLED Azure OpenAI codegen call on an
unauthenticated GET; this module lets an identical repeat question be served for free and
deterministically by keying a stored (validated, grounded) run-doc on the normalized
(query_text, clip) pair.

KEY-NORMALIZATION CONTRACT (the cache's correctness hinges on this):
    cache_key(text, clip_id) = sha256(f"{clip_id}\\x00{normalize_query(text)}").hexdigest()

  - The clip id enters the key RAW (it is already a controlled enum from canned.CLIPS); only
    the free-text question is normalized. A NUL separator keeps clip/text unambiguous.
  - normalize_query is DELIBERATELY conservative (the one design knob): strip + casefold +
    collapse internal whitespace, and NOTHING else (no punctuation stripping, no synonym /
    lemmatization). Rationale: two *meaningfully different* questions must never collide
    (a false merge would serve a wrong cached answer); a slightly lower hit-rate is the safe
    trade. "Does #10 SCORE?" and "  does #10 score?  " share a key; "does #10 score?" and
    "does #11 score?" do NOT.
  - The digest is the FULL sha256 hex (64 chars, ^[0-9a-f]{64}$) and is ALSO the permalink
    `run_id`, so a permalink is stable across processes and identical across repeat questions.
    The route guard depends on the fixed length — never truncate.

CACHE_FORMAT is the deliberate kill-switch: bumping it invalidates every stale entry on read
(treated as a miss + self-healed) without a wall-clock TTL. Bump it whenever the run-doc shape
or the codegen prompt changes such that old cached programs are no longer trustworthy.

The store is runtime-only, gitignored, single-process best-effort (atomic per-entry writes via
temp + os.replace; a tolerant evict-oldest-by-mtime sweep on put). The RunStore Protocol is the
seam a future BlobRunStore slots into.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Protocol

import jsonschema

API_DIR = Path(__file__).resolve().parent
_RD_SCHEMA = json.loads((API_DIR / "schema" / "run_doc.schema.json").read_text())
_RD_VALIDATOR = jsonschema.Draft202012Validator(_RD_SCHEMA)

# Bump to invalidate every stored entry (the no-TTL kill-switch for prompt/schema drift).
CACHE_FORMAT = 1

# A well-formed key is exactly a sha256 hex digest. Used both to derive a key and to refuse a
# path-traversing query param before it ever composes a store path.
KEY_RE = re.compile(r"^[0-9a-f]{64}$")

_WS_RE = re.compile(r"\s+")


def normalize_query(text: str) -> str:
    """Normalize free-text for cache keying: strip, casefold, collapse internal whitespace.

    Conservative by design (see the module docstring's contract) — no punctuation stripping,
    no synonym mapping — so two meaningfully different questions never share a key."""
    t = text.strip().casefold()
    t = _WS_RE.sub(" ", t)
    return t


def cache_key(text: str, clip_id: str) -> str:
    """Content-address (text, clip) as a full sha256 hex digest. This IS the permalink run_id.

    The clip id is RAW (controlled enum); only the question is normalized. A NUL byte separates
    them so no (clip, text) pair can alias another."""
    payload = f"{clip_id}\x00{normalize_query(text)}".encode()
    return hashlib.sha256(payload).hexdigest()


class RunStore(Protocol):
    """The persistence seam. A file-backed impl ships now; a future BlobRunStore implements the
    same two methods. `get` returns the stored run-doc or None (miss / corrupt / stale); `put`
    stores a run-doc envelope. Both are total and never raise on a bad-on-disk entry."""

    def get(self, key: str) -> dict | None: ...

    def put(self, key: str, doc: dict) -> None: ...


def _wrap(doc: dict) -> dict:
    """The stored envelope: a version wrapper around the schema-pure run-doc (so run_id/cached
    never have to live inside the doc — they are derived at serve time by the route)."""
    return {"cache_format": CACHE_FORMAT, "doc": doc}


class FileRunStore:
    """One JSON file per entry at root/<key>.json, written atomically (temp + os.replace). The
    directory listing IS the index; eviction is evict-oldest-by-mtime (FIFO by WRITE time, not
    true LRU — reads do not touch mtime, and we deliberately do NOT touch-on-read to fake LRU).

    get() self-heals: any parse failure, CACHE_FORMAT mismatch, or run_doc schema failure is a
    MISS and the bad file is best-effort deleted. Invalid keys (path traversal) are refused."""

    def __init__(self, root: Path, *, max_entries: int | None = 1000):
        self.root = Path(root)
        self.max_entries = max_entries
        self.root.mkdir(parents=True, exist_ok=True)

    # -- path safety -----------------------------------------------------------------------
    def _path_for(self, key: str) -> Path | None:
        """Resolve key -> file path, or None if the key is unsafe. Defense in depth: never let a
        raw query param compose a store path (run=../../etc/passwd must not become a file read)."""
        if not isinstance(key, str) or not KEY_RE.match(key):
            return None
        return self.root / f"{key}.json"

    # -- read ------------------------------------------------------------------------------
    def get(self, key: str) -> dict | None:
        path = self._path_for(key)
        if path is None:
            return None
        try:
            raw = path.read_text()
        except (FileNotFoundError, OSError):
            return None
        try:
            envelope = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            self._best_effort_delete(path)  # self-heal corrupt JSON
            return None
        if not isinstance(envelope, dict) or envelope.get("cache_format") != CACHE_FORMAT:
            self._best_effort_delete(path)  # stale format -> kill-switch
            return None
        doc = envelope.get("doc")
        if not isinstance(doc, dict) or self._schema_errors(doc):
            self._best_effort_delete(path)  # schema-invalid -> never stream it
            return None
        return doc

    # -- write -----------------------------------------------------------------------------
    def put(self, key: str, doc: dict) -> None:
        path = self._path_for(key)
        if path is None:
            return  # refuse to write at an unsafe key
        text = json.dumps(_wrap(doc), indent=2, ensure_ascii=False) + "\n"
        self._atomic_write(path, text)
        self._evict()

    # -- helpers ---------------------------------------------------------------------------
    @staticmethod
    def _schema_errors(doc: dict) -> list:
        return list(_RD_VALIDATOR.iter_errors(doc))

    @staticmethod
    def _best_effort_delete(path: Path) -> None:
        try:
            path.unlink()
        except OSError:
            pass

    def _atomic_write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(text)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                self._best_effort_delete(Path(tmp))

    def _evict(self) -> None:
        """Soft cap: if over max_entries, delete oldest-by-mtime first. Tolerant of races (a file
        that vanished between listdir and stat is just skipped) — no locks, best-effort."""
        if self.max_entries is None:
            return
        try:
            entries = list(self.root.glob("*.json"))
        except OSError:
            return
        if len(entries) <= self.max_entries:
            return

        def _mtime(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except OSError:
                return float("inf")  # vanished -> sort last, skip on unlink

        entries.sort(key=_mtime)
        overflow = len(entries) - self.max_entries
        for p in entries[:overflow]:
            self._best_effort_delete(p)
