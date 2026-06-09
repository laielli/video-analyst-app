"""
Content-addressed run cache + a swappable persistence seam.

WHY this exists
---------------
Every free-text question fires a BILLED Azure OpenAI codegen call on an unauthenticated
GET. Two identical questions therefore bill twice for the same deterministic answer. This
module is a content-addressed cache keyed on the normalized `(query_text, clip_id)` pair:
`build_free_text_run_doc` looks it up BEFORE codegen, so an identical repeat is served for
free. Because we only ever cache a run-doc that already PASSED validation AND grounded (the
trust boundary is respected), the same key doubles as a stable, shareable permalink id.

KEY-NORMALIZATION CONTRACT (the one DESIGN-ROOM knob)
-----------------------------------------------------
`normalize_query` is the MOST CONSERVATIVE option in the plan: **casefold + strip +
collapse internal whitespace, nothing else.** No punctuation stripping, no synonym/lemma
pass. Rationale: the lowest false-merge risk — two *meaningfully different* questions never
collide; we trade a slightly lower hit rate for never returning a stale/wrong cached verdict.
The key derivation (`cache_key`) is fixed: `sha256(f"{clip_id}\\x00{normalize_query(text)}")`
as a 64-char lowercase hex digest. The clip id is hashed RAW (it is a controlled enum from
`canned.CLIPS`); only the free-text question is normalized.

The `run_id` (permalink) IS this cache key — the full sha256 hex, never truncated. The
`^[0-9a-f]{64}$` route guard depends on the fixed 64-char length.

ENVELOPE + VERSIONING
---------------------
A stored entry is an envelope, not the bare doc: `{"cache_format": CACHE_FORMAT, "doc": ...}`.
`CACHE_FORMAT` is the deliberate kill-switch: bump it when the doc format or codegen prompt
changes and every stale entry is invalidated immediately, without a wall-clock TTL.
`get` validates the envelope version AND the `doc` against `run_doc.schema.json`; any parse
failure / version mismatch / schema failure is treated as a MISS and the bad file is
best-effort deleted (self-healing).

PERSISTENCE BACKEND
-------------------
One JSON file per entry at `root/<key>.json`, written atomically (temp + os.replace),
mirroring the discipline in `scripts/precompute.py`. The directory listing is the index.
Eviction is a soft cap (`max_entries`) with evict-OLDEST-by-mtime on `put` — FIFO by write
time, NOT true LRU (file mtime does not track reads; do not add touch-on-read). Concurrency
posture: single-process, best-effort — atomic per-entry writes + a race-tolerant sweep, no
file locks.

The `RunStore` Protocol + constructor injection is the Azure-Blob seam: a future
`BlobRunStore` implements the same `get`/`put`.
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

# Bump this when the run-doc shape or the codegen prompt changes — it invalidates exactly the
# stale entries on next read, immediately, with no TTL. (Phase 2 kill-switch.)
CACHE_FORMAT = 1

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "run_doc.schema.json"
_RUN_DOC_SCHEMA = json.loads(_SCHEMA_PATH.read_text())
_VALIDATOR = jsonschema.Draft202012Validator(_RUN_DOC_SCHEMA)

# A well-formed key is exactly the sha256 hex digest. The route guards on this too, but the
# store enforces it independently (defense in depth) so a raw query param can never compose a
# path (`../../etc/passwd`, `key/with/slashes`, etc.).
_KEY_RE = re.compile(r"^[0-9a-f]{64}$")

_WS_RE = re.compile(r"\s+")


# ---- key normalization + derivation ------------------------------------------------------

def normalize_query(text: str) -> str:
    """Normalize a free-text question to the cache-key form. CONSERVATIVE default: casefold +
    strip + collapse internal whitespace. Nothing else (no punctuation stripping, no synonyms)
    so two meaningfully different questions never collide. See the module docstring."""
    t = text.strip().casefold()
    t = _WS_RE.sub(" ", t)
    return t


def cache_key(text: str, clip_id: str) -> str:
    """Content address for `(query_text, clip_id)`: a 64-char sha256 hex digest. The clip id is
    hashed RAW (controlled enum); only the question is normalized. This IS the permalink run_id."""
    payload = f"{clip_id}\x00{normalize_query(text)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def is_valid_key(key: str) -> bool:
    """True iff `key` is a well-formed cache key / run_id (exactly 64 lowercase hex chars)."""
    return bool(_KEY_RE.match(key))


# ---- the persistence seam ----------------------------------------------------------------

class RunStore(Protocol):
    """The two-method seam the server depends on. A future BlobRunStore implements the same."""

    def get(self, key: str) -> dict | None: ...

    def put(self, key: str, doc: dict) -> None: ...


class FileRunStore:
    """File-backed RunStore: one JSON envelope per entry at `root/<key>.json`.

    `put` writes atomically (temp + os.replace) and enforces the `max_entries` soft cap by
    evicting the oldest file(s) by mtime. `get` validates the envelope version + the embedded
    run-doc against the schema, returns the bare doc on a hit, and on any failure treats it as
    a miss + best-effort deletes the bad file (self-healing)."""

    def __init__(self, root: Path | str, *, max_entries: int | None = None):
        self.root = Path(root)
        self.max_entries = max_entries

    # -- internal ------------------------------------------------------------------------

    def _path(self, key: str) -> Path:
        # Defense in depth: NEVER let a raw key compose a path. Reject anything that is not a
        # well-formed key (which excludes '/', '\\', '.', os.sep by construction).
        if not is_valid_key(key):
            raise ValueError(f"invalid cache key: {key!r}")
        return self.root / f"{key}.json"

    def _delete_quietly(self, path: Path) -> None:
        try:
            path.unlink()
        except OSError:
            pass  # best-effort self-heal; a concurrent delete is fine

    # -- RunStore -------------------------------------------------------------------------

    def get(self, key: str) -> dict | None:
        """Return the stored run-doc for `key`, or None on a miss. A miss is: no file, a key
        that is not well-formed, a parse error, a `cache_format` mismatch, or a doc that fails
        run_doc.schema.json. Every non-trivial failure best-effort deletes the bad file."""
        if not is_valid_key(key):
            return None
        path = self.root / f"{key}.json"
        try:
            raw = path.read_text()
        except OSError:
            return None  # plain miss (no file) — nothing to delete
        try:
            envelope = json.loads(raw)
        except (ValueError, TypeError):
            self._delete_quietly(path)  # corrupt JSON -> self-heal
            return None
        if not isinstance(envelope, dict) or envelope.get("cache_format") != CACHE_FORMAT:
            self._delete_quietly(path)  # stale format / wrong shape -> kill-switch / self-heal
            return None
        doc = envelope.get("doc")
        if not isinstance(doc, dict) or next(_VALIDATOR.iter_errors(doc), None) is not None:
            self._delete_quietly(path)  # schema-invalid -> self-heal
            return None
        return doc

    def put(self, key: str, doc: dict) -> None:
        """Store `doc` under `key` (atomic write), then evict oldest entries past the cap."""
        path = self._path(key)
        envelope = {"cache_format": CACHE_FORMAT, "doc": doc}
        text = json.dumps(envelope, ensure_ascii=False, indent=2) + "\n"
        self._atomic_write(path, text)
        self._evict()

    # -- atomic write + eviction ---------------------------------------------------------

    def _atomic_write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(text)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _evict(self) -> None:
        """Evict-oldest-by-mtime down to `max_entries`. Race-tolerant: a file vanishing under us
        (concurrent eviction/read) is ignored. FIFO by write time, NOT true LRU."""
        if self.max_entries is None:
            return
        try:
            entries = list(self.root.glob("*.json"))
        except OSError:
            return
        if len(entries) <= self.max_entries:
            return
        # Stat each; skip any that vanished mid-sweep.
        stamped: list[tuple[float, Path]] = []
        for p in entries:
            try:
                stamped.append((p.stat().st_mtime, p))
            except OSError:
                continue
        stamped.sort(key=lambda t: t[0])  # oldest first
        excess = len(stamped) - self.max_entries
        for _, p in stamped[:excess]:
            self._delete_quietly(p)
