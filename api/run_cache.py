"""
Content-addressed run-doc cache + shareable-permalink substrate.

WHY: every free-text question fires a *billed* Azure OpenAI codegen call on an unauthenticated
GET. This module keys a normalized `(query_text, clip)` pair to a sha256 hex digest, persists the
*validated, grounded* run-doc that question produced, and serves an identical repeat for free and
deterministically. Because a cached entry already carries a program that PASSED validation AND
grounded (the trust boundary is respected — we never cache a failure), the same cache key doubles
as a stable `run_id` permalink: sharing the same question twice yields the same link.

KEY-NORMALIZATION CONTRACT (the one knob the plan left open — see DESIGN-ROOM in the plan):
    normalize_query(text) = text.strip().casefold(), with internal runs of whitespace collapsed
    to a single space. NOTHING else — no punctuation stripping, no synonym/lemmatization pass.
    This is the most conservative choice: two *meaningfully different* questions never collide
    (lowest false-merge risk), at the cost of a lower hit rate on punctuation/synonym variants.
    Rationale: a cache hit reuses a previously-grounded answer; merging two questions whose
    answers differ would serve a wrong answer for free, which is strictly worse than a miss.

    cache_key(text, clip_id) = sha256(f"{clip_id}\\x00{normalize_query(text)}"). The clip id is
    part of the key RAW (it is a controlled enum from canned.CLIPS); only the free-text question
    is normalized. The NUL separator prevents (clip="a", text="b") colliding with (clip="ab").

    run_id == cache_key (the full 64-char sha256 hex, never truncated — the route's
    ^[0-9a-f]{64}$ guard depends on the fixed length). The permalink IS the content-address.

PERSISTENCE: one JSON file per entry at `root/<key>.json`, written atomically (temp + os.replace),
mirroring api/scripts/precompute.py's atomic-write discipline. Each file is an envelope
`{"cache_format": CACHE_FORMAT, "doc": <run-doc>}`, NOT a bare doc. `get` validates the envelope
version AND the doc against run_doc.schema.json on every read; any parse failure, version
mismatch, or schema failure is treated as a MISS and the bad file is best-effort deleted
(self-healing). Bumping CACHE_FORMAT is the deliberate kill-switch when the doc format or codegen
prompt changes — it invalidates exactly the stale entries, immediately, without a TTL.

EVICTION: a soft cap of `max_entries`, evict-oldest-by-write-time (file mtime) on `put`. This is
FIFO by write time, NOT true LRU — file mtime does not track reads, and we deliberately do NOT
touch-on-read to fake LRU (that would keep shared links pinned forever and defeat the bound).

CONCURRENCY: single-process, best-effort — atomic per-entry writes + a tolerant eviction sweep
that ignores races. No file locks.

SEAM: `RunStore` is a Protocol with `get`/`put`; `FileRunStore` is the file impl. A future
`BlobRunStore` implements the same two methods — the server injects ONE store at startup.
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
_RD_VALIDATOR = jsonschema.Draft202012Validator(_RD_SCHEMA)

# Bump this to invalidate every stored entry at once (the no-TTL kill-switch): do so whenever the
# run-doc format or the codegen prompt changes such that a replayed stale doc would be wrong.
CACHE_FORMAT = 1

# A well-formed cache key / run_id: exactly 64 lowercase hex chars (a full sha256 digest).
_KEY_RE = re.compile(r"^[0-9a-f]{64}$")

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_query(text: str) -> str:
    """Normalize a free-text question for content-addressing (see the module's KEY-NORMALIZATION
    CONTRACT). Casefold + strip + collapse internal whitespace, nothing else."""
    t = text.strip().casefold()
    t = _WHITESPACE_RE.sub(" ", t)
    return t


def cache_key(text: str, clip_id: str) -> str:
    """The content-address for a (question, clip) pair: a full sha256 hex digest. The clip id is
    raw (controlled enum); only `text` is normalized. A NUL separator prevents boundary
    collisions. This value IS the permalink `run_id`."""
    payload = f"{clip_id}\x00{normalize_query(text)}".encode()
    return hashlib.sha256(payload).hexdigest()


def is_valid_key(key: str) -> bool:
    """True iff `key` is a well-formed cache key / run_id (^[0-9a-f]{64}$). Used by both the route
    guard and the store's defense-in-depth path-traversal refusal."""
    return bool(_KEY_RE.match(key))


@runtime_checkable
class RunStore(Protocol):
    """The persistence seam. A future BlobRunStore implements these two methods."""

    def get(self, key: str) -> dict | None:
        """Return the stored run-doc for `key`, or None on any miss (absent / corrupt / stale
        version / schema-invalid). A corrupt/stale entry is best-effort deleted (self-healing)."""
        ...

    def put(self, key: str, doc: dict) -> None:
        """Store `doc` (a validated, grounded run-doc) under `key`, atomically. Wraps the doc in a
        `{cache_format, doc}` envelope and enforces the entry-count bound."""
        ...


class FileRunStore:
    """File-backed RunStore: one JSON envelope per entry at `root/<key>.json`."""

    def __init__(self, root: Path | str, *, max_entries: int | None = 1000):
        self.root = Path(root)
        self.max_entries = max_entries

    def _path(self, key: str) -> Path:
        # Defense in depth: never let a raw query param compose a path. The route already rejects
        # malformed ids, but the store independently refuses anything that isn't a clean key.
        if not is_valid_key(key):
            raise ValueError(f"invalid cache key: {key!r}")
        return self.root / f"{key}.json"

    def get(self, key: str) -> dict | None:
        # A malformed key is a hard miss (never a file read of an attacker-composed path).
        if not is_valid_key(key):
            return None
        path = self.root / f"{key}.json"
        try:
            raw = path.read_text()
        except (FileNotFoundError, NotADirectoryError):
            return None
        except OSError:
            return None
        try:
            envelope = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            self._discard(path)  # garbage -> miss + self-heal
            return None
        if not isinstance(envelope, dict) or envelope.get("cache_format") != CACHE_FORMAT:
            self._discard(path)  # stale-format kill-switch -> miss + self-heal
            return None
        doc = envelope.get("doc")
        # NOTE: iter_errors returns a GENERATOR (always truthy) — materialize it before testing.
        if not isinstance(doc, dict) or list(_RD_VALIDATOR.iter_errors(doc)):
            # A run-doc that no longer conforms (schema drift) would stream straight into the SSE
            # loop on the cold shared-link path — fail it as a miss and delete it.
            self._discard(path)
            return None
        return doc

    def put(self, key: str, doc: dict) -> None:
        path = self._path(key)  # raises on a malformed key — callers pass cache_key() output
        envelope = {"cache_format": CACHE_FORMAT, "doc": doc}
        text = json.dumps(envelope, ensure_ascii=False, indent=2) + "\n"
        self.root.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, text)
        self._evict()

    def _evict(self) -> None:
        """Soft-cap the entry count: evict oldest-by-mtime until at most max_entries remain.
        Tolerant of races (another process deleting a file mid-sweep) — best-effort."""
        if self.max_entries is None:
            return
        try:
            entries = [p for p in self.root.glob("*.json") if p.is_file()]
        except OSError:
            return
        if len(entries) <= self.max_entries:
            return
        # Oldest first by write time (FIFO, not LRU). Stat failures sort last and are skipped.
        def _mtime(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except OSError:
                return float("inf")

        entries.sort(key=_mtime)
        for p in entries[: len(entries) - self.max_entries]:
            self._discard(p)

    @staticmethod
    def _discard(path: Path) -> None:
        """Best-effort delete; ignore if already gone (self-heal + race tolerance)."""
        try:
            path.unlink()
        except OSError:
            pass


def _atomic_write(path: Path, text: str) -> None:
    """Write `text` to `path` atomically (temp file in the same dir + os.replace), mirroring
    api/scripts/precompute.py:atomic_write. No partial/torn file is ever observable at `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
