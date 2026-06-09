"""
Content-addressed run-doc cache + a small file-backed persistence store.

Why this exists: every free-text question fires a BILLED Azure OpenAI codegen call on an
unauthenticated GET. Identical repeat questions should be served for free and deterministically.
This module is the substrate: it keys a validated+grounded run-doc on the normalized
`(query_text, clip_id)` pair, persists each entry to disk, and exposes the read/write API the
server calls AHEAD of codegen. The same content-address doubles as a shareable permalink id.

KEY-NORMALIZATION CONTRACT (the cache's correctness boundary):
    cache_key(text, clip_id) = sha256( f"{clip_id}\\x00{normalize_query(text)}" ).hexdigest()

  - `clip_id` enters the key RAW. It is already a controlled enum (canned.CLIPS), so it is not
    normalized; only the free-text question is. The NUL separator (`\\x00`) prevents any
    clip_id/text boundary ambiguity.
  - `normalize_query` is DELIBERATELY conservative (the one knob the plan left open). It does:
        casefold (Unicode-aware lowercase) + strip + collapse internal whitespace runs to one
        space. NOTHING else — no punctuation stripping, no synonym/lemmatization. This minimizes
        false merges: two MEANINGFULLY different questions never collide, at the cost of a lower
        hit rate on superficial variants ("Does #10 score?" vs "does 10 score"). The conservative
        choice is correct here because a false merge serves a WRONG cached answer; a false miss
        merely re-bills. See `test_run_cache.py::test_normalize_query_casefold_and_whitespace`.
  - The key derivation (sha256 hex, fixed 64 lowercase-hex chars) is FIXED — the permalink route
    guard (`^[0-9a-f]{64}$`) depends on the length. Only the normalizer is a knob.

The `run_id` (permalink) IS this cache key (full sha256 hex, no truncation) so a permalink is
stable across processes and sharing the same question twice yields the same link.

STORE ENVELOPE + VERSIONING: each entry is `{"cache_format": CACHE_FORMAT, "doc": <run-doc>}`,
NOT the bare doc. `get` validates the version AND the doc against run_doc.schema.json; any parse
failure, version mismatch, or schema failure is treated as a MISS and the bad file is best-effort
deleted (self-healing). Bumping CACHE_FORMAT is the deliberate kill-switch when the doc format or
codegen prompt changes — it invalidates exactly the stale entries, immediately, without a TTL.

CONCURRENCY: single-process, best-effort. Per-entry writes are atomic (temp + os.replace);
eviction tolerates races. No file locks.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Protocol

# Bump this when the run-doc format OR the codegen prompt changes — it invalidates every stale
# entry on the next read without a TTL (Phase 2 kill-switch). Start at 1.
CACHE_FORMAT = 1

# A well-formed cache key / run_id is exactly the sha256 hex digest: 64 lowercase hex chars.
_KEY_RE = re.compile(r"^[0-9a-f]{64}$")

_WS_RE = re.compile(r"\s+")

# ---- run-doc schema (validate-on-read; jsonschema already ships with the api) ----
_SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "run_doc.schema.json"


def _run_doc_validator():
    """Lazily build (and cache) the Draft 2020-12 validator for run_doc.schema.json."""
    global _VALIDATOR
    try:
        return _VALIDATOR
    except NameError:
        import jsonschema  # local import: only the cache-read path needs it

        schema = json.loads(_SCHEMA_PATH.read_text())
        _VALIDATOR = jsonschema.Draft202012Validator(schema)
        return _VALIDATOR


# --------------------------------------------------------------------------------------
# Phase 1 — normalization + content-address
# --------------------------------------------------------------------------------------

def normalize_query(text: str) -> str:
    """Normalize a free-text question to the cache's canonical form (see module contract).

    Conservative by design: casefold + strip + collapse internal whitespace. Nothing else."""
    t = text.strip().casefold()
    t = _WS_RE.sub(" ", t)
    return t


def cache_key(text: str, clip_id: str) -> str:
    """Content-address `(query_text, clip_id)` to a stable 64-char sha256 hex digest.

    clip_id enters raw (it is a controlled enum); only the question is normalized. This value is
    BOTH the store key and the shareable `run_id`."""
    raw = f"{clip_id}\x00{normalize_query(text)}".encode()
    return hashlib.sha256(raw).hexdigest()


def is_valid_key(key: str) -> bool:
    """True iff `key` is a well-formed content-address (exactly 64 lowercase hex chars). The
    permalink route uses this as its `^[0-9a-f]{64}$` input guard."""
    return bool(_KEY_RE.match(key))


# --------------------------------------------------------------------------------------
# Phase 2 — the persistence seam
# --------------------------------------------------------------------------------------

class RunStore(Protocol):
    """The two-method seam the server depends on. A future BlobRunStore implements the same
    interface; constructor injection (server._run_store) is the swap point."""

    def get(self, key: str) -> dict | None: ...

    def put(self, key: str, doc: dict) -> None: ...


class FileRunStore:
    """File-backed RunStore: one JSON file per entry at `root/<key>.json`, written atomically
    (temp + os.replace). Eviction is a soft cap of `max_entries`, evict-oldest-by-mtime on put
    (FIFO by WRITE time, not true LRU — reads do not touch mtime). `get` validates the envelope
    version + the embedded doc against run_doc.schema.json, treating any failure as a self-healing
    miss (delete the bad file). Defense in depth: keys are refused unless they are well-formed
    content-addresses, so a raw query param can never compose a path (`../../etc/passwd` -> miss)."""

    def __init__(self, root: Path | str, *, max_entries: int | None = None):
        self.root = Path(root)
        self.max_entries = max_entries

    def _path(self, key: str) -> Path | None:
        # Never let a raw key compose a path. A well-formed content-address can only be hex, so it
        # cannot contain '/', '\\', '.', or os.sep — but check explicitly (defense in depth).
        if not is_valid_key(key):
            return None
        if any(c in key for c in ("/", "\\", os.sep)) or "." in key:
            return None
        return self.root / f"{key}.json"

    def get(self, key: str) -> dict | None:
        """Return the stored run-doc for `key`, or None on any miss (absent, unreadable, bad
        envelope, stale CACHE_FORMAT, or schema-invalid doc). A corrupt/stale file is best-effort
        deleted so the next ask re-bills cleanly instead of failing forever."""
        path = self._path(key)
        if path is None:
            return None
        try:
            raw = path.read_text()
        except (FileNotFoundError, OSError):
            return None
        try:
            envelope = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            self._evict_bad(path)
            return None
        if not isinstance(envelope, dict) or envelope.get("cache_format") != CACHE_FORMAT:
            # version mismatch (the kill-switch) or a non-envelope blob -> self-heal.
            self._evict_bad(path)
            return None
        doc = envelope.get("doc")
        if not isinstance(doc, dict) or self._schema_errors(doc):
            self._evict_bad(path)
            return None
        return doc

    def put(self, key: str, doc: dict) -> None:
        """Store `doc` under `key` (wrapped in the versioned envelope), atomically, then evict the
        oldest entries if over `max_entries`. A malformed key is a silent no-op (defense in depth);
        the server only ever puts content-address keys."""
        path = self._path(key)
        if path is None:
            return
        envelope = {"cache_format": CACHE_FORMAT, "doc": doc}
        text = json.dumps(envelope, ensure_ascii=False) + "\n"
        self._atomic_write(path, text)
        self._evict_over_cap()

    # ---- internals ----

    def _schema_errors(self, doc: dict) -> bool:
        try:
            return bool(next(_run_doc_validator().iter_errors(doc), None))
        except Exception:  # noqa: BLE001 — a validator failure is itself a miss, not a crash
            return True

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        # Mirrors precompute.atomic_write (proven pattern): temp file in the same dir + os.replace,
        # so a reader never sees a half-written entry and no `.tmp` is left behind on success.
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(text)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _evict_bad(self, path: Path) -> None:
        try:
            path.unlink()
        except OSError:
            pass  # best-effort self-heal; a race that already removed it is fine

    def _evict_over_cap(self) -> None:
        if self.max_entries is None:
            return
        try:
            entries = [p for p in self.root.glob("*.json") if p.is_file()]
        except OSError:
            return
        if len(entries) <= self.max_entries:
            return
        # Evict oldest-by-mtime first (FIFO by write time). Tolerate races: a file another writer
        # already removed just raises FileNotFoundError, which we swallow.
        try:
            entries.sort(key=lambda p: p.stat().st_mtime)
        except OSError:
            return
        for p in entries[: len(entries) - self.max_entries]:
            try:
                p.unlink()
            except OSError:
                pass
