# Replay cache schema (`hero_cache.json`)

The per-clip **replay cache** is the precomputed vision evidence the interpreter replays
with zero live Azure calls (design D1 hero path). `api/interpreter/cache.py` loads it; the
`detect` / `read_text` primitives read from it instead of calling Azure, so the hero run is
deterministic and free.

This is prose, not a JSON Schema — the box/detection shapes already live in
`run_doc.schema.json`'s `$defs` and we do not duplicate them. The cache shape is enforced
in code (`precompute.py`'s `validate_cache_shape`) and in `api/tests/`.

`scripts/precompute.py` writes a cache in exactly this shape from a clip manifest.

## Top-level shape

```jsonc
{
  "clip":      { "id", "width", "height", "duration_ms", "fps" },
  "detect":    { "<ts_ms>": [ <detection>, ... ] },   // keys = sampled-frame timestamps (ms), as strings
  "read_text": { "<det_id>": <read_entry> },          // keys = det_ids
  "events":    [ <event>, ... ]
}
```

A leading `_comment` (any `_`-prefixed key) is allowed as an authoring note and ignored.

### `clip`

| field         | type   | notes                                                              |
| ------------- | ------ | ------------------------------------------------------------------ |
| `id`          | string | clip id (e.g. `single-goal`); matches `canned.CLIPS` registry key. |
| `width`       | int    | analyzed-frame width in px (registry is source of truth).          |
| `height`      | int    | analyzed-frame height in px.                                       |
| `duration_ms` | int    | clip duration in ms.                                               |
| `fps`         | number | the clip's native fps (NOT the sampling fps).                      |

The interpreter copies this block verbatim into the run-doc `clip`. The pipeline reconciles
it against the `canned.CLIPS` registry (registry wins for dims; never edits the registry).

### `detect` — `{ "<ts_ms>": [detection, ...] }`

Keyed by **sampled-frame timestamp in ms, as a string**. `cache.analyzed_frames()` returns
`sorted(int(k) for k in detect)`; `cache.detections_at(ts)` looks up `str(ts)`. Keys MUST
align with the program's sampling stride (`op_sample_frames`: `stride = round(1000/fps)`,
`ts = range(start, end+1, stride)`) for the verdict-bearing frame.

Each detection:

| field        | type        | notes                                                                |
| ------------ | ----------- | -------------------------------------------------------------------- |
| `det_id`     | string      | stable id, the **join key** across `detect`/`read_text`/`events`.    |
| `cls`        | string      | object class (e.g. `person`); `op_detect` filters case-insensitively.|
| `confidence` | number/null | detector confidence (0–1).                                           |
| `box`        | box         | normalized 0–1 `{x,y,w,h}`, rounded to 4 places (`Box.rounded()`).   |

`det_id` minting (pipeline): per frame, sort detections by **descending confidence** and
assign `p0`, `p1`, … (highest = `p0`). A manifest **pin** then *renames* the detection whose
box is nearest the pin's `nearest_box` to the pin's semantic id (e.g. `p_messi`). `det_id`s
are unique within a frame.

### `read_text` — `{ "<det_id>": read_entry }`

Keyed by `det_id`. A missing entry is honest-empty (`op_read_text` skips it cleanly — D-DR5).

| field        | type        | notes                                                            |
| ------------ | ----------- | ---------------------------------------------------------------- |
| `text`       | string      | the read value (e.g. `"10"`).                                    |
| `confidence` | number/null | read confidence, or `null` for pins.                            |
| `source`     | enum        | provenance — see below.                                          |
| `note`       | string?     | optional diagnostic / disclosure text.                          |

Every `read_text` key MUST be a real `det_id` present somewhere in `detect`.

### `source` provenance enum

| value    | meaning                                                                          |
| -------- | -------------------------------------------------------------------------------- |
| `live`   | a real call made **this run** (e.g. the gpt-4o VLM read during precompute).      |
| `cached` | precomputed real Azure output replayed from this cache (detections are `cached`).|
| `pinned` | a hand-verified value (e.g. the hero `#10` OCR pin).                             |

The OCR ladder is binding: **pin** (`pinned`) → **gpt-4o VLM** (`live`, accept if digits) →
**skip** (no entry). Azure Read is deliberately NOT a rung (its 22→77 misread would poison
the cache).

### `events` — `[ event, ... ]`

Mandatory and **not** Azure-derived. Omitting it makes `op_temporal_order` empty → the
verdict flips to "No". `cache.goal_events()` returns events with `type == "goal"` sorted by
`ts_ms`.

| field        | type   | notes                                                                       |
| ------------ | ------ | --------------------------------------------------------------------------- |
| `ts_ms`      | int    | event timestamp in ms.                                                      |
| `type`       | string | `goal` (the only type the interpreter consumes today).                      |
| `scorer_det` | string | **must equal a `det_id` that appears in `detect`**; the verdict's join key. |
| `box`        | box    | normalized 0–1 goal-moment box (overlay), distinct from the detection box.  |

**The verdict linchpin.** The grounded "Yes" exists only if `events[0].scorer_det` is
byte-identical to a `det_id` that (a) carries a `read_text` entry with `text == "10"`,
(b) survives `op_filter text == 10`, and (c) appears in `detect` at some cached ts (so
`op_answer`'s overlay re-scan finds the focus frame). The pipeline asserts all three before
writing — a cache that violates them fails self-validation (exit 1).

## Determinism / idempotency

The pipeline writes **canonical JSON**: `detect` keys sorted numerically, detections sorted
by `det_id`, boxes rounded to 4 places. Re-running on the same inputs yields a **byte-identical**
cache (the JSON only; stills are JPEG re-encodes and excluded from the byte check). Writes are
atomic (temp file + `os.replace`) so a partial write can't yield a truncated read.
