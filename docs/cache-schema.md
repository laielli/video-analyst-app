# Replay cache schema (prose)

The per-clip **replay cache** is the precomputed vision evidence the interpreter replays
with zero live Azure calls (design D1, hero path). It is loaded by
[`api/interpreter/cache.py`](../api/interpreter/cache.py) (`Cache.load`) and produced by
[`api/scripts/precompute.py`](../api/scripts/precompute.py). This is a **prose contract**, not
a JSON Schema file — a standalone `cache.schema.json` would duplicate the `box`/detection
`$defs` already in [`run_doc.schema.json`](../api/schema/run_doc.schema.json). The pipeline and
the pytest suite enforce the shape in code (see `precompute.py:validate_cache_structure` and
`api/tests/test_precompute.py`).

## Top-level shape

```jsonc
{
  "clip":      { "id", "width", "height", "duration_ms", "fps" },
  "detect":    { "<ts_ms>": [ detection, ... ], ... },   // keyed by sampled-frame ms (string)
  "read_text": { "<det_id>": read, ... },                // keyed by det_id (string)
  "events":    [ event, ... ]
}
```

`Cache` reads `clip` directly and `.get(...)`s the other three, so each is optional at load
time — but a usable hero cache needs all four. The pipeline always writes `clip`, `detect`,
`read_text` (possibly empty), and `events`. An optional `regenerated_at` ISO timestamp may be
present at the top level for provenance; it is ignored by `Cache` and excluded from the
idempotency byte-comparison.

### `clip`
| field         | type    | notes                                                            |
|---------------|---------|------------------------------------------------------------------|
| `id`          | string  | clip id, e.g. `"single-goal"`.                                   |
| `width`       | integer | frame width in source pixels.                                    |
| `height`      | integer | frame height in source pixels.                                   |
| `duration_ms` | integer | clip duration in ms.                                             |
| `fps`         | number  | source frame rate (the clip's native fps, e.g. 60 — NOT the sampling fps). |

Source of truth is `canned.CLIPS[id]` (the registry, left unedited); the manifest `clip`
block carries the same values plus the `.mov` `source` path. The pipeline reconciles them:
registry dims win, `ffprobe` (or manifest) fills any gap.

### `detect` — keyed by sampled-frame timestamp (ms, as a string)
Each value is a list of **detection** objects for that frame. Keys are the exact sampled
timestamps the interpreter's `op_sample_frames` derives (`stride = round(1000/fps)`,
`ts = range(start, end+1, stride)`); the pipeline samples on the same stride so the keys align
with what `op_detect` looks up via `cache.detections_at(ts)`.

A **detection**:
| field        | type   | notes                                                              |
|--------------|--------|--------------------------------------------------------------------|
| `det_id`     | string | stable id, the JOIN KEY across `detect` / `read_text` / `events`. Unique within a frame. |
| `cls`        | string | object class, e.g. `"person"`. `op_detect` filters case-insensitively. |
| `confidence` | number\|null | detection confidence (Azure tag confidence), rounded to 4 places. |
| `box`        | box    | normalized 0-1 `{x,y,w,h}`, top-left origin, each ≤ 4 decimals.   |

**det_id minting:** Azure returns no ids. The pipeline mints `p{N}` per frame by **descending
confidence** (highest = `p0`). Manifest **pins** then RENAME a matched det to a semantic id
(`p_messi`) by nearest-box. So the hero frame ends up `p_messi` + `p1..p4`.

### `read_text` — keyed by `det_id`
Each value is a **read** object — the OCR/VLM result for that detection's jersey crop. A
detection with no legible number has **no** entry here (honest-empty, D-DR5).

A **read**:
| field        | type   | notes                                                            |
|--------------|--------|------------------------------------------------------------------|
| `text`       | string | the digits read, e.g. `"10"`.                                    |
| `confidence` | number\|null | read confidence (null for pins).                           |
| `source`     | enum   | provenance: **`live`** (real call this run), **`cached`** (precomputed real Azure output), **`pinned`** (hand-verified). |
| `note`       | string | optional diagnostic / disclosure text.                           |

`source` enum is the demo-honesty contract surfaced in the run-doc readout. The OCR ladder:
**pin** → `pinned`; **gpt-4o VLM** → `live` (when `AZURE_OPENAI_*` available + digits returned);
otherwise **skip** (write no entry). Azure Read is deliberately NOT a rung (its WC-font misread
would poison the cache).

### `events` — goal/scoring moments (list)
Not Azure-derived; carried from the manifest but with `scorer_det` resolved from the pin so it
is byte-identical to a minted det_id.

An **event**:
| field        | type   | notes                                                            |
|--------------|--------|------------------------------------------------------------------|
| `ts_ms`      | integer | event timestamp in ms.                                          |
| `type`       | string | e.g. `"goal"`. `cache.goal_events()` filters `type == "goal"`.  |
| `scorer_det` | string | det_id of the scorer — MUST be a minted det_id that appears in `detect` at some cached ts. |
| `box`        | box    | normalized 0-1 box highlighting the event.                      |

## The verdict linchpin (why these three maps must agree)
The grounded **"Yes — #10 scored the first goal"** exists only when, as one chain:
`events[0].scorer_det` == a `det_id` in `detect`, that det carries a `read_text` entry with
`text == "10"`, and that read survives `op_filter text==10`. If `events` is omitted,
`op_temporal_order` is empty and the verdict flips to "No". The pipeline asserts
`events[i].scorer_det ∈ minted det_ids` AND the scorer det appears in `detect` before writing,
so `op_answer`'s overlay re-scan (`primitives.py:200-203`) finds the focus frame.

## Idempotency
The cache is written canonically: `detect` keys sorted numerically, detections sorted by
`det_id`, boxes rounded to 4 places (matching `Box.rounded()`), `indent=2`, trailing newline.
Re-running with the same inputs yields a byte-identical JSON file. (Stills are JPEG-re-encoded
and are NOT part of the byte-identity guarantee — verify those by dimensions + eyeball.)
