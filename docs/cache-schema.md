# Replay cache schema (`hero_cache.json` shape)

Prose contract for the per-clip replay cache that `api/interpreter/cache.py` loads and the
interpreter replays with **zero live Azure calls** (design D1 hero path). This is intentionally
*not* a JSON Schema file — the box/detection shapes already live as `$defs` in
`api/schema/run_doc.schema.json`, and duplicating them would create two masters to keep in sync.
The pipeline (`api/scripts/precompute.py`) enforces this shape in its self-validating replay
(Phase 6.2) and the pytest suite enforces it structurally (`api/tests/`).

The cache is produced by `precompute.py` from a raw clip + a per-clip manifest
(`clips/<id>/manifest.json`). It carries the exact two-layer evidence the demo needs:
real Azure **detect** boxes (when creds are present) plus the **read_text / events** linkage
that the LLM-generated program joins on.

## Top-level shape

```jsonc
{
  "clip":      { "id", "width", "height", "duration_ms", "fps" },
  "detect":    { "<ts_ms>": [ detection, ... ] },   // keyed by sampled-frame timestamp (ms, string)
  "read_text": { "<det_id>": read },                 // keyed by detection id
  "events":    [ event, ... ]
}
```

`Cache.load(path)` → `Cache` exposes:
- `clip` — the `clip` block verbatim.
- `analyzed_frames()` — sorted `int` ts of every `detect` key.
- `detections_at(ts_ms)` — the detection list for that ts (or `[]`).
- `read_text_for(det_id)` — the read dict for that det_id (or `None`).
- `goal_events()` — `events` with `type == "goal"`, sorted by `ts_ms`.

### `clip`

| field         | type   | notes                                                            |
| ------------- | ------ | ---------------------------------------------------------------- |
| `id`          | string | clip id (e.g. `single-goal`); matches `canned.CLIPS` key.        |
| `width`       | int    | source frame width in px.                                        |
| `height`      | int    | source frame height in px.                                       |
| `duration_ms` | int    | clip duration in ms.                                             |
| `fps`         | number | source frame rate (the clip's, not the sampling fps).           |

`width`/`height`/`duration_ms`/`fps` come from `clips/<id>/manifest.json:clip` (the registry in
`canned.CLIPS` is the source of truth; `ffprobe` is a fallback only when the manifest omits them).

### `detect` — `"<ts_ms>": [ detection, … ]`

Keys are **sampled-frame timestamps in ms, as strings** (e.g. `"4625"`). They MUST line up with
the stride the interpreter samples on: `op_sample_frames` uses `stride = round(1000/fps)` and
`ts = range(start_ms, end_ms+1, stride)`, then `op_detect` looks up `detections_at(ts)` for each.
So the manifest `sampling` block (`start_ms`/`end_ms`/`fps`) must match the program's
`sample_frames` args. The binding invariant is: **every sampled ts the verdict depends on has a
detect key**; other sampled ts may have no key and degrade to an empty detect cleanly.

Each `detection`:

| field        | type    | notes                                                                |
| ------------ | ------- | -------------------------------------------------------------------- |
| `det_id`     | string  | stable id, **unique within a frame**; the join key across detect/read_text/events. |
| `cls`        | string  | object class, lowercased to match `op_detect` (e.g. `person`).       |
| `confidence` | number? | detector confidence (0–1) or `null`.                                 |
| `box`        | box     | normalized 0–1 `{x,y,w,h}`, rounded to 4 places (`Box.rounded()`).   |

**det_id minting:** Azure returns no ids. The pipeline mints `p{N}` per frame by **descending
confidence** (highest = `p0`). A manifest pin then **renames** the matched detection to a semantic
id (`p_messi`) by nearest-box, so the hero linkage is stable and human-readable.

### `read_text` — `"<det_id>": read`

Keyed by `det_id` (a real id present in some `detect` frame). `op_read_text` looks up
`read_text_for(det_id)`; a **missing** entry is honest-empty (D-DR5), not an error.

| field        | type    | notes                                                            |
| ------------ | ------- | --------------------------------------------------------------- |
| `text`       | string  | the read value (e.g. `"10"`). Empty/absent ⇒ step degrades empty.|
| `confidence` | number? | read confidence (0–1) or `null`.                                |
| `source`     | enum    | provenance — one of **`live` \| `cached` \| `pinned`** (see below). |
| `note`       | string? | optional human note (e.g. why the value is pinned).             |

**`source` enum (provenance, for demo honesty):**
- `live` — read this run by a real call (e.g. gpt-4o vision read_jersey_number).
- `cached` — precomputed from a real Azure call in a prior pass.
- `pinned` — hand-verified value (the hero #10, where OCR misreads the stylized font).

**OCR ladder** (binding order): (1) manifest **pin** → `pinned`, verbatim; (2) **gpt-4o VLM**
(`AzureVLM.read_jersey_number`) when `AZURE_OPENAI_*` is set → `live` if it returns digits;
(3) **skip** → write *no* `read_text` entry (honest-empty). Azure Read is deliberately **not** a
rung (its 22→77 misread would poison the cache). On 429 / missing creds / error the VLM rung is
caught, warned, and skipped (unless `--require-ocr`, which hard-fails).

### `events` — `[ event, … ]`

Not Azure-derived; the human goal-moment knowledge. **Mandatory** for the grounded verdict — omit
it and `op_temporal_order` is empty and the answer flips to "No".

| field        | type   | notes                                                                  |
| ------------ | ------ | ---------------------------------------------------------------------- |
| `ts_ms`      | int    | event timestamp in ms.                                                 |
| `type`       | string | `goal` is the consumed type (`op_temporal_order` filters to it).       |
| `scorer_det` | string | **must be a real det_id** that appears in `detect`. The verdict's join key. |
| `box`        | box    | normalized 0–1 `{x,y,w,h}` for the goal overlay.                        |

**Verdict linchpin:** the grounded "Yes" exists only if `events[0].scorer_det` is byte-identical
to a det_id carrying a `read_text` entry with `text == "10"` that survives `op_filter text==10`.
The pipeline derives each `events[].scorer_det` from the **same pin-resolution result** that
renamed the detection (manifest `events[].scorer_pin` → the pin's resolved det_id) — never an
independent literal — and asserts before writing that the scorer det is a minted id present in the
`detect` map (else `op_answer`'s overlay re-scan drops the focus frame while the verdict still
says Yes).

## Determinism (idempotency)

The pipeline writes **canonical JSON**: `detect` keys sorted numerically, detections sorted by
`det_id`, boxes rounded to 4 places, stable key order, trailing newline. Re-running over the same
inputs yields a **byte-identical** cache (idempotency is scoped to the JSON only — re-encoded
JPEG stills are *not* byte-stable, so tests assert their dimensions, not their hash).
