# Replay cache shape (`examples/*_cache.json`)

The per-clip replay cache is the precomputed vision evidence the interpreter replays with
**zero live Azure calls** (design D1, the hero path). `api/interpreter/cache.py:Cache` loads
it; `api/scripts/precompute.py` generates it from a raw clip + manifest.

This is prose, not a JSON Schema file, on purpose: the box/detection `$defs` already live in
`schema/run_doc.schema.json`, and a standalone `cache.schema.json` would duplicate them and
drift. The pipeline self-validates the cache structure in code (see `precompute.py`), and the
real contract is enforced end-to-end by replaying the hero program and validating the emitted
run-doc against `run_doc.schema.json`.

## Top-level object

```jsonc
{
  "clip":      { ... },   // clip metadata block (required)
  "detect":    { ... },   // sampled-frame ts (ms, as string keys) -> [detection]
  "read_text": { ... },   // det_id -> read result
  "events":    [ ... ]    // known moments (goals) — NOT Azure-derived
}
```

`Cache.__init__` reads `clip` (required) and `detect` / `read_text` / `events` (each defaults
to empty if absent). The precompute pipeline always writes all four.

### `clip` (required)

```jsonc
"clip": { "id": "single-goal", "width": 1872, "height": 1042, "duration_ms": 5067, "fps": 60 }
```

| field         | type    | notes                                                            |
|---------------|---------|------------------------------------------------------------------|
| `id`          | string  | clip id; must match the manifest `clip.id`.                      |
| `width`       | integer | source pixel width (the normalization basis for every box).      |
| `height`      | integer | source pixel height.                                             |
| `duration_ms` | integer | full clip duration in ms.                                        |
| `fps`         | number  | source frame rate (the clip's own fps, not the sampling fps).    |

`run_doc.schema.json#/properties/clip` requires `id/width/height/duration_ms` (fps optional);
the pipeline reconciles these against `canned.CLIPS[id]` (the registry is the source of truth;
`ffprobe` is a fallback when the registry lacks dims). The registry is **not** edited.

### `detect` — `"<ts_ms>": [detection, ...]`

Keys are **sampled-frame timestamps in ms, as strings** (`Cache.detections_at(ts)` does
`self._detect.get(str(ts_ms))`). They MUST land on the same stride the interpreter samples on
(`op_sample_frames`: `stride = round(1000/fps)`, `ts = range(start, end+1, stride)`), or the
detect step finds nothing for that frame. At fps=8 the stride is 125ms, so the keys are
`3500, 3625, 3750, ... 5000` — and `4625 = 3500 + 9*125` is in the list.

Each detection:

```jsonc
{ "det_id": "p_messi", "cls": "person", "confidence": 0.753,
  "box": { "x": 0.0294, "y": 0.1891, "w": 0.2212, "h": 0.6276 } }
```

| field        | type           | notes                                                          |
|--------------|----------------|----------------------------------------------------------------|
| `det_id`     | string         | stable id, the **join key** across detect/read_text/events.    |
| `cls`        | string         | detected class; `op_detect` filters case-insensitively.        |
| `confidence` | number \| null | detector confidence (0-1).                                     |
| `box`        | box            | normalized 0-1 `{x,y,w,h}`, rounded to 4 places.               |

**det_id minting** (pipeline): within a frame, `p{N}` by **descending confidence** (highest =
`p0`); ties broken by box position for determinism. A manifest **pin** then *renames* the
matched det (nearest box at the pin's `match.frame_ts_ms`) to its semantic id (`p_messi`)
**across all frames** it appears in — by box overlap — so the join key is stable. det_ids are
unique within a frame.

### `read_text` — `"<det_id>": { ... }`

```jsonc
"p_messi": { "text": "10", "confidence": null, "source": "pinned", "note": "..." }
```

| field        | type           | notes                                                            |
|--------------|----------------|------------------------------------------------------------------|
| `text`       | string         | the read value (e.g. `"10"`). Missing/empty ⇒ omit the entry.    |
| `confidence` | number \| null | read confidence; `null` for pins.                                |
| `source`     | enum           | provenance — one of `live` / `cached` / `pinned` (see below).    |
| `note`       | string (opt)   | disclosure text (e.g. why a value is pinned).                    |

Every key MUST be a real `det_id` that appears in `detect`. A det with no legible number gets
**no** `read_text` entry (D-DR5 honest-empty): `op_read_text` skips it (`read_text_for` returns
`None`).

**`source` enum** (provenance, for demo honesty):

- `live`   — a real model call made *this run* (e.g. a fresh gpt-4o VLM read).
- `cached` — a precomputed real Azure/VLM output replayed from this file (detections are tagged
  this way so the run-doc `detect`/`read_text` steps render `cached`).
- `pinned` — a hand-verified value (the hero #10): trusted over a model that misreads it.

**OCR ladder** the pipeline applies per OCR-target det, in order:
1. **manifest pin** → `source: "pinned"`, verbatim text.
2. **gpt-4o VLM** (`AzureVLM.read_jersey_number`) when `AZURE_OPENAI_*` is set → `source: "live"`
   if it returns digits.
3. **skip** → write no entry (degraded/empty).

Azure AI Vision **Read is deliberately NOT a rung** — it misreads the stylized WC font
(22 → 77) and would poison the cache. On VLM 429 / missing creds / error the pipeline catches,
warns, and skips (rung 3); `--require-ocr` turns a skip into a hard failure (exit 1).

### `events` — `[ { ts_ms, type, scorer_det, box } ]`

```jsonc
"events": [ { "ts_ms": 4000, "type": "goal", "scorer_det": "p_messi",
              "box": { "x": 0.86, "y": 0.34, "w": 0.10, "h": 0.34 } } ]
```

| field        | type    | notes                                                              |
|--------------|---------|--------------------------------------------------------------------|
| `ts_ms`      | integer | event timestamp in ms.                                             |
| `type`       | string  | `Cache.goal_events()` returns those with `type == "goal"`.        |
| `scorer_det` | string  | **must** be a `det_id` present in the `detect` map.               |
| `box`        | box     | normalized 0-1 location overlay for the moment.                   |

This block is **mandatory and not Azure-derived** — it is the verdict linchpin.
`op_temporal_order` returns `subject_is_first_scorer = (first_goal.scorer_det ∈ filtered dets)`,
and `op_answer` reports "Yes" only when that det also survives `filter text == "10"`. If
`events` is empty, `op_temporal_order` is empty and the verdict flips to "No". The pipeline
derives each `scorer_det` from the **pin resolution** (not an independent manifest literal) and
asserts, before writing, that `scorer_det` is a minted det_id that appears in `detect` at some
cached ts (else `op_answer`'s overlay re-scan drops the focus frame while the verdict still says
Yes).

## Determinism / idempotency

The pipeline writes canonical JSON so a re-run is byte-identical (idempotent):
`detect` keys sorted numerically, detections sorted by `det_id`, boxes rounded to 4 places
(matching `Box.rounded()`), stable key order. Stills (JPEGs) are re-encoded and **excluded**
from the byte-identity check (JPEG re-encode is not bit-reproducible; verify by dimensions).
The cache file is written **atomically** (temp file + `os.replace`) so a partial write never
yields truncated JSON to `server.py`'s per-request `Cache.load`.
