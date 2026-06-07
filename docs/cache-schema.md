# Per-clip replay cache shape

The replay cache is the precomputed vision evidence the interpreter replays with **zero live
Azure calls** (design D1, the hero path). It is what `api/interpreter/cache.py:Cache.load`
reads and what `api/scripts/precompute.py` writes. This is a **prose contract**, not a JSON
Schema file — the box/detection `$defs` already live in `api/schema/run_doc.schema.json` and
duplicating them here would risk drift. The pipeline self-validates the cache against this
contract in-process (see `precompute.py:validate_cache`) and the pytest suite enforces it
structurally.

A cache is one JSON object with four top-level keys: `clip`, `detect`, `read_text`, `events`.

## `clip` — clip block

```json
"clip": { "id": "single-goal", "width": 1872, "height": 1042, "duration_ms": 5067, "fps": 60 }
```

| field | type | notes |
|-------|------|-------|
| `id` | string | matches `canned.CLIPS[id]` (the registry is the source of truth; the pipeline reconciles against it and does NOT edit it). |
| `width`, `height` | integer | source pixel dims. Used to convert normalized boxes → source-pixel crops for OCR. |
| `duration_ms` | integer | clip length in ms. |
| `fps` | number | the SOURCE clip fps (60 here), **not** the sampling fps. |

Consumed by `Cache.__init__` as `data["clip"]`; surfaced verbatim as the run-doc `clip` block.

## `detect` — per-frame detections, keyed by sampled-frame ms

```json
"detect": {
  "4625": [
    { "det_id": "p_messi", "cls": "person", "confidence": 0.753,
      "box": { "x": 0.0294, "y": 0.1891, "w": 0.2212, "h": 0.6276 } },
    { "det_id": "p1", "cls": "person", "confidence": 0.598, "box": { ... } }
  ]
}
```

- **Keys** are sampled-frame timestamps in **milliseconds, as strings** (`"4625"`). They MUST
  align with the timestamps `op_sample_frames` derives for the program that replays this cache:
  `stride = round(1000/fps)`, `ts = range(start_ms, end_ms+1, stride)`. With fps 8 over
  3500–5000ms the stride is 125 and `4625 = 3500 + 9*125` is in the list. Sampled ts that have
  no detect key degrade cleanly to empty (`detections_at` returns `[]`).
- **Each detection** is `{ det_id, cls, confidence, box }`:
  - `det_id` (string) — the **join key** across `detect` / `read_text` / `events`. Unique per
    frame. Minted by the pipeline (Azure returns none): `p{N}` by **descending confidence**
    within a frame (`p0` = highest), with manifest pins **renaming** a matched det to a
    semantic id (`p_messi`). The renamed id must be byte-identical wherever it is referenced.
  - `cls` (string) — object class, e.g. `"person"`. Filtered case-insensitively by `op_detect`.
  - `confidence` (number | null) — detector confidence 0–1.
  - `box` (object) — `{x,y,w,h}` in **normalized 0–1** coords (top-left origin), rounded to
    **≤ 4 decimals** (matches `Box.rounded()`).

## `read_text` — OCR results, keyed by `det_id`

```json
"read_text": {
  "p_messi": { "text": "10", "confidence": null, "source": "pinned", "note": "..." }
}
```

- **Keys** are `det_id`s — every key MUST be a real det_id present somewhere in `detect`.
- **Each entry** is `{ text, confidence, source, note? }`:
  - `text` (string) — the read digits. `op_read_text` skips entries with empty/missing text.
  - `confidence` (number | null).
  - `source` — provenance enum, one of:
    - `live` — a real OCR call (gpt-4o VLM) made this pass.
    - `cached` — a precomputed real OCR output replayed from this cache.
    - `pinned` — a hand-verified value (the hero #10).
  - `note` (string, optional) — diagnostic/disclosure text.
- **Missing is legal** (D-DR5 honest-empty): a det with no `read_text` entry means OCR was
  skipped or returned nothing; `op_read_text` degrades that crop to empty rather than guessing.
- **Azure Read is deliberately NOT a source** here — its WC-font misread (22→77) would poison
  the cache. The OCR ladder is: manifest pin → gpt-4o VLM → skip.

## `events` — known goal moments

```json
"events": [
  { "ts_ms": 4000, "type": "goal", "scorer_det": "p_messi",
    "box": { "x": 0.86, "y": 0.34, "w": 0.10, "h": 0.34 } }
]
```

- Each event is `{ ts_ms, type, scorer_det, box }`. `op_temporal_order` reads `type == "goal"`
  events sorted by `ts_ms`; the first goal's `scorer_det` is the **verdict linchpin**.
- `scorer_det` MUST be a real det_id that (a) appears in `detect` at some cached ts and (b) for
  the grounded "Yes", carries a `read_text` entry with `text == "10"`. Azure produces none of
  this — the pipeline derives `scorer_det` from the **pin-resolution result**, not an
  independent manifest literal, and asserts the linkage before writing.
- Omitting `events` makes `op_temporal_order` empty → the verdict flips to "No".

## Determinism (idempotency)

The pipeline writes **canonical JSON**: `detect` keys sorted numerically, detections sorted by
`det_id`, boxes rounded to 4 places, `sort_keys=True`. Re-running over the same inputs yields a
**byte-identical** cache JSON (stills are excluded from the byte-identity check — JPEG re-encode
is not bit-reproducible).
