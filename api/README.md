# api/ — execution + vision layer

The Python side of the Glass-Box Video Analyst (ViperGPT pattern). The **DSL schema** (the
program the LLM emits), the **interpreter** (program -> run-doc, replay mode), and the
**vision harnesses** (T1 clip check) live here. Azure OpenAI codegen + a FastAPI/SSE wrapper
come next.

```
api/
  schema/dsl.schema.json        # the visual-program DSL (ONE schema, three masters — see below)
  schema/run_doc.schema.json    # the execution-trace contract the UI renders (per-step + provenance)
  examples/hero_program.json    # pinned program for "Does #10 score the first goal?"
  examples/hero_cache.json      # replay cache: REAL detect boxes + the hand-verified #10 pin
  examples/hero_run_doc.json    # a worked run-doc (reference)
  interpreter/                  # cache.py · primitives.py · interpreter.py (tree-walking executor)
  scripts/run_program.py        # program + cache -> validated run-doc (zero Azure calls)
  scripts/validate_program.py   # structural (jsonschema) + named-binding validation
  scripts/validate_run_doc.py   # run-doc contract validation + program cross-check
  scripts/ocr_probe.py          # T1: Azure AI Vision read (misreads jersey fonts; see below)
  scripts/vlm_probe.py          # T1: gpt-4o vision read (accurate; quota-gated)
  vision/azure_vision.py        # Azure AI Vision 4.0 adapter: detect + read_text, normalized boxes
  vision/vlm_read.py            # gpt-4o vision read_text backend (Azure OpenAI)
  requirements.txt  .env.example
```

## Run the hero program (replay, no Azure)

```bash
cd api && source .venv/bin/activate && pip install -r requirements.txt
python scripts/run_program.py --out /tmp/hero_run.json
```

Validates the program, interprets it over `hero_cache.json` (real detect boxes + the pinned
#10), and emits a run-doc that re-validates against `run_doc.schema.json`. This is the demo's
replay path — deterministic and free. The run-doc is what the UI renders.

## Serve it (FastAPI + SSE)

```bash
uvicorn server:app --reload --port 8000      # from api/, venv active
```

- `GET /api/health` · `GET /api/catalog` — liveness + the clip/query selectors.
- `GET /api/run_doc?query=hero-10-first-goal` — the whole run-doc at once (debug / scrub).
- `GET /api/run?query=hero-10-first-goal&pace_ms=1200` — **SSE**: streams `meta` (program +
  clip) → one paced `step` per trace entry (D-DR7 ~1.2s autoplay) → `findings` → `done`. The
  UI renders the PROGRAM panel from `meta`, then animates each step as it arrives (D-DR8). Same
  stream for replayed or (later) live-generated runs (D-DR4). Use `pace_ms=0` for tests.

```bash
curl -sN "http://127.0.0.1:8000/api/run?query=hero-10-first-goal&pace_ms=0"
```

## Generate the program live (Azure OpenAI codegen)

ViperGPT Layer 1: `codegen.py` compiles the question into the DSL via Azure OpenAI, sending
`dsl.schema.json` as a **strict structured-output** constraint (valid-by-construction) and
re-checking it with `validation_errors` (the semantics JSON Schema can't express). Execution
stays replay-mode over the pinned cache, so a live-generated program runs deterministically and
free — only codegen hits the network.

```bash
python scripts/codegen_probe.py --run     # question -> program -> validate -> replay over cache
```

**D-DR6 fallback (always safe):** the server uses the live program only if it validates *and*
executes to a grounded answer; on any miss — no `AZURE_OPENAI_*` creds, API error, invalid
program, or cache gap — it silently falls back to the query's pinned program. The chosen
provenance rides the SSE `meta` event as `program_source` (`live` | `pinned`). With no creds,
behavior is identical to before (pinned). Set creds in `api/.env` (see `.env.example`).

## The DSL schema (dsl-json-schema-unification)

`schema/dsl.schema.json` is deliberately one artifact serving three roles:
1. **Codegen constraint** — sent to Azure OpenAI as a strict structured-output schema, so the
   model can only emit valid, whitelisted programs (strip `$schema`/`$id`/`title` before the
   API call; the rest is strict-compatible).
2. **Interpreter validator** — `validate_program.py` checks structure here, then enforces the
   semantics JSON Schema can't express: every binding ref names a real prior step of a
   compatible kind, ids are unique, `answer` is last.
3. **Run-doc base** — the execution trace the UI consumes is this program with a `result`
   added per step (output, confidence, normalized overlay boxes). The `finalized.html` mock's
   `STAGES` array is a hand-rolled preview of that shape.

The load-bearing part is the `detect -> crop -> read_text` named-binding chain (how a detection
box flows into a crop and then into OCR). The hero program wires it as
`frames -> people -> jerseys -> numbers -> tens -> ordered -> result`.

## Validate the schema (no Azure needed)

```bash
cd api
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/validate_program.py            # validates examples/hero_program.json
```

## T1: prove Azure reads the jersey number

The hero clip (`single-goal.mov`, Messi vs Mexico) is visually a slam dunk — `#10` is sharp at
~4.6s. This harness closes T1 by actually calling Azure AI Vision.

**1. Provision a minimal Azure AI Vision resource** (you have an Azure account):

```bash
az login
az group create -n video-analyst-rg -l eastus
az cognitiveservices account create \
  -n va-vision -g video-analyst-rg -l eastus \
  --kind ComputerVision --sku S1 --yes
az cognitiveservices account show     -n va-vision -g video-analyst-rg --query properties.endpoint -o tsv
az cognitiveservices account keys list -n va-vision -g video-analyst-rg --query key1 -o tsv
```

`ComputerVision` / `S1` is the Image Analysis 4.0 resource. It bills by usage (cents for this
demo) and idles near zero. If `eastus` lacks capacity, try `westus2` / `westeurope`.

**2. Configure + run:**

```bash
cp .env.example .env          # paste the endpoint + key1 into AZURE_VISION_*

# T1 check. Full-frame OCR reads the perimeter ad boards, not the jersey — that is
# exactly why the program does detect -> crop -> read_text. The real test crops tight
# to the number band and upscales so the digits clear Read's minimum glyph size:
python scripts/ocr_probe.py --ts 4.6 --crop 230,250,300,240 --scale 3 --expect 10
```

`T1 PASS` means Azure read `10` off the footage — kill-criterion retired, build proceeds.
If Read still misses (jersey numbers are a known-hard OCR case — stylized font, striped
background), the design's sanctioned fallback applies: hand-verify/correct the number while
precomputing the cache and pin it for the hero query (design doc, "The Assignment"). The demo
discloses what is pinned vs live (demo honesty).

## Precompute a clip's replay cache (`scripts/precompute.py`)

The replay cache (`examples/hero_cache.json`) used to be hand-authored. `scripts/precompute.py`
is the orchestrated, **idempotent** pipeline that regenerates it from a raw clip + a per-clip
**manifest** — no hand-annotation. It samples the clip on the SAME stride the program uses,
runs Azure AI Vision `detect` per sampled frame, mints stable `det_id`s, resolves the
manifest's hand-verified jersey pin by nearest box, builds the `events` block, writes the
canonical cache atomically, and **self-validates** by replaying the hero program over the fresh
cache (re-checking the run-doc against `run_doc.schema.json`).

```bash
cd api && source ../.venv/bin/activate
python scripts/precompute.py --clip single-goal --query hero-10-first-goal   # full run
python scripts/precompute.py --clip single-goal --query hero-10-first-goal --dry-run  # plan only
python scripts/precompute.py --manifest ../clips/single-goal/manifest.json --require-ocr
```

- The clip→`.mov` resolution and all human knowledge (sampling window/fps, goal events,
  jersey pins, evidence-frame filenames) live in `clips/<id>/manifest.json` — `canned.py` is
  untouched. A clip with no manifest is a setup error (exit 2); the pipeline never synthesizes
  defaults.
- **OCR ladder:** manifest pin → gpt-4o VLM (`vlm_read.py`) → skip (honest-empty). Azure Read
  is deliberately not a rung (its WC-font misread, 22→77, would poison the cache). On
  429/missing-creds the VLM rung skips and warns; `--require-ocr` opts into hard-fail (exit 1).
- **Stills:** real JPEGs into `../web/public/frames/`, reproducing the exact filenames the
  `EvidencePanel` renders (`goal-4000.jpg`, `scorer-4625.jpg`) — `frameSrc` is not edited.
- **Idempotent:** canonical JSON (sorted keys, det_id-sorted detections, 4-decimal boxes), so
  re-running yields a byte-identical cache. Exit codes: 0 = written + self-validated, 1 =
  produced but invalid/ungrounded (or a required OCR read was skipped), 2 = setup problem.

The exact cache shape is documented in [`../docs/cache-schema.md`](../docs/cache-schema.md)
(prose contract; the box/detection `$defs` already live in `run_doc.schema.json`).

## Tests

```bash
cd api && source ../.venv/bin/activate && pip install -r requirements.txt
python -m pytest -q          # first pytest suite (api/tests/): precompute + interpreter contract
```

Azure (detect + VLM) and ffmpeg are mocked, so the suite runs offline with no `.mov` and no
creds. It asserts the **grounded** verdict chain (`events.scorer_det` → `detect` det_id →
`read_text` text=="10") rather than the hardcoded verdict string, the cache structural
invariants, det_id minting determinism, the OCR ladder, and idempotency.

## What's next (per the locked plan)

- [x] DSL schema + validator; run-doc contract + validator.
- [x] Interpreter: program -> run-doc in replay mode (`run_program.py`), Yes and No paths.
- [x] FastAPI + SSE wrapper (`server.py`): streams `meta -> step* -> findings -> done`, paced.
- [x] **Next.js UI** (`../web`) rendering the SSE stream (the `finalized.html` motion reference is the target).
- [x] **Azure OpenAI codegen** (`codegen.py`): question -> DSL with strict schema validation + the
  pinned-program fallback (D-DR6). Env-gated; `program_source` (`live`/`pinned`) rides the `meta` event.
- [x] **Precompute pipeline** (`scripts/precompute.py`): raw clip + manifest -> replay cache + stills,
  self-validating + idempotent. First pytest suite (`tests/`). Local storage (no Blob yet).
- [ ] **Live `read_text` via gpt-4o vision** once TPM quota clears (`vlm_read.py` is ready; flip cache source from `pinned` to `live`).
- [ ] **Precompute + cache** to Azure Blob (the pipeline writes locally today; `--storage`/Blob is the next step).
