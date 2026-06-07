# What's Next

_Generated 2026-06-06 by `/whats-next`. Each entry is sized for the 5-agent parallel-impl pipeline: parallel planning → plan aggregation → `/refine-plan` + `/finalize-plan` → 5× `claude-opus-4-7` worktree implementations → PR selection._

## Top moves (highest-leverage first)

### 1. Automated clip→cache precompute pipeline — turn a raw clip + query into a replayable run, no hand-annotation

- **Why now:** The whole demo is pinned to one hand-built artifact (`api/examples/hero_cache.json` + hand-extracted `web/public/frames/*.jpg`). There is no orchestrated path from a `.mov` to a cache — only a single-frame probe (`api/scripts/ocr_probe.py:extract_frame`). This is the foundation every other feature stands on: multi-query, multi-clip, and free-text all need real precomputed evidence. Build this first and the next three items get dramatically cheaper.
- **Scope:** ~1,500–3,000 LOC. New `api/scripts/precompute.py` (or `api/ingest/`) + a cache-schema doc; reuses ffmpeg extraction from `ocr_probe.py`, `AzureVision.analyze` (`api/vision/azure_vision.py`) for detect+Read, and the existing `Cache`/run-doc shapes (`api/interpreter/cache.py`, `api/schema/`). Writes frame stills into `web/public/frames/`.
- **Pipeline-ready:** ✅ — real design room: frame-sampling strategy, cache keying + idempotency/incremental re-runs, local-vs-Blob storage, OCR-miss handling (manual-pin override vs Azure Read vs skip), CLI ergonomics, and how it degrades when the gpt-4o VLM path is TPM-gated (fall back to Azure Read). Five agents would diverge meaningfully on storage model and failure posture.
- **Plan artifact:** needs writing — start with a draft, then `/refine-plan`.
- **Status:** [ ] pending

### 2. Free-text query input — the full ViperGPT thesis, end to end

- **Why now:** v1 deliberately deferred free text (design Premise 1/6), but live codegen already landed (PR #2). Letting a user type a question and watch the LLM compile + execute a visible program is the single biggest "wow" and the product's whole point. Highest product leverage once #1 gives novel questions real evidence to run against.
- **Scope:** ~1,500–2,500 LOC across all three layers: a query input in the Glass-Box UI (`web/`), a new `GET /api/run?query_text=…&clip=…` route (`api/server.py`), a clip-aware system prompt (`api/codegen.py`), and an honest degradation path — free text has *no* pinned fallback, so the D-DR5/D-DR6 "couldn't ground this" UX must be designed, not inherited.
- **Pipeline-ready:** ✅ — wide design room: fallback/error UX when codegen or execution can't ground an answer, prompt design + clip-metadata injection, caching of novel generated programs, and input affordances. Five agents would diverge most on the "no grounded answer" experience.
- **Plan artifact:** needs writing — start with a draft, then `/refine-plan`.
- **Status:** [ ] pending

### 3. Test suite + CI — make this portfolio piece reviewable and the above work safe

- **Why now:** Zero tests today, and this is a career/portfolio repo senior engineers will read. Tests de-risk #1 and #2 and let five parallel agents be compared on behavior, not vibes. Compounds across everything that follows.
- **Scope:** ~1,500–3,000 LOC. Python: pytest for interpreter ops (`api/interpreter/primitives.py`), validator semantics (`api/scripts/validate_program.py`), codegen with mocked Azure (`api/codegen.py`), and SSE routes (`api/server.py`). Web: Vitest + Testing Library for the SSE hook (`web/lib/useRun.ts`), panels, and the scrubber. Plus a GitHub Actions workflow (none exists today).
- **Pipeline-ready:** ✅ — design room in mocking strategy (how to fake Azure + EventSource), fixture design, coverage targets, and CI matrix. No external blockers — everything mocks.
- **Plan artifact:** needs writing — start with a draft, then `/refine-plan`.
- **Status:** [ ] pending

### 4. Multi-query / multi-clip catalog — prove repeatability beyond the hero path

- **Why now:** The registry already supports a list (`api/canned.py`), but only one clip + one query exist, and the UI already has selectors. Adding a second clip and 3–4 queries shows the system generalizes — the difference between "a demo" and "a system." Much cheaper once #1 (precompute pipeline) lands, so it sits below it.
- **Scope:** ~800–1,500 LOC: register new clips/queries in `api/canned.py`, generate their caches (via #1's pipeline) + frame stills, and add any query-shape variety the interpreter needs (e.g. a counting or out-of-bounds question that exercises `count`/`filter`/`temporal_order` differently).
- **Pipeline-ready:** ✅ (moderate) — design room mostly in query selection and which new op-paths to stress; lower variance than #1–#3 since it's largely data + registry once the pipeline exists.
- **Plan artifact:** needs writing — start with a draft, then `/refine-plan`.
- **Status:** [ ] pending

## On hold / needs human first

### A. Live gpt-4o VLM `read_text` backend

- **Blocker:** Azure OpenAI TPM quota — `api/vision/vlm_read.py` is written and ready, but the deployment is rate-limited; it can't be exercised. Also small (~300 LOC), so it's one-agent work, not a 5-agent fan-out.
- **Unblocking action:** Raise the TPM quota on the gpt-4o deployment (`mjlaielli-6579-resource`, AIServices/eastus2), then flip `Cache.read_text_for` to call the VLM with an Azure-Read fallback. Until then, #1 uses Azure AI Vision Read for OCR.

### B. Azure production deployment + infrastructure-as-code

- **Blocker:** Needs the user's Azure subscription, secret provisioning, and live deploy credentials to verify — external coordination the pipeline can't do or check.
- **Unblocking action:** Provision/authorize an Azure subscription + grant deploy creds (Container Apps for the API, Static Web Apps for `web/`, Blob for caches, Key Vault for secrets), or confirm it's acceptable to author Bicep/Terraform + a deploy workflow without an end-to-end deploy verification.

## Recently shipped (auto-tracked)

- 2026-06-06 #2: live Azure OpenAI codegen (question → DSL) with D-DR6 fallback — codegen is live; behavior is identical to pinned when creds/quota are absent.
- 2026-06-04 #1: vertical slice — ViperGPT-style video analyst (backend + web UI) — FastAPI + SSE + JSON-DSL interpreter + Azure AI Vision primitives + Next.js Glass-Box run viewer.

## Manual additions

<!-- Anything the user adds below this line is preserved across `/whats-next` runs. -->
