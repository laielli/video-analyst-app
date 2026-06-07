# What's Next

_Generated 2026-06-06 by `/whats-next`. Each entry is sized for the 5-agent parallel-impl pipeline: parallel planning → plan aggregation → `/refine-plan` + `/finalize-plan` → 5× `claude-opus-4-7` worktree implementations → PR selection._

## Top moves (highest-leverage first)

### 1. Free-text query input — the full ViperGPT thesis, end to end

- **Why now:** Both prerequisites just landed — the precompute pipeline (#7, shipped) produces real precomputed evidence, and live Azure OpenAI codegen (#2) already turns a question into a DSL program. Letting a user *type* a question and watch the LLM compile + execute a visible program is the product's whole point and its single biggest "wow." With the foundation in place, this is now the top move.
- **Scope:** ~1,500–2,500 LOC across all three layers: a query input in the Glass-Box UI (`web/`), a new `GET /api/run?query_text=…&clip=…` route (`api/server.py`), a clip-aware system prompt (`api/codegen.py`), and an honest degradation path — free text has *no* pinned fallback, so the D-DR5/D-DR6 "couldn't ground this" UX must be designed, not inherited.
- **Pipeline-ready:** ✅ — wide design room: fallback/error UX when codegen or execution can't ground an answer, prompt design + clip-metadata injection, caching of novel generated programs, and input affordances. Five agents would diverge most on the "no grounded answer" experience.
- **Plan artifact:** needs writing — start with a draft, then `/refine-plan`.
- **Status:** [ ] pending

### 2. Extend the test suite + add CI — the harness exists now; widen coverage and gate it

- **Why now:** PR #7 stood up the **first `api/` pytest harness** (`api/pytest.ini` + `conftest.py` + 22 tests) — but only for the precompute module. The interpreter ops (`primitives.py`/`cache.py`), validator semantics, codegen (mocked Azure), and the server SSE routes are still untested; `web/` has zero tests; and there's no CI. Activation energy is now low (the harness + mock seams exist), and this protects a now-multi-module portfolio codebase senior reviewers will scrutinize.
- **Scope:** ~1,500–3,000 LOC. Python: extend pytest to `api/interpreter/primitives.py`, `api/scripts/validate_program.py`, `api/codegen.py` (reusing PR #7's `from_env` mock seam), and `api/server.py` SSE. Web: Vitest + Testing Library for the SSE hook (`web/lib/useRun.ts`), panels, and the scrubber. Plus a GitHub Actions workflow (none exists today).
- **Pipeline-ready:** ✅ — design room in mocking strategy (fake Azure + `EventSource`), fixture design, coverage targets, and the CI matrix. No external blockers — everything mocks.
- **Plan artifact:** needs writing — start with a draft, then `/refine-plan`.
- **Status:** [ ] pending

### 3. Multi-query / multi-clip catalog — prove repeatability beyond the hero path

- **Why now:** The precompute pipeline (shipped) makes generating new caches cheap, and the registry (`api/canned.py`) + UI selectors already support a list — but only one clip + one query exist. A second clip and 3–4 queries turn "a demo" into "a system."
- **Scope:** ~800–1,500 LOC: author new manifests + programs, register clips/queries in `api/canned.py`, generate caches via `scripts/precompute.py` + stills, and add query-shape variety (e.g. a counting or out-of-bounds question that exercises `count`/`filter`/`temporal_order` differently).
- **Pipeline-ready:** ✅ (moderate) — design room mostly in query selection and which op-paths to stress; lower variance than #1–#2 (more content/data than structural design). **Note:** producing the *real* caches needs live Azure Vision detect + the source video, so the 5 worktree agents author manifests/programs + verify via mocks, while the actual cache generation runs separately with creds (the same live step the precompute pipeline already has).
- **Plan artifact:** needs writing — start with a draft, then `/refine-plan`.
- **Status:** [ ] pending

## On hold / needs human first

### A. Live gpt-4o VLM `read_text` backend

- **Blocker:** Azure OpenAI TPM quota — `api/vision/vlm_read.py` is ready but the deployment is rate-limited, so it can't be exercised. The merged precompute OCR ladder is **pin → gpt-4o VLM → skip** (Azure Read was deliberately dropped), so the VLM is *the* live OCR backend once unblocked. Also small (~300 LOC) — one-agent work, not a 5-agent fan-out.
- **Unblocking action:** Raise the TPM quota on the gpt-4o deployment (`mjlaielli-6579-resource`, AIServices/eastus2). Then live precompute/codegen can read novel jersey numbers without hand-pins; until then non-pinned jerseys honestly skip to diagnostic-empty.

### B. Azure production deployment + infrastructure-as-code

- **Blocker:** Needs the user's Azure subscription, secret provisioning, and live deploy credentials to verify — external coordination the pipeline can't do or check.
- **Unblocking action:** Provision/authorize an Azure subscription + grant deploy creds (Container Apps for the API, Static Web Apps for `web/`, Blob for caches, Key Vault for secrets), or confirm it's acceptable to author Bicep/Terraform + a deploy workflow without an end-to-end deploy verification.

## Recently shipped (auto-tracked)

- 2026-06-06 #7: Automated clip→cache precompute pipeline — `scripts/precompute.py` turns a clip + manifest into the replay cache + stills (pin→VLM→skip OCR ladder, atomic deterministic writes, the events/det_id verdict-linchpin); also stood up the **first `api/` pytest suite (22 tests)**. Completed the prior top entry #1.
- 2026-06-06 #2: live Azure OpenAI codegen (question → DSL) with D-DR6 fallback — codegen is live; behavior is identical to pinned when creds/quota are absent.
- 2026-06-04 #1: vertical slice — ViperGPT-style video analyst (backend + web UI) — FastAPI + SSE + JSON-DSL interpreter + Azure AI Vision primitives + Next.js Glass-Box run viewer.

## Manual additions

<!-- Anything the user adds below this line is preserved across `/whats-next` runs. -->
