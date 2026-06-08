# What's Next

_Generated 2026-06-08 by `/whats-next`. Each entry is sized for the 5-agent parallel-impl pipeline: parallel planning → plan aggregation → `/refine-plan` + `/finalize-plan` → 5× `claude-opus-4-8` worktree implementations → PR selection._

## Top moves (highest-leverage first)

### 1. Multi-clip / multi-query catalog — prove repeatability beyond the single hero path

- **Why now:** The system still rides **one clip (`single-goal`) and one query (`hero-10-first-goal`)** (`api/canned.py` CLIPS/QUERIES). With CI + a real test net now in place (#16), a second clip and 3–4 queries is the move that turns "an impressive demo" into "a system that repeats" — and every new clip/query now lands on the safety net that just shipped. It also stress-tests the free-text codegen prompt against clips whose `hint` differs from the hero's.
- **Scope:** ~1,000–2,000 LOC: author new manifests (`clips/*/manifest.json`) + pinned programs, register clips/queries in `canned.py`, generate caches via `scripts/precompute.py` + stills, and add query-shape variety (a counting or out-of-bounds question that exercises `count`/`filter`/`temporal_order` on paths the hero never hits).
- **Pipeline-ready:** ✅ (moderate) — design room is in query selection and which op-paths to stress; lower structural variance than a new-architecture feature (more content/data). **Note:** the *real* caches need live Azure Vision detect + the source video, so the 5 worktree agents author manifests/programs + verify via mocked detect, while actual cache generation runs separately with creds — the live step the precompute pipeline already isolates.
- **Plan artifact:** needs writing — start with a draft, then `/refine-plan`.
- **Status:** [ ] pending

### 2. Free-text program cache + shareable permalink replays — stop re-billing identical questions

- **Why now:** Every free-text query fires a **billed** Azure codegen call on an unauthenticated GET; the only guard today is a blunt `MAX_CODEGEN_CALLS = 200` per-process counter (`server.py`) + a length cap. A content-addressed cache (hash of normalized `query_text` + `clip` → the validated program / run-doc) is the natural cost lever *and* a feature enabler: a cached, deterministic run is exactly what a **shareable / permalink replay** needs. Now that the SSE + run-doc paths have dedicated tests (#16), reworking the doc-build path is safe.
- **Scope:** ~1,000–2,000 LOC: a cache layer keyed on normalized query+clip, a persistence backend (file now, Blob-ready), wiring into `build_free_text_run_doc` ahead of the codegen call (trust-boundary-respecting — only cache programs that already passed validation), plus a `run_id`/permalink route. Web: a "copy run link" affordance + load-from-id path in `useRun.ts`.
- **Pipeline-ready:** ✅ — design room in key normalization (casing/whitespace/synonyms), program-vs-run-doc cache granularity, eviction/TTL, the persistence seam, and the share-link UX. Five agents would diverge most on cache-key design and cache granularity.
- **Plan artifact:** needs writing — start with a draft, then `/refine-plan`.
- **Status:** [ ] pending

### 3. Trust-boundary hardening — resource + semantic guards on the generated-program executor

- **Why now:** The free-text path now runs **LLM-generated programs** through the interpreter, and `CLAUDE.md` names layer 2 as "the trust boundary: generated code runs here, so isolate and validate it." Today validation is the DSL schema + a few semantic bounds (e.g. the canned-path numeric-bounds gap closed in #14). A dedicated pass — per-op arg-domain validation, max-steps / max-frames execution limits, a bounded envelope, adversarial codegen-output fuzzing, and an explicit reject-before-execute path — hardens the one place untrusted model output runs. The new test harness (#16) makes it verifiable.
- **Scope:** ~1,000–2,000 LOC: extend the validator (`api/scripts/validate_program.py` + `api/schema/`), add interpreter guards (step/frame caps, arg-domain checks, no-partial-run on reject), and an adversarial suite that feeds malformed / oversized / hostile programs and asserts they're rejected before any op executes.
- **Pipeline-ready:** ✅ — wide design room: defensive depth, where validation lives (schema vs interpreter vs both), the limit values, and the reject-vs-degrade posture. Five agents diverge most on how aggressively to lock the boundary.
- **Plan artifact:** needs writing — start with a draft, then `/refine-plan`.
- **Status:** [ ] pending

_Smaller fast-follows (one-agent scope — direct PRs, not a 5-agent fan-out): a real **web ESLint gate** (flat-config `eslint-config-next`, explicitly deferred from the #16 CI slice) and a **calibrated coverage floor** (flip the now-measured `pytest --cov` / vitest coverage to `--cov-fail-under` / thresholds once the natural numbers are read). Both are one obvious path — better as direct PRs than pipeline runs._

## On hold / needs human first

### A. Live gpt-4o VLM `read_text` backend

- **Blocker:** Azure OpenAI TPM quota — `api/vision/vlm_read.py` is ready but the deployment is rate-limited, so it can't be exercised. The merged precompute OCR ladder is **pin → gpt-4o VLM → skip** (Azure Read was deliberately dropped), so the VLM is *the* live OCR backend once unblocked. Also small (~300 LOC) — one-agent work, not a 5-agent fan-out.
- **Unblocking action:** Raise the TPM quota on the gpt-4o deployment (`mjlaielli-6579-resource`, AIServices/eastus2). Then live precompute/codegen can read novel jersey numbers without hand-pins; until then non-pinned jerseys honestly skip to diagnostic-empty.

### B. Azure production deployment + infrastructure-as-code

- **Blocker:** Needs the user's Azure subscription, secret provisioning, and live deploy credentials to verify — external coordination the pipeline can't do or check.
- **Unblocking action:** Provision/authorize an Azure subscription + grant deploy creds (Container Apps for the API, Static Web Apps for `web/`, Blob for caches, Key Vault for secrets), or confirm it's acceptable to author Bicep/Terraform + a deploy workflow without an end-to-end deploy verification.

## Recently shipped (auto-tracked)

- 2026-06-08 #16: CI gate + widen the test net — first `.github/workflows` matrix (Python pytest + web build/tsc/vitest, green on push/PR to `main`) plus dedicated coverage for `interpreter/primitives.py`, the `server.py` SSE stream (sequence + cancel + info-leak guard), the Azure vision adapters, and the entire web side via a new Vitest + Testing Library toolchain. Python 41→105, web 0→35. Completed the prior top entry #1. _(Follow-up #20, in review, tightens 5 assertions the codex challenge flagged as false-confidence.)_
- 2026-06-07 #14: free-text trust-boundary fixups — closed the canned-path numeric-bounds gap and grounded out absent-subject temporal answers; hardened the just-shipped free-text path.
- 2026-06-07 #11: free-text query input — typed question → live Azure codegen → validate → execute, with NO pinned fallback and an honest `_ungrounded` "couldn't ground this" run-doc for every failure mode (`server.py:build_free_text_run_doc`). Completed the then-top entry — the full ViperGPT thesis is live end to end.
- 2026-06-06 #7: automated clip→cache precompute pipeline — `scripts/precompute.py` turns a clip + manifest into the replay cache + stills (pin→VLM→skip OCR ladder, atomic deterministic writes); also stood up the **first `api/` pytest suite**.
- 2026-06-06 #2: live Azure OpenAI codegen (question → DSL) with D-DR6 fallback — codegen is live; behavior is identical to pinned when creds/quota are absent.
- 2026-06-04 #1: vertical slice — ViperGPT-style video analyst (backend + web UI) — FastAPI + SSE + JSON-DSL interpreter + Azure AI Vision primitives + Next.js Glass-Box run viewer.

## Manual additions

<!-- Anything the user adds below this line is preserved across `/whats-next` runs. -->
