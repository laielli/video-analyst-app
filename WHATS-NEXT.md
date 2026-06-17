# What's Next

_Generated 2026-06-16 by `/whats-next` (post the #45→#49→#50→#51 eval-hardening arc; gate now T4 1.00 / oos 1.00 / adv 7/7 on 66 cases at prompt_version `474f8438`). Each entry is sized for the 5-agent parallel-impl pipeline: parallel planning → plan aggregation → `/refine-plan` + `/finalize-plan` → 5× `claude-opus-4-8` worktree implementations → PR selection._

## Top moves (highest-leverage first)

### 1. Widen the visual API surface — one new primitive through all three layers

- **Why now:** The op surface has been frozen at **8 ops** (`api/interpreter/primitives.py` OPS: `sample_frames · detect · crop · read_text · filter · count · temporal_order · answer`) since the vertical slice, and `CLAUDE.md` names this as *the* growth axis: new primitive in layer 2 + model in layer 3, exposed in layer 1's prompt. Every prerequisite has now shipped — validator + limits + adversarial suite (#33), multi-clip catalog (#21), coverage gates (#36), and crucially a **stable, measured eval instrument**: the 66-case bank now scores T4 1.00 / out-of-scope-honest 1.00 / adversarial-safe 7/7 at frozen `prompt_version 474f8438` (#51). The earlier "merge #45 first" sequencing caveat is fully resolved — #45 merged and the prompt has since been hardened and re-baselined three times (#49/#50/#51). That means a new primitive's effect on validity/groundedness/refusal is now **quantified against a 1.00 ceiling, not vibes** — any regression a new shape introduces shows up immediately in the replay gate. Strongest single-primitive candidates: a **scene/caption op** (Azure Image Analysis captioning — a *different* service from the quota-blocked OpenAI VLM, so unblocked), a **spatial-relation filter** (local geometry over detection boxes — "left of", "near", "above"), or a **presence-over-time / dwell op**.
- **Scope:** ~2,000–4,000 LOC across all three layers: `api/schema/dsl.schema.json` + `scripts/validate_program.py` + `limits.py` (new op + arg domains + caps), `interpreter/primitives.py` (new `op_*` + OPS entry + findings plumbing), the codegen op spec in `SYSTEM_PROMPT` (+ a few-shot worked example in the #51 style), precompute caching for the new op's outputs, eval-bank cases for the new shape (`api/eval/question_bank.json`, captured on the PR branch before merge per the established runbook), and web rendering of the new step kind (`EvidencePanel` + `ProgramPanel` + `StepTracker`) — with adversarial cases at every layer.
- **Pipeline-ready:** ✅ — the widest design room on the list: arg-domain design, local-vs-Azure-backed split, cache shape, diagnostic-empty semantics for the new shape, few-shot example construction, and UI evidence rendering. **Plan-phase note:** the 5 plan drafts should *converge on one primitive* (the choice is a planning decision, not a per-agent divergence) so the 5 implementations stay comparable; divergence then lives in arg-domain/cache/defensive-posture/UI/test-thoroughness. **Capture note:** a `SYSTEM_PROMPT` edit stales all 66 committed fixtures uniformly → the phase-aware advisory gate keeps CI green; pair the post-merge live re-capture (~$0.40, runbook in `api/eval/README.md`) with this entry's new eval cases and re-ratchet floors ~5pp under measured.
- **Plan artifact:** needs writing — start with a draft, then `/refine-plan`.
- **Status:** [ ] pending

### 2. Runs gallery / demo front door — make the system's range discoverable

- **Why now:** The ingredients all shipped and nothing consumes them yet: a content-addressed run store with permalinks (#25, `api/run_cache.py` + `?run=<id>`), a multi-clip/multi-query catalog (#21, `/api/catalog`), and a committed eval report that *names* what the system answers well (T4 1.00 across 4 shapes × 2 clips). But the portfolio audience still lands on a single analyst screen (`web/app/page.tsx` is the only page) with one query loaded — there is no way to *discover* the system's range. A gallery/landing surface — clips, canned queries by shape, notable cached free-text runs, each linking into the analyst view via the existing permalink routes — turns the demo from "one impressive run" into "browse what this system can do." High portfolio leverage: it's the first thing a reviewer sees, and it directly showcases the breadth entry 1 keeps adding.
- **Scope:** ~1,000–2,500 LOC: a small API listing surface (extend `/api/catalog` or add a run-listing endpoint over `RunStore`), a gallery/index route in `web/app/` consistent with DESIGN.md tokens (Fraunces / Hanken Grotesk / IBM Plex Mono; paper/ink/teal; sharp 2–4px radius, hairlines), clip stills as thumbnails, navigation into the existing `?query=`/`?run=` load paths, + tests.
- **Pipeline-ready:** ✅ — design room in the IA (landing page vs in-app gallery vs sidebar), what each card surfaces (query shape, clip, confidence, cached-answer preview), static-export vs API-driven listing, curation posture. Must stay strictly inside the locked design system (DESIGN.md tokens are non-negotiable — no Inter/Roboto/system-ui, the slop tell).
- **Plan artifact:** needs writing — start with a draft, then `/refine-plan`.
- **Status:** [ ] pending

### 3. Accessibility + keyboard presentation pass — the agreed design debt, deps long met

- **Why now:** `TODOS.md` carries this as explicitly-deferred (not skipped) design debt from the 2026-05-31 design review, blocked on T-D1/T-D2 (layout + run engine) — both long since shipped. Current a11y coverage is sparse (a handful of `aria-`/`role=` attributes per component, no live-region narration anywhere). The pass: keyboard nav for run controls + the 7-step scrubber, ARIA landmarks for the 3 zones (PROGRAM / EVIDENCE / Findings), screen-reader step narration ("step 4 of 7, read_text, found 10"), focus order, and contrast-audit verification. A portfolio piece failing basic a11y reads badly to exactly the senior-engineer audience this repo targets. Partial mitigations already exist (prefers-reduced-motion D-DR8, contrast-aware tokens D-DR9) — this is the finishing pass on top.
- **Scope:** ~1,000–2,000 LOC, all web-side: `StepTracker` keyboard/scrubber semantics, landmarks + labels across `page.tsx` + the panels, a live-region narration strategy, focus management on run/replay, Testing Library ARIA assertions extending the existing vitest suites (~298 test fns today).
- **Pipeline-ready:** ✅ (moderate) — design room in narration granularity vs announcement fatigue, the focus model, scrubber keyboard semantics, and how much canvas evidence to mirror as offscreen text. Less structural variance than entry 1, but enough for a meaningful 5-way comparison.
- **Plan artifact:** needs writing — start with a draft, then `/refine-plan`.
- **Status:** [ ] pending

## On hold / needs human first

### A. Live gpt-4o VLM `read_text` backend

- **Blocker:** Azure OpenAI TPM quota — `api/vision/vlm_read.py` is ready but the deployment is rate-limited for vision-token-heavy calls, so it can't be exercised. (The #43/#51 captures proved *codegen* quota is fine at 60+ sequential text calls; VLM image calls are far heavier per request.) The merged precompute OCR ladder is **pin → gpt-4o VLM → skip**, so the VLM is *the* live OCR backend once unblocked. Also small (~300 LOC) — one-agent work, not a 5-agent fan-out.
- **Unblocking action:** Raise the TPM quota on the gpt-4o deployment (`mjlaielli-6579-resource`, AIServices/eastus2). Then live precompute/codegen can read novel jersey numbers without hand-pins; until then non-pinned jerseys honestly skip to diagnostic-empty.

### B. Azure production deployment + infrastructure-as-code

- **Blocker:** Needs the user's Azure subscription, secret provisioning, and live deploy credentials to verify — external coordination the pipeline can't do or check.
- **Unblocking action:** Provision/authorize an Azure subscription + grant deploy creds (Container Apps for the API, Static Web Apps for `web/`, Blob for caches, Key Vault for secrets), or confirm it's acceptable to author Bicep/Terraform + a deploy workflow without an end-to-end deploy verification. Entry #2 (gallery/front door) raises the value of unblocking this — a public URL is what the gallery is *for*.

## Recently shipped (auto-tracked)

- 2026-06-16 #51: few-shot worked-examples codegen iteration — one worked example per failing shape, rulebook trimmed 6.8k→4.2k chars (~36%); closed all 6 remaining misses (3 scorer-number paraphrases, single-goal-presence-far-3, and the adversarial count-routing leaks). Lifted T4 0.89→**1.00** and out-of-scope 0.93→**1.00** on the 66-case bank; floors ratcheted to 0.95 at `prompt_version 474f8438`. Stabilized the prompt base entry #1 now extends.
- 2026-06-12 #50: noise-precedence + bank v2 — noise-precedence rule + midpoint exclusivity + hint-number-filter ban; grew the bank +6 → **66 cases**; baseline T4 0.89 / oos 0.93.
- 2026-06-12 #49: analyzed-window-law regression repair — fixed the in-scope grounding regression introduced by #45 (T4 0.42→0.89, count 6/6); dropped the whole-clip count fallback, added the digit-identify shape.
- 2026-06-11 #45 (winner of run `impl-par-20260610-105516`, merged + close-out complete): close-the-measured-prompt-gaps — count-shape single-frame rule + capability-scope honest-grounding rule in `SYSTEM_PROMPT`, phase-aware committed-fixture tests, bank `prompt_version` provenance guard. The prior On-hold §A merge handoff (#44–#48 comparative review, cherry-picks, losers closed, post-merge re-capture, worktree cleanup) is fully done.
- 2026-06-10 #43: live-codegen 400 fix + first real gpt-4o eval baseline — `MAX_PROGRAM_TOKENS` clamped to gpt-4o's 16,384 completion cap (every live codegen call had silently 400'd since #33), 60 captured fixtures (~$0.38), floors recalibrated ~5pp under measured, phase-aware seed tests.
- 2026-06-10 #40: codegen evaluation harness — `api/eval/` package: question bank, capture/replay CLI, monotone 4-tier rubric reusing the production validate→execute path in-process, dual diffable report, CI threshold gate with anti-gaming stale handling. Tests 182→219, coverage 92%.
- 2026-06-09 #36: calibrated coverage floors — `pytest --cov-fail-under=85` + vitest thresholds.
- 2026-06-09 #35: web ESLint gate — flat-config ESLint 9 + `eslint-config-next`, CI lint step.
- 2026-06-09 #33: trust-boundary hardening — `api/limits.py`, `MAX_PROGRAM_BYTES` pre-parse cap, validator step/frame/arg-domain caps, schema static limits, two-check runtime guard, 27-test adversarial suite. Tests 155→183.
- 2026-06-09 #25: free-text run cache + shareable permalinks — content-addressed store, cache hit skips the billed codegen call, `?run=<id>` permalinks, copy-run-link UI.
- 2026-06-09 #21: multi-clip / multi-query catalog — bernabeu-counter clip + 4 query-shape programs, registry generalization, resolver.
- 2026-06-08 #16/#20: CI gate + widen the test net — first workflows matrix (Python + web), Vitest toolchain; tightened 5 assertions from the codex review.
- 2026-06-07 #11/#14: free-text query input + trust-boundary fixups — typed question → live codegen → validate → execute, honest `_ungrounded` everywhere. The full ViperGPT thesis live end to end.
- 2026-06-04/06 #1/#2/#7: vertical slice (ViperGPT-style backend + web UI) → live Azure OpenAI codegen with D-DR6 fallback → automated clip→cache precompute pipeline + first `api/` pytest suite.

## Manual additions

<!-- Anything the user adds below this line is preserved across `/whats-next` runs. -->
