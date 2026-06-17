# What's Next

_Generated 2026-06-17 by `/whats-next` (post #52 — `describe_scene` shipped; op surface 8→9; gate now T4 0.976 / oos 1.00 / adv 8/8 on **71 cases** at prompt_version `1a12f4d66e08159b`). Each entry is sized for the 5-agent parallel-impl pipeline: parallel planning → plan aggregation → `/refine-plan` + `/finalize-plan` → 5× `claude-opus-4-8` worktree implementations → PR selection._

## Top moves (highest-leverage first)

### 1. Runs gallery / demo front door — make the system's range discoverable

- **Why now:** The ingredients all shipped and nothing consumes them yet — and #52 just *widened the range to show*. There's now a content-addressed run store with permalinks (#25, `api/run_cache.py` + `?run=<id>`), a multi-clip/multi-query catalog (#21, `/api/catalog`), **9 op shapes** including the new `describe_scene` ("what is happening"), and a committed eval report that names what the system answers well (T4 0.976 across 5 shapes × 2 clips). But the portfolio audience still lands on a single analyst screen (`web/app/page.tsx` is the only page) with one query loaded — there is no way to *discover* the breadth. A gallery/landing surface — clips, canned queries by shape (now including scene-description), notable cached free-text runs, each linking into the analyst view via the existing permalink routes — turns the demo from "one impressive run" into "browse what this system can do." Highest portfolio leverage on the list: it's the first thing a reviewer sees, and every new primitive (entry 2) makes it richer.
- **Scope:** ~1,000–2,500 LOC: a small API listing surface (extend `/api/catalog` or add a run-listing endpoint over `RunStore`), a gallery/index route in `web/app/` consistent with DESIGN.md tokens (Fraunces / Hanken Grotesk / IBM Plex Mono; paper/ink/teal; sharp 2–4px radius, hairlines), clip stills as thumbnails, navigation into the existing `?query=`/`?run=` load paths, + tests.
- **Pipeline-ready:** ✅ — design room in the IA (landing page vs in-app gallery vs sidebar), what each card surfaces (query shape, clip, confidence, cached-answer preview), static-export vs API-driven listing, curation posture. Must stay strictly inside the locked design system (DESIGN.md tokens are non-negotiable — no Inter/Roboto/system-ui, the slop tell).
- **Plan artifact:** needs writing — start with a draft, then `/refine-plan`.
- **Status:** [ ] pending

### 2. Second visual primitive — spatial-relation filter (or presence-over-time / dwell op)

- **Why now:** #52 proved the full-stack primitive pattern end-to-end (new binding kind + cache slice + answer branch + UI step-kind + eval cell), and `CLAUDE.md` names this as *the* growth axis. The cleanest next primitive is one of the two **layer-2-only** candidates the #52 plan named as alternatives: a **spatial-relation filter** (local geometry over existing detection boxes — "left of", "near", "above", "between") or a **presence-over-time / dwell op** (aggregates detections across sampled frames into a duration). Both answer genuinely new question shapes the 9-op surface can't ("is the keeper left of the ball?", "how long is #7 on screen?"), and crucially both are **deterministic and external-dependency-free** — no Azure visual feature, no TPM quota, no paid live re-capture, and the eval cell scores exactly (not `any_grounded`). That sidesteps the two pain points #52 hit (the ~$0.40 capture step and the non-deterministic caption matcher) while still exercising the layer-2 validator/interpreter/cache/UI pattern.
- **Scope:** ~2,000–4,000 LOC across the layers: `dsl.schema.json` + `validate_program.py` + `limits.py` (new op + arg domains + kind-flow + caps), `interpreter/primitives.py` (new `op_*` + OPS entry + diagnostic-empty semantics), the codegen op spec + few-shot example in `SYSTEM_PROMPT`, an eval-bank cell (`question_bank.json`) with adversarial cases, and web rendering of the new step kind. **Budget note:** the base `SYSTEM_PROMPT` is at ~6694/6700 — adding an op line + routing sentence + worked example *requires freeing headroom first* (the #52 lesson: trimming routing prose has a measurable T4 cost — see Manual additions). Reclaiming prompt budget without losing pinned-phrase routing fidelity is part of this entry's real work, and folding in the presence/count routing fix while re-baselining is a natural opportunistic win.
- **Pipeline-ready:** ✅ — widest design room on the list: the relation vocabulary + geometry thresholds (or the dwell aggregation window/units), arg-domain design, kind-flow membership, diagnostic-empty wording, few-shot construction, prompt-budget reclamation strategy, and UI evidence rendering. The 5 plan drafts should *converge on one primitive* (planning decision, not per-agent divergence) so the 5 implementations stay comparable.
- **Plan artifact:** needs writing — start with a draft, then `/refine-plan`.
- **Status:** [ ] pending

### 3. Accessibility + keyboard presentation pass — the agreed design debt, deps long met

- **Why now:** `TODOS.md` carries this as explicitly-deferred (not skipped) design debt from the 2026-05-31 design review, blocked on T-D1/T-D2 (layout + run engine) — both long since shipped. Current a11y coverage is sparse (a handful of `aria-`/`role=` attributes per component, no live-region narration anywhere). The pass: keyboard nav for run controls + the step scrubber, ARIA landmarks for the 3 zones (PROGRAM / EVIDENCE / Findings), screen-reader step narration ("step 4 of 9, describe_scene, grounded"), focus order, and a contrast-audit verification. A portfolio piece failing basic a11y reads badly to exactly the senior-engineer audience this repo targets. Partial mitigations already exist (prefers-reduced-motion D-DR8, contrast-aware tokens D-DR9) — this is the finishing pass on top.
- **Scope:** ~1,000–2,000 LOC, all web-side: `StepTracker` keyboard/scrubber semantics, landmarks + labels across `page.tsx` + the panels, a live-region narration strategy, focus management on run/replay, Testing Library ARIA assertions extending the existing vitest suites.
- **Pipeline-ready:** ✅ (moderate) — design room in narration granularity vs announcement fatigue, the focus model, scrubber keyboard semantics, and how much canvas evidence to mirror as offscreen text. Less structural variance than entries 1–2, but enough for a meaningful 5-way comparison.
- **Plan artifact:** needs writing — start with a draft, then `/refine-plan`.
- **Status:** [ ] pending

## On hold / needs human first

### A. Live gpt-4o VLM `read_text` backend

- **Blocker:** Azure OpenAI TPM quota — `api/vision/vlm_read.py` is ready but the deployment is rate-limited for vision-token-heavy calls, so it can't be exercised. (The #43/#51 captures proved *codegen* quota is fine at 60+ sequential text calls; VLM image calls are far heavier per request.) The merged precompute OCR ladder is **pin → gpt-4o VLM → skip**, so the VLM is *the* live OCR backend once unblocked. Also small (~300 LOC) — one-agent work, not a 5-agent fan-out.
- **Unblocking action:** Raise the TPM quota on the gpt-4o deployment (`mjlaielli-6579-resource`, AIServices/eastus2). Then live precompute/codegen can read novel jersey numbers without hand-pins; until then non-pinned jerseys honestly skip to diagnostic-empty.

### B. Azure production deployment + infrastructure-as-code

- **Blocker:** Needs the user's Azure subscription, secret provisioning, and live deploy credentials to verify — external coordination the pipeline can't do or check.
- **Unblocking action:** Provision/authorize an Azure subscription + grant deploy creds (Container Apps for the API, Static Web Apps for `web/`, Blob for caches, Key Vault for secrets), or confirm it's acceptable to author Bicep/Terraform + a deploy workflow without an end-to-end deploy verification. Entry #1 (gallery/front door) raises the value of unblocking this — a public URL is what the gallery is *for*.

## Recently shipped (auto-tracked)

- 2026-06-17 #52: widen the visual API surface — **`describe_scene`** primitive (Azure Image Analysis CAPTION) end-to-end through all three layers (op surface **8→9**: new `captions` binding kind + cache slice + `_answer_caption` branch + UI step-kind + `scene` eval shape). Winner of `impl-par-20260616-133102` (5-way parallel-impl; #52 merged over the recommended #56, near-tied). Post-merge live re-capture → **71-case** baseline at prompt_version `1a12f4d66e08159b`, GATE PASS (T4 0.976, oos 1.00, adv 8/8); floors held at 0.95, `min_fixtures` 66→71. Completes prior top-move #1. One deliberately-held regression carried to Manual additions.
- 2026-06-16 #51: few-shot worked-examples codegen iteration — one worked example per failing shape, rulebook trimmed 6.8k→4.2k chars (~36%); closed all 6 remaining misses (3 scorer-number paraphrases, single-goal-presence-far-3, and the adversarial count-routing leaks). Lifted T4 0.89→1.00 and out-of-scope 0.93→1.00 on the 66-case bank; floors ratcheted to 0.95 at `prompt_version 474f8438`.
- 2026-06-12 #50: noise-precedence + bank v2 — noise-precedence rule + midpoint exclusivity + hint-number-filter ban; grew the bank +6 → 66 cases; baseline T4 0.89 / oos 0.93.
- 2026-06-12 #49: analyzed-window-law regression repair — fixed the in-scope grounding regression introduced by #45 (T4 0.42→0.89, count 6/6); dropped the whole-clip count fallback, added the digit-identify shape.
- 2026-06-11 #45 (winner of run `impl-par-20260610-105516`, merged + close-out complete): close-the-measured-prompt-gaps — count-shape single-frame rule + capability-scope honest-grounding rule in `SYSTEM_PROMPT`, phase-aware committed-fixture tests, bank `prompt_version` provenance guard.
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

### Follow-ups from #52 (describe_scene — top-move #1, now merged @ prompt_version `1a12f4d66e08159b`, 71-case baseline)

- **Presence "at least one" → count mis-route (T4 regression).** `bernabeu-presence-far-2` ("Does this frame contain at least one footballer?") now compiles to `sample_frames → detect → count → answer` and grounds to `'5'` instead of `'Yes'` — in-scope T4 slipped 1.00 → 0.976. Root cause: #52 freed prompt budget by trimming the count/scope routing prose in `_RULES`. The T4 floor was deliberately **held at 0.95** (not de-ratcheted) so this stays visible in the gate. Fix: re-add a tight presence-vs-count routing clause to `codegen.py` `_RULES`, but the base prompt is at ~6694/6700 so it needs budget headroom first; the fix requires another paid live re-capture (~$0.40) to re-baseline. Small (~1 prompt edit + recapture), not a 5-agent fan-out. Lesson: trimming routing prose to free prompt budget has a measurable T4 cost.
- **Cosmetic: non-load-bearing test assertion.** `api/tests/test_generalized_answer.py:~1003` has `assert f["answer"] and "pitch" in f["answer"].lower() or f["answer"]` — operator precedence makes the whole expression reduce to `f["answer"]` truthiness, so the `"pitch"` keyword check is dead. Tighten to `assert f["answer"] and "pitch" in f["answer"].lower()`. One-line cleanup, came in with #52.
