# What's Next

_Generated 2026-06-10 by `/whats-next` (post `impl-par-20260610-105516`). Each entry is sized for the 5-agent parallel-impl pipeline: parallel planning → plan aggregation → `/refine-plan` + `/finalize-plan` → 5× `claude-opus-4-8` worktree implementations → PR selection._

## Top moves (highest-leverage first)

### 1. Widen the visual API surface — new primitives through all three layers

- **Why now:** The op surface has been frozen at 8 ops (`api/interpreter/primitives.py` OPS) since the vertical slice, and `CLAUDE.md` names the growth pattern explicitly: new primitive in layer 2 + model in layer 3, exposed in layer 1's prompt. Everything needed to do this safely has now shipped — validator + limits + adversarial suite (#33), multi-clip catalog (#21), coverage gates (#36), and **a measured eval baseline (#40/#43) so each new primitive's effect on validity/groundedness is quantified, not vibes**. Strong candidates: a scene/caption primitive (Azure Image Analysis captioning — a *different* service from the quota-blocked OpenAI VLM), a spatial-relation filter (local geometry over detection boxes), and/or a presence-over-time op. The prior sequencing concern is resolved: the count fix in PR #45 is prompt-only (all 5 parallel implementations independently converged on prompt-rule, no op-semantics change), so the op surface is stable — just merge PR #45 first (see On hold §A) so the SYSTEM_PROMPT base you extend is the fixed one.
- **Scope:** ~2,000–4,000 LOC across all three layers: `api/schema/dsl.schema.json` + `scripts/validate_program.py` + `limits.py` (new op + arg domains + caps), `interpreter/primitives.py` + findings plumbing, the codegen op spec in `SYSTEM_PROMPT`, precompute caching for the new op's outputs, eval-bank cases for the new shapes, web rendering of the new step kind (EvidencePanel + ProgramPanel), and tests at every layer including adversarial cases.
- **Pipeline-ready:** ✅ — the widest design room on the list: which primitives, arg-domain design, local-vs-Azure-backed split, cache shape, diagnostic-empty semantics, UI evidence rendering. **Notes:** regenerating *real* caches needs Azure Vision creds + source video — agents author with the established FakeAzureVision pattern; live precompute runs separately. A `SYSTEM_PROMPT` edit stales all committed eval fixtures uniformly → advisory gate keeps CI green (the phase-aware test machinery for this lands with PR #45); pair the post-merge re-capture with this entry's new eval cases.
- **Plan artifact:** needs writing — start with a draft, then `/refine-plan`.
- **Status:** [ ] pending

### 2. Runs gallery / demo front door — make the system's range discoverable

- **Why now:** The ingredients shipped and nothing consumes them yet: a content-addressed run store with permalinks (#25, `api/run_cache.py` + `?run=<id>`), a multi-clip/multi-query catalog (#21, `/api/catalog`), and a committed eval report that *names* what the system answers well. But the portfolio audience still lands on a single analyst screen with one query loaded — there is no way to *discover* what the system can do. A gallery/landing surface — clips, canned queries by shape, notable cached free-text runs, each linking into the analyst view via existing permalink routes — turns the demo from "one impressive run" into "browse the system's range."
- **Scope:** ~1,000–2,500 LOC: a small API listing surface (extend `/api/catalog` or add a run-listing endpoint over `RunStore`), a gallery/index page in `web/app/` consistent with DESIGN.md tokens (Fraunces/Hanken/Plex Mono, paper/ink/teal, sharp radius), clip stills as thumbnails, navigation into the existing `?query=`/`?run=` load paths + tests.
- **Pipeline-ready:** ✅ — design room in the IA (landing page vs in-app gallery vs sidebar), what each card surfaces, static-export vs API-driven listing, curation posture. Must stay inside the locked design system (DESIGN.md tokens are non-negotiable).
- **Plan artifact:** needs writing — start with a draft, then `/refine-plan`.
- **Status:** [ ] pending

### 3. Accessibility + keyboard presentation pass — the agreed design debt, deps long met

- **Why now:** `TODOS.md` carries this as explicitly-deferred (not skipped) design debt from the 2026-05-31 design review, blocked on T-D1/T-D2 (layout + run engine) — both long since shipped. The pass: keyboard nav for run controls + the step scrubber, ARIA landmarks for the 3 zones, screen-reader step narration ("step 4 of 7, read_text, found 10"), focus order, contrast-audit verification. A portfolio piece failing basic a11y reads badly to exactly the senior-engineer audience this repo targets. Partial mitigations exist (prefers-reduced-motion D-DR8, contrast-aware tokens D-DR9) — this is the finishing pass.
- **Scope:** ~1,000–2,000 LOC, all web-side: `StepTracker` keyboard/scrubber semantics, landmarks + labels across `page.tsx`/panels, a live-region narration strategy, focus management on run/replay, Testing Library ARIA assertions extending the vitest suites.
- **Pipeline-ready:** ✅ (moderate) — design room in narration granularity vs announcement fatigue, the focus model, scrubber keyboard semantics, how much canvas evidence to mirror as offscreen text. Less structural variance than entry 1.
- **Plan artifact:** needs writing — start with a draft, then `/refine-plan`.
- **Status:** [ ] pending

## On hold / needs human first

### A. Merge handoff + post-merge eval runbook for `impl-par-20260610-105516` (the count/refusal prompt fix)

- **Blocker:** Merge authority (the agent classifier blocks self-merge to main) + Azure OpenAI creds for the live re-capture (~$0.40). The 5-agent run is otherwise **complete**: plan drafted → refined (5 reviewers) → codex consult → finalized → 5 worktree implementations → PRs #44–#48, all CI-green → comparative review picked **PR #45** → codex review of #45 passed clean → cherry-picks from #44 (no-hint counting fallback + dry-run `--force` lockstep) applied and pushed to #45 (`193c992`), gate re-verified green.
- **Unblocking action (the close-out sequence):**
  1. `gh pr merge 45 --merge`
  2. Close the losers: `for pr in 44 46 47 48; do gh pr close $pr --delete-branch --comment "superseded by #45 (merged); counting-rule fallback + dry-run lockstep cherry-picked from #44"; done`
  3. Run the post-merge runbook (now documented in `api/eval/README.md` steps 2–6 on the PR): `python eval/run_eval.py capture --force` (~$0.40) → `replay --report eval/report.md --json eval/report.json` → review the lifted T4 / out-of-scope-honest / adversarial-safe rates → ratchet `eval/thresholds.json` floors ~5pp under measured → commit.
  4. Cut the **bank-growth follow-up PR** in the same sitting (deferred by resolved decision 1; mechanics preserved in the plan's Phase 3.5 and the README runbook) so new cases are captured live immediately. Candidate to fold in: the scratch-dir replay provenance bug codex found (`run_eval.py:93-112` reads committed fixtures regardless of `fixtures_dir`).
  5. Clean the 5 pinned worktrees: `git worktree remove --force .claude/worktrees/agent-<id>` (IDs: a4792aca4b58bc612, a44153b2b64dcc3f6, a98778a8becb96aa1, abbacb39cee8cb0d8, aa3fbac3f994f23ab) + delete their local branches (`utils/worktree_remove.sh` / `utils/branch_delete.sh` pattern).

### B. Live gpt-4o VLM `read_text` backend

- **Blocker:** Azure OpenAI TPM quota — `api/vision/vlm_read.py` is ready but the deployment is rate-limited for vision-token-heavy calls, so it can't be exercised. (The #43 capture proved *codegen* quota is fine at 60 sequential text calls; VLM image calls are much heavier per request.) The merged precompute OCR ladder is **pin → gpt-4o VLM → skip**, so the VLM is *the* live OCR backend once unblocked. Also small (~300 LOC) — one-agent work, not a 5-agent fan-out.
- **Unblocking action:** Raise the TPM quota on the gpt-4o deployment (`mjlaielli-6579-resource`, AIServices/eastus2). Then live precompute/codegen can read novel jersey numbers without hand-pins; until then non-pinned jerseys honestly skip to diagnostic-empty.

### C. Azure production deployment + infrastructure-as-code

- **Blocker:** Needs the user's Azure subscription, secret provisioning, and live deploy credentials to verify — external coordination the pipeline can't do or check.
- **Unblocking action:** Provision/authorize an Azure subscription + grant deploy creds (Container Apps for the API, Static Web Apps for `web/`, Blob for caches, Key Vault for secrets), or confirm it's acceptable to author Bicep/Terraform + a deploy workflow without an end-to-end deploy verification. Entry #2 (gallery/front door) raises the value of unblocking this — a public URL is what the gallery is *for*.

## Recently shipped (auto-tracked)

- 2026-06-10 PRs #44–#48 (unmerged, run `impl-par-20260610-105516`): close the measured prompt gaps — count-shape single-frame rule + capability-scope honest-grounding rule in `SYSTEM_PROMPT`, phase-aware committed-fixture tests, bank `prompt_version` provenance guard, README runbook corrections. Winner **#45** awaiting merge handoff (On hold §A); #44/46/47/48 to be closed as superseded.
- 2026-06-10 #43: live-codegen 400 fix + first real gpt-4o eval baseline — `MAX_PROGRAM_TOKENS` clamped to gpt-4o's 16,384 completion cap (every live codegen call had 400'd since #33, invisible creds-free; caught by the harness's first live call), 60 captured fixtures (~$0.38), measured baseline T2 1.00 / T4 0.78 / refusal 0.75, floors recalibrated ~5pp under measured, phase-aware seed tests. Completed the eval entry's live-capture runbook; its findings seeded the prompt-gaps run above.
- 2026-06-10 #40: codegen evaluation harness — `api/eval/` package: 60-case question bank (2 clips × 4 shapes × paraphrase distance + out-of-scope/injection/degenerate), capture/replay CLI, monotone 4-tier rubric reusing the production validate→execute path in-process, dual diffable report, generated seed fixtures, CI threshold gate with anti-gaming stale handling. 5-agent run impl-par-20260609-163634, winner #40, chmod polish cherry-picked from #38. Tests 182→219, coverage 92%.
- 2026-06-09 #36: calibrated coverage floors — `pytest --cov-fail-under=85` + vitest thresholds, calibrated ~5pp under the natural numbers.
- 2026-06-09 #35: web ESLint gate — flat-config ESLint 9 + `eslint-config-next`, CI lint step.
- 2026-06-09 #33: trust-boundary hardening — `api/limits.py`, `MAX_PROGRAM_BYTES` pre-parse cap, validator step/frame/arg-domain caps, schema static limits, two-check runtime guard, 27-test adversarial suite. Tests 155→183.
- 2026-06-09 #25: free-text run cache + shareable permalinks — content-addressed store, cache hit skips the billed codegen call, `?run=<id>` permalinks, copy-run-link UI.
- 2026-06-09 #21: multi-clip / multi-query catalog — bernabeu-counter clip + 4 query-shape programs, registry generalization, resolver.
- 2026-06-08 #20: tightened 5 assertions flagged by the codex review of #16.
- 2026-06-08 #16: CI gate + widen the test net — first workflows matrix (Python + web), Vitest toolchain. Python 41→105, web 0→35.
- 2026-06-07 #14: free-text trust-boundary fixups — canned-path bounds gap + absent-subject temporal grounding.
- 2026-06-07 #11: free-text query input — typed question → live codegen → validate → execute, honest `_ungrounded` everywhere. The full ViperGPT thesis live end to end.
- 2026-06-06 #7: automated clip→cache precompute pipeline + first `api/` pytest suite.
- 2026-06-06 #2: live Azure OpenAI codegen (question → DSL) with D-DR6 fallback.
- 2026-06-04 #1: vertical slice — ViperGPT-style video analyst (backend + web UI).

## Manual additions

<!-- Anything the user adds below this line is preserved across `/whats-next` runs. -->
