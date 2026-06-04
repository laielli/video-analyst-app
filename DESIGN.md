# DESIGN.md

Design system for the Glass-Box Video Analyst. **Source of truth = the locked design
decisions D-DR1..D-DR9** in the approved design doc
(`~/.gstack/projects/laielli-video-analyst-app/michaellaielli-main-design-20260531-142704.md`),
which passed `/plan-design-review` (3/10 → 9/10) and `/plan-eng-review` (CLEARED). This file
mirrors those decisions for in-repo reference. If this file and the design doc ever disagree,
the design doc wins.

> Note: `~/.gstack/projects/.../designs/analyst-screen-20260531/finalized.html` is a
> **non-authoritative motion reference** (it demonstrates the live choreography). Its fonts and
> colors predate this reconciliation and do **not** match the tokens below. Build from these
> tokens, not from that file's CSS.

## Design thesis

**Evidence-forward, glass-box, and live.** The product answers a canned question about a curated
video by having an LLM (Azure OpenAI) compile it into a visible JSON-DSL program, then executing
that program step by step against Azure AI Vision primitives per sampled frame. The UI's job is
to make the reasoning **inspectable**: every answer is traceable to a frame, a producing step, and
a confidence score. Two labeled zones orient the user instantly — `PROGRAM (AI REASONING)` and
`EVIDENCE (INSPECTABLE)` — and the interface must *feel live* (step-by-step execution), never a
static report. The generated program is the conceptual hero even though evidence is visually larger.

Principles: label zones explicitly · show the work, not just the verdict · the answer is earned at
the end of the pipeline · empty/no-result steps render diagnostically (what was searched + why),
never a cold "No evidence."

## Type (D-DR9) — NO Inter / Roboto / Arial / system-ui anywhere (the slop tell)

| Role | Font | Notes |
|------|------|-------|
| Display / headings | **Fraunces** (variable display serif; optical sizing + soft/wonk axes) | Query headline + section labels. Sans alternative if preferred: Clash Display (Fontshare). |
| UI / body / prose | **Hanken Grotesk** (humanist sans) | All UI text and prose. |
| Code / data / overlay labels | **IBM Plex Mono** | The PROGRAM panel, readout values, overlay tags. |

Density: dense-but-readable. Body **≥16px**, code **≥14px mono**. Nothing below these.

## Color (D-DR9) — CSS variables only, no hardcoded hex in components

| Token | Value | Use |
|-------|-------|-----|
| `--surface` | `#FAFAF7` (paper) | page / panel background |
| `--ink` | `#16181D` | text, headings |
| `--accent` | `#0F766E` (deep teal) | the single accent |

**Functional-only status colors** (never decorative; also drive box/step states; each verified
≥4.5:1 on paper at build — design pass 6):

| Token | Value | Meaning |
|-------|-------|---------|
| `--status-running` | `#B45309` (amber) | ACTIVE step |
| `--status-found` | `#15803D` (green) | COMPLETED / evidence found / resolved findings |
| `--status-empty` | `#6B7280` (grey) | PENDING / diagnostic empty (no match) |

## Shape & chrome (D-DR9)

- **Sharp editorial radius: 2–4px.** Not soft/rounded cards.
- **1px hairline rules**, not heavy borders or drop-shadow cards.
- Generous whitespace, minimal chrome.
- Cards only where the card IS the interaction (none in v1).

## Layout & information architecture (D-DR1..D-DR3)

Core screen = the analyst run view, three zones:

```
+--------------------------------------------------------------+
| ZONE 1  Query bar: canned query as editorial headline        |
|         + clip selector + Run/Replay                         |
+----------------------------+---------------------------------+
| ZONE 2 LEFT ~45%           | ZONE 2 RIGHT ~55%               |
| PROGRAM (AI REASONING)     | EVIDENCE (INSPECTABLE)         |
| generated DSL, Python-like | analyzed frame still + boxes   |
| (Shiki/Prism, locked D5);  | + OCR readout w/ confidence;   |
| active line glows per step | each element labeled by the    |
|                            | step that produced it (D-DR2)  |
+----------------------------+---------------------------------+
| ZONE 3  Findings: evidence-grounded answer                   |
| + slim 7-step tracker (scrubber, D-DR3), no vertical rail    |
+--------------------------------------------------------------+
```

- **Evidence-primary 45/55** (D-DR1): EVIDENCE is wider, but every evidence element is labeled
  with its producing step (`read_text → "10"`), so the panel reads as the program's output
  (D-DR2) — the generated program stays the conceptual hero.
- **Bottom 7-step tracker doubles as a scrubber** (D-DR3): `sample_frames → detect → crop →
  read_text → filter → temporal_order → answer`, shows position ("4 of 7"), click-to-replay,
  paired with inline active-line glow. No separate vertical step rail (preserves PROGRAM width).
- **Responsive:** desktop-first **≥1024px**; below that, a "best viewed on desktop" notice, not a
  broken reflow (deferred per TODOS — real sub-desktop layout has no demo payoff).

## Motion & choreography (D-DR4, D-DR7, D-DR8)

- **Streamed authoring (D-DR4):** the program types in line-by-line; identical animation for live
  codegen and cached replay.
- **Hybrid pacing (D-DR7):** auto-advances ~1.2s/step so a public URL self-plays the wow; presenter
  can pause and drag the tracker to dwell on any step. Same model drives the 60s recorded fallback.
- **Sequential cause→effect per step (D-DR8)** — the core legible-orchestration moment:
  - t0: active code line glows (PROGRAM)
  - t+250ms: the evidence that line produces draws onto the frame (box)
  - t+500ms: secondary evidence resolves (OCR "10" pops, label appears)
  - t+1.0s: brief dwell, then tracker advances
- Overlays animate over **pre-extracted stills using normalized 0–1 coords** (learning
  `overlay-normalized-stills`), so boxes are exact and never chase a seeking `<video>`.
- **Motion-optional from day one:** honor `prefers-reduced-motion` by cross-fading instead of
  drawing/sliding (a11y; full pass deferred to TODOS).

## Interaction states (D-DR5, D-DR6)

- **Diagnostic empty (D-DR5):** an empty step dims/outlines the searched region on the frame + a
  plain reason ("read_text found no legible digits") + marks the step "no match". Never a bare
  "No evidence" string.
- **Codegen-failure path (D-DR6):** invalid / non-whitelisted program → silently use the pinned
  known-good program for that query, with a small "replayed" tag. Never a user-facing crash.

## Components

- **Query bar** — canned query as Fraunces editorial headline + clip selector + Run/Replay.
- **PROGRAM panel** — Shiki/Prism Python-like render of the JSON DSL; per-step status using the
  functional status colors (amber active / green done / grey pending); active line glows.
- **EVIDENCE panel** — frame still + normalized-coord overlay boxes + OCR readout; producer-step
  readout: Producer step · Input (from prior step) · Output · Confidence. Each element labeled by
  its step (D-DR2).
- **Findings** — evidence-grounded answer (green `--status-found`); decisive frame pinned, proving
  step highlighted; "partial evidence" note when some steps were empty.
- **7-step tracker** — bottom scrubber, click-to-replay, current marker tracks run position.

## Tech / stack (from the cleared eng review, D1–D16)

- **Web:** Next.js (static export) on Azure Static Web Apps. Code viewer = **Shiki/Prism** (D5,
  not Monaco). Canvas/DOM overlay over normalized-coord stills.
- **API / execution:** FastAPI on Azure Container Apps (scales to zero). Interprets the JSON DSL —
  **no code sandbox in v1** (the program is interpreted data, D3). SSE streams steps; replay-first
  on the hero path (D1).
- **Run-doc contract (D2/D3):** the DSL JSON Schema doubles as the run-document the UI consumes per
  step (`dsl-json-schema-unification`). One schema serves OpenAI structured output, the interpreter
  validator, and the UI view model. No extra view model.
- **Vision primitives:** `detect` + `read_text` → Azure AI Vision Image Analysis 4.0;
  `filter`/`count`/`temporal_order` → local interpreter. Outputs precomputed + cached in Blob.
- **Codegen:** Azure OpenAI gpt-4o, schema-validated, pinned known-good fallback per canned query.

## Reference build

Build target = the locked design doc + approved mockup
`~/.gstack/projects/laielli-video-analyst-app/designs/analyst-screen-20260531/variant-C-locked.png`
(visual layout reference) + `finalized.html` (motion reference only; tokens non-authoritative).
