# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A web-based video analysis app inspired by ViperGPT (https://arxiv.org/pdf/2303.08128).
The core idea: an LLM generates a **visual program** (code) from a natural-language query
about a video; an API/executor layer runs that program, calling individual vision models as
modular subroutines to perform the actual analysis. Azure backs the runtime — both the live
calls to the code-generating LLM and the calls to the various vision models.

### Architecture (the ViperGPT pattern)

When building out features, keep these three layers distinct — they are the heart of the design:

1. **Program generation** — an LLM turns a user query into a visual program. The program is
   code that composes vision primitives; it is *generated*, not hand-written per query.
2. **Execution / API layer** — runs the generated program. Exposes a fixed set of callable
   vision primitives (the "API surface") that programs are allowed to invoke. This layer is
   the trust boundary: generated code runs here, so isolate and validate it.
3. **Vision model calls** — the primitives themselves, each backed by a model (object
   detection, captioning, VQA, etc.), called modularly via Azure.

A new capability is almost always added as a new primitive in layer 2 + its model in layer 3,
then exposed to the LLM in layer 1's prompt/API spec — rather than special-casing it in the LLM.

## Development status

This is a **greenfield** repository. As of this writing only `README.md`, `CLAUDE.md`, and
`.gitignore` are checked in — there is no source code, build system, or test suite yet. Do not
assume build/lint/test commands exist; add them (and update this file) as the codebase grows.

The README mandate is explicit: **start small and build incrementally**, prioritizing a solid
understanding and good design choices at each stage. Favor a minimal vertical slice over broad
scaffolding.

`.gitignore` anticipates the intended stack — treat these as the expected toolchains:
- **Node** (`node_modules/`, `dist/`, `build/`) — likely the web frontend / API.
- **Python** (`.venv/`, `.pytest_cache/`, `.mypy_cache/`) — likely the model/execution layer.
- **Azure** (`.azure/`, `local.settings.json`) — Azure Functions for the backend runtime.

Secrets belong in `.env` / `local.settings.json` (both gitignored); commit a `.env.example` instead.

---

# gstack

Use the `/browse` skill from gstack for all web browsing. Never use `mcp__claude-in-chrome__*` tools.

Available gstack skills:

- `/office-hours`
- `/plan-ceo-review`
- `/plan-eng-review`
- `/plan-design-review`
- `/design-consultation`
- `/design-shotgun`
- `/design-html`
- `/review`
- `/ship`
- `/land-and-deploy`
- `/canary`
- `/benchmark`
- `/browse`
- `/connect-chrome`
- `/qa`
- `/qa-only`
- `/design-review`
- `/setup-browser-cookies`
- `/setup-deploy`
- `/setup-gbrain`
- `/retro`
- `/investigate`
- `/document-release`
- `/document-generate`
- `/codex`
- `/cso`
- `/autoplan`
- `/plan-devex-review`
- `/devex-review`
- `/careful`
- `/freeze`
- `/guard`
- `/unfreeze`
- `/gstack-upgrade`
- `/learn`

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Design system/plan review → invoke /design-consultation or /plan-design-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke /investigate
- QA/testing site behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Visual polish → invoke /design-review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore
