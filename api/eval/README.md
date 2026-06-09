# Codegen evaluation harness

Measure the codegen prompt (ViperGPT Layer 1) with **real model output, not vibes**. The pytest
suite mocks the model, so it proves the *plumbing* (validate → execute → ground) but says nothing
about whether gpt-4o actually emits valid, grounded, correct programs for the questions a user
types. This harness is the calibration layer: a versioned question bank, a live capture into
recorded fixtures, a tiered scoring rubric, a diffable report, and a CI threshold gate.

## Layout

| File | Role |
|------|------|
| `question_bank.json` | ~60 typed cases (2 clips × 4 shapes × phrasing variants + out-of-scope/adversarial/degenerate), each with a declared ground-truth `expect`. |
| `question_bank.schema.json` | JSON Schema for a bank case (drift-guarded against `server.MAX_QUERY_TEXT_LEN` by a test). |
| `bank.py` | Bank loader/filter, the `prompt_version` stamp, and `derive_ground_truth` (count/presence answers are *derived* from pinned programs, never hand-typed). |
| `fixtures.py` | Fixture IO. A fixture records the **untrusted raw model output** (`raw_program`), the analogue of `examples/*_cache.json`. One file per case, named by `case_id`. |
| `fixtures/*.json` | Committed fixtures. Ship as a **generated seed set** (`seed_fixtures.py`); the human capture overwrites them with real gpt-4o output. |
| `scoring.py` | The tiered rubric (T0..T4). Reuses the **exact** server gate: `validation_errors` then `Interpreter.run`. |
| `report.py` | `report.{md,json}` — per-tier funnel + per-shape/clip/bucket tables + failure rows. |
| `thresholds.json` | Committed CI floors (recalibrated after the first real capture). |
| `run_eval.py` | The CLI: `capture` (live, billed, creds-gated) + `replay` (free, CI-safe). |
| `seed_fixtures.py` | Generates the creds-free seed fixtures from the pinned `examples/*_program.json` templates. |

## The tiered rubric (D4)

A strict monotone ladder — a case lands at the highest tier it reaches:

- **T0 parsed** — capture recorded a `raw_program` (not an `error`).
- **T1 schema-valid** — no structural-class validation errors.
- **T2 semantically-valid** — `validation_errors(...) == []`: the **full server gate** (the trust boundary).
- **T3 executes-grounded** — `Interpreter.run(...)["findings"]["grounded"] is True`.
- **T4 answer-correct** — the grounded answer matches `expect.answer` under the shape's comparator.

**Out-of-scope / adversarial cases invert the ladder.** A question a correct program *cannot*
ground (`expect.outcome == "ungrounded"`) **PASSES** when it grounds out honestly (`grounded:false`)
or is cleanly **REJECTED**, and **FAILS** only when a program executes to a fabricated grounded
answer (the dangerous overclaim). The harness rewards **calibrated refusal**, which is the
product's actual contract.

## Replay (free, CI-safe, no creds)

```bash
cd api
python eval/run_eval.py replay                       # scores committed fixtures, writes the report, applies the gate
python eval/run_eval.py replay --shape count          # filter to a shape
python eval/run_eval.py replay --no-gate              # report only, no threshold enforcement
```

Replay re-pushes every fixture's **untrusted** `raw_program` through `validation_errors` **before**
`Interpreter.run`, so a fixture can never smuggle an unvalidated program into a pass. It makes
**zero** codegen/Azure calls and imports no Azure client.

## Capture (live, billed, creds-gated) — the human runbook

> Prereq: `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_KEY` (+ `AZURE_OPENAI_DEPLOYMENT`, default
> `gpt-4o`) in `api/.env`. This is the **only** billed step. Without creds, `capture` exits 2 with a
> provisioning hint. One full run is **< $0.50** (~60 calls).

```bash
cd api
source .venv/bin/activate

# 0) Smoke creds + the harness on ONE case first (sub-cent):
python scripts/codegen_probe.py --question "How many players are visible?"   # exit 0 = creds good
python eval/run_eval.py capture --limit 1
python eval/run_eval.py replay

# 1) Dry preview of cost (no calls):
python eval/run_eval.py capture --dry-run

# 2) Full capture (BILLS tokens; overwrites the seed fixtures; resume-safe):
python eval/run_eval.py capture

# 3) Score the real fixtures + regenerate the committed report:
python eval/run_eval.py replay --report eval/report.md --json eval/report.json

# 4) Review eval/report.md, then commit the baseline:
git add eval/fixtures eval/report.md eval/report.json
git commit -m "eval: capture gpt-4o codegen fixtures + baseline report"

# 5) (optional) Ratchet eval/thresholds.json to ~5pp under the measured rates, then commit.
```

`capture` resumes by default (a case with a fresh fixture is skipped); `--force` re-captures
(still capped by `MAX_CAPTURE_CALLS`). `--only <case_id|shape|clip>` narrows the run.

## Prompt-version invalidation

`prompt_version = sha256(build_system_message(clip))[:16]`. Any edit to `SYSTEM_PROMPT` or a clip
`hint` changes the hash, so every affected fixture goes **STALE**. Replay marks stale fixtures and
the report shouts "re-capture needed". A prompt edit therefore *forces* a re-capture — the correct
coupling.

## CI gate posture (D6)

A **calibrated threshold gate**, replay-only, in the existing `python` job:

- Fails if any in-scope tier rate falls below its floor in `thresholds.json`, if fixtures are
  missing (`min_fixtures`), or if partial staleness exceeds `max_stale`.
- **Uniform** staleness (an honest prompt edit stales *all* fixtures) downgrades to **advisory**
  (exit 0) — an implementer editing the prompt in a creds-less worktree must not red-CI before they
  can re-capture.
- **Partial** staleness is suspicious (a doctored `prompt_version` could silence one regressing
  case), so it **FAILS** beyond `max_stale` and never excludes a case from the denominator.

## Determinism disclaimer

`temperature=0` does **not** guarantee bit-identical gpt-4o output. **Fixtures are a frozen
snapshot; replay scores the snapshot, not a fresh call.** Replay itself is fully deterministic
(parse/validate/execute over committed bytes + caches), so CI never flakes. The harness never
asserts golden-program equality — it scores coarse tier outcomes, and a re-capture shift surfaces
as a `report.json` diff, never silently.

## Escape discipline

The bank deliberately carries hostile strings (injection payloads, RTL/unicode noise). The report
escapes control/bidi characters repr-style so they can't garble CI logs and diffs.

## A code-review caveat

Fixtures are committed, hand-editable files whose `prompt_version` is self-reported. The gate
catches deletion (`min_fixtures`) and partial staleness, but a doctored-but-valid `raw_program`
under the current hash remains possible — as for any committed test fixture. Treat a fixture diff as
code under review.
