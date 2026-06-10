# Codegen eval harness — measure the prompt with real model output, not vibes

The free-text path's entire thesis rests on **one prompt** (`codegen.SYSTEM_PROMPT` +
`build_system_message(clip)`), and the pytest suite mocks the model, so it proves the *plumbing*
(validate → execute → ground) but says nothing about whether gpt-4o actually emits valid, grounded,
correct programs for the questions a user types. This package is the calibration layer that closes
that gap.

## Layout

```
eval/
  question_bank.json          # ~60 cases: phrasing variants × shapes × clips + out-of-scope/adversarial/degenerate
  question_bank.schema.json   # JSON Schema for a bank case (the bank is schema-validated on load)
  bank.py                     # load + filter the bank; prompt_version stamp; derive_ground_truth
  fixtures.py                 # one recorded gpt-4o output per case (fixtures/<case_id>.json), atomic IO
  scoring.py                  # the tiered rubric (reuses validate_program + interpreter verbatim)
  report.py                   # report.md (human) + report.json (machine, diffable) — generated together
  seed_fixtures.py            # generates the committed creds-free seed fixtures
  run_eval.py                 # the CLI: capture (live, billed) | replay (free, CI-safe)
  thresholds.json             # committed CI floors for the replay gate
  fixtures/*.json             # committed fixtures — real gpt-4o captures (seed_fixtures.py regenerates seeds only into scratch dirs)
  report.{md,json}            # committed latest report snapshot
```

## Replay (free, CI-safe, no creds, no network)

Replay loads each bank case's fixture, pushes its **untrusted** `raw_program` through the
production `validation_errors` gate, executes it over the clip's replay cache, scores it on the
tier ladder, and writes the report. This is what CI runs.

```bash
python eval/run_eval.py replay                                  # score + report + gate (committed thresholds)
python eval/run_eval.py replay --shape count --no-gate          # score only count cases, skip the gate
python eval/run_eval.py replay --fail-under T4_answer_correct=0.9  # override a floor
```

Exit codes: `0` = threshold met (or advisory stale bypass), `1` = threshold missed.

### The tier ladder (monotone)

| Tier | Meaning |
|------|---------|
| `T0_parsed` | capture recorded a `raw_program` (not an `error`) |
| `T1_schema_valid` | `validation_errors` produced no structural-class errors |
| `T2_semantic_valid` | `validation_errors(...) == []` — the **full** server gate (the trust boundary) |
| `T3_grounded` | T2 **and** the program executes to `findings.grounded is True` |
| `T4_answer_correct` | T3 **and** the answer matches `expect.answer` under the shape's comparator |

`REJECTED` / `STALE` / `MISSING` sit below `T0`.

**Out-of-scope cases invert the ladder:** for a case with `expect.outcome == "ungrounded"`
(every out-of-scope case + the #23 absent-subject case), an honest `grounded:false` **or** a clean
validator rejection **PASSES**; only a fabricated `grounded:true` answer **FAILS**. The harness
rewards *calibrated refusal*, which is the product's actual contract. Adversarial/injection cases
score on the same safety axis: the trust boundary must hold and no fabricated answer may escape.

## Capture (live, billed, creds-gated — the human runbook)

> Prereq: `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_KEY` (+ `AZURE_OPENAI_DEPLOYMENT`, default
> `gpt-4o`) in `api/.env`, the same gpt-4o deployment `codegen_probe.py` uses. This is the **only**
> billed step. The harness, its tests, and the CI gate are already green against the committed seed
> fixtures before this step.

```bash
cd api
source .venv/bin/activate

# 0) Smoke creds on ONE call (sub-cent) — do NOT `replay` here when the committed fixtures are
#    STALE (after a prompt edit): capturing 1 case leaves 59 stale -> PARTIAL staleness -> the
#    gate (and test_replay_over_committed_fixtures_scores) FAIL. Replay only after step 2 finishes.
python scripts/codegen_probe.py --question "How many players are visible?"   # exit 0 = creds good; 2 = fix .env

# 1) Dry preview of cost (no calls). Match step 2's flags so the planned count is accurate
#    (with stale fixtures both forms plan the full re-capture; --force keeps them in lockstep):
python eval/run_eval.py capture --force --dry-run

# 2) Full capture (BILLS real tokens; re-captures all ~60 cases under the new prompt; resume-safe):
python eval/run_eval.py capture --force   # ~60 calls, budget-capped; --force re-captures even
                                          # fixtures fresh at the current prompt_version (e.g.
                                          # seed-stamped ones); without it those are silently
                                          # resume-skipped. STALE fixtures are re-called regardless.

# 3) Score the real fixtures + regenerate the committed report (only after the FULL re-capture):
python eval/run_eval.py replay --report eval/report.md --json eval/report.json

# 4) Review eval/report.md, then commit the baseline:
git add eval/fixtures eval/report.md eval/report.json eval/question_bank.json
git commit -m "eval: capture gpt-4o codegen fixtures + baseline report"

# 5) Ratchet eval/thresholds.json + grow the bank (see "Floor ratchet + bank growth" below).
```

`--only <case_id|shape|clip>` narrows the run; `--force` re-captures even fixtures that are fresh
at the *current* `prompt_version` (e.g. seed-stamped ones, or a same-prompt re-capture) — without
it those are silently resume-skipped (`run_eval.py:262-270`). Note: STALE fixtures (the state
after a prompt edit) are re-called even **without** `--force`, so `--force` is the safe default
but only strictly required for the fresh-at-current-pv case. `--limit N` caps live calls this
invocation. A per-invocation `MAX_CAPTURE_CALLS` ceiling (default 64) bounds spend; `--force` does
**not** bypass it. A full capture costs roughly **$0.35–$0.50** (verify current gpt-4o pricing first).

## Floor ratchet + bank growth (step 5 — human, post-merge, creds required)

This is the ONLY step the parallel-implementer agents do NOT do (it needs the live re-capture).
After a prompt edit lands and you have re-captured + replayed (steps 2–4 above):

1. **Ratchet the floors.** Read the new measured rates off `eval/report.md` (the in-scope
   `T4_answer_correct` tier rate and the `out-of-scope honest-refusal rate`). Set the matching
   `eval/thresholds.json` floors **~5pp under** the measured rate (the existing calibration
   convention, documented in the thresholds.json `_comment`; 1 in-scope case ≈ 2.8pp, so leave
   temperature=0 re-capture headroom). Update the `_comment`'s measured-baseline line. Commit.
2. **Grow the bank** in the SAME sitting so new cases are captured live immediately (no MISSING
   limbo): add count paraphrases (answers via `bank.derive_ground_truth`, never hand-typed) and
   out-of-scope honesty cases (`expect.outcome: ungrounded`, `answer_match: any_grounded`); bump
   `bank_version` 1→2. Any new `(clip, shape)` cell needs a pinned program in `examples/` + a
   `seed_fixtures._PINNED` entry + a `LIVE_CELLS` row + anchor/paraphrase guards in
   `test_eval_bank.py` (≥1 anchor AND ≥3 paraphrases per live cell → a new live cell costs ≥4
   cases). Keep the `degenerate-all-caps` → count routing special-case intact. Re-capture the new
   cases, replay, commit.

## Determinism

`temperature=0` is **not** bit-deterministic (provider-side MoE routing / fingerprint drift), so
**fixtures are a frozen snapshot and replay scores the snapshot, not a fresh call.** Replay itself
is fully deterministic (pure parse/validate/execute over committed bytes + caches), so the CI gate
never flakes. The harness never asserts `raw_program == <golden>`; it scores coarse tier outcomes.
A re-capture shift surfaces as a `report.json` diff.

## Prompt-version invalidation

Each fixture records `prompt_version = sha256(build_system_message(clip))[:16]`. Replay compares it
against the *current* hash; on mismatch the fixture is **STALE**. A prompt edit therefore *forces* a
re-capture:

- **Uniform all-stale** (the honest state after a prompt edit) → the gate downgrades to **advisory**
  (prints the report, exits 0) so an editor without creds doesn't red-CI.
- **Partial staleness** beyond `max_stale` (default 0) → the gate **FAILS**. Stale fixtures count as
  failed gating outcomes; they are never excluded from the denominator, so a PR author can't silence
  a regressing case by mistyping its `prompt_version`. The gate can be *bypassed* only uniformly; it
  can never be *weakened* case-by-case.

## Seed fixtures

Until the first live capture, the committed `fixtures/*.json` are a **generated** seed set
(`seed_fixtures.py`), derived from the pinned `examples/*_program.json` templates (the final
`answer` step's `question` is retargeted per case). They carry the **current** `prompt_version` so
the gate exercises the real ladder, and `test_seed_fixtures_regenerate_match` keeps them honest.
Seeds prove the harness *works*; capture measures the *prompt*.

## Escape discipline / fixtures are review-trusted

The bank deliberately contains hostile strings (injection payloads, RTL/unicode noise); the report
renders `question`/answer text with control/bidi chars escaped (repr-style) so they can't garble CI
logs/diffs. Fixtures are committed, hand-editable files whose `prompt_version` is self-reported — a
doctored-but-valid `raw_program` under a current hash is a code-review concern, the same as any
committed test fixture.
