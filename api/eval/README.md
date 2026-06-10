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
  fixtures/*.json             # committed fixtures: real gpt-4o captures (model: gpt-4o, ISO captured_at)
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

# 0) Smoke creds only (sub-cent) — do NOT `capture --limit 1` then `replay` after a prompt edit:
#    capturing 1 case makes it fresh while the other 59 stay stale -> partial staleness -> gate
#    FAILs (and the phase-aware committed-fixtures test fails too). Smoke creds with the probe,
#    then do the FULL re-capture in step 2 before any replay.
python scripts/codegen_probe.py --question "How many players are visible?"   # exit 0 = creds good; 2 = fix .env

# 1) Dry preview of cost (no calls):
python eval/run_eval.py capture --dry-run

# 2) Full capture (BILLS real tokens; re-captures ALL cases under the current prompt; resume-safe):
#    `--force` re-captures even fixtures that are fresh at the current prompt_version (e.g.
#    seed-stamped ones, or a same-prompt re-run); without it those are silently resume-skipped.
#    After a prompt edit the committed fixtures are STALE, so capture would re-call them even
#    without --force — but --force is the safe default that also covers the seed-stamped/same-prompt
#    cases (verified against run_eval.py:262-270).
python eval/run_eval.py capture --force   # ~60 calls, budget-capped; a re-run skips fresh cases

# 3) Score the real fixtures + regenerate the committed report:
python eval/run_eval.py replay --report eval/report.md --json eval/report.json

# 4) Review eval/report.md, then commit the baseline:
git add eval/fixtures eval/report.md eval/report.json
git commit -m "eval: capture gpt-4o codegen fixtures + baseline report"

# 5) Ratchet eval/thresholds.json (creds-NOT-required, but pairs with this re-capture):
#    - read the new measured T4_answer_correct / out_of_scope_honest rates from eval/report.md,
#    - set each thresholds.json floor ~5pp UNDER the measured rate (the calibration convention in
#      the thresholds.json _comment; ~2.8pp == 1 in-scope case, so leave temperature=0 jitter room),
#    - commit. This is the ONLY step the implementer agents do NOT do (they have no creds to
#      re-capture, so they cannot measure the lifted rates).

# 6) Bank-growth follow-up (pair with this same sitting so new cases are captured live now,
#    no MISSING limbo): grow question_bank.json for newly-exposed failure modes — count
#    paraphrases via bank.derive_ground_truth (never hand-typed), out-of-scope honesty cases with
#    expect.outcome "ungrounded" / answer_match "any_grounded"; bump bank_version 1->2; any new
#    (clip, shape) cell needs a pinned program in examples/ + a seed_fixtures._PINNED entry +
#    LIVE_CELLS + anchor/paraphrase guards (>=1 anchor AND >=3 paraphrases per live cell, so a new
#    cell costs >=4 cases); keep the degenerate-all-caps -> count routing special-case intact.
```

`--only <case_id|shape|clip>` narrows the run; `--force` re-captures even fixtures that are fresh
at the current prompt_version (resume is the default — fresh fixtures are silently skipped without
it); `--limit N` caps live calls this invocation. A per-invocation `MAX_CAPTURE_CALLS`
ceiling (default 64) bounds spend; `--force` does **not** bypass it. A full capture costs roughly
**$0.35–$0.50** (verify current gpt-4o pricing first).

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

The committed `fixtures/*.json` are now **real gpt-4o captures** (since the first live capture);
the **seed** set (`seed_fixtures.py`) is the creds-free stand-in that proves the harness *works*
before any capture. Seeds are derived from the pinned `examples/*_program.json` templates (the
final `answer` step's `question` is retargeted per case) and carry the **current** `prompt_version`
so the gate exercises the real ladder.

Seeds are regenerated ONLY into scratch dirs — `seed_fixtures.generate_all(fixtures_dir=tmp)`,
which is exactly what `test_replay_over_regenerated_seeds_reaches_targets` and
`test_seed_fixtures_regenerate_match` do. **Never run bare `generate_all()`**: its default
`fixtures_dir` is the committed dir, so it would clobber the real captures with seeds (an
anti-gaming violation). Capture measures the *prompt*; seeds prove the *machinery*.

## Escape discipline / fixtures are review-trusted

The bank deliberately contains hostile strings (injection payloads, RTL/unicode noise); the report
renders `question`/answer text with control/bidi chars escaped (repr-style) so they can't garble CI
logs/diffs. Fixtures are committed, hand-editable files whose `prompt_version` is self-reported — a
doctored-but-valid `raw_program` under a current hash is a code-review concern, the same as any
committed test fixture.
