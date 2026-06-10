# `api/eval/` — codegen evaluation harness

Measure the **codegen prompt** (ViperGPT Layer 1) against real gpt-4o output, not vibes. The whole
free-text thesis rests on one prompt (`codegen.SYSTEM_PROMPT` + `build_system_message(clip)` hint
injection); this is the calibration layer that catches a silent regression when the prompt, a
primitive, or the model changes.

```
eval/
  question_bank.json          # ~50 typed cases: 2 clips × 4 shapes × phrasing variants + oos/injection/degenerate
  question_bank.schema.json   # JSON Schema for a bank case (additionalProperties:false)
  bank.py                     # load + schema-validate the bank; prompt_version stamp; derive_ground_truth
  fixtures.py                 # one fixture file per case (untrusted recorded model output); atomic IO
  scoring.py                  # the tiered rubric (T0..T4), reusing the production validate + execute path
  report.py                   # report.json (machine) + report.md (human, generated from the json)
  seed_fixtures.py            # generates the committed seed fixtures (creds-free stand-in for capture)
  run_eval.py                 # the CLI: capture (live) / replay (CI-safe)
  thresholds.json             # committed CI floors (calibrated)
  fixtures/<case_id>.json     # committed seed fixtures (overwritten by a real capture)
  report.{md,json}            # latest reviewable snapshot (committed)
```

## Two modes (one scoring path)

```bash
# REPLAY — free, CI-safe, no creds, no network. Implementers + CI run this.
python eval/run_eval.py replay [--report eval/report.md] [--json eval/report.json] \
       [--shape S] [--clip C] [--fail-under TIER=RATE] [--no-gate]

# CAPTURE — live, BILLED, creds-gated (exit 2 without creds). The human runs this once after merge.
python eval/run_eval.py capture [--only X] [--limit N] [--force] [--dry-run]
```

Run both from `api/` (the bare `python eval/run_eval.py ...` form bootstraps its own `sys.path`).

## Tiered rubric (D4)

A strict **monotone ladder** — each tier presupposes the one below, so a case lands at the highest
tier it reaches:

| Tier | Meaning | Production code reused |
|------|---------|------------------------|
| T0 parsed | capture recorded a `raw_program` (not an `error`) with a program shape | `codegen.generate`'s post-parse guard |
| T1 schema-valid | `validation_errors` has no structural/size errors | `validate_program` |
| T2 semantically-valid | `validation_errors(...) == []` — the **full trust boundary** | `validate_program` |
| T3 executes-grounded | T2 AND `Interpreter.run(...).findings.grounded is True` | `interpreter` |
| T4 answer-correct | T3 AND the answer matches `expect.answer` under the shape's comparator | harness-specific |

Replay pushes every fixture's **untrusted** `raw_program` through `validation_errors` **before**
`Interpreter.run` — the same order `server.build_free_text_run_doc` uses. A fixture can never be
scored T2+ without passing the real gate; a validator-rejected program is never executed.

**Out-of-scope inversion (the load-bearing call).** A case with `expect.outcome == "ungrounded"`
(every out-of-scope case + the #23 absent-subject case) **PASSES** when the run grounds out honestly
(`grounded:false`, any reason) **OR** is cleanly rejected, and **FAILS** only when it grounds a
fabricated answer. The harness rewards calibrated refusal, not just confident answers.

## Ground truth — derived, never hand-typed

Count and presence answers are **derived** by running the pinned `examples/*_program.json` over the
committed cache (`bank.derive_ground_truth`, guarded by `test_count_ground_truth_derived_from_pinned`)
so the answer key can't drift from the cache. Canonical case texts are verbatim from
`canned.QUERIES`. The five known answers: hero #10 → `Yes`, count → `5`, #7 → `Yes`, scorer → `7`,
#23 → honestly ungrounded.

## Determinism disclaimer

`temperature=0` does **not** guarantee bit-identical gpt-4o output (provider nondeterminism, MoE
routing, fingerprint drift). **Fixtures are a frozen snapshot; replay scores the snapshot, not a
fresh call.** Replay is fully deterministic (pure parse/validate/execute over committed bytes +
caches), so the CI gate never flakes. The harness never asserts golden-program equality — it scores
coarse tier outcomes, and a re-capture shift surfaces as a `report.json` diff.

## Prompt-version invalidation

Each fixture records `prompt_version = sha256(build_system_message(clip))[:16]` —
`build_system_message` is the full prompt surface the model sees (base rules + the per-clip injected
block; it already begins with `SYSTEM_PROMPT`). Replay compares it against the current hash:

- **Uniform all-stale** (the honest state after a prompt edit) → the gate downgrades to **advisory**
  (prints the report, exits 0) so an editor without creds is not red-CI'd before they can re-capture.
- **Partial staleness** is suspicious, not excludable: stale fixtures count as **failures** beyond
  `max_stale` (default 0). The gate can be *bypassed* only uniformly; it can never be *weakened*
  case-by-case. `min_fixtures` catches deletion.

A prompt edit therefore *forces* a re-capture — the correct coupling.

## Escape discipline

The bank deliberately contains hostile strings (injection payloads, RTL/unicode noise). The report
renders all `question`/answer text **repr-escaped** so a payload can't garble CI logs or diffs.

> **Doctored fixtures are a code-review concern.** Fixtures are committed, attacker-editable files
> whose `prompt_version` is self-reported. Partial staleness fails the gate and `min_fixtures`
> catches deletion, but a doctored-but-valid `raw_program` under the current hash remains possible —
> as it is for any committed test fixture. Review fixture diffs like any other test data.

## Live-capture runbook (human, once after merge — the only billed step)

> Prereq: `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_KEY` (+ `AZURE_OPENAI_DEPLOYMENT`, default
> `gpt-4o`) in `api/.env` — the gpt-4o deployment `codegen_probe.py` uses. The harness, its tests,
> and the CI gate are already green against the committed **seed** fixtures before this step.

```bash
cd api
source .venv/bin/activate

# 0) Smoke creds + the harness on ONE case (cheap, sub-cent):
python scripts/codegen_probe.py --question "How many players are visible?"   # exit 0 = creds good; 2 = fix .env
python eval/run_eval.py capture --limit 1
python eval/run_eval.py replay            # confirm it scores + reports

# 1) Dry preview of cost (no calls): prints "will make N live calls (~$X)"
python eval/run_eval.py capture --dry-run

# 2) Full capture (BILLS ~$0.35–$0.50; overwrites seed fixtures with real gpt-4o output; resume-safe):
python eval/run_eval.py capture           # ~50 calls, budget-capped; re-run after a crash skips done cases

# 3) Score the real fixtures + regenerate the committed report:
python eval/run_eval.py replay --report eval/report.md --json eval/report.json

# 4) Review eval/report.md (per-shape rates, overclaim/failure rows), then commit the baseline:
git add eval/fixtures eval/report.md eval/report.json
git commit -m "eval: capture gpt-4o codegen fixtures + baseline report"

# 5) (optional) Ratchet eval/thresholds.json to ~5pp under the measured rates, then commit.
```

Cost stays an **offline estimate** (`usage` is not recorded — the harness adds no production usage
seam); verify current gpt-4o pricing before the run.

## Regenerating the seed fixtures

The committed `fixtures/*.json` are **generated**, not hand-authored — `seed_fixtures.py` lifts the
pinned `examples/*_program.json` templates and retargets the final `answer` step's question. After a
bank edit, regenerate and commit:

```bash
python eval/seed_fixtures.py      # rewrites eval/fixtures/*.json from the current bank
```

`test_seed_fixtures_regenerate_match` fails if the committed seeds drift from a regeneration.
