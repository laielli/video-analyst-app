# Codegen evaluation harness (`api/eval/`)

The calibration layer for **ViperGPT Layer 1** (program generation). It measures the codegen prompt
(`codegen.SYSTEM_PROMPT` + per-clip `hint`) against **real gpt-4o output** with a tiered scoring
rubric — instead of trusting that the prompt still works after every edit.

The pytest suite proves the *plumbing* (validate → execute → ground) with a mocked model. This
harness answers the question the mocks can't: **does gpt-4o actually emit valid, grounded, correct
programs for the questions a user types?**

## Two modes (one scoring path)

```bash
# REPLAY (free, CI-safe, no creds, no network) — implementers + CI run this:
python eval/run_eval.py replay [--report eval/report.md] [--json eval/report.json]
                               [--shape S] [--clip C] [--fail-under <tier>=<rate>] [--no-gate]

# CAPTURE (live, billed, creds required) — the human runs this once after merge:
python eval/run_eval.py capture [--only <case_id|shape|clip>] [--limit N] [--force] [--dry-run]
```

- **replay** loads each bank case's recorded fixture, pushes its *untrusted* `raw_program` through
  the production `validation_errors` gate, executes it over the committed clip cache, scores it, and
  writes the report. Fully deterministic — the CI gate runs this and never flakes.
- **capture** is the only billed step. It is **creds-gated**: with `AZURE_OPENAI_*` unset it prints a
  setup hint and exits **2** (never a silent no-op). It calls `codegen.generate` **directly**,
  bypassing the server (so it never touches `MAX_CODEGEN_CALLS`), and records the raw model output as
  one fixture per case. Resume-by-default; `--force` disables resume-skip (not the cost ceiling).

Exit codes mirror `scripts/codegen_probe.py`: `0` ok / threshold met, `1` threshold missed,
`2` setup problem (no creds on capture).

## The tiered rubric (D4)

A strict monotone ladder — a case lands at the highest tier it reaches:

| Tier | Meaning | How |
|---|---|---|
| REJECTED | no output / failed parse or validate | error tag, or `validation_errors` non-empty |
| T0 parsed | capture recorded a `raw_program` list | (not an `error`) |
| T1 schema-valid | no structural/size-class validator errors | one `validation_errors` call |
| T2 semantically-valid | `validation_errors == []` (the full server gate) | the trust boundary |
| T3 executes-grounded | T2 AND `findings.grounded is True` | `Interpreter.run` |
| T4 answer-correct | T3 AND the answer matches ground truth | shape comparator |

T1/T2 reuse `validate_program.validation_errors`, T3 reuses `interpreter.Interpreter`, T4 is the only
harness-specific logic. Ground truth is **deterministic** (5 known answers over committed caches) —
no LLM judge. Count/presence answers are **derived** from the pinned `examples/*_program.json`, never
hand-typed (`bank.derive_ground_truth`).

**Out-of-scope cases invert the ladder.** For a case that expects `outcome: "ungrounded"` (every
out-of-scope / injection / degenerate case, and the absent-subject #23 cases), an honest
`grounded:false` **or** a clean validator rejection is a **PASS**; a `grounded:true` with a fabricated
answer is a **FAIL** (the dangerous overclaim). The harness rewards *calibrated refusal*.

## Fixtures (D2)

One file per case at `eval/fixtures/<case_id>.json`, committed. A fixture records the **raw,
untrusted** model output (`raw_program`) — not the validated program, not the run-doc — so replay
re-runs the *real* trust boundary on every fixture. A fixture can never smuggle an unvalidated
program into a "pass". `usage` is reserved and always `null` (no production usage seam — Resolved #1).

Each fixture carries a `prompt_version = sha256(build_system_message(clip))[:16]`. Replay compares it
to the *current* prompt hash:
- match → scored normally;
- mismatch → marked **STALE** (D6 stale policy below).

**Seed fixtures.** The committed `fixtures/*.json` are **generated** seeds
(`seed_fixtures.py`), not real captures: each in-scope seed copies the matching pinned program and
retargets only the final `answer` step's question to the case text, so it grounds to the cell's
expected answer. They make replay + tests + the CI gate green creds-free. The human capture
**overwrites** them with real gpt-4o output; `git diff eval/report.json` then surfaces any case the
real model fails that the seed passed.

## CI gate (D6)

One replay-only step in the existing `python` job, after pytest:

```yaml
- name: Codegen eval (replay-mode regression gate)
  working-directory: api
  run: python eval/run_eval.py replay --fail-under T2_semantic_valid=1.0 --fail-under T4_answer_correct=0.85
```

Floors live in `eval/thresholds.json` (a one-line ratchet). The gate fails when an in-scope tier rate
falls below its floor, fixtures are missing, or **partial staleness** exceeds `max_stale`. The
**only** bypass is *uniform* staleness: if **all** fixtures are stale (an honest prompt edit), the
gate downgrades to advisory (exit 0) so an editor without creds isn't red-CI'd before re-capturing.
Partial staleness never excludes a case — it FAILS, so a PR author can't silence one regressing case
by mistyping its `prompt_version`. A doctored-but-valid `raw_program` under a current hash remains
possible (as for any committed fixture) and is a code-review concern.

## Determinism

`temperature=0` does **not** guarantee bit-identical gpt-4o output. **Fixtures are a frozen
snapshot; replay scores the snapshot, not a fresh call.** The harness never asserts golden-program
equality — it scores coarse *tier outcomes*. Residual jitter rarely flips a tier; when it does, the
report's per-case diff shows it.

## Escape discipline

The bank deliberately carries hostile strings (injection payloads, RTL/unicode/control noise). The
report escapes (`repr`-style) every echoed `question`/answer/detail so a payload can't garble CI logs
or `git diff`s.

## Live-capture runbook (the human runs this once, after merge)

> Prereq: `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_KEY` (+ `AZURE_OPENAI_DEPLOYMENT`, default `gpt-4o`)
> in `api/.env` — the same gpt-4o deployment `codegen_probe.py` uses. This is the **only** billed step.

```bash
cd api
source .venv/bin/activate

# 0) Smoke creds + the harness on ONE case (cheap, sub-cent):
python scripts/codegen_probe.py --question "How many players are visible?"   # exit 0 = creds good
python eval/run_eval.py capture --limit 1
python eval/run_eval.py replay            # confirm it scores + reports

# 1) Dry preview of cost (no calls): prints "will make N live calls (~$X)"
python eval/run_eval.py capture --dry-run

# 2) Full capture (BILLS real tokens; overwrites seed fixtures; resume-safe ~$0.35–$0.50):
python eval/run_eval.py capture

# 3) Score the real fixtures + regenerate the committed report:
python eval/run_eval.py replay --report eval/report.md --json eval/report.json

# 4) Review eval/report.md (per-shape rates, overclaim/failure rows), then commit:
git add eval/fixtures eval/report.md eval/report.json
git commit -m "eval: capture gpt-4o codegen fixtures + baseline report"

# 5) (optional) Ratchet eval/thresholds.json to ~5pp under the measured rates, then commit.
```

## Files

- `question_bank.json` / `question_bank.schema.json` — the ~57-case bank + its schema.
- `bank.py` — loader, `prompt_version`, `derive_ground_truth`.
- `fixtures.py` — fixture IO (atomic, one file per case).
- `scoring.py` — the tiered rubric (reuses `validate_program` + `interpreter`).
- `report.py` — the dual `report.{json,md}` builder.
- `seed_fixtures.py` — the creds-free seed generator.
- `run_eval.py` — the `capture`/`replay` CLI + the threshold gate.
- `thresholds.json` — committed CI floors.
- `fixtures/*.json` — committed seed fixtures (overwritten by the human capture).
- `report.{md,json}` — the latest reviewable snapshot (committed).
