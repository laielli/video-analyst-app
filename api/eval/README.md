# Codegen evaluation harness

Measure the codegen prompt with **real model output, not vibes**. The free-text product thesis
rests on one prompt (`codegen.SYSTEM_PROMPT` + `build_system_message(clip)` hint injection); this
harness is the calibration layer that catches a silent regression — a prompt edit, new primitive,
or model swap that drops a query shape's hit rate — before it ships.

It pairs a versioned **question bank** with two CLI modes that share one scoring path:

- **capture** (live, billed, creds-gated) records gpt-4o's program output per case into
  `fixtures/<case_id>.json` — the analogue of the committed `examples/*_cache.json`.
- **replay** (free, deterministic, CI-safe) pushes each recorded *untrusted* program through the
  **same production gate the server uses** (`validate_program.validation_errors` then
  `interpreter.Interpreter.run`), scores it on a monotone tier ladder, and writes a diffable report.

The harness **measures** the prompt; it never changes it. A failing tier is a finding for a
follow-up, not a fix here.

## Layout

| File | Role |
|---|---|
| `question_bank.json` | ~58 typed cases (clip × shape × phrasing/paraphrase + out-of-scope/injection/degenerate), each with a co-located ground-truth `expect` block. |
| `question_bank.schema.json` | JSON Schema for a bank case (the lint gate). |
| `bank.py` | `load_bank` / `filter_cases` / `prompt_version` / `derive_ground_truth`. |
| `fixtures.py` | `Fixture` dataclass + atomic fixture IO + `is_fresh`. |
| `scoring.py` | The tiered rubric (T0→T4 + the inverted out-of-scope contract). Reuses the production validate→execute path. |
| `report.py` | `build_report` → machine `report.json` + human `report.md`. |
| `seed_fixtures.py` | Generates the committed seed fixtures (the creds-free stand-in for a capture). |
| `run_eval.py` | The CLI (`capture` / `replay`) + the threshold gate. |
| `thresholds.json` | Committed CI floors (a one-line diff to ratchet). |
| `fixtures/*.json` | Committed recorded model output (seeds today; real gpt-4o after capture). |
| `report.{md,json}` | The latest committed report snapshot. |

## The tier ladder (scoring)

A strict monotone ladder — a case lands at the highest tier it reaches:

| Tier | Meaning |
|---|---|
| **T0 parsed** | capture recorded a `raw_program` (not an `error`). |
| **T1 schema-valid** | clears the structural/byte-cap gate (no structural-class `validation_errors`). |
| **T2 semantically-valid** | `validation_errors(...) == []` — the full server trust boundary. |
| **T3 executes-grounded** | T2 + `Interpreter.run(...)["findings"]["grounded"] is True`. |
| **T4 answer-correct** | T3 + `findings.answer` matches `expect.answer` under the shape's comparator. |

T1/T2 reuse `validate_program`; T3 reuses `interpreter`; T4's comparators (`exact` / `ci_contains`
/ `numeric` / `any_grounded`) are the only harness-specific logic.

**Out-of-scope inverts.** For a case whose `expect.outcome == "ungrounded"` (every out-of-scope,
injection, degenerate, and the #23 absent-subject case) the ladder inverts: an honest
`grounded == False` **or** a clean validator/runtime rejection is a **PASS**; an executed
`grounded == True` (a fabricated answer) is the **FAIL**. The harness rewards *calibrated refusal*,
which is the product's actual contract.

## Replay (free, CI-safe)

```bash
cd api
python eval/run_eval.py replay                 # score over committed fixtures, write the report, gate
python eval/run_eval.py replay --no-gate        # report only
python eval/run_eval.py replay --shape count    # filter
python eval/run_eval.py replay --fail-under T4_answer_correct=0.85   # override a floor
```

No creds, no network, no Azure client import. Deterministic over recorded `raw_program` + committed
caches, so the CI gate never flakes.

## The CI gate (D6)

A **calibrated threshold gate, replay-only**, in the existing `python` job (`thresholds.json`
floors + `--fail-under` overrides). It exits non-zero if an in-scope tier rate falls below its
floor, if `min_fixtures` is unmet, or on partial staleness beyond `max_stale`.

**Stale handling.** `prompt_version = sha256(build_system_message(clip))[:16]` is stamped into each
fixture; replay compares it to the current hash.

- **Uniform all-stale** (the honest state after a `SYSTEM_PROMPT` edit) → the gate downgrades to
  **advisory** (prints the report, exits 0) so an editor who lacks creds is not red-CI'd before they
  can re-capture.
- **Partial staleness** is suspicious, not excludable: stale fixtures count as **FAILED** gating
  outcomes beyond `max_stale` (default 0) — so a PR author cannot silence one regressing case by
  mistyping its `prompt_version`. The gate can be bypassed only uniformly, never weakened case-by-case.

## Capture (live, billed — the human runbook)

> Prereq: `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_KEY` (and `AZURE_OPENAI_DEPLOYMENT`, default
> `gpt-4o`) in `api/.env`, the same gpt-4o deployment `codegen_probe.py` uses. This is the **only**
> billed step. The harness, its tests, and the CI gate are already green against the committed seed
> fixtures before this step. A full ~58-call capture costs **well under $1**.

```bash
cd api
source .venv/bin/activate

# 0) Smoke creds + the harness on ONE case first (cheap):
python scripts/codegen_probe.py --question "How many players are visible?"   # exit 0 = creds good
python eval/run_eval.py capture --limit 1
python eval/run_eval.py replay            # confirm it scores + reports

# 1) Dry preview of cost (no calls):
python eval/run_eval.py capture --dry-run

# 2) Full capture (BILLS; overwrites seed fixtures with real gpt-4o output; resume-safe):
python eval/run_eval.py capture           # budget-capped at MAX_CAPTURE_CALLS; re-run skips done cases

# 3) Score the real fixtures + regenerate the committed report:
python eval/run_eval.py replay --report eval/report.md --json eval/report.json

# 4) Review eval/report.md, then commit the baseline:
git add eval/fixtures eval/report.md eval/report.json
git commit -m "eval: capture gpt-4o codegen fixtures + baseline report"

# 5) (optional) Ratchet eval/thresholds.json to ~5pp under the measured rates, then commit.
```

Capture conveniences: `--only <case_id|shape|clip>` narrows; `--limit N` caps calls this
invocation (clamped to the ceiling); `--force` disables resume-skip (it does **not** bypass the
ceiling); `--dry-run` prints the call count + cost and makes no calls. **Resume by default**: a case
whose fixture already exists at the current `prompt_version` is skipped unless `--force`.

## Determinism

`temperature=0` does **not** guarantee bit-identical gpt-4o output (provider-side nondeterminism,
MoE routing, fingerprint drift). The harness does not pretend otherwise: **fixtures are a frozen
snapshot; replay scores the snapshot, not a fresh call.** Replay is fully deterministic, so the gate
never flakes. The harness never asserts golden-program equality — it scores coarse tier outcomes,
and a re-capture shift surfaces as a clean `report.json` diff. The report carries a determinism
footer.

## Why fixtures store the raw (untrusted) program

A fixture stores the **raw model output** (`raw_program`), not the validated program and not the
run-doc. Replay re-runs the *real* `validation_errors` + `Interpreter.run` on it, so a fixture can
never smuggle an unvalidated program into a "pass". Storing a pre-validated program would bake the
current validator's verdict into the fixture and hide future regressions.

**Code-review note.** Fixtures are committed, hand-editable files whose `prompt_version` is
self-reported. The gate's anti-gaming rules (partial staleness fails; `min_fixtures` catches
deletion) close the obvious holes, but a doctored-but-valid `raw_program` under a current hash
remains a code-review concern — as it is for any committed test fixture.

## Escaping discipline

The bank deliberately carries hostile strings (injection payloads, RTL/unicode noise). The report
renders case `question` text and echoed answers **repr-style** (control/bidi chars escaped) so raw
rendering never garbles CI logs or diffs.
