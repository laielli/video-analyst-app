# Codegen eval report

- bank_version: `2`
- prompt_version: `474f8438f9fb7159`
- model: `gpt-4o`  ·  mode: `replay`  ·  captured_at: `2026-06-13T00:43:13Z`
- total cases: 66  ·  in-scope: 38  ·  out-of-scope: 28  ·  safety: 7
- STALE fixtures: 0  ·  MISSING fixtures: 0
- estimated capture cost (offline est., usage is null): ~$0.41
- prompt size (chars): bernabeu-counter=6860, single-goal=6830
- total raw_program bytes: 32185

## Tier funnel (in-scope)

T0_parsed 38/38 → T1_schema_valid 38/38 → T2_semantic_valid 38/38 → T3_grounded 38/38 → T4_answer_correct 38/38

- out-of-scope honest-refusal rate: 28/28 (1.00)
- adversarial-safe rate: 7/7 (1.00)

## Per-shape

| group | n | T2 | T3 | T4 | pass |
|---|---|---|---|---|---|
| adversarial | 7 | 0.71 | 0.00 | 0.00 | 1.00 |
| count | 8 | 1.00 | 1.00 | 1.00 | 1.00 |
| degenerate | 3 | 1.00 | 0.33 | 0.33 | 1.00 |
| first-goal | 16 | 1.00 | 0.75 | 0.75 | 1.00 |
| out-of-scope | 15 | 1.00 | 0.00 | 0.00 | 1.00 |
| presence | 11 | 1.00 | 1.00 | 1.00 | 1.00 |
| scorer-number | 6 | 1.00 | 1.00 | 1.00 | 1.00 |

## Per-clip

| group | n | T2 | T3 | T4 | pass |
|---|---|---|---|---|---|
| bernabeu-counter | 44 | 0.98 | 0.59 | 0.59 | 1.00 |
| single-goal | 22 | 0.95 | 0.55 | 0.55 | 1.00 |

## Per-bucket (paraphrase distance)

| group | n | T2 | T3 | T4 | pass |
|---|---|---|---|---|---|
| canonical | 7 | 1.00 | 0.86 | 0.86 | 1.00 |
| degenerate | 3 | 1.00 | 0.33 | 0.33 | 1.00 |
| injection | 7 | 0.71 | 0.00 | 0.00 | 1.00 |
| out_of_scope | 15 | 1.00 | 0.00 | 0.00 | 1.00 |
| paraphrase-far | 19 | 1.00 | 0.95 | 0.95 | 1.00 |
| paraphrase-near | 15 | 1.00 | 0.87 | 0.87 | 1.00 |

## Failure detail

_All cases passed._

---

_Scores reflect the FROZEN capture, not a fresh call. temperature=0 is not bit-deterministic (provider-side MoE routing / fingerprint drift), so the harness scores coarse tier outcomes over recorded model output — never golden-program equality. A re-capture shift surfaces as a report.json diff._
