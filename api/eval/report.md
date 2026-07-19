# Codegen eval report

- bank_version: `4`
- prompt_version: `1a12f4d66e08159b`
- model: `gpt-4o`  ·  mode: `replay`  ·  captured_at: `2026-07-19T22:26:54Z`
- total cases: 71  ·  in-scope: 42  ·  out-of-scope: 29  ·  safety: 8
- STALE fixtures: 0  ·  MISSING fixtures: 0
- estimated capture cost (offline est., usage is null): ~$0.44
- prompt size (chars): bernabeu-counter=6914, single-goal=6884
- total raw_program bytes: 33623

## Tier funnel (in-scope)

T0_parsed 42/42 → T1_schema_valid 42/42 → T2_semantic_valid 42/42 → T3_grounded 42/42 → T4_answer_correct 41/42

- out-of-scope honest-refusal rate: 29/29 (1.00)
- adversarial-safe rate: 8/8 (1.00)

## Per-shape

| group | n | T2 | T3 | T4 | pass |
|---|---|---|---|---|---|
| adversarial | 7 | 0.57 | 0.00 | 0.00 | 1.00 |
| count | 8 | 1.00 | 1.00 | 1.00 | 1.00 |
| degenerate | 3 | 1.00 | 0.33 | 0.33 | 1.00 |
| first-goal | 16 | 1.00 | 0.75 | 0.75 | 1.00 |
| out-of-scope | 15 | 1.00 | 0.00 | 0.00 | 1.00 |
| presence | 11 | 1.00 | 1.00 | 0.91 | 0.91 |
| scene | 5 | 0.80 | 0.80 | 0.80 | 1.00 |
| scorer-number | 6 | 1.00 | 1.00 | 1.00 | 1.00 |

## Per-clip

| group | n | T2 | T3 | T4 | pass |
|---|---|---|---|---|---|
| bernabeu-counter | 44 | 0.98 | 0.59 | 0.57 | 0.98 |
| single-goal | 27 | 0.89 | 0.59 | 0.59 | 1.00 |

## Per-bucket (paraphrase distance)

| group | n | T2 | T3 | T4 | pass |
|---|---|---|---|---|---|
| canonical | 8 | 1.00 | 0.88 | 0.88 | 1.00 |
| degenerate | 3 | 1.00 | 0.33 | 0.33 | 1.00 |
| injection | 8 | 0.50 | 0.00 | 0.00 | 1.00 |
| out_of_scope | 15 | 1.00 | 0.00 | 0.00 | 1.00 |
| paraphrase-far | 20 | 1.00 | 0.95 | 0.90 | 0.95 |
| paraphrase-near | 17 | 1.00 | 0.88 | 0.88 | 1.00 |

## Failure detail

| case_id | reached | detail | flag |
|---|---|---|---|
| `bernabeu-presence-far-2` | T3_grounded | T3 only (answer '3' != expected 'Yes') |  |

---

_Scores reflect the FROZEN capture, not a fresh call. temperature=0 is not bit-deterministic (provider-side MoE routing / fingerprint drift), so the harness scores coarse tier outcomes over recorded model output — never golden-program equality. A re-capture shift surfaces as a report.json diff._
