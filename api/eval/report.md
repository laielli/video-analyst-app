# Codegen eval report

- bank_version: `1`
- prompt_version: `f3a5ad27035af279`
- model: `seed`  ·  mode: `replay`  ·  captured_at: `seed`
- total cases: 60  ·  in-scope: 36  ·  out-of-scope: 24  ·  safety: 5
- STALE fixtures: 0  ·  MISSING fixtures: 0
- estimated capture cost (offline est., usage is null): ~$0.38
- prompt size (chars): bernabeu-counter=3017, single-goal=2987
- total raw_program bytes: 28787

## Tier funnel (in-scope)

T0_parsed 36/36 → T1_schema_valid 36/36 → T2_semantic_valid 36/36 → T3_grounded 36/36 → T4_answer_correct 36/36

- out-of-scope honest-refusal rate: 24/24 (1.00)
- adversarial-safe rate: 5/5 (1.00)

## Per-shape

| group | n | T2 | T3 | T4 | pass |
|---|---|---|---|---|---|
| adversarial | 5 | 0.00 | 0.00 | 0.00 | 1.00 |
| count | 6 | 1.00 | 1.00 | 1.00 | 1.00 |
| degenerate | 3 | 1.00 | 0.33 | 0.33 | 1.00 |
| first-goal | 16 | 1.00 | 0.75 | 0.75 | 1.00 |
| out-of-scope | 13 | 1.00 | 0.00 | 0.00 | 1.00 |
| presence | 11 | 1.00 | 1.00 | 1.00 | 1.00 |
| scorer-number | 6 | 1.00 | 1.00 | 1.00 | 1.00 |

## Per-clip

| group | n | T2 | T3 | T4 | pass |
|---|---|---|---|---|---|
| bernabeu-counter | 40 | 0.90 | 0.60 | 0.60 | 1.00 |
| single-goal | 20 | 0.95 | 0.60 | 0.60 | 1.00 |

## Per-bucket (paraphrase distance)

| group | n | T2 | T3 | T4 | pass |
|---|---|---|---|---|---|
| canonical | 7 | 1.00 | 0.86 | 0.86 | 1.00 |
| degenerate | 3 | 1.00 | 0.33 | 0.33 | 1.00 |
| injection | 5 | 0.00 | 0.00 | 0.00 | 1.00 |
| out_of_scope | 13 | 1.00 | 0.00 | 0.00 | 1.00 |
| paraphrase-far | 18 | 1.00 | 0.94 | 0.94 | 1.00 |
| paraphrase-near | 14 | 1.00 | 0.86 | 0.86 | 1.00 |

## Failure detail

_All cases passed._

---

_Scores reflect the FROZEN capture, not a fresh call. temperature=0 is not bit-deterministic (provider-side MoE routing / fingerprint drift), so the harness scores coarse tier outcomes over recorded model output — never golden-program equality. A re-capture shift surfaces as a report.json diff._
