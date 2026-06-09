# Codegen eval report

- bank_version: `1`
- prompt_version (current): `f3a5ad27035af279`
- captured_at: `1970-01-01T00:00:00Z`
- model: `seed`
- mode: `replay`
- total cases: 60 (in-scope 35, out-of-scope 20, adversarial 5)
- estimated capture cost: ~$0.38 (offline estimate; `usage` is null per Resolved #1)
- prompt size (chars): bernabeu-counter=3017, single-goal=2987
- total raw_program bytes: 30084
- STALE fixtures: 0; MISSING fixtures: 0

## Tier funnel (in-scope)

`T0_parsed 35/35 -> T1_schema_valid 35/35 -> T2_semantic_valid 35/35 -> T3_executes_grounded 35/35 -> T4_answer_correct 35/35`

- out-of-scope honest-refusal rate: 1.00
- adversarial safe rate: 1.00

## Per-shape (in-scope tier rates)

| shape | n | passed | T2 | T3 | T4 |
|---|---|---|---|---|---|
| adversarial | 5 | 1.00 | 0.00 | 0.00 | 0.00 |
| count | 6 | 1.00 | 1.00 | 1.00 | 1.00 |
| degenerate | 3 | 1.00 | 1.00 | 1.00 | 1.00 |
| first-goal | 16 | 1.00 | 1.00 | 1.00 | 1.00 |
| out-of-scope | 14 | 1.00 | 0.00 | 0.00 | 0.00 |
| presence | 10 | 1.00 | 1.00 | 1.00 | 1.00 |
| scorer-number | 6 | 1.00 | 1.00 | 1.00 | 1.00 |

## Per-clip (in-scope tier rates)

| clip | n | passed | T2 | T3 | T4 |
|---|---|---|---|---|---|
| bernabeu-counter | 40 | 1.00 | 1.00 | 1.00 | 1.00 |
| single-goal | 20 | 1.00 | 1.00 | 1.00 | 1.00 |

## Per-bucket / paraphrase distance

| bucket | n | passed | T2 | T3 | T4 |
|---|---|---|---|---|---|
| degenerate | 3 | 1.00 | 1.00 | 1.00 | 1.00 |
| in_scope_canonical | 6 | 1.00 | 1.00 | 1.00 | 1.00 |
| in_scope_far | 16 | 1.00 | 1.00 | 1.00 | 1.00 |
| in_scope_near | 12 | 1.00 | 1.00 | 1.00 | 1.00 |
| injection | 5 | 1.00 | 0.00 | 0.00 | 0.00 |
| out_of_scope | 18 | 1.00 | 0.00 | 0.00 | 0.00 |

## Failure detail (0)

_none — every case met its target tier._

---

_Scores reflect the frozen capture, not a fresh call. `temperature=0` does not guarantee bit-identical gpt-4o output; replay is deterministic over the recorded `raw_program` + fixed caches. A re-capture shift surfaces as a `report.json` diff, never silent. Hostile question text is escaped repr-style (the bank contains injection/unicode payloads)._
