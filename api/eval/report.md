# Codegen eval report

- bank_version: `1`
- prompt_version (current): `3aefdf60232f9d4f,7dd70c3354541f33`
- captured_at: `seed`
- model: `seed`
- mode: `replay`
- total cases: 58 (in-scope 34, out-of-scope 24, adversarial 8)
- estimated capture cost: ~$0.36 (offline estimate — `usage` is null per Resolved #1)
- prompt size [bernabeu-counter]: 3017 chars
- prompt size [single-goal]: 2987 chars
- total raw_program bytes: 23597
- STALE fixtures: 0  ·  MISSING fixtures: 0

## Tier funnel (in-scope)

`T0_parsed 34/34 → T1_schema 34/34 → T2_semantic 34/34 → T3_grounded 34/34 → T4_correct 34/34`

- out-of-scope honest-refusal: 24/24 (rate 1.00)
- adversarial-safe: 8/8 (rate 1.00)

## Per-shape (in-scope)

| shape | n | T2 | T3 | T4 |
|---|---|---|---|---|
| count | 6 | 1.00 | 1.00 | 1.00 |
| first-goal | 12 | 1.00 | 1.00 | 1.00 |
| presence | 10 | 1.00 | 1.00 | 1.00 |
| scorer-number | 6 | 1.00 | 1.00 | 1.00 |

## Per-clip (in-scope)

| clip | n | T2 | T3 | T4 |
|---|---|---|---|---|
| bernabeu-counter | 23 | 1.00 | 1.00 | 1.00 |
| single-goal | 11 | 1.00 | 1.00 | 1.00 |

## Per-bucket (phrasing / paraphrase distance)

| bucket | n | pass rate |
|---|---|---|
| canonical | 7 | 1.00 |
| degenerate | 3 | 1.00 |
| injection | 5 | 1.00 |
| out_of_scope | 12 | 1.00 |
| paraphrase-far | 22 | 1.00 |
| paraphrase-near | 9 | 1.00 |

## Failure detail

_No cases below target tier._

_Determinism: scores reflect the FROZEN capture, not a fresh call. `temperature=0` does not guarantee bit-identical gpt-4o output; replay scores recorded `raw_program` over committed caches, so the gate never flakes. A re-capture shift surfaces as a `report.json` diff._
