# Codegen eval report

- bank_version: `1`
- prompt_version: `3aefdf60232f9d4f`
- captured_at: `2026-06-09T00:00:00Z`
- model: `seed`
- mode: `replay`
- total cases: 57
- estimated capture cost (offline): $0.36 (usage is null per Resolved #1)
- prompt size (chars): bernabeu-counter=3017, single-goal=2987
- total raw_program bytes: 28520
- STALE fixtures: 0 | MISSING fixtures: 0

## Tier funnel (in-scope groundable)

T0_parsed 34/34 -> T1_schema 34/34 -> T2_semantic 34/34 -> T3_grounded 34/34 -> T4_correct 34/34

- out-of-scope honest-refusal rate: 1.00 (23 cases)
- adversarial-safe rate: 1.00 (5 cases)

## Per-shape (T2 / T3 / T4 reach rates)

| shape | n | T2 | T3 | T4 |
|---|---|---|---|---|
| count | 6 | 1.00 | 1.00 | 1.00 |
| first-goal | 12 | 1.00 | 1.00 | 1.00 |
| presence | 10 | 1.00 | 1.00 | 1.00 |
| scorer-number | 6 | 1.00 | 1.00 | 1.00 |

## Per-clip (T2 / T3 / T4 reach rates)

| clip | n | T2 | T3 | T4 |
|---|---|---|---|---|
| bernabeu-counter | 23 | 1.00 | 1.00 | 1.00 |
| single-goal | 11 | 1.00 | 1.00 | 1.00 |

## Per-bucket / paraphrase-distance (pass rate)

| bucket | n | pass rate |
|---|---|---|
| canonical | 7 | 1.00 |
| far | 13 | 1.00 |
| injection | 5 | 1.00 |
| near | 17 | 1.00 |
| out_of_scope | 15 | 1.00 |

## Failure detail (0 case(s) below target)

_(no failures)_

_Determinism: scores reflect the FROZEN capture, not a fresh call. temperature=0 does not guarantee bit-identical gpt-4o output; this report scores recorded fixtures. Re-capture shifts surface as a report.json diff, never a silent change._
