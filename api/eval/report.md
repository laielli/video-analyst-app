# Codegen eval report

- bank_version: `1`
- prompt_version: `bec96ea8aefb6a6e`
- model: `gpt-4o`  ·  mode: `replay`  ·  captured_at: `2026-06-11T02:21:01Z`
- total cases: 60  ·  in-scope: 36  ·  out-of-scope: 24  ·  safety: 5
- STALE fixtures: 0  ·  MISSING fixtures: 0
- estimated capture cost (offline est., usage is null): ~$0.38
- prompt size (chars): bernabeu-counter=4914, single-goal=4884
- total raw_program bytes: 33843

## Tier funnel (in-scope)

T0_parsed 36/36 → T1_schema_valid 36/36 → T2_semantic_valid 36/36 → T3_grounded 19/36 → T4_answer_correct 15/36

- out-of-scope honest-refusal rate: 24/24 (1.00)
- adversarial-safe rate: 5/5 (1.00)

## Per-shape

| group | n | T2 | T3 | T4 | pass |
|---|---|---|---|---|---|
| adversarial | 5 | 0.60 | 0.00 | 0.00 | 1.00 |
| count | 6 | 1.00 | 0.00 | 0.00 | 0.00 |
| degenerate | 3 | 1.00 | 0.00 | 0.00 | 0.67 |
| first-goal | 16 | 1.00 | 0.75 | 0.75 | 1.00 |
| out-of-scope | 13 | 1.00 | 0.00 | 0.00 | 1.00 |
| presence | 11 | 1.00 | 0.09 | 0.09 | 0.09 |
| scorer-number | 6 | 1.00 | 1.00 | 0.33 | 0.33 |

## Per-clip

| group | n | T2 | T3 | T4 | pass |
|---|---|---|---|---|---|
| bernabeu-counter | 40 | 0.97 | 0.30 | 0.20 | 0.60 |
| single-goal | 20 | 0.95 | 0.35 | 0.35 | 0.75 |

## Per-bucket (paraphrase distance)

| group | n | T2 | T3 | T4 | pass |
|---|---|---|---|---|---|
| canonical | 7 | 1.00 | 0.43 | 0.43 | 0.57 |
| degenerate | 3 | 1.00 | 0.00 | 0.00 | 0.67 |
| injection | 5 | 0.60 | 0.00 | 0.00 | 1.00 |
| out_of_scope | 13 | 1.00 | 0.00 | 0.00 | 1.00 |
| paraphrase-far | 18 | 1.00 | 0.56 | 0.44 | 0.50 |
| paraphrase-near | 14 | 1.00 | 0.43 | 0.29 | 0.43 |

## Failure detail

| case_id | reached | detail | flag |
|---|---|---|---|
| `bernabeu-count-canonical` | T2_semantic_valid | T2 only (grounded:false, no-grounded-answer) |  |
| `bernabeu-count-far-1` | T2_semantic_valid | T2 only (grounded:false, no-grounded-answer) |  |
| `bernabeu-count-far-2` | T2_semantic_valid | T2 only (grounded:false, no-grounded-answer) |  |
| `bernabeu-count-far-3` | T2_semantic_valid | T2 only (grounded:false, no-grounded-answer) |  |
| `bernabeu-count-near-1` | T2_semantic_valid | T2 only (grounded:false, no-grounded-answer) |  |
| `bernabeu-count-near-2` | T2_semantic_valid | T2 only (grounded:false, no-grounded-answer) |  |
| `bernabeu-presence-anchor` | T2_semantic_valid | T2 only (grounded:false, no-grounded-answer) |  |
| `bernabeu-presence-far-1` | T2_semantic_valid | T2 only (grounded:false, no-grounded-answer) |  |
| `bernabeu-presence-far-2` | T2_semantic_valid | T2 only (grounded:false, no-grounded-answer) |  |
| `bernabeu-presence-near-1` | T2_semantic_valid | T2 only (grounded:false, no-grounded-answer) |  |
| `bernabeu-presence-near-2` | T2_semantic_valid | T2 only (grounded:false, no-grounded-answer) |  |
| `bernabeu-scorer-number-far-2` | T3_grounded | T3 only (answer 'Yes' != expected '7') |  |
| `bernabeu-scorer-number-far-3` | T3_grounded | T3 only (answer 'Yes' != expected '7') |  |
| `bernabeu-scorer-number-near-1` | T3_grounded | T3 only (answer 'Yes' != expected '7') |  |
| `bernabeu-scorer-number-near-2` | T3_grounded | T3 only (answer 'Yes' != expected '7') |  |
| `degenerate-all-caps` | T2_semantic_valid | T2 only (grounded:false, no-grounded-answer) |  |
| `single-goal-presence-anchor` | T2_semantic_valid | T2 only (grounded:false, no-grounded-answer) |  |
| `single-goal-presence-far-2` | T2_semantic_valid | T2 only (grounded:false, no-grounded-answer) |  |
| `single-goal-presence-far-3` | T2_semantic_valid | T2 only (grounded:false, no-grounded-answer) |  |
| `single-goal-presence-near-1` | T2_semantic_valid | T2 only (grounded:false, no-grounded-answer) |  |
| `single-goal-presence-near-2` | T2_semantic_valid | T2 only (grounded:false, no-grounded-answer) |  |

---

_Scores reflect the FROZEN capture, not a fresh call. temperature=0 is not bit-deterministic (provider-side MoE routing / fingerprint drift), so the harness scores coarse tier outcomes over recorded model output — never golden-program equality. A re-capture shift surfaces as a report.json diff._
