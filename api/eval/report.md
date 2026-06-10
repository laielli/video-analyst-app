# Codegen eval report

- bank_version: 1
- prompt_version: f3a5ad27035af279
- captured_at: 1970-01-01T00:00:00Z
- model: seed
- mode: replay
- total cases: 53 (in-scope 29, out-of-scope 14, injection 5, degenerate 5)
- estimated capture cost (offline): ~$0.33 (53 calls @ ~900in/400out tokens)  (usage is null — Resolved #1; estimate only)
- prompt size [bernabeu-counter]: 3017 chars
- prompt size [single-goal]: 2987 chars
- total raw_program bytes: 23248
- STALE fixtures: 0    MISSING fixtures: 0

## Tier funnel (in-scope)

`T0_parsed 29/29 -> T1_schema 29/29 -> T2_semantic 29/29 -> T3_grounded 29/29 -> T4_correct 29/29`

- out-of-scope honest-refusal: 14/14 (rate 1.00)
- adversarial/injection safe: 5/5 (rate 1.00)
- degenerate safe: 5/5 (rate 1.00)

## Per-shape (in-scope T2/T3/T4 rates)

| shape | n | T2 | T3 | T4 |
|---|---|---|---|---|
| count | 6 | 1.00 | 1.00 | 1.00 |
| first-goal | 12 | 1.00 | 1.00 | 1.00 |
| presence | 5 | 1.00 | 1.00 | 1.00 |
| scorer-number | 6 | 1.00 | 1.00 | 1.00 |

## Per-clip (in-scope T2/T3/T4 rates)

| clip | n | T2 | T3 | T4 |
|---|---|---|---|---|
| bernabeu-counter | 18 | 1.00 | 1.00 | 1.00 |
| single-goal | 11 | 1.00 | 1.00 | 1.00 |

## Per-bucket / paraphrase-distance (pass rate)

| bucket | n | pass rate |
|---|---|---|
| canonical | 5 | 1.00 |
| near | 10 | 1.00 |
| far | 14 | 1.00 |
| out_of_scope | 14 | 1.00 |
| injection | 5 | 1.00 |
| degenerate | 5 | 1.00 |

## Failure / below-target detail

_All cases passed at or above their target tier._

---

_Question/answer text is rendered repr-escaped: the bank deliberately contains hostile strings (injection payloads, RTL/unicode noise) and raw rendering could garble CI logs and diffs._

_Scores reflect the FROZEN capture (recorded gpt-4o output), not a fresh model call. temperature=0 does not guarantee bit-identical output; replay is deterministic over the committed fixtures + caches. Re-capture on a prompt edit and review the report.json diff._
