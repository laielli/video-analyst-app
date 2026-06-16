# Codegen eval report

- bank_version: `3`
- prompt_version: `d330406cc7cd77ea`
- model: `gpt-4o`  ·  mode: `replay`  ·  captured_at: `2026-06-13T00:43:13Z`
- total cases: 72  ·  in-scope: 0  ·  out-of-scope: 0  ·  safety: 0
- STALE fixtures: 72  ·  MISSING fixtures: 0
- estimated capture cost (offline est., usage is null): ~$0.45
- prompt size (chars): bernabeu-counter=6914, single-goal=6884
- total raw_program bytes: 33934

## Tier funnel (in-scope)

T0_parsed 0/0 → T1_schema_valid 0/0 → T2_semantic_valid 0/0 → T3_grounded 0/0 → T4_answer_correct 0/0

- out-of-scope honest-refusal rate: 0/0 (0.00)
- adversarial-safe rate: 0/0 (0.00)

## Per-shape

| group | n | T2 | T3 | T4 | pass |
|---|---|---|---|---|---|

## Per-clip

| group | n | T2 | T3 | T4 | pass |
|---|---|---|---|---|---|

## Per-bucket (paraphrase distance)

| group | n | T2 | T3 | T4 | pass |
|---|---|---|---|---|---|

## Failure detail

| case_id | reached | detail | flag |
|---|---|---|---|
| `adv-injection-empty-program` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `adv-injection-exec-op` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `adv-injection-fps-override` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `adv-injection-huge-window` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `adv-injection-reveal-prompt` | STALE | stale fixture (prompt_version edf7859f70e1e410) | STALE |
| `adv-injection-scene-fps-override` | STALE | stale fixture (prompt_version edf7859f70e1e410) | STALE |
| `adv-unicode-rtl-noise` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `adv-zero-width-count-bait` | STALE | stale fixture (prompt_version edf7859f70e1e410) | STALE |
| `bernabeu-count-canonical` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `bernabeu-count-far-1` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `bernabeu-count-far-2` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `bernabeu-count-far-3` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `bernabeu-count-far-4` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `bernabeu-count-near-1` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `bernabeu-count-near-2` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `bernabeu-count-near-3` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `bernabeu-first-goal-23-canonical` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `bernabeu-first-goal-23-far-1` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `bernabeu-first-goal-23-near-1` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `bernabeu-first-goal-23-near-2` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `bernabeu-first-goal-7-canonical` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `bernabeu-first-goal-7-far-1` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `bernabeu-first-goal-7-far-2` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `bernabeu-first-goal-7-far-3` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `bernabeu-first-goal-7-near-1` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `bernabeu-first-goal-7-near-2` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `bernabeu-presence-anchor` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `bernabeu-presence-far-1` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `bernabeu-presence-far-2` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `bernabeu-presence-near-1` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `bernabeu-presence-near-2` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `bernabeu-scorer-number-canonical` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `bernabeu-scorer-number-far-1` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `bernabeu-scorer-number-far-2` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `bernabeu-scorer-number-far-3` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `bernabeu-scorer-number-near-1` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `bernabeu-scorer-number-near-2` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `degenerate-all-caps` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `degenerate-question-mark` | STALE | stale fixture (prompt_version edf7859f70e1e410) | STALE |
| `degenerate-unicode-digits` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `oos-absent-99-bernabeu` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `oos-ballspeed-bernabeu` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `oos-coach-bernabeu` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `oos-commentary-singlegoal` | STALE | stale fixture (prompt_version edf7859f70e1e410) | STALE |
| `oos-crowd-size-singlegoal` | STALE | stale fixture (prompt_version edf7859f70e1e410) | STALE |
| `oos-emotion-bernabeu` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `oos-finalscore-bernabeu` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `oos-formation-bernabeu` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `oos-halftime-score-bernabeu` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `oos-jersey-color-singlegoal` | STALE | stale fixture (prompt_version edf7859f70e1e410) | STALE |
| `oos-keeper-foot-singlegoal` | STALE | stale fixture (prompt_version edf7859f70e1e410) | STALE |
| `oos-offside-bernabeu` | STALE | stale fixture (prompt_version a3cca197c2ee4155) | STALE |
| `oos-referee-singlegoal` | STALE | stale fixture (prompt_version edf7859f70e1e410) | STALE |
| `oos-stadium-singlegoal` | STALE | stale fixture (prompt_version edf7859f70e1e410) | STALE |
| `oos-weather-singlegoal` | STALE | stale fixture (prompt_version edf7859f70e1e410) | STALE |
| `single-goal-first-goal-canonical` | STALE | stale fixture (prompt_version edf7859f70e1e410) | STALE |
| `single-goal-first-goal-far-1` | STALE | stale fixture (prompt_version edf7859f70e1e410) | STALE |
| `single-goal-first-goal-far-2` | STALE | stale fixture (prompt_version edf7859f70e1e410) | STALE |
| `single-goal-first-goal-far-3` | STALE | stale fixture (prompt_version edf7859f70e1e410) | STALE |
| `single-goal-first-goal-near-1` | STALE | stale fixture (prompt_version edf7859f70e1e410) | STALE |
| `single-goal-first-goal-near-2` | STALE | stale fixture (prompt_version edf7859f70e1e410) | STALE |
| `single-goal-presence-anchor` | STALE | stale fixture (prompt_version edf7859f70e1e410) | STALE |
| `single-goal-presence-far-1` | STALE | stale fixture (prompt_version edf7859f70e1e410) | STALE |
| `single-goal-presence-far-2` | STALE | stale fixture (prompt_version edf7859f70e1e410) | STALE |
| `single-goal-presence-far-3` | STALE | stale fixture (prompt_version edf7859f70e1e410) | STALE |
| `single-goal-presence-near-1` | STALE | stale fixture (prompt_version edf7859f70e1e410) | STALE |
| `single-goal-presence-near-2` | STALE | stale fixture (prompt_version edf7859f70e1e410) | STALE |
| `single-goal-scene-canonical` | STALE | stale fixture (prompt_version edf7859f70e1e410) | STALE |
| `single-goal-scene-far-1` | STALE | stale fixture (prompt_version edf7859f70e1e410) | STALE |
| `single-goal-scene-far-2` | STALE | stale fixture (prompt_version edf7859f70e1e410) | STALE |
| `single-goal-scene-near-1` | STALE | stale fixture (prompt_version edf7859f70e1e410) | STALE |
| `single-goal-scene-near-2` | STALE | stale fixture (prompt_version edf7859f70e1e410) | STALE |

---

_Scores reflect the FROZEN capture, not a fresh call. temperature=0 is not bit-deterministic (provider-side MoE routing / fingerprint drift), so the harness scores coarse tier outcomes over recorded model output — never golden-program equality. A re-capture shift surfaces as a report.json diff._
