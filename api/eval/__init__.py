"""
Codegen evaluation harness — measure the codegen prompt with real model output, not vibes.

This is the calibration layer for ViperGPT Layer 1 (program generation). It pairs a versioned
question bank (`question_bank.json`) with two CLI modes (`run_eval.py capture` / `replay`):

  - capture (live, billed, creds-gated) records gpt-4o's program output per case into
    `fixtures/<case_id>.json` — the analogue of the committed `examples/*_cache.json`.
  - replay (free, deterministic, CI-safe) pushes each recorded *untrusted* program through the
    SAME production gate the server uses (`validate_program.validation_errors` +
    `interpreter.Interpreter`), scores it on a monotone tier ladder, and writes a diffable report.

The harness MEASURES the prompt; it never changes it. See `README.md` for the capture/replay
runbook, the determinism disclaimer, and the prompt-version invalidation rule.

Package-name note: `eval/` shadows the builtin `eval()` only as an import path; no module here
ever calls the bare `eval()` builtin, so the shadow is inert.
"""
