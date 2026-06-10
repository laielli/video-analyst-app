"""
Codegen evaluation harness — measure the codegen prompt with real model output, not vibes.

This is the calibration layer for ViperGPT Layer 1 (program generation): a versioned question
bank, a live `capture` of gpt-4o output into recorded fixtures, a tiered scoring rubric that
reuses the *exact* server validate -> execute path, a diffable report, and a replay-mode CI gate.

Two modes share one scoring path (see `run_eval.py`):
  - capture: live, billed, creds-gated (exit 2 without creds), writes fixtures. Human runs it.
  - replay:  recorded fixtures, free, deterministic, CI-safe. Implementers + CI run it.

Runbook + design notes: `api/eval/README.md`. The package name `eval` shadows the Python
builtin `eval()` only as a package PATH (`import eval.run_eval`); no module here ever calls the
bare `eval()` builtin, so the shadow is inert.
"""
