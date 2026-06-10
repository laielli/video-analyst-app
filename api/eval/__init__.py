"""Codegen evaluation harness — measure the prompt with real model output, not vibes.

This package is the calibration layer for ViperGPT Layer 1 (program generation): a versioned
question bank, a capture/replay CLI, a tiered scoring rubric reusing the production
validate->execute path, a diffable report, and a CI threshold gate. See README.md for the
capture/replay/thresholds workflow and the live-capture runbook.

Note on the package name: `eval/` shadows the Python builtin `eval()` only as a *package path*
(`import eval.run_eval`); no module here ever calls the bare `eval()` builtin, so the shadow is
inert.
"""
