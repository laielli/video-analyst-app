"""Codegen evaluation harness — measure the codegen prompt with real model output, not vibes.

This package is the calibration layer for ViperGPT Layer 1 (program generation). It holds a
versioned question bank, a capture/replay CLI, a tiered scoring rubric that reuses the *exact*
production validate -> execute path, a diffable report, and a CI threshold gate. See README.md
for the capture/replay runbook.

Package-name note: `eval/` shadows the Python builtin `eval()` only as an import path
(`import eval.run_eval`). No module here ever calls the bare `eval()` builtin, so the shadow is
inert. Verified importable under pytest's `pythonpath = . scripts`.
"""
