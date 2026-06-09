"""Codegen evaluation harness (the calibration layer for ViperGPT Layer 1).

This package measures the codegen prompt (`codegen.SYSTEM_PROMPT` + per-clip hint) against real
gpt-4o output via a tiered scoring rubric, instead of trusting vibes. It is a tested measurement
*library* + a thin CLI (`run_eval.py`), a sibling of `interpreter/` — not a one-shot probe.

Two modes share one scoring path:
  - capture (live, billed, creds-gated, exit-2 on no creds) — writes recorded model-output fixtures.
  - replay  (free, CI-safe, no creds, no network)          — scores the recorded fixtures.

See `api/eval/README.md` for the capture/replay runbook, the determinism disclaimer, and the
prompt-version invalidation rule.

Package-name note: `eval/` shadows the Python builtin `eval()` only as a *package path*
(`import eval.run_eval`); no module here ever calls the bare `eval()` builtin, so the shadow is
inert.
"""
