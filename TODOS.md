# TODOS

Design debt deferred from /plan-design-review (2026-05-31). Review scope was the "visual
core" (passes 1–4 + mockups); the items below (passes 5–6 + design system) were agreed-deferred
at review start, not skipped.

## Responsive / mobile layout
- **What:** A real layout for <1024px (+ a tablet breakpoint), or an explicit desktop gate.
- **Why:** The two-column PROGRAM+EVIDENCE screen has no defined sub-desktop behavior; today it would reflow arbitrarily.
- **Pros:** Survives a recruiter opening the link on a phone; no broken first impression.
- **Cons:** Genuine work (the code panel doesn't shrink gracefully); near-zero payoff for the laptop/projector demo path.
- **Context:** Decision at review — desktop-first ≥1024px; below that, a "best viewed on desktop" notice instead of a broken reflow. Revisit if the public URL gets real mobile traffic. **Reaffirmed 2026-05-31** in the /plan-eng-review delta review (the throwaway `finalized.html` added 900/620px breakpoints, but that's an incidental motion-reference detail; the real Next.js build keeps the desktop-gate).
- **Depends on:** T-D1 (base layout) first.

## Full accessibility pass
- **What:** Keyboard nav for run controls + step scrubber, ARIA landmarks for the 3 zones, screen-reader step narration ("step 4 of 7, read_text, found 10"), focus order, full contrast audit.
- **Why:** A portfolio piece failing basic a11y reads badly to senior engineers — and it's the right thing.
- **Pros:** Shows craft; keyboard-driven step advance is genuinely nice when presenting.
- **Cons:** Narrating an animated canvas overlay to a screen reader is non-trivial.
- **Context:** Mitigations already in the build — animation is motion-optional (prefers-reduced-motion, D-DR8); token colors are contrast-aware (D-DR9). This is the full pass on top.
- **Depends on:** T-D1, T-D2 (layout + run engine) first.

## DESIGN.md / formal design system — DONE (2026-05-31)
- **What:** Promote the inline locked tokens (D-DR9) into a DESIGN.md.
- **Status:** Done. `DESIGN.md` created at repo root and reconciled to D-DR9 (Fraunces +
  Hanken Grotesk + IBM Plex Mono; paper/ink/deep-teal + functional status colors; sharp
  2–4px radius + hairlines). It mirrors the design doc and defers to it as source of truth.
  (Created during /design-html; corrected to D-DR9 during the /plan-eng-review delta review —
  the first draft had drifted to Inter/JetBrains, the banned slop tell.)
