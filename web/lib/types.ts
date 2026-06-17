// Mirrors api/schema/run_doc.schema.json — the contract the backend streams.

export type Box = { x: number; y: number; w: number; h: number };
export type Tone = "azure" | "teal" | "green" | "amber" | "dim";

export type Overlay = {
  box: Box;
  label?: string;
  confidence?: number | null;
  tone: Tone;
  kind?: "box" | "crop" | "goal" | "ball";
};

export type StepStatus = "pending" | "active" | "done" | "empty" | "error";
export type Source = "live" | "cached" | "pinned";
// Provenance of the PROGRAM (Q1 -> A): "ungrounded" = a free-text query that couldn't be
// grounded and has no pinned fallback — the honest "couldn't ground this" state.
export type ProgramSource = "live" | "pinned" | "ungrounded";

export type StepResult = {
  id: string;
  op: string;
  producer?: string;
  status: StepStatus;
  source: Source;
  inputs?: string[];
  input_label?: string;
  output_label?: string;
  confidence?: number | null;
  evidence?: { frame_ts_ms: number; overlays: Overlay[] };
  note?: string;
};

export type ProgramStep = {
  id: string;
  op: string;
  args: Record<string, unknown>;
};

export type Clip = { id: string; width: number; height: number; duration_ms: number; fps?: number };

export type Findings = {
  // null when grounded === false (the honest "couldn't ground this" state, Q1 -> A).
  answer: string | null;
  verdict: string | null;
  // The ungrounded discriminator: false => no answer for the question asked; reason is a fixed
  // enum (e.g. "codegen-disabled", "no-grounded-answer"). A grounded run has grounded:true,
  // reason:null. May be absent on legacy/canned docs — treat missing as grounded.
  grounded?: boolean;
  reason?: string | null;
  supporting_step?: string;
  partial?: boolean;
  partial_note?: string;
};

export type Meta = {
  query: string;
  clip: Clip;
  program: ProgramStep[];
  total_steps: number;
  pace_ms: number;
  // Provenance of the program: "live" = compiled this run by Azure OpenAI codegen and
  // validated; "pinned" = the known-good fallback (no creds, or live failed — D-DR6);
  // "ungrounded" = a free-text query that couldn't be grounded (no pinned fallback).
  program_source?: ProgramSource;
  // Content-addressed permalink id, present ONLY on a grounded free-text run (fresh or replayed).
  // Its presence gates the "Copy run link" affordance; `${origin}/?run=<run_id>` is the share URL.
  run_id?: string;
  // True when this run-doc was served from the run cache (a free-text cache hit OR a permalink
  // replay) rather than freshly computed. Derived by the route at serve time, never persisted.
  cached?: boolean;
};

export type CatalogQuery = { id: string; text: string; clip: string };

// ---- Gallery (the static front-door curation; no /api/catalog coupling — see plan Resolved #4) ----
// One entry per canned query, committed in web/lib/curated.json and build-imported by gallery.ts.
// `shape` is the lowercase machine label the repo already uses (count/first-goal/scorer-number/
// scene); display text is minted only at render time. `clipLabel` is the human string from
// canned.CLIPS[clip].label — carried here so cards resolve labels with zero API coupling. Both
// are kept honest by api/tests/test_gallery_curation_parity.py.
export type CurationEntry = {
  id: string;
  text: string;
  clip: string;
  clipLabel: string;
  shape: string;
};

// A card's thumbnail: a committed still (`img`) or a token-built CSS placeholder for a clip with
// no committed frame (the bernabeu-counter cards — see plan Phase 2 / R1).
export type GalleryThumb = { kind: "img"; src: string } | { kind: "placeholder" };

// The presentational view-model GalleryCard consumes — fully resolved, no registry lookups in JSX.
export type GalleryCard = {
  queryId: string;
  clipId: string;
  clipLabel: string;
  text: string;
  shape: string;
  href: string;
  thumb: GalleryThumb;
};

// A shape section: a machine shape label + its cards (empty groups suppressed by groupByShape).
export type GallerySection = { shape: string; cards: GalleryCard[] };
