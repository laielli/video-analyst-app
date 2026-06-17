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

// ---- Gallery / demo front door -------------------------------------------------------------
// The static curation entry (one per canned query), build-imported from web/lib/curated.json and
// kept honest against canned.QUERIES + the eval bank by api/tests/test_gallery_curation_parity.py.
// `shape` is the lowercase MACHINE label the repo uses (count / first-goal / scorer-number /
// scene); the UI maps machine -> display text only at render time.
export type CurationEntry = {
  id: string;
  text: string;
  clip: string;
  clipLabel: string;
  shape: string;
};

// A token-driven thumbnail: a committed still, or a CSS placeholder for clips with no still.
export type GalleryThumb =
  | { kind: "img"; src: string }
  | { kind: "placeholder" };

// The per-card view-model GalleryCard consumes (assembled by web/lib/gallery.ts, no DOM).
export type GalleryCardVM = {
  queryId: string;
  clipId: string;
  clipLabel: string;
  text: string;
  shape: string; // lowercase machine label
  href: string; // /?query=<id> (URL-encoded)
  thumb: GalleryThumb;
};

// A shape section: a machine shape label + its cards, in deterministic order (empty groups
// suppressed). The UI title-cases `shape` for display.
export type GallerySection = { shape: string; cards: GalleryCardVM[] };
